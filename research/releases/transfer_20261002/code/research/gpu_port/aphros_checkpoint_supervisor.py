#!/usr/bin/env python3
"""Run Aphros in restartable segments and maintain atomic A/B checkpoints."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

try:
    import fcntl
except ImportError:  # Unit tests and config generation also run on Windows.
    fcntl = None


CHECKPOINT_FIELDS = (
    "vf",
    "vx",
    "vy",
    "p",
    "fluxx",
    "fluxy",
    "fluxxp",
    "fluxyp",
    "fluxeb",
    "ebvf",
)
DUMP_RE = re.compile(r"Dump\s+n=(?P<index>\d+)\s+t=(?P<time>[-+0-9.eE]+)")
STEP_RE = re.compile(
    r"STEP=(?P<step>\d+)\s+t=(?P<time>[-+0-9.eE]+)\s+dt=(?P<dt>[-+0-9.eE]+)"
)
SET_DOUBLE_RE = re.compile(
    r"^\s*set\s+double\s+(?P<name>\S+)\s+(?P<value>\S+)", re.MULTILINE
)
RAW_DTYPE = np.dtype("<f8")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_log_guard(
    path: Path, target_time: float, minimum_dt: float = 0.0
) -> dict | None:
    """Detect native aborts and timestep collapse between field dumps.

    A native abort can exit with code zero and leave the last healthy dump.
    Never use that dump or the exit code to certify a successful trajectory.
    """
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        tail = stream.read().decode(errors="replace")
    if "abortvel exceeded" in tail or "nabort = " in tail:
        return {
            "valid": False,
            "violations": ["native solver abort reported in run.log"],
            "source": "runtime_log",
        }
    # Ignore an incomplete last token while the solver is writing its log.
    matches = list(re.finditer(r"STEP=(\d+)\s+t=(\S+)\s+dt=(\S+)(?=\s)", tail))
    if not matches:
        return None
    match = matches[-1]
    try:
        current_time, step_dt = float(match[2]), float(match[3])
    except ValueError:
        return None  # Writer may not yet have completed the last line.
    violations = []
    if not np.isfinite([current_time, step_dt]).all():
        violations.append("nonfinite step time or timestep")
    elif current_time < target_time - 1e-10 and step_dt <= minimum_dt:
        violations.append(
            f"timestep {step_dt:.9g} is below the diagnostic floor {minimum_dt:.9g}"
        )
    if violations:
        return {
            "valid": False,
            "violations": violations,
            "source": "runtime_log",
            "step": int(match[1]),
            "simulation_time": current_time if np.isfinite(current_time) else None,
            "dt": step_dt if np.isfinite(step_dt) else None,
        }
    return None


def runtime_identity(solver: Path, solver_library: Path) -> dict[str, str]:
    for path in (solver, solver_library):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "solver_path": str(solver),
        "solver_sha256": sha256(solver),
        "solver_library_path": str(solver_library),
        "solver_library_sha256": sha256(solver_library),
    }


def read_raw(path: Path, expected_cells: int) -> np.ndarray:
    values = np.fromfile(path, dtype=RAW_DTYPE)
    if values.size != expected_cells:
        raise ValueError(
            f"{path} has {values.size} float64 values; expected {expected_cells}"
        )
    return values


def audit_initial_state(
    input_dir: Path,
    *,
    initial_mass_relative_tolerance: float,
    face_divergence_tolerance: float = 1e-10,
    shared_face_tolerance: float = 1e-12,
    momentum_relative_tolerance: float = 5e-3,
    momentum_absolute_tolerance: float | None = None,
    max_liquid_speed: float = 10.0,
    max_all_speed: float = 50.0,
) -> dict:
    """Validate both VOF encodings and the target-MAC velocity contract."""

    report_path = input_dir / "initial_state.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    nx = int(report["nx"])
    ny = int(report["ny"])
    expected_cells = nx * ny
    expected_size = expected_cells * RAW_DTYPE.itemsize
    dx = float(report["cell_size"])
    target_mass = float(report["target_liquid_volume"])
    fields = {
        name: read_raw(input_dir / f"{name}.raw", expected_cells)
        for name in (
            "q",
            "vf",
            "cs",
            "vx",
            "vy",
            "fluxx",
            "fluxxp",
            "fluxy",
            "fluxyp",
            "fluxeb",
        )
    }
    q = fields["q"]
    vf = fields["vf"]
    cs = fields["cs"]
    finite = {name: bool(np.isfinite(value).all()) for name, value in fields.items()}
    violations: list[str] = []
    if not all(finite.values()):
        violations.append("one or more input fields contain NaN or infinity")

    tolerance = 64.0 * np.finfo(np.float64).eps
    if finite["cs"] and (
        float(cs.min()) < -tolerance or float(cs.max()) > 1 + tolerance
    ):
        violations.append("cs.raw is outside [0, 1]")
    if (
        finite["q"]
        and finite["cs"]
        and (float(q.min()) < -tolerance or float(np.max(q - cs)) > tolerance)
    ):
        violations.append("q.raw does not satisfy 0 <= q <= cs")
    if finite["vf"] and (
        float(vf.min()) < -tolerance or float(vf.max()) > 1 + tolerance
    ):
        violations.append("vf.raw is outside [0, 1]")

    q_vf_cs_max = (
        float(np.max(np.abs(q - vf * cs)))
        if finite["q"] and finite["vf"] and finite["cs"]
        else float("inf")
    )
    encoding_tolerance = 128.0 * np.finfo(np.float64).eps
    if q_vf_cs_max > encoding_tolerance:
        violations.append(f"q.raw and vf.raw*cs.raw disagree by {q_vf_cs_max:.9g}")

    aperture_mass = float(q.sum() * dx * dx) if finite["q"] else float("nan")
    normalized_mass = (
        float((vf * cs).sum() * dx * dx)
        if finite["vf"] and finite["cs"]
        else float("nan")
    )
    aperture_relative_error = abs(aperture_mass - target_mass) / target_mass
    normalized_relative_error = abs(normalized_mass - target_mass) / target_mass
    if not np.isfinite(aperture_relative_error) or (
        aperture_relative_error > initial_mass_relative_tolerance
    ):
        violations.append(
            "q.raw target-mass relative error "
            f"{aperture_relative_error:.9g} exceeds "
            f"{initial_mass_relative_tolerance:.9g}"
        )
    if not np.isfinite(normalized_relative_error) or (
        normalized_relative_error > initial_mass_relative_tolerance
    ):
        violations.append(
            "vf.raw*cs.raw target-mass relative error "
            f"{normalized_relative_error:.9g} exceeds "
            f"{initial_mass_relative_tolerance:.9g}"
        )

    speed = np.hypot(fields["vx"], fields["vy"])
    liquid_regular = (cs >= 0.5) & (vf > 0.01)
    momentum = np.asarray(
        [
            np.sum(q * fields["vx"]) * dx * dx,
            np.sum(q * fields["vy"]) * dx * dx,
        ]
    )
    target_momentum = np.asarray(report["target_liquid_momentum"], dtype=float)
    momentum_absolute_error = float(np.linalg.norm(momentum - target_momentum))
    momentum_relative_error = float(
        momentum_absolute_error
        / max(float(np.linalg.norm(target_momentum)), np.finfo(float).tiny)
    )
    speed_all_max = float(speed[cs > 0].max())
    speed_liquid_max = float(speed[liquid_regular].max())
    if momentum_absolute_tolerance is not None:
        if momentum_absolute_error > momentum_absolute_tolerance:
            violations.append(
                "initial liquid-momentum absolute error "
                f"{momentum_absolute_error:.9g} exceeds {momentum_absolute_tolerance:.9g}"
            )
    elif momentum_relative_error > momentum_relative_tolerance:
        violations.append(
            "initial liquid-momentum relative error "
            f"{momentum_relative_error:.9g} exceeds {momentum_relative_tolerance:.9g}"
        )
    if speed_liquid_max > max_liquid_speed:
        violations.append(
            f"initial regular-liquid speed {speed_liquid_max:.9g} exceeds "
            f"{max_liquid_speed:.9g}"
        )
    if speed_all_max > max_all_speed:
        violations.append(
            f"initial all-fluid speed {speed_all_max:.9g} exceeds {max_all_speed:.9g}"
        )
    face = {
        name: fields[name].reshape(ny, nx)
        for name in ("fluxx", "fluxxp", "fluxy", "fluxyp", "fluxeb")
    }
    face_divergence = (
        face["fluxxp"] - face["fluxx"] + face["fluxyp"] - face["fluxy"]
    ) / dx
    face_divergence_l2 = float(np.sqrt(np.mean(face_divergence**2)))
    face_divergence_linf = float(np.max(np.abs(face_divergence)))
    shared_x_mismatch = float(
        np.max(np.abs(face["fluxxp"] - np.roll(face["fluxx"], -1, axis=1)))
    )
    shared_y_mismatch = float(
        np.max(np.abs(face["fluxyp"][:-1] - face["fluxy"][1:])) if ny > 1 else 0.0
    )
    if face_divergence_linf > face_divergence_tolerance:
        violations.append(
            "target-MAC normal-velocity divergence "
            f"{face_divergence_linf:.9g} exceeds {face_divergence_tolerance:.9g}"
        )
    if max(shared_x_mismatch, shared_y_mismatch) > shared_face_tolerance:
        violations.append(
            "target-MAC shared-face mismatch "
            f"{max(shared_x_mismatch, shared_y_mismatch):.9g} exceeds "
            f"{shared_face_tolerance:.9g}"
        )
    if float(np.max(np.abs(face["fluxeb"]))) > shared_face_tolerance:
        violations.append("initial embedded-boundary flux must be zero")
    result = {
        "schema": "aphros-initial-preflight-v3",
        "audited_utc": utc_now(),
        "valid": not violations,
        "violations": violations,
        "nx": nx,
        "ny": ny,
        "expected_cells": expected_cells,
        "expected_field_bytes": expected_size,
        "cell_size": dx,
        "target_physical_liquid_volume": target_mass,
        "aperture_liquid_volume": aperture_mass,
        "normalized_embedded_liquid_volume": normalized_mass,
        "aperture_relative_mass_error": aperture_relative_error,
        "normalized_relative_mass_error": normalized_relative_error,
        "maximum_abs_q_minus_vf_cs": q_vf_cs_max,
        "liquid_momentum": momentum.tolist(),
        "target_liquid_momentum": target_momentum.tolist(),
        "relative_liquid_momentum_error": momentum_relative_error,
        "absolute_liquid_momentum_error": momentum_absolute_error,
        "maximum_speed_all_fluid_cells": speed_all_max,
        "maximum_speed_liquid_regular_cells": speed_liquid_max,
        "normal_velocity_divergence_l2": face_divergence_l2,
        "normal_velocity_divergence_linf": face_divergence_linf,
        "shared_x_face_mismatch_linf": shared_x_mismatch,
        "shared_y_face_mismatch_linf": shared_y_mismatch,
        "finite_fields": finite,
        "semantics": {
            "initial": "q.raw is aperture q=f*cs; init_vf_is_aperture=1",
            "initial_flux": (
                "raw Cartesian face fields are normal velocities; "
                "flux_init_raw_is_velocity=1 converts with Aphros EB areas"
            ),
            "restart": "checkpoint vf.raw is normalized f; init_vf_is_aperture=0",
        },
    }
    return result


def audit_state_fields(
    paths: dict[str, Path],
    *,
    source: str,
    simulation_time: float,
    expected_cells: int,
    grid_shape: tuple[int, int],
    cell_area: float,
    target_mass: float,
    runtime_reference_mass: float | None,
    runtime_reference_vof_mass: float | None,
    reference_cs: np.ndarray,
    initial_mass_relative_tolerance: float,
    mass_relative_tolerance: float,
    vof_mass_relative_tolerance: float,
    vf_bound_tolerance: float,
    geometry_relative_tolerance: float,
    max_liquid_speed: float,
    max_all_speed: float,
    flux_divergence_tolerance: float,
    shared_face_tolerance: float,
    target_momentum: np.ndarray | None = None,
    initial_momentum_relative_tolerance: float = 5e-3,
    initial_momentum_absolute_tolerance: float | None = None,
) -> dict:
    """Validate the physical state, not just the existence of raw files."""

    fields: dict[str, np.ndarray] = {}
    violations: list[str] = []
    for name, path in paths.items():
        try:
            fields[name] = read_raw(path, expected_cells)
        except (OSError, ValueError) as exc:
            violations.append(str(exc))
    finite = {name: bool(np.isfinite(value).all()) for name, value in fields.items()}
    nonfinite = sorted(name for name, valid in finite.items() if not valid)
    if nonfinite:
        violations.append("non-finite fields: " + ", ".join(nonfinite))

    required = set(CHECKPOINT_FIELDS)
    missing = sorted(required - fields.keys())
    if missing:
        violations.append("missing audited fields: " + ", ".join(missing))

    vf = fields.get("vf")
    ebvf = fields.get("ebvf")
    vf_min = vf_max = physical_mass = vof_transport_mass = float("nan")
    initial_mass_relative_error = runtime_mass_relative_error = float("nan")
    runtime_vof_mass_relative_error = float("nan")
    geometry_relative_l1 = geometry_maximum_abs = float("nan")
    if vf is not None and finite.get("vf", False):
        vf_min = float(vf.min())
        vf_max = float(vf.max())
        if vf_min < -vf_bound_tolerance or vf_max > 1 + vf_bound_tolerance:
            violations.append(
                f"vf bounds [{vf_min:.9g}, {vf_max:.9g}] exceed tolerance"
            )
    if ebvf is not None and finite.get("ebvf", False):
        ebvf_min = float(ebvf.min())
        ebvf_max = float(ebvf.max())
        if ebvf_min < -vf_bound_tolerance or ebvf_max > 1 + vf_bound_tolerance:
            violations.append(
                f"ebvf bounds [{ebvf_min:.9g}, {ebvf_max:.9g}] exceed tolerance"
            )
        geometry_difference = np.abs(ebvf - reference_cs)
        reference_fluid_volume = float(reference_cs.sum() * cell_area)
        geometry_relative_l1 = float(
            geometry_difference.sum() * cell_area / reference_fluid_volume
        )
        geometry_maximum_abs = float(geometry_difference.max())
        if geometry_relative_l1 > geometry_relative_tolerance:
            violations.append(
                "embedded geometry relative L1 error "
                f"{geometry_relative_l1:.9g} exceeds "
                f"{geometry_relative_tolerance:.9g}"
            )
    if (
        vf is not None
        and ebvf is not None
        and finite.get("vf", False)
        and finite.get("ebvf", False)
    ):
        # Aphros advances the conditional VOF fraction f on the Cartesian
        # mesh. The physical liquid aperture in an embedded cell is q=f*cs.
        physical_mass = float((vf * ebvf).sum() * cell_area)
        vof_transport_mass = float(vf.sum() * cell_area)
        initial_mass_relative_error = abs(physical_mass - target_mass) / target_mass
        if runtime_reference_mass is None:
            runtime_reference_mass = physical_mass
            runtime_reference_vof_mass = vof_transport_mass
            if initial_mass_relative_error > initial_mass_relative_tolerance:
                violations.append(
                    "initial embedded physical-mass relative error "
                    f"{initial_mass_relative_error:.9g} exceeds "
                    f"{initial_mass_relative_tolerance:.9g}"
                )
        if runtime_reference_vof_mass is None:
            runtime_reference_vof_mass = vof_transport_mass
        runtime_mass_relative_error = (
            abs(physical_mass - runtime_reference_mass) / runtime_reference_mass
        )
        runtime_vof_mass_relative_error = (
            abs(vof_transport_mass - runtime_reference_vof_mass)
            / runtime_reference_vof_mass
        )
        if runtime_mass_relative_error > mass_relative_tolerance:
            violations.append(
                "runtime embedded physical-mass relative change "
                f"{runtime_mass_relative_error:.9g} exceeds "
                f"{mass_relative_tolerance:.9g}"
            )
        if runtime_vof_mass_relative_error > vof_mass_relative_tolerance:
            violations.append(
                "runtime Cartesian VOF-mass relative change "
                f"{runtime_vof_mass_relative_error:.9g} exceeds "
                f"{vof_mass_relative_tolerance:.9g}"
            )

    momentum = None
    momentum_error = None
    momentum_absolute_error = None
    if all(
        name in fields and finite.get(name, False)
        for name in ("vx", "vy", "vf", "ebvf")
    ):
        momentum = np.array(
            [np.sum(fields[name] * vf * ebvf) * cell_area for name in ("vx", "vy")]
        )
        if target_momentum is not None:
            target_momentum = np.asarray(target_momentum, dtype=float)
            if target_momentum.shape != (2,) or not np.isfinite(target_momentum).all():
                violations.append("invalid target liquid momentum")
            else:
                momentum_absolute_error = float(np.linalg.norm(momentum - target_momentum))
                momentum_error = float(
                    momentum_absolute_error / max(np.linalg.norm(target_momentum), 1e-30)
                )
                # Momentum can evolve physically later. This gate verifies loading,
                # including all cut cells, only at the initial time.
                if simulation_time == 0:
                    if initial_momentum_absolute_tolerance is not None:
                        if momentum_absolute_error > initial_momentum_absolute_tolerance:
                            violations.append(
                                "initial loaded liquid-momentum absolute error "
                                f"{momentum_absolute_error:.9g} exceeds {initial_momentum_absolute_tolerance:.9g}"
                            )
                    elif momentum_error > initial_momentum_relative_tolerance:
                        violations.append(
                            "initial loaded liquid-momentum relative error "
                            f"{momentum_error:.9g} exceeds {initial_momentum_relative_tolerance:.9g}"
                        )
    speed_max = speed_liquid_max = pressure_min = pressure_max = float("nan")
    if all(name in fields and finite.get(name, False) for name in ("vx", "vy")):
        speed = np.hypot(fields["vx"], fields["vy"])
        speed_max = float(speed.max())
        if vf is not None and ebvf is not None and finite.get("ebvf", False):
            liquid_regular = (ebvf >= 0.5) & (vf > 0.01)
            if np.any(liquid_regular):
                speed_liquid_max = float(speed[liquid_regular].max())
            if speed_liquid_max > max_liquid_speed:
                violations.append(
                    "maximum liquid regular-cell speed "
                    f"{speed_liquid_max:.9g} exceeds {max_liquid_speed:.9g}"
                )
        if speed_max > max_all_speed:
            violations.append(
                f"maximum all-cell speed {speed_max:.9g} exceeds {max_all_speed:.9g}"
            )
    if "p" in fields and finite.get("p", False):
        pressure_min = float(fields["p"].min())
        pressure_max = float(fields["p"].max())

    flux_divergence_l2 = flux_divergence_linf = float("nan")
    shared_x_mismatch = shared_y_mismatch = float("nan")
    flux_names = ("fluxx", "fluxxp", "fluxy", "fluxyp", "fluxeb")
    if all(name in fields and finite.get(name, False) for name in flux_names):
        ny, nx = grid_shape
        if nx * ny != expected_cells:
            raise ValueError("grid shape disagrees with expected cell count")
        flux = {name: fields[name].reshape(ny, nx) for name in flux_names}
        divergence = (
            flux["fluxxp"]
            - flux["fluxx"]
            + flux["fluxyp"]
            - flux["fluxy"]
            + flux["fluxeb"]
        ) / cell_area
        flux_divergence_l2 = float(np.sqrt(np.mean(divergence**2)))
        flux_divergence_linf = float(np.max(np.abs(divergence)))
        shared_x_mismatch = float(
            np.max(np.abs(flux["fluxxp"] - np.roll(flux["fluxx"], -1, axis=1)))
        )
        shared_y_mismatch = float(
            np.max(np.abs(flux["fluxyp"][:-1] - flux["fluxy"][1:])) if ny > 1 else 0.0
        )
        if simulation_time > 1e-12 and flux_divergence_linf > flux_divergence_tolerance:
            violations.append(
                "volume-flux divergence "
                f"{flux_divergence_linf:.9g} exceeds {flux_divergence_tolerance:.9g}"
            )
        if max(shared_x_mismatch, shared_y_mismatch) > shared_face_tolerance:
            violations.append(
                "checkpoint shared-face mismatch "
                f"{max(shared_x_mismatch, shared_y_mismatch):.9g} exceeds "
                f"{shared_face_tolerance:.9g}"
            )

    return {
        "schema": "aphros-semantic-state-audit-v5",
        "audited_utc": utc_now(),
        "source": source,
        "simulation_time": simulation_time,
        "valid": not violations,
        "violations": violations,
        "physical_liquid_volume": physical_mass,
        "target_physical_liquid_volume": target_mass,
        "initial_mass_relative_error": initial_mass_relative_error,
        "liquid_momentum": momentum.tolist() if momentum is not None else None,
        "target_liquid_momentum": target_momentum.tolist()
        if target_momentum is not None
        else None,
        "momentum_relative_difference_from_initial_target": momentum_error,
        "momentum_absolute_difference_from_initial_target": momentum_absolute_error,
        "runtime_reference_liquid_volume": runtime_reference_mass,
        "runtime_mass_relative_error": runtime_mass_relative_error,
        "vof_transport_volume": vof_transport_mass,
        "runtime_reference_vof_volume": runtime_reference_vof_mass,
        "runtime_vof_mass_relative_error": runtime_vof_mass_relative_error,
        "vf_min": vf_min,
        "vf_max": vf_max,
        "geometry_relative_l1_error": geometry_relative_l1,
        "geometry_maximum_abs_error": geometry_maximum_abs,
        "maximum_speed_all_cells": speed_max,
        "maximum_speed_liquid_regular_cells": speed_liquid_max,
        "pressure_min": pressure_min,
        "pressure_max": pressure_max,
        "volume_flux_divergence_l2": flux_divergence_l2,
        "volume_flux_divergence_linf": flux_divergence_linf,
        "shared_x_face_mismatch_linf": shared_x_mismatch,
        "shared_y_face_mismatch_linf": shared_y_mismatch,
        "finite_fields": finite,
    }


def append_jsonl(path: Path, value: object) -> None:
    def json_safe(item: object) -> object:
        if isinstance(item, float) and not np.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: json_safe(member) for key, member in item.items()}
        if isinstance(item, (list, tuple)):
            return [json_safe(member) for member in item]
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(json_safe(value), sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_tmax(config: Path) -> float:
    values = {
        match.group("name"): float(match.group("value"))
        for match in SET_DOUBLE_RE.finditer(config.read_text(encoding="utf-8"))
    }
    return values["tmax"]


def parse_dump_times(log_path: Path) -> dict[int, float]:
    result: dict[int, float] = {}
    if not log_path.exists():
        return result
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for match in DUMP_RE.finditer(text):
        result[int(match.group("index"))] = float(match.group("time"))
    return result


def parse_step_rows(log_path: Path) -> list[dict[str, float | int]]:
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    result: list[dict[str, float | int]] = []
    for match in STEP_RE.finditer(text):
        dt_text = match.group("dt")
        mantissa = re.split(r"[eE]", dt_text, maxsplit=1)[0]
        decimal_places = len(mantissa.rsplit(".", 1)[1]) if "." in mantissa else 0
        result.append(
            {
                "step": int(match.group("step")),
                "time": float(match.group("time")),
                "dt": float(dt_text),
                "dt_decimal_places": decimal_places,
            }
        )
    return result


def parse_stat_rows(stat_path: Path) -> list[dict[str, float]]:
    if not stat_path.exists():
        return []
    lines = stat_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return []
    header = lines[0].split()
    rows: list[dict[str, float]] = []
    for line in lines[1:]:
        words = line.split()
        if len(words) != len(header):
            continue
        try:
            rows.append(dict(zip(header, map(float, words))))
        except ValueError:
            continue
    return rows


def completed_dumps(
    segment: Path, expected_size: int
) -> list[dict[str, float | int | Path]]:
    dump_times = parse_dump_times(segment / "run.log")
    step_rows = parse_step_rows(segment / "run.log")
    stat_rows = parse_stat_rows(segment / "stat.dat")
    result: list[dict[str, float | int | Path]] = []
    for marker in segment.glob(".dumpdone_*"):
        try:
            index = int(marker.name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if index not in dump_times:
            continue
        paths = {
            field: segment / f"{field}_{index:04d}.raw" for field in CHECKPOINT_FIELDS
        }
        if any(
            not path.is_file() or path.stat().st_size != expected_size
            for path in paths.values()
        ):
            continue
        dump_time = dump_times[index]
        if not stat_rows:
            continue
        stat = min(stat_rows, key=lambda row: abs(row.get("t", -1e300) - dump_time))
        stat_dt = abs(stat.get("dt", 0.0))
        if abs(stat.get("t", -1e300) - dump_time) > max(1e-10, 1.1 * stat_dt):
            continue
        step = round(stat.get("step", 0.0))
        # run.log prints a shortened time.  stat.dat carries the value used by
        # the solver and avoids treating the restart's t0 dump as new progress.
        simulation_time = float(stat["t"])
        # The dt stored on a stat row is the interval that produced that row.
        # Therefore the first later stat row contains, at full precision, the
        # dt that must be used to advance from this dumped state.  A STEP line
        # is only a fallback: older Aphros builds printed it with 8 digits.
        following_stat_rows = [
            row
            for row in stat_rows
            if round(row.get("step", -1.0)) == step + 1
            and row.get("t", -1e300)
            > simulation_time + max(1e-14, 1e-10 * max(stat_dt, 1e-12))
        ]
        following_stat = (
            min(following_stat_rows, key=lambda row: row["t"])
            if following_stat_rows
            else None
        )
        next_step_row = (
            min(step_rows, key=lambda row: abs(float(row["time"]) - simulation_time))
            if step_rows
            else None
        )
        step_dt_is_predicted = bool(
            next_step_row is not None
            and abs(float(next_step_row["time"]) - simulation_time)
            <= max(5e-8, 1e-4 * max(stat_dt, 1e-12))
        )
        if following_stat is not None:
            restart_dt = abs(float(following_stat["dt"]))
            restart_dt_source = "following_stat_exact"
        elif step_dt_is_predicted:
            restart_dt = float(next_step_row["dt"])
            restart_dt_source = (
                "step_log_full_precision"
                if int(next_step_row.get("dt_decimal_places", 0)) >= 15
                else "step_log_truncated"
            )
        else:
            restart_dt = stat_dt
            restart_dt_source = "previous_stat_fallback"
        result.append(
            {
                "index": index,
                "time": simulation_time,
                "dt": stat.get("dt", 0.0),
                "restart_dt": restart_dt,
                "restart_dt_is_predicted": restart_dt_source
                != "previous_stat_fallback",
                "restart_dt_is_exact": restart_dt_source
                in ("following_stat_exact", "step_log_full_precision"),
                "restart_dt_source": restart_dt_source,
                "iter": round(stat.get("iter", 0.0)),
                "step": step,
                "next_step": 0 if abs(dump_time) < 1e-15 else step + 1,
                "segment": segment,
            }
        )
    return sorted(result, key=lambda item: (float(item["time"]), int(item["index"])))


def validate_checkpoint(slot: Path, expected_size: int) -> dict | None:
    try:
        if not (slot / "COMPLETE").is_file():
            return None
        metadata = json.loads((slot / "metadata.json").read_text(encoding="utf-8"))
        for field in CHECKPOINT_FIELDS:
            entry = metadata["files"][f"{field}.raw"]
            path = slot / f"{field}.raw"
            if not path.is_file() or path.stat().st_size != expected_size:
                return None
            if entry["size"] != expected_size or sha256(path) != entry["sha256"]:
                return None
        metadata["slot_path"] = str(slot)
        return metadata
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None


def recover_checkpoint(checkpoint_root: Path, expected_size: int) -> dict | None:
    candidates: list[dict] = []
    latest_path = checkpoint_root / "latest.json"
    if latest_path.is_file():
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            slot = checkpoint_root / latest["slot"]
            metadata = validate_checkpoint(slot, expected_size)
            if metadata is not None:
                candidates.append(metadata)
        except (KeyError, OSError, json.JSONDecodeError):
            pass
    for name in ("A", "B"):
        metadata = validate_checkpoint(checkpoint_root / name, expected_size)
        if metadata is not None:
            candidates.append(metadata)
    return max(candidates, key=lambda item: item["sequence"], default=None)


def commit_checkpoint(
    checkpoint_root: Path,
    dump: dict,
    previous: dict | None,
    expected_size: int,
    reason: str,
    semantic_audit: dict,
    wall_seconds_since_previous: float | None = None,
    checkpoint_deadline_seconds: float | None = None,
) -> dict:
    current_slot = previous.get("slot") if previous else None
    slot_name = "B" if current_slot == "A" else "A"
    sequence = int(previous.get("sequence", -1)) + 1 if previous else 0
    temporary = checkpoint_root / f".{slot_name}.tmp.{os.getpid()}"
    destination = checkpoint_root / slot_name
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    source_segment = Path(dump["segment"])
    files: dict[str, dict[str, int | str]] = {}
    index = int(dump["index"])
    for field in CHECKPOINT_FIELDS:
        source = source_segment / f"{field}_{index:04d}.raw"
        target = temporary / f"{field}.raw"
        shutil.copy2(source, target)
        size = target.stat().st_size
        if size != expected_size:
            raise RuntimeError(f"Incomplete checkpoint field {target}: {size}")
        files[target.name] = {"size": size, "sha256": sha256(target)}
    metadata = {
        "schema": "aphros-atomic-checkpoint-v2",
        "sequence": sequence,
        "slot": slot_name,
        "created_utc": utc_now(),
        "created_unix": time.time(),
        "reason": reason,
        "source_segment": source_segment.name,
        "source_dump_index": index,
        "time": float(dump["time"]),
        # The stat row contains the step that produced this state.  A restart
        # needs the dt already predicted for the following step instead.
        "dt": float(dump.get("restart_dt", dump["dt"])),
        "dt_used_to_reach_state": float(dump["dt"]),
        "restart_dt_is_predicted": bool(dump.get("restart_dt_is_predicted", False)),
        "restart_dt_is_exact": bool(dump.get("restart_dt_is_exact", False)),
        "restart_dt_source": str(
            dump.get("restart_dt_source", "previous_stat_fallback")
        ),
        "iter": int(dump["iter"]),
        "step": int(dump["step"]),
        "next_step": int(dump["next_step"]),
        "semantic_audit": semantic_audit,
        "wall_seconds_since_previous": wall_seconds_since_previous,
        "checkpoint_deadline_seconds": checkpoint_deadline_seconds,
        "checkpoint_deadline_met": (
            wall_seconds_since_previous <= checkpoint_deadline_seconds
            if wall_seconds_since_previous is not None
            and checkpoint_deadline_seconds is not None
            else None
        ),
        "files": files,
    }
    atomic_json(temporary / "metadata.json", metadata)
    (temporary / "COMPLETE").write_text("complete\n", encoding="ascii")
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(temporary, destination)
    atomic_json(
        checkpoint_root / "latest.json",
        {
            "schema": "aphros-checkpoint-pointer-v1",
            "slot": slot_name,
            "sequence": sequence,
            "time": metadata["time"],
            "updated_utc": utc_now(),
        },
    )
    metadata["slot_path"] = str(destination)
    return metadata


def write_segment_config(
    path: Path,
    base_config: Path,
    input_dir: Path,
    checkpoint: dict | None,
    overrides: list[str] | None = None,
    initial_velocity_init: str = "raw",
    initial_flux_init: str = "reconstruct",
    restart_pressure_init: str = "raw",
    restart_flux_init: str = "raw",
) -> None:
    lines = [
        f"include {base_config}",
        "",
        f"set string eb_list_path {input_dir / 'body.dat'}",
    ]
    if checkpoint is None:
        initial_vf_path = input_dir / "q.raw"
        lines += [
            f"set string init_vf_raw_path {initial_vf_path}",
            "set int init_vf_is_aperture 1",
            "set string pressure_init zero",
        ]
        if initial_velocity_init == "raw":
            lines += [
                "set string vel_init raw",
                f"set string vel_init_raw_path_0 {input_dir / 'vx.raw'}",
                f"set string vel_init_raw_path_1 {input_dir / 'vy.raw'}",
            ]
        elif initial_velocity_init == "zero":
            lines.append("set string vel_init zero")
        else:
            raise ValueError(
                f"unsupported initial velocity mode: {initial_velocity_init}"
            )
        if initial_flux_init == "raw":
            lines += [
                "set string flux_init raw",
                f"set string flux_init_raw_path_0 {input_dir / 'fluxx.raw'}",
                f"set string flux_init_raw_path_1 {input_dir / 'fluxy.raw'}",
                f"set string flux_init_raw_positive_path_0 {input_dir / 'fluxxp.raw'}",
                f"set string flux_init_raw_positive_path_1 {input_dir / 'fluxyp.raw'}",
                f"set string flux_init_raw_embed_path {input_dir / 'fluxeb.raw'}",
                "set int flux_init_raw_is_velocity 1",
            ]
        elif initial_flux_init == "reconstruct":
            # Let Aphros construct a face field consistent with its own
            # embedded geometry and cell-centred velocity.  Importing target-
            # grid normal velocities as if they were an EB-compatible flux
            # creates an O(1/dx) startup pressure impulse near cut cells.
            lines.append("set string flux_init zero")
        else:
            raise ValueError(f"unsupported initial flux mode: {initial_flux_init}")
        lines += [
            "set double restart_time 0",
            "set int restart_iter 0",
            "set int restart_step 0",
            "set double dump_field_t0 0",
        ]
    else:
        slot = Path(checkpoint["slot_path"])
        lines += [
            f"set string init_vf_raw_path {slot / 'vf.raw'}",
            "set int init_vf_is_aperture 0",
            f"set string vel_init_raw_path_0 {slot / 'vx.raw'}",
            f"set string vel_init_raw_path_1 {slot / 'vy.raw'}",
        ]
        if restart_pressure_init == "raw":
            lines += [
                "set string pressure_init raw",
                f"set string pressure_init_raw_path {slot / 'p.raw'}",
            ]
        else:
            lines.append("set string pressure_init zero")
        if restart_flux_init == "raw":
            lines += [
                "set string flux_init raw",
                f"set string flux_init_raw_path_0 {slot / 'fluxx.raw'}",
                f"set string flux_init_raw_path_1 {slot / 'fluxy.raw'}",
                f"set string flux_init_raw_positive_path_0 {slot / 'fluxxp.raw'}",
                f"set string flux_init_raw_positive_path_1 {slot / 'fluxyp.raw'}",
                f"set string flux_init_raw_embed_path {slot / 'fluxeb.raw'}",
                "set int flux_init_raw_is_velocity 0",
            ]
        else:
            # A null flux pointer makes Proj reconstruct face fluxes from the
            # restored cell velocity and the current embedded geometry.
            lines.append("set string flux_init zero")
        lines += [
            f"set double restart_time {checkpoint['time']:.17g}",
            f"set int restart_iter {checkpoint['iter']}",
            f"set int restart_step {checkpoint['next_step']}",
            f"set double dt0 {checkpoint['dt']:.17g}",
            # Keep the original output-event phase.  Resetting t0 to the
            # checkpoint time changes event-clamped time steps after restart.
            "set double dump_field_t0 0",
        ]
    if overrides:
        lines += [""] + overrides
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def newest_dump(segment: Path, expected_size: int) -> dict | None:
    dumps = completed_dumps(segment, expected_size)
    return dumps[-1] if dumps else None


def dump_field_paths(dump: dict) -> dict[str, Path]:
    segment = Path(dump["segment"])
    index = int(dump["index"])
    return {field: segment / f"{field}_{index:04d}.raw" for field in CHECKPOINT_FIELDS}


def checkpoint_field_paths(checkpoint: dict) -> dict[str, Path]:
    slot = Path(checkpoint["slot_path"])
    return {field: slot / f"{field}.raw" for field in CHECKPOINT_FIELDS}


def checkpoint_reaches_target(checkpoint: dict | None, target_time: float) -> bool:
    if checkpoint is None:
        return False
    tolerance = max(1e-12, 64 * np.finfo(float).eps * max(1.0, abs(target_time)))
    return float(checkpoint["time"]) >= target_time - tolerance


def exit_dump_can_be_promoted(last_dump: dict | None, return_code: int | None) -> bool:
    """Accept exact restart states, or a final state after clean completion."""
    return bool(
        last_dump is not None
        and (last_dump.get("restart_dt_is_exact") or return_code == 0)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--solver-library", type=Path)
    parser.add_argument("--checkpoint-seconds", type=float, default=3600.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--max-restarts", type=int, default=20)
    parser.add_argument("--max-stalled-restarts", type=int, default=2)
    parser.add_argument(
        "--target-time",
        type=float,
        help="Override tmax for a qualification prefix without changing production config.",
    )
    parser.add_argument("--dump-field-dt", type=float)
    parser.add_argument("--dtmax", type=float)
    parser.add_argument("--dt-growth-max", type=float)
    parser.add_argument("--cfl", type=float)
    parser.add_argument("--cfla", type=float)
    parser.add_argument("--minimum-dt", type=float, default=0.0)
    parser.add_argument("--abort-speed", type=float)
    parser.add_argument("--restart-dt-scale", type=float, default=1.0)
    # Aphros and the offline polygon integrator differ at cut-cell roundoff.
    # The measured L10 startup discrepancy is 1.76e-6; 1e-5 still rejects the
    # aperture/normalized semantic bug (9.01e-3) by almost three orders.
    parser.add_argument("--initial-mass-relative-tolerance", type=float, default=1e-5)
    parser.add_argument("--mass-relative-tolerance", type=float, default=2e-3)
    parser.add_argument("--vof-mass-relative-tolerance", type=float, default=5e-4)
    parser.add_argument("--vf-bound-tolerance", type=float, default=1e-12)
    parser.add_argument("--geometry-relative-tolerance", type=float, default=1e-5)
    parser.add_argument("--max-liquid-speed", type=float, default=10.0)
    parser.add_argument("--max-all-speed", type=float, default=50.0)
    parser.add_argument(
        "--initial-momentum-relative-tolerance", type=float, default=5e-3
    )
    parser.add_argument("--initial-momentum-absolute-tolerance", type=float)
    parser.add_argument("--flux-divergence-tolerance", type=float, default=1e-8)
    parser.add_argument("--shared-face-tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--initial-velocity-init",
        choices=("raw", "zero"),
        default="raw",
    )
    parser.add_argument(
        "--initial-flux-init",
        choices=("raw", "reconstruct"),
        default="reconstruct",
    )
    parser.add_argument(
        "--restart-pressure-init", choices=("raw", "zero"), default="raw"
    )
    parser.add_argument(
        "--restart-flux-init",
        choices=("raw", "reconstruct"),
        default="raw",
    )
    args = parser.parse_args()

    if not 0 < args.checkpoint_seconds <= 3600:
        parser.error("--checkpoint-seconds must be in (0, 3600]")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.initial_momentum_absolute_tolerance is not None and (
        not np.isfinite(args.initial_momentum_absolute_tolerance)
        or args.initial_momentum_absolute_tolerance < 0
    ):
        parser.error("--initial-momentum-absolute-tolerance must be finite and nonnegative")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if not np.isfinite(args.minimum_dt) or args.minimum_dt < 0:
        parser.error("--minimum-dt must be finite and nonnegative")
    if not np.isfinite(args.restart_dt_scale) or not 0 < args.restart_dt_scale <= 1:
        parser.error("--restart-dt-scale must be in (0, 1]")
    if args.abort_speed is not None and (
        not np.isfinite(args.abort_speed) or args.abort_speed <= 0
    ):
        parser.error("--abort-speed must be finite and positive")

    run_root = args.run_root.resolve()
    base_config = args.base_config.resolve()
    input_dir = args.input_dir.resolve()
    args.solver = args.solver.resolve()
    solver_library = (
        args.solver_library.resolve()
        if args.solver_library is not None
        else args.solver.parent.parent / "lib" / "libaphros.so"
    )
    expected_runtime = runtime_identity(args.solver, solver_library)
    run_root.mkdir(parents=True, exist_ok=True)
    segments_root = run_root / "segments"
    checkpoint_root = run_root / "checkpoints"
    segments_root.mkdir(exist_ok=True)
    checkpoint_root.mkdir(exist_ok=True)
    status_path = run_root / "status.json"
    audit_log_path = run_root / "semantic_audit.jsonl"

    try:
        preflight = audit_initial_state(
            input_dir,
            initial_mass_relative_tolerance=args.initial_mass_relative_tolerance,
            face_divergence_tolerance=args.flux_divergence_tolerance,
            shared_face_tolerance=args.shared_face_tolerance,
            momentum_relative_tolerance=args.initial_momentum_relative_tolerance,
            momentum_absolute_tolerance=args.initial_momentum_absolute_tolerance,
            max_liquid_speed=args.max_liquid_speed,
            max_all_speed=args.max_all_speed,
        )
    # Any parser, I/O, or numerical failure makes the initial state unsafe.
    # Persisting the reason is more useful than losing it to a traceback only.
    except Exception as exc:  # noqa: BLE001
        preflight = {
            "schema": "aphros-initial-preflight-v3",
            "audited_utc": utc_now(),
            "valid": False,
            "violations": [f"preflight could not be completed: {exc}"],
        }
    atomic_json(run_root / "initial_preflight.json", preflight)
    if not preflight["valid"]:
        atomic_json(
            status_path,
            {
                "schema": "aphros-supervisor-status-v2",
                "updated_utc": utc_now(),
                "state": "failed_preflight",
                "preflight": preflight,
            },
        )
        print(
            "Initial-state preflight failed: " + "; ".join(preflight["violations"]),
            flush=True,
        )
        return 3

    expected_cells = int(preflight["expected_cells"])
    expected_size = int(preflight["expected_field_bytes"])
    cell_area = float(preflight["cell_size"]) ** 2
    target_mass = float(preflight["target_physical_liquid_volume"])
    reference_cs = read_raw(input_dir / "cs.raw", expected_cells)

    lock_stream = (run_root / "supervisor.lock").open("a+")
    if fcntl is None:
        raise RuntimeError("The Aphros supervisor must run under Linux/WSL")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another supervisor already holds the run lock", flush=True)
        return 0

    tmax = args.target_time if args.target_time is not None else read_tmax(base_config)
    if tmax <= 0:
        raise ValueError("target time must be positive")
    overrides: list[str] = []
    if args.target_time is not None:
        overrides.append(f"set double tmax {args.target_time:.17g}")
    if args.dump_field_dt is not None:
        overrides.append(f"set double dump_field_dt {args.dump_field_dt:.17g}")
    if args.dtmax is not None:
        overrides.append(f"set double dtmax {args.dtmax:.17g}")
    if args.dt_growth_max is not None:
        if args.dt_growth_max < 1:
            parser.error("--dt-growth-max must be at least one")
        overrides.append(f"set double dt_growth_max {args.dt_growth_max:.17g}")
    if args.cfl is not None:
        overrides.append(f"set double cfl {args.cfl:.17g}")
    if args.cfla is not None:
        overrides.append(f"set double cfla {args.cfla:.17g}")
    checkpoint = recover_checkpoint(checkpoint_root, expected_size)
    if args.abort_speed is not None:
        overrides.append(f"set double abortvel {args.abort_speed:.17g}")
    if args.restart_dt_scale != 1 and checkpoint is None:
        parser.error("--restart-dt-scale requires a saved checkpoint")
    runtime_reference_mass = (
        float(checkpoint["semantic_audit"]["runtime_reference_liquid_volume"])
        if checkpoint is not None
        and checkpoint.get("semantic_audit", {}).get("runtime_reference_liquid_volume")
        is not None
        else None
    )
    runtime_reference_vof_mass = (
        float(checkpoint["semantic_audit"]["runtime_reference_vof_volume"])
        if checkpoint is not None
        and checkpoint.get("semantic_audit", {}).get("runtime_reference_vof_volume")
        is not None
        else None
    )
    stop_requested = False
    child: subprocess.Popen | None = None

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        print(
            f"Received signal {signum}; preserving the newest complete dump", flush=True
        )
        stop_requested = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def status(state: str, **extra: object) -> None:
        value = {
            "schema": "aphros-supervisor-status-v2",
            "updated_utc": utc_now(),
            "state": state,
            "supervisor_pid": os.getpid(),
            "solver_pid": child.pid
            if child is not None and child.poll() is None
            else None,
            "target_time": tmax,
            "runtime_fingerprint": expected_runtime,
            "initial_velocity_init": args.initial_velocity_init,
            "initial_flux_init": args.initial_flux_init,
            "restart_pressure_init": args.restart_pressure_init,
            "restart_flux_init": args.restart_flux_init,
            "restart_dt_scale": args.restart_dt_scale,
            "minimum_dt": args.minimum_dt,
            "abort_speed": args.abort_speed,
            "initial_preflight": {
                "valid": preflight["valid"],
                "aperture_relative_mass_error": preflight[
                    "aperture_relative_mass_error"
                ],
                "normalized_relative_mass_error": preflight[
                    "normalized_relative_mass_error"
                ],
                "maximum_abs_q_minus_vf_cs": preflight["maximum_abs_q_minus_vf_cs"],
            },
            "checkpoint": (
                {
                    "slot": checkpoint["slot"],
                    "sequence": checkpoint["sequence"],
                    "time": checkpoint["time"],
                    "created_utc": checkpoint["created_utc"],
                }
                if checkpoint
                else None
            ),
        }
        value.update(extra)
        atomic_json(status_path, value)

    def audit_paths(
        paths: dict[str, Path], source: str, simulation_time: float
    ) -> dict:
        nonlocal runtime_reference_mass, runtime_reference_vof_mass
        audit = audit_state_fields(
            paths,
            source=source,
            simulation_time=simulation_time,
            expected_cells=expected_cells,
            grid_shape=(int(preflight["ny"]), int(preflight["nx"])),
            cell_area=cell_area,
            target_mass=target_mass,
            runtime_reference_mass=runtime_reference_mass,
            runtime_reference_vof_mass=runtime_reference_vof_mass,
            reference_cs=reference_cs,
            initial_mass_relative_tolerance=args.initial_mass_relative_tolerance,
            mass_relative_tolerance=args.mass_relative_tolerance,
            vof_mass_relative_tolerance=args.vof_mass_relative_tolerance,
            vf_bound_tolerance=args.vf_bound_tolerance,
            geometry_relative_tolerance=args.geometry_relative_tolerance,
            max_liquid_speed=args.max_liquid_speed,
            max_all_speed=args.max_all_speed,
            flux_divergence_tolerance=args.flux_divergence_tolerance,
            shared_face_tolerance=args.shared_face_tolerance,
            target_momentum=np.asarray(preflight["target_liquid_momentum"]),
            initial_momentum_relative_tolerance=args.initial_momentum_relative_tolerance,
            initial_momentum_absolute_tolerance=args.initial_momentum_absolute_tolerance,
        )
        if runtime_reference_mass is None and audit["valid"]:
            runtime_reference_mass = float(audit["physical_liquid_volume"])
            runtime_reference_vof_mass = float(audit["vof_transport_volume"])
        append_jsonl(audit_log_path, audit)
        return audit

    if checkpoint is not None:
        if runtime_reference_mass is None:
            status(
                "failed_restart_contract",
                error=(
                    "Checkpoint predates the aperture-VOF mass contract and "
                    "has no runtime mass reference"
                ),
            )
            print(
                "Refusing to extend a checkpoint without an aperture-VOF "
                "runtime mass reference",
                flush=True,
            )
            return 3
        recovered_audit = audit_paths(
            checkpoint_field_paths(checkpoint),
            f"checkpoint/{checkpoint['slot']}",
            float(checkpoint["time"]),
        )
        if not recovered_audit["valid"]:
            status(
                "failed_validation",
                error="The checksum-valid restart checkpoint failed semantic validation",
                semantic_audit=recovered_audit,
            )
            print(
                "Recovered checkpoint failed semantic validation: "
                + "; ".join(recovered_audit["violations"]),
                flush=True,
            )
            return 3

    if checkpoint_reaches_target(checkpoint, tmax):
        status("complete", progress_time=checkpoint["time"], progress_percent=100.0)
        print("Existing checkpoint already reaches tmax", flush=True)
        return 0

    if checkpoint is not None and not bool(
        checkpoint.get("restart_dt_is_exact", False)
    ):
        status(
            "failed_restart_contract",
            error="Checkpoint does not contain a full-precision next dt",
            restart_dt_source=checkpoint.get("restart_dt_source", "legacy_unknown"),
        )
        print(
            "Refusing to extend a checkpoint without a full-precision next dt",
            flush=True,
        )
        return 3

    existing_indices = []
    for path in segments_root.glob("segment_*"):
        try:
            existing_indices.append(int(path.name.rsplit("_", 1)[1]))
        except ValueError:
            pass
    segment_index = max(existing_indices, default=-1) + 1
    restart_count = 0
    stalled_restart_count = 0
    progress_time_epsilon = 1e-7
    last_checkpoint_wall = time.monotonic()

    def checkpoint_deadline_failure(wall_interval: float) -> dict:
        deadline_message = (
            "checkpoint wall-time deadline missed: "
            f"{wall_interval:.1f}s > {args.checkpoint_seconds:.1f}s"
        )
        print(deadline_message, flush=True)
        return {
            "schema": "aphros-checkpoint-deadline-audit-v1",
            "audited_utc": utc_now(),
            "valid": False,
            "violations": [deadline_message],
            "wall_seconds_since_previous": wall_interval,
            "checkpoint_deadline_seconds": args.checkpoint_seconds,
        }

    def promote_dump(
        dump: dict,
        reason: str,
        semantic_audit: dict | None,
    ) -> dict | None:
        """Atomically promote one dump and enforce the recovery-time contract."""

        nonlocal checkpoint, last_checkpoint_wall
        wall_interval = time.monotonic() - last_checkpoint_wall
        checkpoint = commit_checkpoint(
            checkpoint_root,
            dump,
            checkpoint,
            expected_size,
            reason,
            semantic_audit,
            wall_seconds_since_previous=wall_interval,
            checkpoint_deadline_seconds=args.checkpoint_seconds,
        )
        last_checkpoint_wall = time.monotonic()
        if checkpoint["checkpoint_deadline_met"]:
            return None
        return checkpoint_deadline_failure(wall_interval)

    while not stop_requested:
        if runtime_identity(args.solver, solver_library) != expected_runtime:
            status(
                "failed_runtime_integrity",
                error="solver binary/library changed before a segment launch",
            )
            print(
                "Refusing to launch a segment with a changed solver runtime",
                flush=True,
            )
            return 3
        segment_start_time = float(checkpoint["time"]) if checkpoint else 0.0
        segment = segments_root / f"segment_{segment_index:04d}"
        segment.mkdir()
        config = segment / "config.conf"
        segment_overrides = list(overrides)
        # Explicit temporal-policy change; preserve the seed's exact metadata.
        if checkpoint is not None and args.restart_dt_scale != 1:
            segment_overrides.append(
                f"set double dt0 {checkpoint['dt'] * args.restart_dt_scale:.17g}"
            )
        write_segment_config(
            config,
            base_config,
            input_dir,
            checkpoint,
            overrides=segment_overrides,
            initial_velocity_init=args.initial_velocity_init,
            initial_flux_init=args.initial_flux_init,
            restart_pressure_init=args.restart_pressure_init,
            restart_flux_init=args.restart_flux_init,
        )
        log_path = segment / "run.log"
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = str(args.threads)
        prefix_lib = str(args.solver.parent.parent / "lib")
        environment["LD_LIBRARY_PATH"] = (
            prefix_lib
            + ":/usr/local/cuda/lib64:"
            + environment.get("LD_LIBRARY_PATH", "")
        )
        print(
            f"Starting {segment.name} from "
            + (
                f"checkpoint t={checkpoint['time']:.9g}"
                if checkpoint
                else "initial state"
            ),
            flush=True,
        )
        with log_path.open("ab", buffering=0) as log_stream:
            child = subprocess.Popen(
                [str(args.solver), str(config)],
                cwd=segment,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            last_dump: dict | None = None
            last_audit: dict | None = None
            last_audited_key: tuple[str, int] | None = None
            validation_failure: dict | None = None
            while child.poll() is None and not stop_requested:
                validation_failure = runtime_log_guard(log_path, tmax, args.minimum_dt)
                if validation_failure is not None:
                    print(
                        "Runtime guard: " + "; ".join(validation_failure["violations"]),
                        flush=True,
                    )
                    child.terminate()
                    break
                candidate = newest_dump(segment, expected_size)
                if candidate is not None:
                    candidate_key = (segment.name, int(candidate["index"]))
                    if candidate_key != last_audited_key:
                        last_audit = audit_paths(
                            dump_field_paths(candidate),
                            f"{segment.name}/dump_{int(candidate['index']):04d}",
                            float(candidate["time"]),
                        )
                        last_audited_key = candidate_key
                        if not last_audit["valid"]:
                            validation_failure = last_audit
                            print(
                                "Semantic validation failed: "
                                + "; ".join(last_audit["violations"]),
                                flush=True,
                            )
                            child.terminate()
                            break
                    last_dump = candidate
                    candidate_time = float(candidate["time"])
                    # Audit a dump as soon as it is complete, but promote it
                    # only after the following stat row reveals the exact dt
                    # needed to continue from that state. Re-evaluating the
                    # same candidate on every poll is intentional.
                    has_exact_restart_dt = bool(candidate.get("restart_dt_is_exact"))
                    should_commit = (
                        last_audit is not None
                        and last_audit["valid"]
                        and has_exact_restart_dt
                        and (
                            checkpoint is None
                            or candidate_time
                            > float(checkpoint["time"]) + progress_time_epsilon
                        )
                    )
                    if should_commit:
                        validation_failure = promote_dump(
                            candidate,
                            "initial" if checkpoint is None else "field_dump",
                            last_audit,
                        )
                        if validation_failure is not None:
                            child.terminate()
                            break
                        print(
                            f"Committed checkpoint {checkpoint['slot']} "
                            f"sequence={checkpoint['sequence']} "
                            f"t={checkpoint['time']:.9g}",
                            flush=True,
                        )
                wall_without_checkpoint = time.monotonic() - last_checkpoint_wall
                if wall_without_checkpoint > args.checkpoint_seconds:
                    validation_failure = checkpoint_deadline_failure(
                        wall_without_checkpoint
                    )
                    child.terminate()
                    break
                progress_time = (
                    float(last_dump["time"])
                    if last_dump is not None
                    else float(checkpoint["time"])
                    if checkpoint
                    else 0.0
                )
                status(
                    "running",
                    segment=segment.name,
                    restart_count=restart_count,
                    progress_time=progress_time,
                    progress_percent=min(100.0, 100.0 * progress_time / tmax),
                    latest_dump_index=(int(last_dump["index"]) if last_dump else None),
                    latest_semantic_audit=last_audit,
                )
                time.sleep(args.poll_seconds)

            return_code = child.poll()
            if return_code is None:
                try:
                    return_code = child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    return_code = child.wait()

        validation_failure = validation_failure or runtime_log_guard(
            log_path, tmax, args.minimum_dt
        )
        if validation_failure is not None:
            status(
                "failed_validation",
                segment=segment.name,
                return_code=return_code,
                progress_time=float(checkpoint["time"]) if checkpoint else 0.0,
                semantic_audit=validation_failure,
            )
            return 3

        last_dump = newest_dump(segment, expected_size)
        last_audit = None
        if last_dump is not None:
            last_audit = audit_paths(
                dump_field_paths(last_dump),
                f"{segment.name}/dump_{int(last_dump['index']):04d}:exit",
                float(last_dump["time"]),
            )
            if not last_audit["valid"]:
                status(
                    "failed_validation",
                    segment=segment.name,
                    return_code=return_code,
                    progress_time=float(checkpoint["time"]) if checkpoint else 0.0,
                    semantic_audit=last_audit,
                )
                return 3
        if exit_dump_can_be_promoted(last_dump, return_code) and (
            checkpoint is None
            or float(last_dump["time"])
            > float(checkpoint["time"]) + progress_time_epsilon
        ):
            validation_failure = promote_dump(
                last_dump,
                "shutdown" if stop_requested else "segment_exit",
                last_audit,
            )
            print(
                f"Committed exit checkpoint {checkpoint['slot']} "
                f"sequence={checkpoint['sequence']} t={checkpoint['time']:.9g}",
                flush=True,
            )
            if validation_failure is not None:
                status(
                    "failed_validation",
                    segment=segment.name,
                    return_code=return_code,
                    progress_time=float(checkpoint["time"]),
                    semantic_audit=validation_failure,
                )
                return 3

        if stop_requested:
            status(
                "stopped",
                segment=segment.name,
                return_code=return_code,
                progress_time=float(checkpoint["time"]) if checkpoint else 0.0,
            )
            return 0
        if return_code == 0:
            progress_time = float(checkpoint["time"]) if checkpoint else 0.0
            state = (
                "complete"
                if checkpoint_reaches_target(checkpoint, tmax)
                else "stopped_early"
            )
            status(
                state,
                segment=segment.name,
                return_code=return_code,
                progress_time=progress_time,
                progress_percent=(
                    100.0
                    if state == "complete"
                    else min(100.0, 100.0 * progress_time / tmax)
                ),
            )
            return 0 if state == "complete" else 2

        restart_count += 1
        made_progress = checkpoint is not None and (
            float(checkpoint["time"]) > segment_start_time + progress_time_epsilon
        )
        stalled_restart_count = 0 if made_progress else stalled_restart_count + 1
        if (
            checkpoint is None
            or restart_count > args.max_restarts
            or stalled_restart_count > args.max_stalled_restarts
        ):
            status(
                "failed",
                segment=segment.name,
                return_code=return_code,
                restart_count=restart_count,
                stalled_restart_count=stalled_restart_count,
                error=(
                    "No valid checkpoint, restart limit exceeded, or repeated "
                    "restart attempts made no forward progress"
                ),
            )
            return return_code or 1
        status(
            "restarting",
            segment=segment.name,
            return_code=return_code,
            restart_count=restart_count,
            stalled_restart_count=stalled_restart_count,
            progress_time=checkpoint["time"],
        )
        print(
            f"Solver exited with code {return_code}; restarting from "
            f"t={checkpoint['time']:.9g} in 10 seconds",
            flush=True,
        )
        time.sleep(10)
        segment_index += 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
