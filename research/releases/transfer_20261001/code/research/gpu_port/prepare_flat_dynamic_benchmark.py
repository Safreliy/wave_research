"""Prepare a preregistered flat harmonic native transfer-propagation test.

The exact branch uses closed-form phase-cell velocity means for the prescribed
initial harmonic field. It is an exact *input* reference, not an exact solution
of the ensuing viscous two-phase Navier--Stokes equations. All branches share
the same VOF, embedded bed, gas velocities, and MAC face velocities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
from pathlib import Path

import numpy as np
import scipy
import shapely

from aphros_checkpoint_supervisor import audit_initial_state
from liquid_dirichlet_transfer import boundary_traces, transfer
from prepare_aphros_initial_state import (
    DOMAIN_LENGTH,
    EMBEDDED_BOUNDARY_OFFSET,
    VERTICAL_ORIGIN,
    fluid_aperture,
    write_body_polygon,
)


MODE = 16
WAVE_NUMBER = 2.0 * np.pi * MODE / DOMAIN_LENGTH
POTENTIAL_AMPLITUDE = 0.02
BACKGROUND_CURRENT = 0.0
BED_SOLVER_Y = 0.53
SURFACE_SOLVER_Y = 1.5000625
GRAVITY = 1.0
SURFACE_TENSION = 0.001
DENSITY_LIQUID = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def harmonic_source(bed: float, surface: float, ntrace: int) -> dict:
    x = DOMAIN_LENGTH * np.arange(ntrace) / ntrace
    depth = surface - bed
    k = WAVE_NUMBER
    amplitude = POTENTIAL_AMPLITUDE
    norm = np.cosh(k * depth)
    surface_phi = amplitude * np.cos(k * x)
    bottom_phi = surface_phi / norm
    surface_q = amplitude * k * np.tanh(k * depth) * np.cos(k * x)
    return {
        "surface_x": x,
        "surface_z": np.full_like(x, surface),
        "bottom_x": x,
        "bottom_z": np.full_like(x, bed),
        "surface_potential": surface_phi,
        "bottom_potential": bottom_phi,
        "surface_normal_derivative": surface_q,
        "bottom_normal_derivative": np.zeros_like(x),
        "surface_velocity_x": -amplitude * k * np.sin(k * x),
        "surface_velocity_z": surface_q,
        "source_volume": DOMAIN_LENGTH * depth,
        "target_momentum_x": 0.0,
        "target_momentum_z": 0.0,
    }


def exact_means(xcentres: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                bed: float, surface: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    k = WAVE_NUMBER
    norm = np.cosh(k * (surface - bed))
    sinc = np.sinc(k * h / (2.0 * np.pi))
    dz = hi - lo
    u = np.zeros((len(lo), len(xcentres)), dtype=float)
    w = np.zeros_like(u)
    wet = dz > 0
    vertical_u = (np.sinh(k * (hi[wet] - bed)) - np.sinh(k * (lo[wet] - bed))) / dz[wet]
    vertical_w = (np.cosh(k * (hi[wet] - bed)) - np.cosh(k * (lo[wet] - bed))) / dz[wet]
    u[wet] = (-POTENTIAL_AMPLITUDE * sinc / norm
              * vertical_u[:, None] * np.sin(k * xcentres)[None, :])
    w[wet] = (POTENTIAL_AMPLITUDE * sinc / norm
              * vertical_w[:, None] * np.cos(k * xcentres)[None, :])
    return u, w


def bilinear_centroid_means(psi: np.ndarray, q: np.ndarray, zedge: np.ndarray,
                            bed: float, surface: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    lo = np.maximum(zedge[:-1], bed)
    hi = np.minimum(zedge[1:], surface)
    az = (0.5 * (lo + hi) - zedge[:-1]) / h
    p00 = psi[:-1]
    p01 = psi[1:]
    p10 = np.roll(p00, -1, axis=1)
    p11 = np.roll(p01, -1, axis=1)
    mixed = p11 - p10 - p01 + p00
    ax = 0.5
    u = (p01 - p00 + mixed * ax) / h
    w = -(p10 - p00 + mixed * az[:, None]) / h
    if np.any((q > 0) & ((az[:, None] < -1e-12) | (az[:, None] > 1+1e-12))):
        raise ValueError("invalid liquid centroid")
    return u, w


def weighted_error(a: np.ndarray, b: np.ndarray, q: np.ndarray) -> float:
    numerator = float(np.sum(q * np.sum((a - b) ** 2, axis=0)))
    denominator = float(np.sum(q * np.sum(b * b, axis=0)))
    return math.sqrt(numerator / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, choices=(512, 1024, 2048), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trace-multiplier", type=int, choices=(1, 4), default=1)
    parser.add_argument("--physical-trace", choices=("linear", "parent_hermite_matched"),
                        default="linear")
    parser.add_argument("--common-input-root", type=Path)
    args = parser.parse_args()
    if args.trace_multiplier == 4 and args.common_input_root is None and args.nx != 2048:
        parser.error("4x trace control needs frozen original exact-input root")
    if args.physical_trace == "parent_hermite_matched":
        if args.trace_multiplier != 1 or args.common_input_root is None:
            parser.error("matched trace requires default trace factor and frozen common input")
    elif args.trace_multiplier == 1 and args.common_input_root is not None:
        parser.error("common-input-root applies only to 4x trace control or matched trace")
    args.root.mkdir(parents=True, exist_ok=False)
    nx, ny = args.nx, args.nx // 16
    h = DOMAIN_LENGTH / nx
    zedge = VERTICAL_ORIGIN + np.arange(ny + 1) * h
    xedge = np.arange(nx + 1) * h
    x = xedge[:-1]
    bed = VERTICAL_ORIGIN + BED_SOLVER_Y + EMBEDDED_BOUNDARY_OFFSET
    surface = VERTICAL_ORIGIN + SURFACE_SOLVER_Y
    depth = surface - bed
    omega = math.sqrt((GRAVITY * WAVE_NUMBER
                       + SURFACE_TENSION * WAVE_NUMBER ** 3 / DENSITY_LIQUID)
                      * math.tanh(WAVE_NUMBER * depth))
    period = 2.0 * math.pi / omega
    reference_time = 0.1 * period
    ntrace = 512
    trace_factor = (8 if nx <= 1024 else 16) * args.trace_multiplier
    protocol = {
        "schema": "flat-harmonic-native-dynamic-benchmark-v1",
        "claim_scope": "same-grid error propagation relative to an exact initial phase-mean input, not exact Navier--Stokes trajectory",
        "geometry": {"period": DOMAIN_LENGTH, "nx": nx, "ny": ny,
                     "bed_solver_y_before_native_offset": BED_SOLVER_Y,
                     "bed_solver_y_effective": BED_SOLVER_Y + EMBEDDED_BOUNDARY_OFFSET,
                     "surface_solver_y": SURFACE_SOLVER_Y, "depth": depth,
                     "horizontal_mode": MODE, "wave_number": WAVE_NUMBER},
        "initial_field": {"potential_amplitude": POTENTIAL_AMPLITUDE,
                          "background_current": BACKGROUND_CURRENT,
                          "boundary_source_nodes": ntrace,
                          "boundary_trace_refinement_factor": trace_factor,
                          "trace_multiplier_against_default": args.trace_multiplier},
        "physical_trace": args.physical_trace,
        "physics": {"gravity": GRAVITY, "surface_tension": SURFACE_TENSION,
                    "liquid_density": DENSITY_LIQUID},
        "linear_capillary_gravity_period": period,
        "target_time_one_tenth_period": reference_time,
        "controls": ["bilinear streamfunction curl at liquid centroids from same sampled psi",
                     "closed-form exact initial phase means"],
        "methods": ["fitted", "bilinear_centroid", "exact_initial_means"],
        "primary_metric": "global liquid-volume-weighted velocity relative L2 at identical times against exact-initial-means native branch",
        "secondary_metrics": ["VOF relative L2", "column depth relative L2"],
        "same_runtime_fields": ["q", "vf", "cs", "body.dat", "gas velocities", "MAC face velocities"],
        "comparison_limit": "exact initial means are not exact later-time physical fields; the branch shares Aphros truncation and projection errors",
        "code_sha256": sha256(Path(__file__)),
        "transfer_code_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("liquid_dirichlet_transfer.py", "cutcell_transfer.py",
                         "physical_trace.py")
        },
    }
    if args.common_input_root is not None:
        protocol["reused_reference_native_root"] = (
            f"/opt/gpu-cfd/flat_harmonic_dynamic_20261001/"
            f"flat_native_L{int(round(math.log2(nx)))}"
        )
        protocol["common_input_root"] = str(args.common_input_root.resolve())
    (args.root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    source = harmonic_source(bed, surface, ntrace)
    np.savez_compressed(args.root / "harmonic_source.npz", **source)
    curves, trace_report = boundary_traces(source, factor=trace_factor)
    data, transfer_report = transfer(
        curves, x, zedge, backend="numpy", gas=True,
        continuation_order=2, cutcell_moments=True,
        physical_trace=args.physical_trace,
    )
    q = data["volume_fraction"]
    cs = fluid_aperture(xedge, zedge, source["bottom_x"], source["bottom_z"])
    if np.max(q-cs) > 1e-10 or abs(q.sum()*h*h-DOMAIN_LENGTH*depth) > 1e-9:
        raise ValueError("analytic/native liquid geometry mismatch")
    vf = np.divide(q, cs, out=np.zeros_like(q), where=cs > 0)
    original_reference = args.common_input_root
    if original_reference is not None:
        old_q = np.fromfile(original_reference / "q.raw", dtype="<f8").reshape(ny, nx)
        if np.max(np.abs(q-old_q)) > 1e-12:
            raise ValueError("trace intervention changed liquid geometry")
        q = old_q
        cs = np.fromfile(original_reference / "cs.raw", dtype="<f8").reshape(ny, nx)
        vf = np.fromfile(original_reference / "vf.raw", dtype="<f8").reshape(ny, nx)
        protocol["reused_exact_input_sha256"] = {
            p.name: sha256(p) for p in original_reference.iterdir() if p.is_file()
        }
        (args.root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    lo = np.maximum(zedge[:-1], bed)
    hi = np.minimum(zedge[1:], surface)
    true_u, true_w = exact_means(x + h/2, lo, hi, bed, surface, h)
    centroid_u, centroid_w = bilinear_centroid_means(data["mac_streamfunction"], q, zedge, bed, surface, h)
    base_u = np.asarray(data["cell_velocity_x_projected"])
    base_w = np.asarray(data["cell_velocity_z_projected"])
    fields = {
        "fitted": np.array([base_u.copy(), base_w.copy()]),
        "bilinear_centroid": np.array([base_u.copy(), base_w.copy()]),
        "exact_initial_means": np.array([base_u.copy(), base_w.copy()]),
    }
    wet = q > 0
    if original_reference is not None:
        original_exact = np.stack([
            np.fromfile(original_reference / f"{name}.raw", dtype="<f8").reshape(ny, nx)
            for name in ("vx", "vy")
        ])
        if np.max(np.abs((original_exact - np.array([true_u, true_w]))[:, wet])) > 1e-12:
            raise ValueError("old exact phase means differ from frozen analytic target")
        for field in fields.values():
            field[:, ~wet] = original_exact[:, ~wet]
    fields["bilinear_centroid"][0][wet] = centroid_u[wet]
    fields["bilinear_centroid"][1][wet] = centroid_w[wet]
    fields["exact_initial_means"][0][wet] = true_u[wet]
    fields["exact_initial_means"][1][wet] = true_w[wet]
    # A uniform periodic current has no periodic scalar potential; superpose
    # its curl-free velocity after transferring the zero-mean harmonic field.
    for field in fields.values():
        field[0][cs > 0] += BACKGROUND_CURRENT
        field[:, cs <= 0] = 0
    face_x = np.asarray(data["face_velocity_x_projected"]).copy()
    face_x += BACKGROUND_CURRENT
    face_y = np.asarray(data["face_velocity_z_projected"])
    common = {
        "q": q, "vf": vf, "cs": cs,
        "fluxx": face_x, "fluxxp": np.roll(face_x, -1, axis=1),
        "fluxy": face_y[:-1], "fluxyp": face_y[1:],
        "fluxeb": np.zeros_like(q),
    }
    if original_reference is not None:
        for name in common:
            common[name] = np.fromfile(original_reference / f"{name}.raw", dtype="<f8").reshape(ny, nx)
    target_momentum = [BACKGROUND_CURRENT * DOMAIN_LENGTH * depth, 0.0]
    reports = {}
    for label, velocity in fields.items():
        folder = args.root / label
        folder.mkdir()
        if original_reference is not None:
            for name in (*common, "body.dat"):
                shutil.copy2(original_reference / (name if name == "body.dat" else f"{name}.raw"),
                             folder / (name if name == "body.dat" else f"{name}.raw"))
        else:
            for name, array in common.items():
                np.asarray(array, dtype="<f8").tofile(folder / f"{name}.raw")
            write_body_polygon(folder / "body.dat", source["bottom_x"],
                               np.full_like(source["bottom_z"], bed - EMBEDDED_BOUNDARY_OFFSET))
        if label == "exact_initial_means" and original_reference is not None:
            for name in ("vx.raw", "vy.raw", "initial_state.json"):
                shutil.copy2(original_reference / name, folder / name)
        else:
            np.asarray(velocity[0], dtype="<f8").tofile(folder / "vx.raw")
            np.asarray(velocity[1], dtype="<f8").tofile(folder / "vy.raw")
        momentum = [float(np.sum(q * velocity[i]) * h*h) for i in range(2)]
        initial = {"schema": "flat-harmonic-native-initial-v1", "nx": nx, "ny": ny,
                   "cell_size": h, "target_liquid_volume": DOMAIN_LENGTH*depth,
                   "target_liquid_momentum": target_momentum,
                   "liquid_momentum": momentum,
                   "branch": label, "protocol_sha256": sha256(args.root / "protocol.json")}
        if label != "exact_initial_means" or original_reference is None:
            (folder / "initial_state.json").write_text(json.dumps(initial, indent=2) + "\n")
        preflight = audit_initial_state(
            folder, initial_mass_relative_tolerance=1e-8,
            face_divergence_tolerance=1e-10, shared_face_tolerance=1e-10,
            momentum_absolute_tolerance=1e-6,
        )
        if not preflight["valid"]:
            raise ValueError(f"{label} preflight: {preflight['violations']}")
        reports[label] = {"preflight": preflight,
                          "input_hashes": {p.name: sha256(p) for p in folder.iterdir() if p.is_file()},
                          "initial_global_liquid_velocity_error_against_exact_harmonic":
                          weighted_error(velocity, fields["exact_initial_means"], q)}
    report = {"schema": "flat-harmonic-native-preparation-v1",
              "protocol_sha256": sha256(args.root / "protocol.json"),
              "source_sha256": sha256(args.root / "harmonic_source.npz"),
              "trace_report": trace_report,
              "transfer_report": transfer_report,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "scipy": scipy.__version__, "shapely": shapely.__version__},
              "branches": reports}
    (args.root / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"root": str(args.root), "period": period,
                      "reference_time": reference_time,
                      "initial_errors": {key: value["initial_global_liquid_velocity_error_against_exact_harmonic"]
                                         for key, value in reports.items()}}, indent=2))


if __name__ == "__main__":
    main()
