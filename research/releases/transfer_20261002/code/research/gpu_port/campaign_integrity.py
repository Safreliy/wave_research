"""Integrity and runtime-identity contracts for sealed Aphros campaigns."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_value(text: str, value_type: str, key: str) -> str:
    matches = re.findall(
        rf"(?m)^set\s+{re.escape(value_type)}\s+{re.escape(key)}\s+([^\s#]+)",
        text,
    )
    if len(matches) != 1:
        raise ValueError(f"expected one {key} setting, found {len(matches)}")
    return matches[0]


def verified_manifest(bundle: Path) -> tuple[dict, str]:
    """Load a sealed campaign definition and verify every static artifact."""
    manifest_path = bundle / "campaign_manifest.json"
    manifest_sha256 = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "aphros-gpu-convergence-campaign-v6":
        raise ValueError("campaign is not a sealed v6 bundle")
    if manifest.get("scientific_mode") not in {"convergence", "ablation"}:
        raise ValueError("campaign has no explicit scientific mode")
    if manifest.get("initial_flux_mode") not in {"raw", "reconstruct"}:
        raise ValueError("campaign has no valid initial-flux contract")
    if manifest.get("initial_velocity_mode") not in {"raw", "zero"}:
        raise ValueError("campaign has no valid initial-velocity contract")
    if manifest.get("restart_pressure_mode") not in {"raw", "zero"}:
        raise ValueError("campaign has no valid restart-pressure contract")
    if manifest.get("restart_flux_mode") not in {"raw", "reconstruct"}:
        raise ValueError("campaign has no valid restart-flux contract")
    if manifest.get("initial_dt_policy") != "per-case dt0 equals dtmax":
        raise ValueError("campaign has no valid initial time-step policy")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("campaign has no cases")
    for case in cases:
        initial_dt = float(case.get("initial_dt", 0))
        dtmax = float(case.get("dtmax", 0))
        if initial_dt <= 0 or initial_dt != dtmax:
            raise ValueError(
                f"case {case.get('name', '<unnamed>')} must start with dt0=dtmax"
            )
    growth = float(manifest.get("dt_growth_max", 0))
    if not 1 < growth <= 2:
        raise ValueError("campaign time-step growth limit must be in (1, 2]")
    if manifest.get("exact_target_time") is not True:
        raise ValueError("campaign does not require exact target-time landing")
    if manifest.get("initial_autodt") is not True:
        raise ValueError("campaign does not compute stability limits before step zero")
    checkpoint_seconds = float(manifest.get("checkpoint_seconds", 0))
    if not 0 < checkpoint_seconds <= 3600:
        raise ValueError("campaign checkpoint deadline must be in (0, 3600]")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("campaign manifest has no artifact inventory")
    for relative, expected in artifacts.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe campaign artifact path: {relative}")
        path = bundle / relative_path
        if not path.is_file():
            raise ValueError(f"campaign artifact is missing: {relative}")
        if path.stat().st_size != int(expected["size"]):
            raise ValueError(f"campaign artifact size mismatch: {relative}")
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"campaign artifact fingerprint mismatch: {relative}")
    for case in cases:
        name = str(case["name"])
        config = (bundle / "configs" / f"{name}.conf").read_text(encoding="utf-8")
        expected_numeric = {
            ("int", "bsx"): int(case["nx"]) // 64,
            ("int", "bsy"): int(case["vertical_cells"]) // 8,
            ("double", "cfl"): float(case["cfl"]),
            ("double", "cfla"): float(case["cfl"]),
            ("double", "dt0"): float(case["initial_dt"]),
            ("double", "dtmax"): float(case["dtmax"]),
            ("double", "tmax"): float(case["target_time"]),
            ("double", "dt_growth_max"): growth,
            ("double", "dump_field_dt"): float(case["dump_field_dt"]),
            ("double", "embed_aperture_eps"): 0.0,
            ("int", "initial_autodt"): 1,
            ("int", "exact_tmax"): 1,
        }
        for (value_type, key), expected_value in expected_numeric.items():
            actual = float(config_value(config, value_type, key))
            if actual != expected_value:
                raise ValueError(
                    f"case {name} config/manifest mismatch for {key}: "
                    f"{actual} != {expected_value}"
                )
    for level, source_text in manifest["projected_sources"].items():
        source = Path(source_text)
        if not source.is_file():
            raise ValueError(f"projected source is missing for {level}: {source}")
        if sha256(source) != manifest["projected_source_sha256"][level]:
            raise ValueError(f"projected source fingerprint mismatch for {level}")
    return manifest, manifest_sha256


def assert_manifest_unchanged(manifest_path: Path, expected_sha256: str) -> None:
    if sha256(manifest_path) != expected_sha256:
        raise RuntimeError("campaign manifest changed while the campaign was running")


def runtime_fingerprint(solver: Path, solver_library: Path) -> dict[str, str]:
    for path in (solver, solver_library):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "solver_path": str(solver),
        "solver_sha256": sha256(solver),
        "solver_library_path": str(solver_library),
        "solver_library_sha256": sha256(solver_library),
    }
