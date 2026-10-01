"""Check what changes when the frozen flat source trace is refined fourfold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from liquid_dirichlet_transfer import boundary_traces, transfer
from prepare_aphros_initial_state import DOMAIN_LENGTH, VERTICAL_ORIGIN


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--original-root", type=Path, required=True)
    p.add_argument("--refined-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    original_protocol = json.loads((args.original_root / "protocol.json").read_text())
    refined_protocol = json.loads((args.refined_root / "protocol.json").read_text())
    n = original_protocol["geometry"]["nx"]
    if refined_protocol["geometry"] != original_protocol["geometry"]:
        raise ValueError("geometry differs")
    source_files = [root / "harmonic_source.npz" for root in (args.original_root, args.refined_root)]
    with np.load(source_files[0]) as aa, np.load(source_files[1]) as bb:
        if set(aa.files) != set(bb.files) or any(not np.array_equal(aa[key], bb[key]) for key in aa.files):
            raise ValueError("source boundary data differ")
        source = {key: aa[key] for key in aa.files}
    h = DOMAIN_LENGTH / n
    x = np.arange(n)*h
    z = VERTICAL_ORIGIN + np.arange(n//16+1)*h
    factors = [original_protocol["initial_field"]["boundary_trace_refinement_factor"],
               refined_protocol["initial_field"]["boundary_trace_refinement_factor"]]
    if factors[1] != 4*factors[0]:
        raise ValueError("not a fourfold trace test")
    data = []
    curves = []
    for factor in factors:
        curve, _ = boundary_traces(source, factor=factor)
        state, _ = transfer(curve, x, z, backend="numpy", gas=True,
                            continuation_order=2, cutcell_moments=True)
        curves.append(curve)
        data.append(state)
    a, b = data
    q = a["volume_fraction"]
    if not np.array_equal(q, b["volume_fraction"]):
        raise ValueError("liquid geometry q changed")
    surface_fields = {}
    for name in ("surface", "bottom"):
        surface_fields[name] = {
            field: float(np.max(np.abs(curves[0][name][field]-curves[1][name][field][::4])))
            for field in ("x", "z", "psi", "psi_x", "psi_z")
        }
    result = {
        "schema": "flat-trace-fourfold-diagnosis-v1",
        "nx": n, "original_factor": factors[0], "refined_factor": factors[1],
        "original_source_sha256": sha256(source_files[0]),
        "refined_source_sha256": sha256(source_files[1]),
        "source_arrays_exactly_equal": True,
        "q_arrays_exactly_equal": True,
        "shared_trace_node_max_differences": surface_fields,
        "mac_streamfunction_max_abs_change": float(np.max(np.abs(a["mac_streamfunction"]-b["mac_streamfunction"]))),
        "mac_streamfunction_relative_l2_change": float(np.linalg.norm(a["mac_streamfunction"]-b["mac_streamfunction"])/np.linalg.norm(a["mac_streamfunction"])),
        "mac_face_x_max_abs_change": float(np.max(np.abs(a["face_velocity_x_projected"]-b["face_velocity_x_projected"]))),
        "mac_face_z_max_abs_change": float(np.max(np.abs(a["face_velocity_z_projected"]-b["face_velocity_z_projected"]))),
        "fitted_u_max_abs_change_wet": float(np.max(np.abs(a["cell_velocity_x_projected"][q>0]-b["cell_velocity_x_projected"][q>0]))),
        "fitted_w_max_abs_change_wet": float(np.max(np.abs(a["cell_velocity_z_projected"][q>0]-b["cell_velocity_z_projected"][q>0]))),
        "code_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
