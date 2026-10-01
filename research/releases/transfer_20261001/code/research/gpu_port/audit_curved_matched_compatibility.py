"""Check whether the flat matched trace accepts the frozen steep BIE state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from liquid_dirichlet_transfer import boundary_traces
from physical_trace import ParentChordTrace


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=16)
    parser.add_argument("--nx", type=int, default=1024)
    args = parser.parse_args()
    with np.load(args.boundary) as archive:
        source = {name: archive[name].copy() for name in archive.files}
    curves, trace_report = boundary_traces(source, factor=args.factor)
    z0 = source["surface_z"]
    x0 = source["surface_x"]
    x = curves["surface"]["x"]
    z = curves["surface"]["z"]
    length = 64.0
    rejection = None
    try:
        ParentChordTrace(curves).flat_surface_jet_means(np.linspace(0, length, args.nx + 1))
    except ValueError as error:
        rejection = str(error)
    result = {
        "schema": "matched-curved-source-compatibility-v1",
        "source_path": str(args.boundary.resolve()),
        "source_sha256": sha(args.boundary),
        "code_sha256": sha(Path(__file__)),
        "physical_trace_source_sha256": sha(Path(__file__).with_name("physical_trace.py")),
        "boundary_trace_source_sha256": sha(Path(__file__).with_name("liquid_dirichlet_transfer.py")),
        "source_nodes": len(x0),
        "resample_factor": args.factor,
        "resampled_nodes": len(x),
        "receiver_nx": args.nx,
        "source_surface_height_span": float(np.ptp(z0)),
        "resampled_surface_height_span": float(np.ptp(z)),
        "flat_surface_height_tolerance": 1e-10,
        "source_nonpositive_x_segments": int(np.count_nonzero(np.diff(x0) <= 0)),
        "resampled_nonpositive_x_segments": int(np.count_nonzero(np.diff(x) <= 0)),
        "resampled_min_dx": float(np.min(np.diff(x))),
        "flat_matched_jet_rejection": rejection,
        "supported_by_current_method": rejection is None,
        "trace_preflight_pass": bool(trace_report["flux_preflight"]["pass"]),
        "interpretation": "This audits method-domain compatibility only. It is not a native trajectory or physical accuracy test.",
    }
    if rejection is None:
        raise AssertionError("Expected steep source to be rejected by flat-only method")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
