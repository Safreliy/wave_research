"""Export accepted BIE snapshots for a bounded handoff-time sensitivity screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


RESEARCH = Path(__file__).resolve().parents[1]
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from vof_handoff import export_handoff  # noqa: E402


def slice_snapshot(source: Path, destination: Path, requested_time: float) -> dict[str, object]:
    with np.load(source) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    times = np.asarray(data["time"], dtype=float)
    index = int(np.argmin(np.abs(times - requested_time)))
    count = len(times)
    snapshot: dict[str, np.ndarray] = {}
    for key, value in data.items():
        array = np.asarray(value)
        if array.ndim and array.shape[0] == count:
            snapshot[key] = array[index : index + 1]
        else:
            snapshot[key] = array
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **snapshot)
    return {
        "requested_time": requested_time,
        "selected_time": float(times[index]),
        "snapshot_index": index,
        "snapshot_archive": str(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--nz", type=int, default=192)
    parser.add_argument("--max-marker-spacing-cv", type=float, default=0.5)
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    for requested_time in args.times:
        tag = f"{requested_time:.6f}".replace(".", "p")
        snapshot_path = args.output_directory / f"bie_snapshot_t{tag}.npz"
        record = slice_snapshot(args.source, snapshot_path, requested_time)
        output_prefix = args.output_directory / f"handoff_source_t{tag}_x{args.nx}"
        metadata = export_handoff(
            snapshot_path,
            output_prefix,
            selection="last_admissible",
            nx=args.nx,
            nz=args.nz,
            export_face_fluxes=True,
            max_marker_spacing_cv=args.max_marker_spacing_cv,
        )
        record["handoff_prefix"] = str(output_prefix)
        record["handoff_metadata"] = metadata
        records.append(record)

    report = {
        "schema": "three-time-handoff-source-screen-v1",
        "source": str(args.source),
        "grid": {"nx": args.nx, "nz": args.nz},
        "maximum_marker_spacing_cv": args.max_marker_spacing_cv,
        "records": records,
        "evidence_class": "accepted-source transfer sensitivity; not overlap coupling",
    }
    report_path = args.output_directory / "source_export_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
