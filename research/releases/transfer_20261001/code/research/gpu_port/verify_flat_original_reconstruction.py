"""Check that the current compatible builder reproduces frozen original inputs.

The executed original builder source was not archived. This checks only the
numerical input files, not historical source identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--reconstructed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    relevant = [p for p in sorted(args.reconstructed.rglob("*"))
                if p.is_file() and (p.suffix == ".raw" or p.name == "body.dat")]
    if len(relevant) != 33:
        raise ValueError(f"expected 33 native raw/body inputs, found {len(relevant)}")
    old_protocol = json.loads((args.original / "protocol.json").read_text())
    used_frozen_reference_copy = bool(old_protocol.get("common_input_root"))
    matches = []
    for reconstructed in relevant:
        relative = reconstructed.relative_to(args.reconstructed)
        original = args.original / relative
        original_hash = sha256(original)
        reconstructed_hash = sha256(reconstructed)
        if original_hash != reconstructed_hash:
            raise ValueError(f"reconstruction differs: {relative}")
        name = relative.name
        copied = (used_frozen_reference_copy and
                  (name in {"q.raw", "vf.raw", "cs.raw", "fluxx.raw", "fluxxp.raw",
                            "fluxy.raw", "fluxyp.raw", "fluxeb.raw", "body.dat"}
                   or (relative.parts[0] == "exact_initial_means" and name in {"vx.raw", "vy.raw"})))
        matches.append({"path": relative.as_posix(), "sha256": original_hash,
                        "workflow_origin": "copied_frozen_reference" if copied else "newly_computed"})
    copied_count = sum(item["workflow_origin"] == "copied_frozen_reference" for item in matches)
    report = {
        "schema": "flat-original-input-reconstruction-v1",
        "claim_limit": "Current compatible preparation workflow reproduces all 33 numerical raw/body inputs byte-for-byte; refined workflows deliberately copy frozen exact-reference/common arrays. Executed earlier builder source snapshots are unavailable.",
        "original_protocol_sha256": sha256(args.original / "protocol.json"),
        "executed_original_builder_sha256_recorded_in_protocol": old_protocol["code_sha256"],
        "current_builder_sha256": sha256(Path(__file__).with_name("prepare_flat_dynamic_benchmark.py")),
        "current_builder_source_same_as_executed_original": False,
        "workflow_copied_frozen_reference_file_count": copied_count,
        "workflow_newly_computed_file_count": len(matches) - copied_count,
        "matching_inputs": matches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"matched_inputs": len(matches), "report": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
