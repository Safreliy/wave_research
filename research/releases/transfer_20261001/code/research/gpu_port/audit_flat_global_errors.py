"""Recompute global liquid-volume-weighted errors in retained flat tests.

This reads the frozen per-cell arrays, not the runner's summary metrics.  The
analytic centroid control evaluates the *exact* harmonic field at centroids;
it is an oracle, not a reconstruction from the sampled streamfunction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = args.source_root / "result.json"
    original = json.loads(summary.read_text(encoding="utf-8"))
    rows = []
    for row in original["rows"]:
        path = args.source_root / row["arrays"]
        if sha256(path) != row["arrays_sha256"]:
            raise ValueError(f"array digest mismatch: {path}")
        with np.load(path) as arrays:
            q = arrays["q"]
            ref = arrays["exact_phase_mean"]
            if ref.shape[0] != 2 or ref.shape[1:] != q.shape:
                raise ValueError(f"invalid array shape: {path}")
            if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
                raise ValueError(f"invalid liquid fraction: {path}")
            denominator = float(np.sum(q * np.sum(ref * ref, axis=0)))
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError(f"invalid norm denominator: {path}")
            errors = {}
            for label, key in (
                ("fitted", "fitted_phase_mean"),
                ("adjacent_mac", "arithmetic_mac_mean"),
                ("exact_centroid_oracle", "exact_centroid_oracle"),
            ):
                field = arrays[key]
                if field.shape != ref.shape or not np.isfinite(field).all():
                    raise ValueError(f"invalid {key}: {path}")
                numerator = float(np.sum(q * np.sum((field - ref) ** 2, axis=0)))
                errors[label] = float(np.sqrt(numerator / denominator))
            rows.append({
                "nx": row["nx"],
                "surface_liquid_fraction": row["surface_liquid_fraction"],
                "arrays": row["arrays"],
                "arrays_sha256": sha256(path),
                "global_liquid_weighted_relative_l2": errors,
                "adjacent_mac_over_fitted": errors["adjacent_mac"] / errors["fitted"],
                "fitted_over_exact_centroid_oracle": errors["fitted"] / errors["exact_centroid_oracle"],
            })
    output = {
        "schema": "flat-global-liquid-error-audit-v1",
        "source_summary_sha256": sha256(summary),
        "norm": "sqrt(sum_cells q*|U-U_exact_phase_mean|^2 / sum_cells q*|U_exact_phase_mean|^2)",
        "scope": "all liquid cells in each analytic flat-domain case, before native projection",
        "interpretation": "fitted beats adjacent-MAC arithmetic but loses to exact-field centroid oracle in every retained case; the latter is not the same-streamfunction bilinear-centroid baseline",
        "rows": rows,
        "code_sha256": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
