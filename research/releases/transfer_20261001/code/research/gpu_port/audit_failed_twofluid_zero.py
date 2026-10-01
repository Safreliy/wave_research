"""Localize the first-step zero-state loss in the physical-wave pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def field(path: Path, shape: tuple[int, int]) -> np.ndarray:
    data = np.fromfile(path, dtype="<f8")
    if data.size != np.prod(shape):
        raise ValueError((str(path), data.size, shape))
    return data.reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.case_root
    shape = (32, 512)
    h2 = (64 / 512) ** 2
    seg = root / "runs/full/zero/segments/segment_0000"
    initial = root / "zero"
    out: dict[str, object] = {}
    arrays = {}
    for kind in ("vf", "ebvf", "vx", "vy", "fluxx", "fluxy", "fluxxp", "fluxyp", "fluxeb"):
        arrays[kind] = [field(seg / f"{kind}_{i:04d}.raw", shape) for i in (0, 1)]
        input_path = initial / f"{kind}.raw"
        arrays[kind].append(field(input_path, shape) if input_path.exists() else None)
        out[kind] = {
            "t0_vs_input_linf": (float(np.max(np.abs(arrays[kind][0] - arrays[kind][2])))
                                 if arrays[kind][2] is not None else None),
            "firstdump_vs_t0_linf": float(np.max(np.abs(arrays[kind][1] - arrays[kind][0]))),
        }
    vol0 = arrays["vf"][0] * arrays["ebvf"][0]
    vol1 = arrays["vf"][1] * arrays["ebvf"][1]
    delta = (vol1 - vol0) * h2
    out["physical_liquid_volume"] = {
        "t0": float(np.sum(vol0) * h2),
        "firstdump": float(np.sum(vol1) * h2),
        "difference": float(np.sum(delta)),
        "row_difference": [float(x) for x in np.sum(delta, axis=1)],
        "top_20_changed_cells": [
            {"j": int(j), "i": int(i), "dv": float(delta[j, i]),
             "vf0": float(arrays["vf"][0][j, i]),
             "vf1": float(arrays["vf"][1][j, i]),
             "eb0": float(arrays["ebvf"][0][j, i]),
             "eb1": float(arrays["ebvf"][1][j, i])}
            for j, i in (np.unravel_index(k, shape) for k in np.argsort(np.abs(delta).ravel())[-20:][::-1])
        ],
    }
    out["firstdump_speed"] = {
        "linf": float(np.max(np.hypot(arrays["vx"][1], arrays["vy"][1]))),
        "nonzero_cells": int(np.count_nonzero(np.hypot(arrays["vx"][1], arrays["vy"][1]) > 1e-8)),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
