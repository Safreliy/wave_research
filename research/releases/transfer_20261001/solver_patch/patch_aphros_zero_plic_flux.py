"""Apply an exact-zero displacement identity to an isolated Aphros source copy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError((old, text.count(old)))
    return text.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    header = args.source_root / "src/solver/reconst.h"
    before = header.read_bytes()
    text = before.decode()
    changes = {
        "const Vect& n, Scal a, const Vect& h, Scal dx, size_t d) {\n    // Acceptor":
            "const Vect& n, Scal a, const Vect& h, Scal dx, size_t d) {\n"
            "    if (dx == 0.) return 0.; // zero transported volume; avoid degenerate box\n"
            "    // Acceptor",
        "Vect n, Scal a, Vect h, Scal dx, Scal dxu, size_t d) {\n    const Scal u =":
            "Vect n, Scal a, Vect h, Scal dx, Scal dxu, size_t d) {\n"
            "    if (dx == 0.) return 0.; // no transported volume\n"
            "    const Scal u =",
        "Vect n, Scal a, Vect h, Scal q, Scal dt, size_t d) {\n    const Scal s =":
            "Vect n, Scal a, Vect h, Scal q, Scal dt, size_t d) {\n"
            "    if (q == 0.) return 0.; // exact zero flux, independent of PLIC normal\n"
            "    const Scal s =",
        "Vect n, Scal a, Vect h, Scal q, Scal qu, Scal dt, size_t d) {\n    const Scal s =":
            "Vect n, Scal a, Vect h, Scal q, Scal qu, Scal dt, size_t d) {\n"
            "    if (q == 0.) return 0.; // exact zero flux, independent of stretching\n"
            "    const Scal s =",
    }
    for old, new in changes.items():
        text = replace_once(text, old, new)
    after = text.encode()
    header.write_bytes(after)
    print(json.dumps({"path": str(header), "before_sha256": sha(before),
                      "after_sha256": sha(after), "guards": list(changes)}, indent=2))


if __name__ == "__main__":
    main()
