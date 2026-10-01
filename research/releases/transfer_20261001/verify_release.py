"""Verify every explicitly allowlisted file and selected archive member."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "SHA256SUMS.json").read_text(encoding="utf-8"))
    expected = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(entry["path"])
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size != entry["size"] or sha_file(path) != entry["sha256"]:
            raise AssertionError(entry["path"])
        expected.add(relative.as_posix())
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    } - {"SHA256SUMS.json"}
    if actual != expected:
        raise AssertionError({"missing": sorted(expected - actual), "unlisted": sorted(actual - expected)})
    subset_members = 0
    for name in ("native_baseline_selected_outputs.tar.gz", "twofluid_half_amplitude_outputs.tar.gz"):
        with tarfile.open(ROOT / "archives" / name, "r:gz") as archive:
            sub = json.loads(archive.extractfile("subset_manifest.json").read())
            for item in sub["members"]:
                blob = archive.extractfile(item["path"]).read()
                if len(blob) != item["size"] or hashlib.sha256(blob).hexdigest() != item["sha256"]:
                    raise AssertionError(name + "!" + item["path"])
                subset_members += 1
    print(json.dumps({"verified_files": len(expected), "verified_subset_members": subset_members}))


if __name__ == "__main__":
    main()
