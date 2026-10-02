"""Summarize isolated guard source/build and one-step native causal evidence."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def member_json(archive: Path, member: str) -> dict:
    with tarfile.open(archive, "r:gz") as stream:
        target = stream.extractfile(member)
        if target is None:
            raise FileNotFoundError(member)
        return json.load(target)


def main() -> None:
    home = Path(__file__).resolve().parents[1] / "results/publication_package/transfer_physical_pilot_20261001"
    root = home / "source_build_guard"
    sources = {variant: root / f"aphros_guard_{variant}_source.tar.gz" for variant in ("baseline", "patched")}
    runtimes = {variant: root / f"aphros_guard_{variant}_runtime.tar.gz" for variant in ("baseline", "guarded")}
    source_manifests = {name: member_json(path, "SOURCE_TREE_MANIFEST.json") for name, path in sources.items()}
    old = {x["path"]: x["sha256"] for x in source_manifests["baseline"]["files"]}
    new = {x["path"]: x["sha256"] for x in source_manifests["patched"]["files"]}
    if old.keys() != new.keys():
        raise ValueError("source file sets differ")
    differences = {path: {"baseline_sha256": old[path], "patched_sha256": new[path]}
                   for path in old if old[path] != new[path]}
    if set(differences) != {"src/solver/reconst.h", "src/util/gitgen.cpp"}:
        raise ValueError(differences)
    regression = {name: (root / f"zero_flux_regression_{name}.txt").read_text().splitlines()
                  for name in ("old", "guarded")}
    nonzero = {name: [row for row in rows if row.startswith("nonzero ")]
               for name, rows in regression.items()}
    zero = {name: [row for row in rows if row.startswith("zero ")]
            for name, rows in regression.items()}
    if nonzero["old"] != nonzero["guarded"]:
        raise ValueError("nonzero flux outputs differ")
    if len(zero["guarded"]) != 16 or not all(row.endswith("pass=1") for row in zero["guarded"]):
        raise ValueError("guarded zero tests failed")
    zero_archive = home / "twofluid_zero_diagnostic_outputs.tar.gz"
    with tarfile.open(zero_archive, "r:gz") as stream:
        def run(variant: str) -> dict:
            base = f"guard_causal_runs/{variant}_zero/"
            status = json.load(stream.extractfile(base + "status.json"))
            audits = [json.loads(x) for x in stream.extractfile(base + "semantic_audit.jsonl").read().decode().splitlines()]
            return {"state": status["state"],
                    "time": audits[-1]["simulation_time"],
                    "final_physical_mass_relative_error": audits[-1]["runtime_mass_relative_error"],
                    "final_cartesian_vof_mass_relative_error": audits[-1]["runtime_vof_mass_relative_error"],
                    "solver_sha256": status["runtime_fingerprint"]["solver_sha256"],
                    "library_sha256": status["runtime_fingerprint"]["solver_library_sha256"]}
        contrast = {variant: run(variant) for variant in ("baseline", "guarded")}
    runtime_manifests = {name: member_json(path, "RUNTIME_MANIFEST.json") for name, path in runtimes.items()}
    report = {
        "schema": "isolated-aphros-zero-flux-guard-build-evidence-v1",
        "upstream_repository": "https://github.com/cselab/aphros.git",
        "upstream_head_commit": "b60ce3da52c19935fa24c778f62f02141eaf7f80",
        "license_in_source_tars": "LICENSE (MIT)",
        "source_archives": {name: {"sha256": sha(path), "files": len(source_manifests[name]["files"]),
                                    "archive": path.name} for name, path in sources.items()},
        "source_file_hash_differences": differences,
        "gitgen_note": "src/util/gitgen.cpp differs only in generated dirty-file list, which adds src/solver/reconst.h; physical source difference is the guarded header",
        "guard_patch_sha256": sha(root / "zero_flux_guard.patch"),
        "runtime_archives": {name: {"sha256": sha(path), "files": runtime_manifests[name]["files"],
                                     "archive": path.name} for name, path in runtimes.items()},
        "build_state_files": {path.name: sha(path) for path in root.glob("guard_build_*") if path.is_file()},
        "build_environment": "WSL Linux on authorized GPU host; GCC 13.3, CMake 3.28, Ninja, CUDA toolkit 13.0, existing AMGX shared library; CMakeCache and build graph archived",
        "successful_build_commands": [
            "env APHROS_PREFIX=/opt/gpu-cfd/prefix_zero_flux_guard_baseline_20261001 cmake -S /opt/gpu-cfd/aphros_zero_flux_guard_20261001/src -B /opt/gpu-cfd/aphros_zero_flux_guard_build_20261001 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 -DCUDAToolkit_NVCC_EXECUTABLE=/usr/local/cuda-13.0/bin/nvcc -DCUDA_CUDART=/usr/local/cuda-13.0/lib64/libcudart.so -DUSE_AMGX=ON -DUSE_AVX=ON -DUSE_BACKEND_LOCAL=ON -DUSE_BACKEND_NATIVE=ON -DUSE_BACKEND_CUBISM=OFF -DUSE_CONF2PY=ON -DUSE_DIM2=ON -DUSE_DIM3=OFF -DUSE_HDF=OFF -DUSE_HYPRE=OFF -DUSE_MFER=ON -DUSE_MPI=ON -DUSE_OPENMP=ON -DUSE_TESTS=OFF -DFIND_HDF=OFF -DUSE_EXPLORER=OFF",
            "cmake --build /opt/gpu-cfd/aphros_zero_flux_guard_build_20261001 --target ap.mfer -j 12",
            "cmake --build /opt/gpu-cfd/aphros_zero_flux_guard_build_20261001 --target ap.conf2py -j 12",
            "cmake --install /opt/gpu-cfd/aphros_zero_flux_guard_build_20261001",
            "env APHROS_PREFIX=/opt/gpu-cfd/prefix_zero_flux_guard_20261001 cmake -S /opt/gpu-cfd/aphros_zero_flux_guard_20261001/src -B /opt/gpu-cfd/aphros_zero_flux_guard_build_20261001",
            "cmake --build /opt/gpu-cfd/aphros_zero_flux_guard_build_20261001 --target ap.mfer ap.conf2py -j 12",
            "cmake --install /opt/gpu-cfd/aphros_zero_flux_guard_build_20261001",
        ],
        "regression": {"guarded_zero_cases": len(zero["guarded"]),
                       "guarded_zero_passed": sum(row.endswith("pass=1") for row in zero["guarded"]),
                       "unpatched_zero_failed": sum(row.endswith("pass=0") for row in zero["old"]),
                       "nonzero_cases_bitwise_equal": len(nonzero["old"])},
        "native_zero_one_step_contrast": contrast,
        "native_zero_archive_sha256": sha(zero_archive),
        "causal_limit": "fresh unpatched and guarded builds come from the same isolated source copy, apart from reconst.h and generated version metadata; different install prefixes alter binary paths/RPATH but not solver equations",
        "original_campaign_solver_unchanged": "prior dynamic runs remain tied to original /opt/gpu-cfd/prefix binary hashes, not these isolated runtimes",
        "code_sha256": sha(Path(__file__)),
    }
    target = root / "build_manifest.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"path": str(target), "sha256": sha(target),
                      "source_differences": list(differences),
                      "native_contrast": contrast}, indent=2))


if __name__ == "__main__":
    main()
