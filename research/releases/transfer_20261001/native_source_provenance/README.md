# Aphros receiver source provenance (read-only host snapshot, 2026-10-01)

The runtime used `/opt/gpu-cfd/prefix/bin/ap.mfer` and
`/opt/gpu-cfd/prefix/lib/libaphros.so` on the authorized GPU host. Their
SHA-256 values are in `native_source_provenance.json` and match the dynamic
launch protocols. The shared library is byte-identical to
`/opt/gpu-cfd/aphros-build/libaphros.so`. That build directory's
`CMakeCache.txt` points to `/opt/gpu-cfd/aphros/src` and install prefix
`/opt/gpu-cfd/prefix`; its install manifest names both runtime paths. Its
Ninja log records `libaphros.so` and `ap.mfer` builds. These records identify
the build tree and deployed library, not a cryptographic digest of every
source file at compile time.

The source repository reports origin
`https://github.com/cselab/aphros.git` and currently has HEAD
`b60ce3da52c19935fa24c778f62f02141eaf7f80` plus the tracked changes
in `source_worktree_patch.diff`. This patch is an inspected **current worktree
snapshot**. No immutable source snapshot or per-source build hashes were
found, so the exact dirty patch used to compile the September binary is **not
proved**. Do not claim that applying this patch recreates the exact binary.
`libaphros_rebuild_commands.txt` and `ap_mfer_rebuild_commands.txt` are
commands reported by Ninja's dependency graph at inspection time, not an
executed command log. `aphros_build_ninja_log.txt` is the executed build log.

The installed `ap.mfer` has a different SHA-256 from its build-tree copy,
while `file` reports the same ELF BuildID SHA-1 for both. The install step may
have changed binary bytes; this check supports a common link output but does
not establish byte identity. The separate
`/opt/gpu-cfd/build_consistent_v52` tree installs to
`/opt/gpu-cfd/prefix_consistent_v52`, so it is not the runtime prefix used by
the recorded dynamic runs.

`aphros_LICENSE` is the source repository's MIT licence. The complete Aphros
repository, external AMGX/CUDA/MPI dependencies, and GPU/WSL environment are
not bundled here. No remote build or solver process was changed during this
inspection.
