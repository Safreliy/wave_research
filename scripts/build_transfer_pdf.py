"""Build the current transfer article with installed MiKTeX, without installs.

The older PowerShell builder writes auxiliary files beside the manuscript and
can read stale aux data. This fallback keeps all TeX intermediates in build/,
checks the source/figures did not change mid-build, and publishes only a fully
resolved PDF. Run: python scripts/build_transfer_pdf.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO / "manuscript"
SOURCE = SOURCE_DIR / "transfer_article.tex"
BUILD_DIR = REPO / "build" / "transfer_article_pdf"
FINAL_PDF = SOURCE_DIR / "transfer_article.pdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inputs() -> dict[str, str]:
    figures = sorted((SOURCE_DIR / "figures").glob("*.pdf"))
    if not figures:
        raise FileNotFoundError("No manuscript PDF figures found")
    files = [
        SOURCE,
        *figures,
        *sorted((SOURCE_DIR / "tables").glob("*.tex")),
    ]
    return {path.relative_to(REPO).as_posix(): sha256(path) for path in files}


def unresolved(log: str) -> list[str]:
    patterns = (
        r"LaTeX Warning: There were undefined references",
        r"Package natbib Warning: There were undefined citations",
        r"(?:Citation|Reference) [`'].+? undefined",
        r"Rerun to get cross-references right",
        r"Label\(s\) may have changed",
    )
    return [pattern for pattern in patterns if re.search(pattern, log, re.IGNORECASE)]


def pdf_pages(path: Path) -> int | None:
    executable = shutil.which("pdfinfo")
    if not executable:
        return None
    result = subprocess.run([executable, str(path)], capture_output=True, text=True)
    if result.returncode:
        return None
    match = re.search(r"^Pages:\s+(\d+)", result.stdout, re.MULTILINE)
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--check-only", action="store_true", help="Build and validate without replacing manuscript/transfer_article.pdf")
    args = parser.parse_args()
    if not 2 <= args.max_passes <= 5:
        parser.error("--max-passes must be between 2 and 5")
    executable = shutil.which("pdflatex")
    if executable is None:
        parser.error("Installed MiKTeX pdflatex was not found; no installation attempted")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    initial_inputs = inputs()
    env = os.environ.copy()
    # MiKTeX searches the directory of an absolute .tex source before cwd.
    # Put build/ first so the old manuscript/transfer_article.aux cannot win.
    separator = ";" if os.name == "nt" else ":"
    env["TEXINPUTS"] = separator.join((str(BUILD_DIR), str(SOURCE_DIR), env.get("TEXINPUTS", "")))
    command = [
        executable,
        "--disable-installer",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-no-shell-escape",
        "-file-line-error",
        f"-aux-directory={BUILD_DIR}",
        f"-output-directory={BUILD_DIR}",
        str(SOURCE),
    ]
    passes: list[dict[str, object]] = []
    for number in range(1, args.max_passes + 1):
        result = subprocess.run(command, cwd=BUILD_DIR, env=env, capture_output=True, text=True, errors="replace")
        (BUILD_DIR / f"pass_{number}.console.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        log_path = BUILD_DIR / "transfer_article.log"
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        problems = unresolved(log)
        overfull = len(re.findall(r"Overfull \\hbox", log))
        passes.append({"number": number, "exit_code": result.returncode, "unresolved": problems, "overfull_hboxes": overfull})
        if result.returncode or re.search(r"^! (?:LaTeX|Package|Emergency stop)", log, re.MULTILINE):
            raise RuntimeError(f"pdflatex pass {number} failed; see {log_path} and pass_{number}.console.txt")
        if inputs() != initial_inputs:
            raise RuntimeError("Manuscript source or PDF figures changed during compilation; final PDF left untouched")
        if number >= 2 and not problems:
            break
    else:
        raise RuntimeError(f"Unresolved citations/references after {args.max_passes} passes; final PDF left untouched")

    built = BUILD_DIR / "transfer_article.pdf"
    if not built.is_file() or built.stat().st_size < 10_000 or not built.open("rb").read(5) == b"%PDF-":
        raise RuntimeError("No valid PDF produced; final PDF left untouched")
    if inputs() != initial_inputs:
        raise RuntimeError("Manuscript source or PDF figures changed after compilation; final PDF left untouched")

    if not args.check_only:
        temporary = FINAL_PDF.with_suffix(".pdf.tmp")
        shutil.copyfile(built, temporary)
        os.replace(temporary, FINAL_PDF)
    output = built if args.check_only else FINAL_PDF
    diagnostic = {
        "schema": "transfer-pdf-local-miktex-build-v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "compiler": executable,
        "installer_disabled": True,
        "command": command,
        "texinputs": env["TEXINPUTS"],
        "passes": passes,
        "input_sha256": initial_inputs,
        "check_only": args.check_only,
        "pdf_sha256": sha256(output),
        "pdf_bytes": output.stat().st_size,
        "pdf_pages": pdf_pages(output),
    }
    diagnostic_name = "diagnostics_preflight.json" if args.check_only else "diagnostics.json"
    (BUILD_DIR / diagnostic_name).write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pdf": str(output), "pages": diagnostic["pdf_pages"], "sha256": diagnostic["pdf_sha256"], "passes": passes}, indent=2))


if __name__ == "__main__":
    main()
