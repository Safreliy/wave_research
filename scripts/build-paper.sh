#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
manuscript_dir="$repo_dir/manuscript"

(
  cd "$manuscript_dir"
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
)

test -f "$manuscript_dir/main.pdf"
cp "$manuscript_dir/main.pdf" "$repo_dir/paper.pdf"
printf 'Built %s\n' "$repo_dir/paper.pdf"
