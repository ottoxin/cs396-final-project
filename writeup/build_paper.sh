#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT/.latex-build-acl"
SOURCE_TEX="$ROOT/carm_acl.tex"
FINAL_PDF="$ROOT/carm_final.pdf"
BUILD_PDF="$BUILD_DIR/carm_acl.pdf"

cd "$ROOT"

# Keep one canonical PDF and clear legacy clutter from older paper rounds.
rm -f \
  "$ROOT/carm_acl.pdf" \
  "$ROOT/carm_acl_round1_reviewed.pdf" \
  "$ROOT/carm_paper.pdf" \
  "$ROOT/carm_paper_round0_original.pdf" \
  "$ROOT/carm_paper_round1.pdf" \
  "$ROOT/carm_paper_round2.pdf" \
  "$ROOT/carm_paper_w3_final.pdf" \
  "$ROOT/carm_acl.aux" \
  "$ROOT/carm_acl.bbl" \
  "$ROOT/carm_acl.blg" \
  "$ROOT/carm_acl.fdb_latexmk" \
  "$ROOT/carm_acl.fls" \
  "$ROOT/carm_acl.log" \
  "$ROOT/carm_acl.out"

rm -rf \
  "$ROOT/.latex-build" \
  "$ROOT/__pycache__" \
  "$ROOT/figures/__pycache__"

latexmk -pdf -interaction=nonstopmode -outdir="$BUILD_DIR" "$SOURCE_TEX"
cp "$BUILD_PDF" "$FINAL_PDF"
rm -f "$BUILD_PDF"

printf 'Built %s\n' "$FINAL_PDF"
