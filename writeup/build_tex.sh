#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  build_tex.sh [--clean | --distclean] [path/to/file.tex]

Examples:
  ./build_tex.sh carm_proposal.tex
  ./build_tex.sh cs396-final-project/writeup/carm_proposal.tex
  ./build_tex.sh --clean carm_proposal.tex

Optional bibliography sync:
  ZOTERO_SYNC=1 ./build_tex.sh carm_proposal.tex
EOF
}

BUILD_DIR_NAME=".latex-build"

load_texlive_module() {
  if command -v latexmk >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
    return 0
  fi

  if [[ -n "${LMOD_CMD:-}" ]]; then
    eval "$("$LMOD_CMD" bash load texlive/2020)"
  elif [[ -n "${MODULESHOME:-}" && -f "${MODULESHOME}/init/bash" ]]; then
    # shellcheck disable=SC1090
    source "${MODULESHOME}/init/bash"
    module load texlive/2020
  elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash
    module load texlive/2020
  else
    echo "Unable to load texlive/2020 automatically. Load a TeX toolchain, then rerun." >&2
    exit 1
  fi

  if ! command -v latexmk >/dev/null 2>&1 || ! command -v bibtex >/dev/null 2>&1; then
    echo "TeX toolchain is still unavailable after loading texlive/2020." >&2
    exit 1
  fi
}

resolve_tex_file() {
  local input="$1"
  if [[ ! -f "$input" ]]; then
    echo "TeX file not found: $input" >&2
    exit 1
  fi

  local tex_dir
  tex_dir="$(cd "$(dirname "$input")" && pwd)"
  local tex_name
  tex_name="$(basename "$input")"

  printf '%s\n%s\n' "$tex_dir" "$tex_name"
}

build_paths() {
  tex_stem="${tex_name%.tex}"
  build_dir="${tex_dir}/${BUILD_DIR_NAME}"
  build_pdf="${build_dir}/${tex_stem}.pdf"
  root_pdf="${tex_dir}/${tex_stem}.pdf"
}

remove_root_aux_files() {
  rm -f \
    "${tex_stem}.aux" \
    "${tex_stem}.bbl" \
    "${tex_stem}.bcf" \
    "${tex_stem}.blg" \
    "${tex_stem}.fdb_latexmk" \
    "${tex_stem}.fls" \
    "${tex_stem}.log" \
    "${tex_stem}.out" \
    "${tex_stem}.run.xml" \
    "${tex_stem}.synctex.gz"
}

copy_final_pdf() {
  if [[ -f "${build_pdf}" ]]; then
    cp -f "${build_pdf}" "${root_pdf}"
  else
    echo "Expected PDF was not produced: ${build_pdf}" >&2
    exit 1
  fi
}

maybe_sync_zotero_bib() {
  if [[ "${ZOTERO_SYNC:-0}" != "1" ]]; then
    return 0
  fi

  local sync_script="${tex_dir}/sync_zotero_bib.sh"
  if [[ ! -x "$sync_script" ]]; then
    echo "ZOTERO_SYNC=1 was set, but ${sync_script} is missing or not executable." >&2
    exit 1
  fi

  "$sync_script"
}

action="build"
tex_input="carm_proposal.tex"

while (($#)); do
  case "$1" in
    --clean)
      action="clean"
      shift
      ;;
    --distclean)
      action="distclean"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      tex_input="$1"
      shift
      ;;
  esac
done

load_texlive_module

mapfile -t tex_parts < <(resolve_tex_file "$tex_input")
tex_dir="${tex_parts[0]}"
tex_name="${tex_parts[1]}"
build_paths

cd "$tex_dir"

case "$action" in
  build)
    maybe_sync_zotero_bib
    mkdir -p "${build_dir}"
    remove_root_aux_files
    latexmk \
      -pdf \
      -auxdir="${BUILD_DIR_NAME}" \
      -outdir="${BUILD_DIR_NAME}" \
      -interaction=nonstopmode \
      -halt-on-error \
      "$tex_name"
    copy_final_pdf
    echo "Built ${root_pdf} (aux files in ${build_dir})"
    ;;
  clean)
    mkdir -p "${build_dir}"
    latexmk -c -auxdir="${BUILD_DIR_NAME}" -outdir="${BUILD_DIR_NAME}" "$tex_name"
    remove_root_aux_files
    find "${build_dir}" -mindepth 1 -maxdepth 1 ! -name '*.pdf' -delete
    echo "Cleaned auxiliary files for ${tex_dir}/${tex_name} (kept PDFs)"
    ;;
  distclean)
    mkdir -p "${build_dir}"
    latexmk -C -auxdir="${BUILD_DIR_NAME}" -outdir="${BUILD_DIR_NAME}" "$tex_name"
    remove_root_aux_files
    rm -f "${root_pdf}"
    rm -rf "${build_dir}"
    echo "Removed build artifacts for ${tex_dir}/${tex_name}"
    ;;
esac
