#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./sync_zotero_bib.sh [output.bib]

Environment:
  ZOTERO_LIBRARY_ID      Required. Numeric Zotero user or group ID.
  ZOTERO_LIBRARY_TYPE    Optional. "user" (default) or "group".
  ZOTERO_COLLECTION_KEY  Optional. Restrict export to a specific collection key.
  ZOTERO_API_KEY         Required for this private collection.
  ZOTERO_EXPORT_FORMAT   Optional. "bibtex" (default) or "biblatex".
  ZOTERO_PAGE_LIMIT      Optional. Items per request, default 100.
  ZOTERO_BIB_FILE        Optional default output path when no positional arg is passed.

Examples:
  ./sync_zotero_bib.sh
  ZOTERO_BIB_FILE=my_refs.bib ./sync_zotero_bib.sh
EOF
}

load_env_file() {
  local env_file="$1"
  local -a zotero_vars=(
    ZOTERO_LIBRARY_ID
    ZOTERO_LIBRARY_TYPE
    ZOTERO_COLLECTION_KEY
    ZOTERO_API_KEY
    ZOTERO_EXPORT_FORMAT
    ZOTERO_PAGE_LIMIT
    ZOTERO_BIB_FILE
  )

  if [[ -f "$env_file" ]]; then
    local name
    for name in "${zotero_vars[@]}"; do
      if [[ -v $name ]]; then
        export "__ORIG_${name}=${!name}"
      fi
    done

    # Source a repo-local config so the user only sets Zotero values once.
    # shellcheck disable=SC1090
    source "$env_file"

    for name in "${zotero_vars[@]}"; do
      local original_name="__ORIG_${name}"
      if [[ -v $original_name ]]; then
        export "${name}=${!original_name}"
        unset "$original_name"
      fi
    done
  fi
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

validate_integer() {
  local value="$1"
  local name="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be a positive integer. Got: $value" >&2
    exit 1
  fi
}

fetch_page() {
  local url="$1"
  local headers_file="$2"
  local body_file="$3"

  local curl_args=(
    -sS
    -L
    -D "$headers_file"
    -o "$body_file"
    -w "%{http_code}"
  )

  if [[ -n "${ZOTERO_API_KEY:-}" ]]; then
    curl_args=(-H "Zotero-API-Key: ${ZOTERO_API_KEY}" "${curl_args[@]}")
  fi

  local status
  status="$(curl "${curl_args[@]}" "$url")"

  if [[ "$status" != "200" ]]; then
    echo "Zotero API request failed with HTTP $status" >&2
    echo "URL: $url" >&2
    if [[ -s "$body_file" ]]; then
      sed -n '1,40p' "$body_file" >&2
    fi
    exit 1
  fi
}

maybe_generate_aliases() {
  local output_file="$1"
  local script_dir="$2"
  local alias_script="${script_dir}/generate_ref_aliases.py"
  local alias_map="${script_dir}/citekey_aliases.json"
  local alias_output="${script_dir}/ref.bib"

  if [[ ! -f "$alias_script" || ! -f "$alias_map" ]]; then
    return 0
  fi

  python3 "$alias_script" \
    --input "$output_file" \
    --mapping "$alias_map" \
    --output "$alias_output"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_command curl
require_command python3

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
load_env_file "${script_dir}/.zotero.env"

library_type="${ZOTERO_LIBRARY_TYPE:-user}"
library_id="${ZOTERO_LIBRARY_ID:-}"
collection_key="${ZOTERO_COLLECTION_KEY:-}"
export_format="${ZOTERO_EXPORT_FORMAT:-bibtex}"
page_limit="${ZOTERO_PAGE_LIMIT:-100}"
output_file="${1:-${ZOTERO_BIB_FILE:-cs396_zotero.bib}}"

if [[ -z "$library_id" ]]; then
  echo "ZOTERO_LIBRARY_ID is required." >&2
  usage >&2
  exit 1
fi

if [[ -z "${ZOTERO_API_KEY:-}" ]]; then
  echo "ZOTERO_API_KEY is required for this private collection." >&2
  exit 1
fi

if [[ "$library_type" != "user" && "$library_type" != "group" ]]; then
  echo "ZOTERO_LIBRARY_TYPE must be 'user' or 'group'. Got: $library_type" >&2
  exit 1
fi

if [[ "$export_format" != "biblatex" && "$export_format" != "bibtex" ]]; then
  echo "ZOTERO_EXPORT_FORMAT must be 'biblatex' or 'bibtex'. Got: $export_format" >&2
  exit 1
fi

validate_integer "$library_id" "ZOTERO_LIBRARY_ID"
validate_integer "$page_limit" "ZOTERO_PAGE_LIMIT"

if (( page_limit < 1 || page_limit > 100 )); then
  echo "ZOTERO_PAGE_LIMIT must be between 1 and 100. Got: $page_limit" >&2
  exit 1
fi

base_url="https://api.zotero.org/${library_type}s/${library_id}"
if [[ -n "$collection_key" ]]; then
  endpoint="${base_url}/collections/${collection_key}/items/top"
else
  endpoint="${base_url}/items/top"
fi

tmp_dir="$(mktemp -d)"
tmp_output="${tmp_dir}/bibliography.bib"
trap 'rm -rf "$tmp_dir"' EXIT

headers_file="${tmp_dir}/headers_0.txt"
body_file="${tmp_dir}/body_0.bib"
first_url="${endpoint}?format=${export_format}&limit=${page_limit}&start=0"

fetch_page "$first_url" "$headers_file" "$body_file"

total_results="$(
  awk -F': ' 'tolower($1)=="total-results" {gsub(/\r/, "", $2); print $2; exit}' "$headers_file"
)"

if [[ -z "$total_results" ]]; then
  echo "Could not read Zotero total-results header." >&2
  exit 1
fi

validate_integer "$total_results" "Zotero total-results"

: > "$tmp_output"
if [[ -s "$body_file" ]]; then
  cat "$body_file" >> "$tmp_output"
  printf '\n' >> "$tmp_output"
fi

for ((start = page_limit; start < total_results; start += page_limit)); do
  headers_file="${tmp_dir}/headers_${start}.txt"
  body_file="${tmp_dir}/body_${start}.bib"
  page_url="${endpoint}?format=${export_format}&limit=${page_limit}&start=${start}"

  fetch_page "$page_url" "$headers_file" "$body_file"

  if [[ -s "$body_file" ]]; then
    cat "$body_file" >> "$tmp_output"
    printf '\n' >> "$tmp_output"
  fi
done

mv "$tmp_output" "$output_file"
echo "Synced ${total_results} Zotero items to ${output_file}"

maybe_generate_aliases "$output_file" "$script_dir"
