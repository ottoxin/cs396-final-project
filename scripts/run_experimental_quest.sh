#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_experimental_quest.sh <output_dir> <config_path> <input_jsonl> <manifest_path>
EOF
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*" | tee -a "${RUN_LOG}"
}

run_cmd() {
  quote_cmd "$@" >> "${COMMANDS_FILE}"
  "$@" 2>&1 | tee -a "${RUN_LOG}"
}

if [[ $# -ne 4 ]]; then
  usage >&2
  exit 2
fi

OUTPUT_DIR="$1"
CONFIG_PATH="$2"
INPUT_JSONL="$3"
MANIFEST_PATH="$4"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
RUN_SCRIPT="${REPO_ROOT}/scripts/run_experimental_small_data.py"
QWEN_TEST="${REPO_ROOT}/tests/test_qwen_inference_optin.py"

mkdir -p "${OUTPUT_DIR}"
COMMANDS_FILE="${OUTPUT_DIR}/commands.log"
RUN_LOG="${OUTPUT_DIR}/run.log"
: > "${COMMANDS_FILE}"
: > "${RUN_LOG}"

cd "${REPO_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing virtualenv python: ${PYTHON_BIN}" >&2
  exit 1
fi

export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export HF_HOME="${REPO_ROOT}/.hf_home"
export HF_HUB_CACHE="${REPO_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${REPO_ROOT}/.hf_cache/datasets"
export PYTHONUNBUFFERED=1

log "start experimental qwen output_dir=${OUTPUT_DIR}"
log "repo_root=${REPO_ROOT}"
log "slurm_job_id=${SLURM_JOB_ID:-none}"
log "python_bin=${PYTHON_BIN}"

{
  printf 'created_utc=%s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-none}"
  printf 'slurm_job_name=%s\n' "${SLURM_JOB_NAME:-none}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'config_path=%s\n' "${CONFIG_PATH}"
  printf 'input_jsonl=%s\n' "${INPUT_JSONL}"
  printf 'manifest_path=%s\n' "${MANIFEST_PATH}"
} > "${OUTPUT_DIR}/run_context.txt"

run_cmd cp "${CONFIG_PATH}" "${OUTPUT_DIR}/$(basename "${CONFIG_PATH}")"
run_cmd cp "${MANIFEST_PATH}" "${OUTPUT_DIR}/$(basename "${MANIFEST_PATH}")"

if command -v git >/dev/null 2>&1; then
  quote_cmd git rev-parse HEAD >> "${COMMANDS_FILE}"
  git rev-parse HEAD > "${OUTPUT_DIR}/git_rev.txt"
  quote_cmd git status --short >> "${COMMANDS_FILE}"
  git status --short > "${OUTPUT_DIR}/git_status.txt"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  quote_cmd nvidia-smi -L >> "${COMMANDS_FILE}"
  nvidia-smi -L > "${OUTPUT_DIR}/nvidia_smi.txt" 2>&1 || true
  quote_cmd nvidia-smi >> "${COMMANDS_FILE}"
  nvidia-smi >> "${OUTPUT_DIR}/nvidia_smi.txt" 2>&1 || true
fi

run_cmd env RUN_QWEN_INFERENCE_TESTS=1 "${PYTHON_BIN}" -m pytest "${QWEN_TEST}" -q

run_cmd "${PYTHON_BIN}" "${RUN_SCRIPT}" \
  --config "${CONFIG_PATH}" \
  --input_jsonl "${INPUT_JSONL}" \
  --output_dir "${OUTPUT_DIR}"

log "completed experimental qwen run"
