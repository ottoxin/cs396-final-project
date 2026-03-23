#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_carm_quest.sh <output_root> <config_path> <input_jsonl> <manifest_path> <eval_split>
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

if [[ $# -ne 5 ]]; then
  usage >&2
  exit 2
fi

OUTPUT_ROOT="$1"
CONFIG_PATH="$2"
INPUT_JSONL="$3"
MANIFEST_PATH="$4"
EVAL_SPLIT="$5"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${REPO_ROOT}/scripts/runtime_env.sh"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
TRAIN_SCRIPT="${REPO_ROOT}/scripts/train_carm.py"
EVAL_SCRIPT="${REPO_ROOT}/scripts/evaluate_carm.py"
QWEN_TEST="${REPO_ROOT}/tests/test_qwen_inference_optin.py"

mkdir -p "${OUTPUT_ROOT}"
COMMANDS_FILE="${OUTPUT_ROOT}/commands.log"
RUN_LOG="${OUTPUT_ROOT}/run.log"
TRAIN_DIR="${OUTPUT_ROOT}/train"
EVAL_DIR="${OUTPUT_ROOT}/eval_${EVAL_SPLIT}"
: > "${COMMANDS_FILE}"
: > "${RUN_LOG}"

cd "${REPO_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing virtualenv python: ${PYTHON_BIN}" >&2
  exit 1
fi

export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
setup_hf_runtime_env "${REPO_ROOT}"

log "start carm control output_root=${OUTPUT_ROOT}"
log "repo_root=${REPO_ROOT}"
log "slurm_job_id=${SLURM_JOB_ID:-none}"
log "python_bin=${PYTHON_BIN}"

{
  printf 'created_utc=%s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-none}"
  printf 'slurm_job_name=%s\n' "${SLURM_JOB_NAME:-none}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'hf_runtime_root=%s\n' "${HF_RUNTIME_ROOT}"
  printf 'hf_home=%s\n' "${HF_HOME}"
  printf 'hf_hub_cache=%s\n' "${HF_HUB_CACHE}"
  printf 'hf_datasets_cache=%s\n' "${HF_DATASETS_CACHE}"
  printf 'config_path=%s\n' "${CONFIG_PATH}"
  printf 'input_jsonl=%s\n' "${INPUT_JSONL}"
  printf 'manifest_path=%s\n' "${MANIFEST_PATH}"
  printf 'eval_split=%s\n' "${EVAL_SPLIT}"
} > "${OUTPUT_ROOT}/run_context.txt"

run_cmd cp "${CONFIG_PATH}" "${OUTPUT_ROOT}/$(basename "${CONFIG_PATH}")"
run_cmd cp "${MANIFEST_PATH}" "${OUTPUT_ROOT}/$(basename "${MANIFEST_PATH}")"

if command -v git >/dev/null 2>&1; then
  quote_cmd git rev-parse HEAD >> "${COMMANDS_FILE}"
  git rev-parse HEAD > "${OUTPUT_ROOT}/git_rev.txt"
  quote_cmd git status --short >> "${COMMANDS_FILE}"
  git status --short > "${OUTPUT_ROOT}/git_status.txt"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  quote_cmd nvidia-smi -L >> "${COMMANDS_FILE}"
  nvidia-smi -L > "${OUTPUT_ROOT}/nvidia_smi.txt" 2>&1 || true
  quote_cmd nvidia-smi >> "${COMMANDS_FILE}"
  nvidia-smi >> "${OUTPUT_ROOT}/nvidia_smi.txt" 2>&1 || true
fi

run_cmd env RUN_QWEN_INFERENCE_TESTS=1 "${PYTHON_BIN}" -m pytest "${QWEN_TEST}" -q

run_cmd "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --config "${CONFIG_PATH}" \
  --train_jsonl "${INPUT_JSONL}" \
  --output_dir "${TRAIN_DIR}"

run_cmd "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
  --config "${CONFIG_PATH}" \
  --input_jsonl "${INPUT_JSONL}" \
  --output_dir "${EVAL_DIR}" \
  --model_ckpt "${TRAIN_DIR}/carm_heads.pt" \
  --split "${EVAL_SPLIT}"

log "completed carm control run"
