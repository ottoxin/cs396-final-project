#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_carm_quest.sh [options]

Options:
  --run-id <id>
  --config <path>
  --input-jsonl <path>
  --manifest-path <path>
  --output-root <path>
  --eval-split <split>
  --job-name <name>
  --time <hh:mm:ss>
  --cpus-per-task <n>
  --mem <size>
  --account <allocation>
  --partition <partition>
  --gres <gres>
  --test-only
EOF
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

test_only=0
run_id=""
config_path=""
input_jsonl=""
manifest_path=""
output_root=""
eval_split="test_id"
job_name="carm-control-qwen"
wall_time="12:00:00"
cpus_per_task="8"
mem="64G"
account="p33196"
partition="gengpu"
gres="gpu:1"
while (($# > 0)); do
  case "$1" in
    --run-id)
      run_id="$2"
      shift
      ;;
    --config)
      config_path="$2"
      shift
      ;;
    --input-jsonl)
      input_jsonl="$2"
      shift
      ;;
    --manifest-path)
      manifest_path="$2"
      shift
      ;;
    --output-root)
      output_root="$2"
      shift
      ;;
    --eval-split)
      eval_split="$2"
      shift
      ;;
    --job-name)
      job_name="$2"
      shift
      ;;
    --time)
      wall_time="$2"
      shift
      ;;
    --cpus-per-task)
      cpus_per_task="$2"
      shift
      ;;
    --mem)
      mem="$2"
      shift
      ;;
    --account)
      account="$2"
      shift
      ;;
    --partition)
      partition="$2"
      shift
      ;;
    --gres)
      gres="$2"
      shift
      ;;
    --test-only)
      test_only=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
RUN_SCRIPT="${REPO_ROOT}/scripts/run_carm_quest.sh"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

RUN_ID="${run_id:-RUN-CTRL-0001}"
CONFIG_PATH="${config_path:-${REPO_ROOT}/configs/hf_5way_qwen_caption_derived_10pct_protocol_family_lowmem.yaml}"
INPUT_JSONL="${input_jsonl:-${REPO_ROOT}/data/cache/hf_5way/prepared/carm_vqa_5way_10pct_protocol_family_seed7.jsonl}"
MANIFEST_PATH="${manifest_path:-${REPO_ROOT}/data/cache/hf_5way/prepared/carm_vqa_5way_10pct_protocol_family_seed7.manifest.json}"
OUTPUT_ROOT="${output_root:-${REPO_ROOT}/outputs/carm/${RUN_ID}}"

mkdir -p "${OUTPUT_ROOT}"

printf -v wrapped_cmd '%q ' \
  bash "${RUN_SCRIPT}" "${OUTPUT_ROOT}" "${CONFIG_PATH}" "${INPUT_JSONL}" "${MANIFEST_PATH}" "${eval_split}"

cmd=("${SBATCH_BIN}")
if [[ "${test_only}" -eq 1 ]]; then
  cmd+=(--test-only)
fi
cmd+=(
  --account="${account}"
  --partition="${partition}"
  --gres="${gres}"
  --time="${wall_time}"
  --cpus-per-task="${cpus_per_task}"
  --mem="${mem}"
  --job-name="${job_name}"
  --output="${OUTPUT_ROOT}/slurm-%j.out"
  --error="${OUTPUT_ROOT}/slurm-%j.err"
  --wrap="${wrapped_cmd}"
)

command_file="${OUTPUT_ROOT}/sbatch_command.txt"
submit_output_file="${OUTPUT_ROOT}/sbatch_submit_output.txt"
metadata_file="${OUTPUT_ROOT}/submission_metadata.txt"

command_string=$(quote_cmd "${cmd[@]}")
printf '%s\n' "${command_string}" | tee "${command_file}"

{
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'created_utc=%s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  printf 'config_path=%s\n' "${CONFIG_PATH}"
  printf 'input_jsonl=%s\n' "${INPUT_JSONL}"
  printf 'manifest_path=%s\n' "${MANIFEST_PATH}"
  printf 'output_root=%s\n' "${OUTPUT_ROOT}"
  printf 'eval_split=%s\n' "${eval_split}"
  printf 'sbatch_bin=%s\n' "${SBATCH_BIN}"
  printf 'job_name=%s\n' "${job_name}"
  printf 'wall_time=%s\n' "${wall_time}"
  printf 'cpus_per_task=%s\n' "${cpus_per_task}"
  printf 'mem=%s\n' "${mem}"
  printf 'account=%s\n' "${account}"
  printf 'partition=%s\n' "${partition}"
  printf 'gres=%s\n' "${gres}"
  printf 'test_only=%s\n' "${test_only}"
} > "${metadata_file}"

echo "Submitting ${RUN_ID}"
if submit_output=$("${cmd[@]}" 2>&1); then
  printf '%s\n' "${submit_output}" | tee "${submit_output_file}"
else
  status=$?
  printf '%s\n' "${submit_output}" | tee "${submit_output_file}" >&2
  exit "${status}"
fi
