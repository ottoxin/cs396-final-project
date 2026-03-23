#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_baseline_refresh_quest.sh --mode <tune|eval> [options]

Options:
  --mode <tune|eval>
  --run-id <id>
  --config <path>
  --input-jsonl <path>
  --manifest-path <path>
  --output-dir <path>
  --job-name <name>
  --time <hh:mm:ss>
  --cpus-per-task <n>
  --mem <size>
  --account <allocation>
  --partition <partition>
  --gres <gres>
  --dependency <slurm-dependency>
  --tuned-thresholds-json <path>
  --test-only
EOF
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

test_only=0
mode=""
run_id=""
config_path=""
input_jsonl=""
manifest_path=""
output_dir=""
job_name="carm-baseline-refresh"
wall_time="24:00:00"
cpus_per_task="8"
mem="64G"
account="p33196"
partition="gengpu"
gres="gpu:1"
dependency=""
tuned_thresholds_json=""

while (($# > 0)); do
  case "$1" in
    --mode)
      mode="$2"
      shift
      ;;
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
    --output-dir)
      output_dir="$2"
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
    --dependency)
      dependency="$2"
      shift
      ;;
    --tuned-thresholds-json)
      tuned_thresholds_json="$2"
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

if [[ "${mode}" != "tune" && "${mode}" != "eval" ]]; then
  echo "--mode must be tune or eval" >&2
  exit 2
fi

if [[ "${mode}" == "eval" && -z "${tuned_thresholds_json}" ]]; then
  echo "--tuned-thresholds-json is required for eval mode" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
RUN_SCRIPT="${REPO_ROOT}/scripts/run_baseline_refresh_quest.sh"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

RUN_ID="${run_id:-RUN-BASE-REFRESH}"
CONFIG_PATH="${config_path:-${REPO_ROOT}/configs/hf_5way_qwen_caption_derived.yaml}"
INPUT_JSONL="${input_jsonl:-${REPO_ROOT}/data/cache/hf_5way/prepared/carm_vqa_5way.jsonl}"
MANIFEST_PATH="${manifest_path:-${REPO_ROOT}/data/cache/hf_5way/prepared/carm_vqa_5way.manifest.json}"
OUTPUT_DIR="${output_dir:-${REPO_ROOT}/outputs/baselines/${RUN_ID}}"

mkdir -p "${OUTPUT_DIR}"

if [[ -n "${tuned_thresholds_json}" ]]; then
  printf -v wrapped_cmd '%q ' \
    bash "${RUN_SCRIPT}" "${mode}" "${OUTPUT_DIR}" "${CONFIG_PATH}" "${INPUT_JSONL}" "${MANIFEST_PATH}" "${tuned_thresholds_json}"
else
  printf -v wrapped_cmd '%q ' \
    bash "${RUN_SCRIPT}" "${mode}" "${OUTPUT_DIR}" "${CONFIG_PATH}" "${INPUT_JSONL}" "${MANIFEST_PATH}"
fi

cmd=("${SBATCH_BIN}")
if [[ "${test_only}" -eq 1 ]]; then
  cmd+=(--test-only)
fi
if [[ -n "${dependency}" ]]; then
  cmd+=(--dependency="${dependency}")
fi
cmd+=(
  --account="${account}"
  --partition="${partition}"
  --gres="${gres}"
  --time="${wall_time}"
  --cpus-per-task="${cpus_per_task}"
  --mem="${mem}"
  --job-name="${job_name}"
  --output="${OUTPUT_DIR}/slurm-%j.out"
  --error="${OUTPUT_DIR}/slurm-%j.err"
  --wrap="${wrapped_cmd}"
)

command_file="${OUTPUT_DIR}/sbatch_command.txt"
submit_output_file="${OUTPUT_DIR}/sbatch_submit_output.txt"
metadata_file="${OUTPUT_DIR}/submission_metadata.txt"

command_string=$(quote_cmd "${cmd[@]}")
printf '%s\n' "${command_string}" | tee "${command_file}"

{
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'mode=%s\n' "${mode}"
  printf 'created_utc=%s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  printf 'config_path=%s\n' "${CONFIG_PATH}"
  printf 'input_jsonl=%s\n' "${INPUT_JSONL}"
  printf 'manifest_path=%s\n' "${MANIFEST_PATH}"
  printf 'output_dir=%s\n' "${OUTPUT_DIR}"
  printf 'tuned_thresholds_json=%s\n' "${tuned_thresholds_json:-none}"
  printf 'sbatch_bin=%s\n' "${SBATCH_BIN}"
  printf 'job_name=%s\n' "${job_name}"
  printf 'wall_time=%s\n' "${wall_time}"
  printf 'cpus_per_task=%s\n' "${cpus_per_task}"
  printf 'mem=%s\n' "${mem}"
  printf 'account=%s\n' "${account}"
  printf 'partition=%s\n' "${partition}"
  printf 'gres=%s\n' "${gres}"
  printf 'dependency=%s\n' "${dependency:-none}"
  printf 'test_only=%s\n' "${test_only}"
} > "${metadata_file}"

echo "Submitting ${RUN_ID} (${mode})"
if submit_output=$("${cmd[@]}" 2>&1); then
  printf '%s\n' "${submit_output}" | tee "${submit_output_file}"
else
  status=$?
  printf '%s\n' "${submit_output}" | tee "${submit_output_file}" >&2
  exit "${status}"
fi
