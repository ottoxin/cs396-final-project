#!/usr/bin/env bash

setup_hf_runtime_env() {
  local repo_root="${1:?repo_root is required}"
  local runtime_root="${repo_root}/data/cache/hf_runtime"

  export HF_RUNTIME_ROOT="${runtime_root}"
  export HF_HOME="${runtime_root}/home"
  export HF_HUB_CACHE="${runtime_root}/cache"
  export HF_DATASETS_CACHE="${runtime_root}/cache/datasets"
  export PYTHONUNBUFFERED=1

  mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"
}
