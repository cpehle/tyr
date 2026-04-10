#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

usage() {
  cat <<'EOF'
Usage: scripts/runpod/run_bench.sh [bench_mha_h100_train.sh args...]

Creates or reuses the managed Runpod pod, syncs the repo, bootstraps the remote
toolchain, and runs scripts/gpu/bench_mha_h100_train.sh on the pod.
EOF
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

load_runpod_config
require_runpod_auth

pod_id="$("${RUNPOD_SCRIPT_DIR}/create_or_resume.sh")"
"${RUNPOD_SCRIPT_DIR}/sync_repo.sh" "${pod_id}"

run_id="${TYR_RUNPOD_RUN_ID:-mha_h100_train_$(date -u +%Y%m%dT%H%M%SZ)}"
remote_log="${TYR_RUNPOD_RESULTS_DIR%/}/${run_id}.log"
args_quoted=""
if [[ $# -gt 0 ]]; then
  printf -v args_quoted ' %q' "$@"
fi

state_set last_run_id "${run_id}"
state_set last_remote_log "${remote_log}"

remote_script=$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "${TYR_RUNPOD_REMOTE_REPO_DIR}")
export TYR_RUNPOD_VOLUME_MOUNT_PATH=$(printf '%q' "${TYR_RUNPOD_VOLUME_MOUNT_PATH}")
export TYR_RUNPOD_REMOTE_BOOTSTRAP_PACKAGES=$(printf '%q' "${TYR_RUNPOD_REMOTE_BOOTSTRAP_PACKAGES}")
./scripts/runpod/bootstrap.sh
mkdir -p $(printf '%q' "${TYR_RUNPOD_RESULTS_DIR}")
./scripts/gpu/bench_mha_h100_train.sh${args_quoted} 2>&1 | tee $(printf '%q' "${remote_log}")
EOF
)

log "running benchmark ${run_id} on pod ${pod_id}"
remote_bash "${pod_id}" "${remote_script}"
log "benchmark log saved to ${remote_log}"
