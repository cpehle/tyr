#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

usage() {
  cat <<'EOF'
Usage: scripts/runpod/sync_repo.sh [pod-id]

Rsyncs the local tyr working tree to the managed Runpod pod without copying
credentials or local build/output artifacts.
EOF
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

load_runpod_config
require_runpod_auth

pod_id="${1:-$(state_get pod_id)}"
[[ -n "${pod_id}" ]] || die "no pod id available; run create_or_resume.sh first"

load_ssh_env "${pod_id}"
remote_bash "${pod_id}" "mkdir -p $(printf '%q' "${TYR_RUNPOD_REMOTE_REPO_DIR}") $(printf '%q' "${TYR_RUNPOD_RESULTS_DIR}")"

revision="$(local_revision)"
log "syncing ${revision} to ${pod_id}:${TYR_RUNPOD_REMOTE_REPO_DIR}"

rsync -az --delete \
  -e "${RUNPOD_RSYNC_SHELL}" \
  --exclude '.git/' \
  --exclude '.lake/' \
  --exclude '.runpod-state/' \
  --exclude '.codex/' \
  --exclude '.claude/' \
  --exclude '.model-cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.venv*/' \
  --exclude 'build/' \
  --exclude 'out/' \
  --exclude 'output/' \
  --exclude 'docbuild/' \
  --exclude 'checkpoints/' \
  --exclude 'data/' \
  --exclude 'logs/' \
  --exclude 'orb.db/' \
  --exclude 'cc/build/' \
  --exclude 'cc/src/generated/' \
  --exclude 'scripts/__pycache__/' \
  --exclude 'external/mlx' \
  "${REPO_ROOT}/" \
  "${RUNPOD_SSH_TARGET}:${TYR_RUNPOD_REMOTE_REPO_DIR}/"

state_set last_sync_revision "${revision}"
state_set remote_repo_dir "${TYR_RUNPOD_REMOTE_REPO_DIR}"
log "sync complete"
