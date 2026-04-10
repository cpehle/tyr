#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

usage() {
  cat <<'EOF'
Usage: scripts/runpod/create_or_resume.sh

Creates a fresh Runpod pod if no managed pod is currently RUNNING.
The script never stores credentials in the repo. Configure Runpod auth via
`runpodctl doctor` or the RUNPOD_API_KEY environment variable.
EOF
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

load_runpod_config
require_runpod_auth

pod_id="$(state_get pod_id)"
if [[ -n "${pod_id}" ]]; then
  status="$(pod_status "${pod_id}" || true)"
  status="$(normalize_status "${status}")"
  if [[ "${status}" == "RUNNING" ]]; then
    log "reusing tracked pod ${pod_id}"
    wait_for_ssh "${pod_id}"
    echo "${pod_id}"
    exit 0
  fi
fi

if pod_id="$(find_running_pod_by_name 2>/dev/null)"; then
  log "reusing running pod by name ${TYR_RUNPOD_POD_NAME}: ${pod_id}"
  state_set pod_id "${pod_id}"
  wait_for_ssh "${pod_id}"
  echo "${pod_id}"
  exit 0
fi

pod_id="$(create_pod)"
wait_for_pod_running "${pod_id}"
wait_for_ssh "${pod_id}"
log "pod is running: ${pod_id}"
echo "${pod_id}"
