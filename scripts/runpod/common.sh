#!/usr/bin/env bash
set -euo pipefail

RUNPOD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RUNPOD_SCRIPT_DIR}/../.." && pwd)"
RUNPOD_UTIL="${RUNPOD_SCRIPT_DIR}/util.py"

log() {
  echo "[runpod] $*" >&2
}

die() {
  echo "[runpod] error: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

require_runpod_auth() {
  require_cmd runpodctl
  require_cmd python3
  if ! runpodctl user >/dev/null 2>&1; then
    die "Runpod CLI is not authenticated. Run 'runpodctl doctor' or export RUNPOD_API_KEY in your shell."
  fi
}

load_runpod_config() {
  if [[ -n "${TYR_RUNPOD_CONFIG:-}" ]]; then
    [[ -f "${TYR_RUNPOD_CONFIG}" ]] || die "TYR_RUNPOD_CONFIG does not exist: ${TYR_RUNPOD_CONFIG}"
    # shellcheck disable=SC1090
    source "${TYR_RUNPOD_CONFIG}"
  fi

  : "${TYR_RUNPOD_PROFILE:=h100}"
  : "${TYR_RUNPOD_POD_NAME:=tyr-h100}"
  : "${TYR_RUNPOD_GPU_ID:=NVIDIA H100 80GB HBM3}"
  : "${TYR_RUNPOD_GPU_COUNT:=1}"
  : "${TYR_RUNPOD_CLOUD_TYPE:=COMMUNITY}"
  : "${TYR_RUNPOD_PREFERRED_LOCATION:=United States}"
  : "${TYR_RUNPOD_DATA_CENTER_IDS:=}"
  : "${TYR_RUNPOD_CONTAINER_DISK_GB:=50}"
  : "${TYR_RUNPOD_LOCAL_VOLUME_GB:=50}"
  : "${TYR_RUNPOD_VOLUME_NAME:=tyr-runpod-cache}"
  : "${TYR_RUNPOD_VOLUME_SIZE_GB:=200}"
  : "${TYR_RUNPOD_VOLUME_MOUNT_PATH:=/workspace}"
  : "${TYR_RUNPOD_RESULTS_DIR:=${TYR_RUNPOD_VOLUME_MOUNT_PATH%/}/results/tyr}"
  : "${TYR_RUNPOD_REMOTE_REPO_DIR:=${TYR_RUNPOD_VOLUME_MOUNT_PATH%/}/tyr}"
  : "${TYR_RUNPOD_PORTS:=22/tcp}"
  : "${TYR_RUNPOD_VCPU:=8}"
  : "${TYR_RUNPOD_MEM_GB:=64}"
  : "${TYR_RUNPOD_REMOTE_BOOTSTRAP_PACKAGES:=1}"

  if [[ -z "${TYR_RUNPOD_TEMPLATE_ID:-}" && -z "${TYR_RUNPOD_IMAGE:-}" ]]; then
    die "Set TYR_RUNPOD_TEMPLATE_ID or TYR_RUNPOD_IMAGE. The workflow will not guess an image."
  fi

  case "${TYR_RUNPOD_CLOUD_TYPE}" in
    COMMUNITY|SECURE)
      ;;
    *)
      die "TYR_RUNPOD_CLOUD_TYPE must be COMMUNITY or SECURE"
      ;;
  esac

  if [[ "${TYR_RUNPOD_CLOUD_TYPE}" == "COMMUNITY" && -z "${TYR_RUNPOD_MAX_HOURLY_USD:-}" ]]; then
    die "Set TYR_RUNPOD_MAX_HOURLY_USD when using COMMUNITY cloud so pod creation is price-capped."
  fi

  RUNPOD_STATE_DIR="${REPO_ROOT}/.runpod-state"
  RUNPOD_STATE_FILE="${RUNPOD_STATE_DIR}/${TYR_RUNPOD_PROFILE}.json"
  mkdir -p "${RUNPOD_STATE_DIR}"
}

state_get() {
  python3 "${RUNPOD_UTIL}" state-get "${RUNPOD_STATE_FILE}" "$1"
}

state_set() {
  python3 "${RUNPOD_UTIL}" state-set "${RUNPOD_STATE_FILE}" "$1" "$2" >/dev/null
}

json_field() {
  python3 "${RUNPOD_UTIL}" field "$@"
}

pod_from_list() {
  python3 "${RUNPOD_UTIL}" pod-from-list "$1"
}

volume_from_list() {
  python3 "${RUNPOD_UTIL}" volume-from-list "$1"
}

choose_datacenter_json() {
  python3 "${RUNPOD_UTIL}" datacenter-for-gpu \
    "${TYR_RUNPOD_GPU_ID}" \
    "${TYR_RUNPOD_DATA_CENTER_IDS}" \
    "${TYR_RUNPOD_PREFERRED_LOCATION}"
}

normalize_status() {
  echo "$1" | tr '[:lower:]' '[:upper:]'
}

pod_status() {
  local pod_id="$1"
  local pod_json
  if ! pod_json="$(runpodctl pod get "${pod_id}" -o json 2>/dev/null)"; then
    return 1
  fi
  printf '%s' "${pod_json}" | json_field status desiredStatus desired_status 2>/dev/null | head -n 1
}

resolve_data_center_id() {
  local state_dc
  state_dc="$(state_get data_center_id)"
  if [[ -n "${state_dc}" ]]; then
    echo "${state_dc}"
    return
  fi

  local dc_json dc_id
  dc_json="$(runpodctl datacenter list -o json)"
  dc_id="$(printf '%s' "${dc_json}" | choose_datacenter_json | json_field id)"
  echo "${dc_id}"
}

ensure_volume() {
  if [[ -z "${TYR_RUNPOD_VOLUME_NAME}" ]]; then
    return 0
  fi

  local volume_id
  volume_id="$(state_get volume_id)"
  if [[ -n "${volume_id}" ]]; then
    if runpodctl network-volume get "${volume_id}" -o json >/dev/null 2>&1; then
      echo "${volume_id}"
      return
    fi
    log "tracked volume ${volume_id} no longer exists; recreating metadata"
    state_set volume_id ""
  fi

  local existing volume_json
  existing="$(runpodctl network-volume list -o json)"
  if volume_json="$(printf '%s' "${existing}" | volume_from_list "${TYR_RUNPOD_VOLUME_NAME}" 2>/dev/null)"; then
    volume_id="$(printf '%s' "${volume_json}" | json_field id)"
    state_set volume_id "${volume_id}"
    if printf '%s' "${volume_json}" | json_field dataCenterId data_center_id >/dev/null 2>&1; then
      state_set data_center_id "$(printf '%s' "${volume_json}" | json_field dataCenterId data_center_id)"
    fi
    echo "${volume_id}"
    return
  fi

  local data_center_id create_json
  data_center_id="$(resolve_data_center_id)"
  log "creating network volume ${TYR_RUNPOD_VOLUME_NAME} in ${data_center_id}"
  create_json="$(runpodctl network-volume create -o json \
    --name "${TYR_RUNPOD_VOLUME_NAME}" \
    --size "${TYR_RUNPOD_VOLUME_SIZE_GB}" \
    --data-center-id "${data_center_id}")"
  volume_id="$(printf '%s' "${create_json}" | json_field id)"
  [[ -n "${volume_id}" ]] || die "failed to create network volume"
  state_set volume_id "${volume_id}"
  state_set data_center_id "${data_center_id}"
  echo "${volume_id}"
}

find_running_pod_by_name() {
  local pods_json pod_json status
  pods_json="$(runpodctl pod list -o json)"
  if ! pod_json="$(printf '%s' "${pods_json}" | pod_from_list "${TYR_RUNPOD_POD_NAME}" 2>/dev/null)"; then
    return 1
  fi
  status="$(printf '%s' "${pod_json}" | json_field status desiredStatus desired_status 2>/dev/null | head -n 1 || true)"
  status="$(normalize_status "${status}")"
  if [[ "${status}" == "RUNNING" ]]; then
    printf '%s' "${pod_json}" | json_field id
    return 0
  fi
  return 1
}

create_pod() {
  local data_center_id volume_id pod_json pod_id
  local create_cmd=()

  data_center_id="$(resolve_data_center_id)"
  state_set data_center_id "${data_center_id}"
  volume_id="$(ensure_volume || true)"

  if [[ "${TYR_RUNPOD_CLOUD_TYPE}" == "COMMUNITY" ]]; then
    log "creating COMMUNITY pod ${TYR_RUNPOD_POD_NAME} in ${data_center_id} with price cap \$${TYR_RUNPOD_MAX_HOURLY_USD}/hr"
    create_cmd=(
      runpodctl create pod -o json
      --communityCloud
      --cost "${TYR_RUNPOD_MAX_HOURLY_USD}"
      --gpuType "${TYR_RUNPOD_GPU_ID}"
      --gpuCount "${TYR_RUNPOD_GPU_COUNT}"
      --name "${TYR_RUNPOD_POD_NAME}"
      --containerDiskSize "${TYR_RUNPOD_CONTAINER_DISK_GB}"
      --volumeSize "${TYR_RUNPOD_LOCAL_VOLUME_GB}"
      --dataCenterId "${data_center_id}"
      --ports "${TYR_RUNPOD_PORTS}"
      --startSSH
      --vcpu "${TYR_RUNPOD_VCPU}"
      --mem "${TYR_RUNPOD_MEM_GB}"
    )
    if [[ -n "${TYR_RUNPOD_TEMPLATE_ID:-}" ]]; then
      create_cmd+=(--templateId "${TYR_RUNPOD_TEMPLATE_ID}")
    else
      create_cmd+=(--imageName "${TYR_RUNPOD_IMAGE}")
    fi
    if [[ -n "${volume_id}" ]]; then
      create_cmd+=(--networkVolumeId "${volume_id}")
    fi
    create_cmd+=(--volumePath "${TYR_RUNPOD_VOLUME_MOUNT_PATH}")
  else
    log "creating SECURE pod ${TYR_RUNPOD_POD_NAME} in ${data_center_id}"
    create_cmd=(
      runpodctl pod create -o json
      --cloud-type SECURE
      --gpu-id "${TYR_RUNPOD_GPU_ID}"
      --gpu-count "${TYR_RUNPOD_GPU_COUNT}"
      --name "${TYR_RUNPOD_POD_NAME}"
      --container-disk-in-gb "${TYR_RUNPOD_CONTAINER_DISK_GB}"
      --data-center-ids "${data_center_id}"
      --ports "${TYR_RUNPOD_PORTS}"
      --ssh
      --volume-mount-path "${TYR_RUNPOD_VOLUME_MOUNT_PATH}"
      --volume-in-gb "${TYR_RUNPOD_LOCAL_VOLUME_GB}"
    )
    if [[ -n "${TYR_RUNPOD_TEMPLATE_ID:-}" ]]; then
      create_cmd+=(--template-id "${TYR_RUNPOD_TEMPLATE_ID}")
    else
      create_cmd+=(--image "${TYR_RUNPOD_IMAGE}")
    fi
    if [[ -n "${volume_id}" ]]; then
      create_cmd+=(--network-volume-id "${volume_id}")
    fi
  fi

  pod_json="$("${create_cmd[@]}")"

  pod_id="$(printf '%s' "${pod_json}" | json_field id)"
  [[ -n "${pod_id}" ]] || die "failed to extract pod id from Runpod output"
  state_set pod_id "${pod_id}"
  if [[ -n "${volume_id}" ]]; then
    state_set volume_id "${volume_id}"
  fi
  echo "${pod_id}"
}

wait_for_pod_running() {
  local pod_id="$1"
  local attempts="${2:-60}"
  local sleep_seconds="${3:-10}"
  local status=""

  for _ in $(seq 1 "${attempts}"); do
    status="$(pod_status "${pod_id}" || true)"
    status="$(normalize_status "${status}")"
    if [[ "${status}" == "RUNNING" ]]; then
      return 0
    fi
    sleep "${sleep_seconds}"
  done
  die "pod ${pod_id} did not become RUNNING (last status: ${status:-unknown})"
}

wait_for_ssh() {
  local pod_id="$1"
  local attempts="${2:-30}"
  local sleep_seconds="${3:-5}"

  for _ in $(seq 1 "${attempts}"); do
    if runpodctl ssh info "${pod_id}" -o json >/dev/null 2>&1; then
      if load_ssh_env "${pod_id}" >/dev/null 2>&1; then
        if "${RUNPOD_SSH_PREFIX[@]}" "true" >/dev/null 2>&1; then
          return 0
        fi
      fi
    fi
    sleep "${sleep_seconds}"
  done
  die "ssh did not become ready for pod ${pod_id}"
}

load_ssh_env() {
  local pod_id="$1"
  eval "$(
    runpodctl ssh info "${pod_id}" -o json | python3 "${RUNPOD_UTIL}" ssh-export
  )"
}

remote_bash() {
  local pod_id="$1"
  local remote_script="$2"
  local quoted
  load_ssh_env "${pod_id}"
  printf -v quoted '%q' "${remote_script}"
  "${RUNPOD_SSH_PREFIX[@]}" "bash -lc ${quoted}"
}

local_revision() {
  local revision dirty=""
  revision="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  if [[ -n "$(git -C "${REPO_ROOT}" status --short --untracked-files=normal)" ]]; then
    dirty="+dirty"
  fi
  echo "${revision}${dirty}"
}
