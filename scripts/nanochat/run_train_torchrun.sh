#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${SKIP_MODULES:-0}" != "1" ]] && [[ -f ./load_modules.sh ]]; then
  if command -v module >/dev/null 2>&1; then
    source ./load_modules.sh >/dev/null
  else
    echo "warning: module command not available; skipping load_modules.sh" >&2
  fi
fi

export LEAN_CC="${REPO_ROOT}/scripts/lean_cc_wrapper.sh"
export LEAN_CC_FAST="${LEAN_CC_FAST:-1}"
gcccore_lib=""
if [[ -n "${EBROOTGCCCORE:-}" ]]; then
  gcccore_lib=":${EBROOTGCCCORE}/lib64"
fi
export LD_LIBRARY_PATH="${REPO_ROOT}/external/libtorch/lib:${REPO_ROOT}/cc/build${gcccore_lib}:${LD_LIBRARY_PATH:-}"

torchrun_from_path="$(command -v torchrun || true)"
default_torchrun="${torchrun_from_path:-/grid/it/data/elzar/easybuild/software/Anaconda3/2023.07-2/bin/torchrun}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${default_torchrun}}"
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "error: torchrun launcher not found at ${TORCHRUN_BIN}" >&2
  echo "set TORCHRUN_BIN to a working torchrun binary on this host" >&2
  exit 2
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "error: NPROC_PER_NODE must be a positive integer (got ${NPROC_PER_NODE})" >&2
  exit 2
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  lake --quiet build TrainNanoChat
fi

if [[ $# -eq 0 ]]; then
  set -- --debug --iterations 2 --data data/nanochat --val data/nanochat
fi

echo "Running TrainNanoChat with torchrun (${NPROC_PER_NODE} processes)"
echo "Torchrun: ${TORCHRUN_BIN}"

exec env TYR_DEVICE="${TYR_DEVICE:-cuda}" \
  "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" --no_python \
  ./.lake/build/bin/TrainNanoChat "$@"
