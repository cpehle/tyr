#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source ./load_modules.sh >/dev/null

export LEAN_CC="${REPO_ROOT}/scripts/lean_cc_wrapper.sh"
export LEAN_CC_FAST="${LEAN_CC_FAST:-1}"
export LD_LIBRARY_PATH="${REPO_ROOT}/external/libtorch/lib:${REPO_ROOT}/cc/build:${EBROOTGCCCORE}/lib64:${LD_LIBRARY_PATH:-}"

TORCHRUN_BIN="${TORCHRUN_BIN:-/grid/it/data/elzar/easybuild/software/Anaconda3/2023.07-2/bin/torchrun}"
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "error: torchrun launcher not found at ${TORCHRUN_BIN}" >&2
  exit 2
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
PIPELINE_EXE="${PIPELINE_EXE:-NanoChatPipeline}"
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  lake --quiet build "${PIPELINE_EXE}"
fi

if [[ "${QUICK_MODE_FLAG:-0}" == "1" ]]; then
  export QUICK_MODE=1
else
  unset QUICK_MODE || true
fi

if [[ "${ENABLE_RL_FLAG:-0}" == "1" ]]; then
  export ENABLE_RL=1
else
  unset ENABLE_RL || true
fi

export NANOCHAT_DIR="${NANOCHAT_DIR:-${HOME}/.cache/nanochat}"
export DATA_PATH="${DATA_PATH:-base_data}"
export MODEL_DEPTH="${MODEL_DEPTH:-20}"
export VOCAB_SIZE="${VOCAB_SIZE:-65536}"
export NUM_SHARDS="${NUM_SHARDS:-240}"
if [[ -z "${PRETRAIN_DATA_PATH:-}" && -d "${REPO_ROOT}/data/nanochat" ]]; then
  export PRETRAIN_DATA_PATH="${REPO_ROOT}/data/nanochat"
fi
if [[ -z "${MIDTRAIN_DATA_PATH:-}" && -n "${PRETRAIN_DATA_PATH:-}" ]]; then
  export MIDTRAIN_DATA_PATH="${PRETRAIN_DATA_PATH}"
fi

if [[ "${CLEAR_PIPELINE_CHECKPOINT:-1}" == "1" ]]; then
  rm -f "${REPO_ROOT}/.pipeline_checkpoint.json" "${NANOCHAT_DIR}/.pipeline_checkpoint.json"
fi

echo "Running ${PIPELINE_EXE} with torchrun (${NPROC_PER_NODE} processes)"
echo "Torchrun: ${TORCHRUN_BIN}"
echo "Executable: ${PIPELINE_EXE}"
echo "QUICK_MODE=${QUICK_MODE_FLAG:-0} ENABLE_RL=${ENABLE_RL_FLAG:-0}"
echo "NANOCHAT_DIR=${NANOCHAT_DIR}"
echo "DATA_PATH=${DATA_PATH}"
echo "PRETRAIN_DATA_PATH=${PRETRAIN_DATA_PATH:-<unset>}"
echo "MIDTRAIN_DATA_PATH=${MIDTRAIN_DATA_PATH:-<unset>}"

exec env TYR_DEVICE="${TYR_DEVICE:-cuda}" \
  "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" --no_python \
  "./.lake/build/bin/${PIPELINE_EXE}"
