#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Disable module paging for non-interactive and detached launches.
export LMOD_PAGER="${LMOD_PAGER:-none}"
export MODULES_PAGER="${MODULES_PAGER:-cat}"
export PAGER="${PAGER:-cat}"

source ./load_modules.sh >/dev/null

export LEAN_CC="${REPO_ROOT}/scripts/lean_cc_wrapper.sh"
export LEAN_CC_FAST="${LEAN_CC_FAST:-1}"
export LD_LIBRARY_PATH="${REPO_ROOT}/external/libtorch/lib:${REPO_ROOT}/cc/build:${EBROOTGCCCORE}/lib64:${LD_LIBRARY_PATH:-}"

TORCHRUN_BIN="${TORCHRUN_BIN:-/grid/it/data/elzar/easybuild/software/Anaconda3/2023.07-2/bin/torchrun}"
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "error: torchrun launcher not found at ${TORCHRUN_BIN}" >&2
  exit 2
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
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

if [[ "${NANOCHAT_USE_RUN_ID_DIR:-0}" == "1" ]]; then
  export NANOCHAT_RUN_ID="${NANOCHAT_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
  export NANOCHAT_DIR="${NANOCHAT_DIR:-${HOME}/.cache/nanochat_${NANOCHAT_RUN_ID}}"
else
  export NANOCHAT_DIR="${NANOCHAT_DIR:-${HOME}/.cache/nanochat}"
fi
mkdir -p "${NANOCHAT_DIR}"

export DATA_PATH="${DATA_PATH:-base_data}"
export MODEL_DEPTH="${MODEL_DEPTH:-20}"
export PARAM_DATA_RATIO="${PARAM_DATA_RATIO:-20}"
export VOCAB_SIZE="${VOCAB_SIZE:-65536}"
export TOKENIZER_MAX_CHARS="${TOKENIZER_MAX_CHARS:-2000000000}"
export TOKENIZER_DOC_CAP="${TOKENIZER_DOC_CAP:-10000}"
export INITIAL_DATA_SHARDS="${INITIAL_DATA_SHARDS:-8}"
export NUM_SHARDS="${NUM_SHARDS:-240}"
export WANDB_RUN="${WANDB_RUN:-dummy}"
if [[ -z "${WANDB_ENABLED:-}" ]]; then
  if [[ "${WANDB_RUN}" == "dummy" ]]; then
    export WANDB_ENABLED=0
  else
    export WANDB_ENABLED=1
  fi
fi

# Pretrain defaults (nanochat speedrun d20).
export PRETRAIN_ITERS="${PRETRAIN_ITERS:-21400}"
export PRETRAIN_EXTENSION_ITERS="${PRETRAIN_EXTENSION_ITERS:-0}"
export PRETRAIN_DEVICE_BATCH_SIZE="${PRETRAIN_DEVICE_BATCH_SIZE:-32}"
export PRETRAIN_TOTAL_BATCH_SIZE="${PRETRAIN_TOTAL_BATCH_SIZE:-524288}"
export PRETRAIN_VAL_INTERVAL="${PRETRAIN_VAL_INTERVAL:-250}"
export PRETRAIN_LOG_INTERVAL="${PRETRAIN_LOG_INTERVAL:-10}"
export PRETRAIN_CHECKPOINT_INTERVAL="${PRETRAIN_CHECKPOINT_INTERVAL:-21400}"

# Midtrain defaults aligned to current Lean pipeline implementation.
export MIDTRAIN_ITERS="${MIDTRAIN_ITERS:-811}"
export MIDTRAIN_EXTENSION_ITERS="${MIDTRAIN_EXTENSION_ITERS:-0}"
export MIDTRAIN_DEVICE_BATCH_SIZE="${MIDTRAIN_DEVICE_BATCH_SIZE:-32}"
export MIDTRAIN_TOTAL_BATCH_SIZE="${MIDTRAIN_TOTAL_BATCH_SIZE:-524288}"
export MIDTRAIN_VAL_INTERVAL="${MIDTRAIN_VAL_INTERVAL:-150}"
export MIDTRAIN_LOG_INTERVAL="${MIDTRAIN_LOG_INTERVAL:-10}"
export MIDTRAIN_CHECKPOINT_INTERVAL="${MIDTRAIN_CHECKPOINT_INTERVAL:-811}"

# SFT defaults.
export SFT_EPOCHS="${SFT_EPOCHS:-1}"
export SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-4}"
export SFT_TARGET_EXAMPLES_PER_STEP="${SFT_TARGET_EXAMPLES_PER_STEP:-32}"

# GRPO defaults (used only if ENABLE_RL=1).
export GRPO_NUM_SAMPLES="${GRPO_NUM_SAMPLES:-16}"
export GRPO_MAX_NEW_TOKENS="${GRPO_MAX_NEW_TOKENS:-256}"
export GRPO_EXAMPLES_PER_STEP="${GRPO_EXAMPLES_PER_STEP:-16}"

if [[ -z "${PRETRAIN_DATA_PATH:-}" ]]; then
  if [[ -d "${NANOCHAT_DIR}/base_data_bin" ]]; then
    export PRETRAIN_DATA_PATH="${NANOCHAT_DIR}/base_data_bin"
  elif [[ -d "${NANOCHAT_DIR}/base_data" ]] && compgen -G "${NANOCHAT_DIR}/base_data/*.bin" > /dev/null; then
    export PRETRAIN_DATA_PATH="${NANOCHAT_DIR}/base_data"
  elif [[ -d "${REPO_ROOT}/data/nanochat" ]]; then
    export PRETRAIN_DATA_PATH="${REPO_ROOT}/data/nanochat"
  fi
fi
if [[ -z "${MIDTRAIN_DATA_PATH:-}" && -n "${PRETRAIN_DATA_PATH:-}" ]]; then
  export MIDTRAIN_DATA_PATH="${PRETRAIN_DATA_PATH}"
fi

export REQUIRE_BIN_DATA_PATHS="${REQUIRE_BIN_DATA_PATHS:-1}"
if [[ "${REQUIRE_BIN_DATA_PATHS}" == "1" ]]; then
  if [[ -z "${PRETRAIN_DATA_PATH:-}" ]]; then
    echo "error: PRETRAIN_DATA_PATH is unset. Set PRETRAIN_DATA_PATH to a directory containing .bin token shards." >&2
    exit 2
  fi
  if [[ -z "${MIDTRAIN_DATA_PATH:-}" ]]; then
    echo "error: MIDTRAIN_DATA_PATH is unset. Set MIDTRAIN_DATA_PATH to a directory containing .bin token shards." >&2
    exit 2
  fi
  if ! compgen -G "${PRETRAIN_DATA_PATH}/*.bin" > /dev/null; then
    echo "error: PRETRAIN_DATA_PATH has no .bin files: ${PRETRAIN_DATA_PATH}" >&2
    exit 2
  fi
  if ! compgen -G "${MIDTRAIN_DATA_PATH}/*.bin" > /dev/null; then
    echo "error: MIDTRAIN_DATA_PATH has no .bin files: ${MIDTRAIN_DATA_PATH}" >&2
    exit 2
  fi
fi

if [[ "${CLEAR_PIPELINE_CHECKPOINT:-1}" == "1" ]]; then
  rm -f "${REPO_ROOT}/.pipeline_checkpoint.json" "${NANOCHAT_DIR}/.pipeline_checkpoint.json"
fi

echo "Running ${PIPELINE_EXE} with torchrun (${NPROC_PER_NODE} processes)"
echo "Torchrun: ${TORCHRUN_BIN}"
echo "Executable: ${PIPELINE_EXE}"
echo "QUICK_MODE=${QUICK_MODE_FLAG:-0} ENABLE_RL=${ENABLE_RL_FLAG:-0}"
echo "NANOCHAT_DIR=${NANOCHAT_DIR}"
echo "NANOCHAT_USE_RUN_ID_DIR=${NANOCHAT_USE_RUN_ID_DIR:-0}"
echo "WANDB_RUN=${WANDB_RUN} WANDB_ENABLED=${WANDB_ENABLED}"
echo "MODEL_DEPTH=${MODEL_DEPTH} PARAM_DATA_RATIO=${PARAM_DATA_RATIO} VOCAB_SIZE=${VOCAB_SIZE}"
echo "TOKENIZER_MAX_CHARS=${TOKENIZER_MAX_CHARS} TOKENIZER_DOC_CAP=${TOKENIZER_DOC_CAP}"
echo "INITIAL_DATA_SHARDS=${INITIAL_DATA_SHARDS} NUM_SHARDS=${NUM_SHARDS}"
echo "PRETRAIN_ITERS=${PRETRAIN_ITERS} PRETRAIN_DEVICE_BATCH_SIZE=${PRETRAIN_DEVICE_BATCH_SIZE} PRETRAIN_TOTAL_BATCH_SIZE=${PRETRAIN_TOTAL_BATCH_SIZE}"
echo "MIDTRAIN_ITERS=${MIDTRAIN_ITERS} MIDTRAIN_DEVICE_BATCH_SIZE=${MIDTRAIN_DEVICE_BATCH_SIZE} MIDTRAIN_TOTAL_BATCH_SIZE=${MIDTRAIN_TOTAL_BATCH_SIZE}"
echo "SFT_EPOCHS=${SFT_EPOCHS} SFT_DEVICE_BATCH_SIZE=${SFT_DEVICE_BATCH_SIZE} SFT_TARGET_EXAMPLES_PER_STEP=${SFT_TARGET_EXAMPLES_PER_STEP}"
echo "DATA_PATH=${DATA_PATH}"
echo "PRETRAIN_DATA_PATH=${PRETRAIN_DATA_PATH:-<unset>}"
echo "MIDTRAIN_DATA_PATH=${MIDTRAIN_DATA_PATH:-<unset>}"
echo "REQUIRE_BIN_DATA_PATHS=${REQUIRE_BIN_DATA_PATHS}"

exec env TYR_DEVICE="${TYR_DEVICE:-cuda}" \
  "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" --no_python \
  "./.lake/build/bin/${PIPELINE_EXE}"
