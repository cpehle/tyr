#!/usr/bin/env bash
set -euo pipefail

source ./load_modules.sh

export LEAN_CC="$PWD/scripts/lean_cc_wrapper.sh"
export LEAN_CC_FAST=1
export LD_LIBRARY_PATH="$PWD/external/libtorch/lib:$PWD/cc/build:${EBROOTGCCCORE}/lib64:${LD_LIBRARY_PATH:-}"

echo "[1/5] Build Lean targets"
lake --quiet build GenerateGpuKernels Tyr.GPU.Kernels.ThunderKittensFlashAttn RunThunderKittensMhaH100Train

echo "[2/5] Generate CUDA translation unit"
lake env ./.lake/build/bin/GenerateGpuKernels Tyr.GPU.Kernels.ThunderKittensFlashAttn --out-dir cc/src/generated

echo "[3/5] Build C++/CUDA runtime library"
make -C cc -j"$(nproc)"

echo "[4/5] Build benchmark executable"
lake --quiet build RunThunderKittensMhaH100Train

echo "[5/5] Run benchmark"
lake env ./.lake/build/bin/RunThunderKittensMhaH100Train --benchmark --warmup 20 --bench-iters 500 --lr 200.0 --noise 0.5 "$@"
