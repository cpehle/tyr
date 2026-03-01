#!/usr/bin/env bash
set -euo pipefail

source ./load_modules.sh

export LEAN_CC="$PWD/scripts/lean_cc_wrapper.sh"
export LEAN_CC_FAST=1
export LD_LIBRARY_PATH="$PWD/external/libtorch/lib:$PWD/cc/build:${EBROOTGCCCORE:+${EBROOTGCCCORE}/lib64:}${LD_LIBRARY_PATH:-}"

cpu_count() {
  local count
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  if command -v getconf >/dev/null 2>&1; then
    count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
    if [[ "$count" =~ ^[0-9]+$ ]] && [[ "$count" -gt 0 ]]; then
      echo "$count"
      return
    fi
  fi
  if command -v sysctl >/dev/null 2>&1; then
    count="$(sysctl -n hw.logicalcpu 2>/dev/null || true)"
    if ! [[ "$count" =~ ^[0-9]+$ ]] || [[ "$count" -lt 1 ]]; then
      count="$(sysctl -n hw.ncpu 2>/dev/null || true)"
    fi
    if [[ "$count" =~ ^[0-9]+$ ]] && [[ "$count" -gt 0 ]]; then
      echo "$count"
      return
    fi
  fi
  echo 1
}

echo "[1/5] Build Lean targets"
lake --quiet build GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 RunMhaH100Train

echo "[2/5] Generate CUDA translation unit"
lake env ./.lake/build/bin/GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 --out-dir cc/src/generated

echo "[3/5] Build C++/CUDA runtime library"
make -C cc -j"$(cpu_count)"

echo "[4/5] Build benchmark executable"
lake --quiet build RunMhaH100Train

echo "[5/5] Run benchmark"
lake env ./.lake/build/bin/RunMhaH100Train --benchmark --warmup 20 --bench-iters 500 --lr 200.0 --noise 0.5 "$@"
