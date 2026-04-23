#!/usr/bin/env bash
set -euo pipefail

source ./load_modules.sh

export LEAN_CC="$PWD/scripts/lean_cc_wrapper.sh"
export LEAN_CC_FAST=1

echo "[1/2] Build GPU-backed benchmark target"
lake -R run buildGpuTarget -- Tyr.GPU.Kernels.MhaH100 RunMhaH100Train

echo "[2/2] Run benchmark"
lake -R run runBuiltTarget -- RunMhaH100Train --benchmark --warmup 20 --bench-iters 500 --lr 200.0 --noise 0.5 "$@"
