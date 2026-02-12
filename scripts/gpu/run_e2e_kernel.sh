#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <KernelModule> <RunnerExe> <Label> [ExtraLeanBuildTarget ...]" >&2
  echo "Example: $0 Tyr.GPU.Kernels.Rotary RunThunderKittensRotary rotary" >&2
  exit 2
fi

kernel_module="$1"
runner_exe="$2"
label="$3"
shift 3
extra_build_targets=("$@")

source ./load_modules.sh

export LEAN_CC="$PWD/scripts/lean_cc_wrapper.sh"
export LEAN_CC_FAST=1
export LD_LIBRARY_PATH="$PWD/external/libtorch/lib:$PWD/cc/build:${EBROOTGCCCORE}/lib64:${LD_LIBRARY_PATH:-}"

trials="${E2E_TRIALS:-1}"
if ! [[ "$trials" =~ ^[0-9]+$ ]] || [[ "$trials" -lt 1 ]]; then
  echo "E2E_TRIALS must be a positive integer (got: $trials)" >&2
  exit 2
fi

echo "[1/6] Build Lean kernel + generator (${label})"
lake --quiet build GenerateGpuKernels "$kernel_module" "${extra_build_targets[@]}"

echo "[2/6] Generate CUDA translation unit (${label})"
lake env ./.lake/build/bin/GenerateGpuKernels "$kernel_module" --out-dir cc/src/generated

echo "[3/6] Build C++/CUDA runtime library"
make -C cc -j"$(nproc)"

echo "[4/6] Build Lean executable (${runner_exe})"
lake --quiet build "$runner_exe"

for i in $(seq 1 "$trials"); do
  echo "[5/6] (${i}/${trials}) Regenerate fixture tensors (${label})"
  lake env ./.lake/build/bin/"${runner_exe}" --gen-only --regen

  echo "[6/6] (${i}/${trials}) Run end-to-end check (${label})"
  lake env ./.lake/build/bin/"${runner_exe}"
done
