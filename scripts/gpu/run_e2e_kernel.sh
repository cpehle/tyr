#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <KernelModule> <RunnerExe> <Label> [ExtraLeanBuildTarget ...]" >&2
  echo "Example: $0 Tyr.GPU.Kernels.Rotary RunRotary rotary" >&2
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

trials="${E2E_TRIALS:-1}"
if ! [[ "$trials" =~ ^[0-9]+$ ]] || [[ "$trials" -lt 1 ]]; then
  echo "E2E_TRIALS must be a positive integer (got: $trials)" >&2
  exit 2
fi

echo "[1/3] Build GPU-backed target (${label})"
lake -R run buildGpuTarget -- "$kernel_module" "$runner_exe" "${extra_build_targets[@]}"

for i in $(seq 1 "$trials"); do
  echo "[2/3] (${i}/${trials}) Regenerate fixture tensors (${label})"
  lake -R run runBuiltTarget -- "${runner_exe}" --gen-only --regen

  echo "[3/3] (${i}/${trials}) Run end-to-end check (${label})"
  lake -R run runBuiltTarget -- "${runner_exe}"
done
