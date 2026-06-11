import LeanTest
import Tyr.Torch
import Examples.GPU.Parity
import Examples.GPU.RunLayerNorm
import Examples.GPU.RunRMSNorm
import Examples.GPU.RunMhaGB10
import Examples.GPU.RunRKCombine

namespace Tests.GPUGB10E2E

open LeanTest
open torch

private def runGpuBoolTestIf
    (label : String)
    (enabled : IO Bool)
    (reason : String)
    (action : IO Bool)
    : IO Unit := do
  if !(← torch.cuda_is_available) then
    IO.println s!"[skip] {label}: CUDA unavailable"
  else if !(← enabled) then
    IO.println s!"[skip] {label}: {reason}"
  else
    let ok ← action
    LeanTest.assertTrue ok label

@[test]
def testMhaGB10TorchParity : IO Unit :=
  runGpuBoolTestIf
    "mha_gb10_tyr_vs_torch"
    Examples.GPU.isBlackwellFamily
    "requires TYR_GPU_FAMILY=BLACKWELL"
    Examples.GPU.RunMhaGB10.runOnce

@[test]
def testLayerNormGB10Float32TorchParity : IO Unit :=
  runGpuBoolTestIf
    "layernorm_gb10_f32_tyr_vs_torch"
    Examples.GPU.isBlackwellFamily
    "requires TYR_GPU_FAMILY=BLACKWELL"
    Examples.GPU.RunLayerNorm.runFloat32Once

@[test]
def testLayerNormGB10BFloat16TorchParity : IO Unit :=
  runGpuBoolTestIf
    "layernorm_gb10_bf16_tyr_vs_torch"
    Examples.GPU.isBlackwellFamily
    "requires TYR_GPU_FAMILY=BLACKWELL"
    Examples.GPU.RunLayerNorm.runBFloat16Once

@[test]
def testRMSNormGB10Float32TorchParity : IO Unit :=
  runGpuBoolTestIf
    "rmsnorm_gb10_f32_tyr_vs_torch"
    Examples.GPU.isBlackwellFamily
    "requires TYR_GPU_FAMILY=BLACKWELL"
    Examples.GPU.RunRMSNorm.runFloat32Once

@[test]
def testRMSNormGB10BFloat16TorchParity : IO Unit :=
  runGpuBoolTestIf
    "rmsnorm_gb10_bf16_tyr_vs_torch"
    Examples.GPU.isBlackwellFamily
    "requires TYR_GPU_FAMILY=BLACKWELL"
    Examples.GPU.RunRMSNorm.runBFloat16Once

@[test]
def testRKCombineGB10TorchParity : IO Unit :=
  runGpuBoolTestIf
    "rk_combine_gb10_tyr_vs_torch"
    Examples.GPU.isBlackwellFamily
    "requires TYR_GPU_FAMILY=BLACKWELL"
    Examples.GPU.RunRKCombine.runOnce

end Tests.GPUGB10E2E
