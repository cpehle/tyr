/- End-to-end fused residual + RMSNorm validation. -/
import Tyr.Torch
import Tyr.GPU.Kernels.FusedRMSNorm
import Examples.GPU.Parity
import Examples.GPU.FixtureRunner

namespace Examples.GPU.RunRMSNorm

open torch
open Tyr.GPU.Kernels

private def launchBlockX : UInt64 := 32
private def seqLen : UInt64 := 64
private def hiddenDim : UInt64 := 1024
private def rmsNormEps : Float := 1.0e-6

def suiteName : String := "rmsnorm"

def fixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/rmsnorm64x1024"⟩
  names := #["x", "residual", "weight", "expected_out", "expected_resid"]
}

def fixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fixtureSpec name

private abbrev RMSNormInput := T #[1, seqLen, hiddenDim]
private abbrev RMSNormWeight := T #[hiddenDim]

private structure RMSNormInputs where
  x : RMSNormInput
  residual : RMSNormInput
  weight : RMSNormWeight

private def rmsNormReference
    (inputs : RMSNormInputs)
    : RMSNormInput × RMSNormInput :=
  let expectedResid := inputs.x + inputs.residual
  let expectedOut := torch.nn.rmsNormWeighted expectedResid inputs.weight rmsNormEps
  (expectedOut, expectedResid)

private def rmsNormReferenceBFloat16
    (inputs : RMSNormInputs)
    : RMSNormInput × RMSNormInput :=
  let expectedResid := inputs.x + inputs.residual
  let expectedOut32 : RMSNormInput :=
    torch.nn.rmsNormWeighted
      (torch.toFloat' expectedResid)
      (torch.toFloat' inputs.weight)
      rmsNormEps
  let expectedOut := torch.toBFloat16' expectedOut32
  (expectedOut, expectedResid)

private def asBFloat16Inputs (inputs : RMSNormInputs) : RMSNormInputs := {
  x := torch.toBFloat16' inputs.x
  residual := torch.toBFloat16' inputs.residual
  weight := torch.toBFloat16' inputs.weight
}

private def inputDType? (inputs : RMSNormInputs) : Except String DType := do
  let dtype := inputs.x.dtype
  if inputs.residual.dtype != dtype then
    throw s!"residual dtype {inputs.residual.dtype} does not match x dtype {dtype}"
  if inputs.weight.dtype != dtype then
    throw s!"weight dtype {inputs.weight.dtype} does not match x dtype {dtype}"
  pure dtype

private def logInputDTypes (label : String) (inputs : RMSNormInputs) : IO Unit := do
  IO.println <|
    s!"[{suiteName}:{label}] input_dtypes x={inputs.x.dtype} residual={inputs.residual.dtype} " ++
    s!"weight={inputs.weight.dtype}"

private def compareResults
    (mode : String)
    (expectedOut : RMSNormInput)
    (expectedResid : RMSNormInput)
    (out : RMSNormInput)
    (outResid : RMSNormInput)
    (rtol atol : Float)
    : IO Bool := do
  let outCheck := compareTensors s!"rmsnorm.{mode}.output" (torch.toFloat' expectedOut) (torch.toFloat' out) rtol atol
  let residCheck := compareTensors s!"rmsnorm.{mode}.residual" (torch.toFloat' expectedResid) (torch.toFloat' outResid) rtol atol
  logTensorCheck outCheck
  logTensorCheck residCheck
  pure (outCheck.ok && residCheck.ok)

private def runInputs (label : String) (inputs : RMSNormInputs) : IO Bool := do
  logInputDTypes label inputs
  let stream ← torch.cuda_current_stream
  let blackwell ← isBlackwellFamily
  match inputDType? inputs with
  | .error msg =>
      IO.eprintln s!"[{suiteName}:{label}] unsupported_dtype_mix error={msg}"
      pure false
  | .ok .Float32 => do
      let (expectedOut, expectedResid) := rmsNormReference inputs
      let out := torch.zeros_like inputs.x
      let outResid := torch.zeros_like inputs.residual
      if blackwell then
        tkFusedRMSNormResidual1024F32Blackwell.launch
          inputs.x inputs.residual inputs.weight out outResid
          1 1 1 launchBlockX 1 1 0 stream
      else
        tkFusedRMSNormResidual1024F32.launch
          inputs.x inputs.residual inputs.weight out outResid
          1 1 1 launchBlockX 1 1 0 stream
      let _ ← torch.cuda_synchronize
      compareResults (if blackwell then "blackwell.f32" else "f32") expectedOut expectedResid out outResid 5e-3 5e-3
  | .ok .BFloat16 => do
      let (expectedOut, expectedResid) := rmsNormReferenceBFloat16 inputs
      let out := torch.zeros_like inputs.x
      let outResid := torch.zeros_like inputs.residual
      if blackwell then
        tkFusedRMSNormResidual1024Blackwell.launch
          inputs.x inputs.residual inputs.weight out outResid
          1 1 1 launchBlockX 1 1 0 stream
      else
        tkFusedRMSNormResidual1024.launch
          inputs.x inputs.residual inputs.weight out outResid
          1 1 1 launchBlockX 1 1 0 stream
      let _ ← torch.cuda_synchronize
      compareResults (if blackwell then "blackwell.bf16" else "bf16") expectedOut expectedResid out outResid 2e-2 2e-2
  | .ok dtype =>
      IO.eprintln s!"[{suiteName}:{label}] unsupported_dtype dtype={dtype}"
      pure false

private def randomInputs (dtype : DType) : IO RMSNormInputs := do
  let device := Device.CUDA 0
  let base : RMSNormInputs := {
    x := ← torch.rand #[1, seqLen, hiddenDim] false device
    residual := ← torch.rand #[1, seqLen, hiddenDim] false device
    weight := ← torch.rand #[hiddenDim] false device
  }
  match dtype with
  | .Float32 => pure base
  | .BFloat16 => pure (asBFloat16Inputs base)
  | _ => throw <| IO.userError s!"unsupported random rmsnorm dtype {dtype}"

def generateFixtures : IO Unit := do
  if !(← requireCuda suiteName) then
    throw <| IO.userError "CUDA is not available; cannot generate rmsnorm fixtures."

  IO.FS.createDirAll fixtureSpec.dir
  let inputs ← randomInputs .Float32
  let vendoredInputs := asBFloat16Inputs inputs
  let (expectedOut, expectedResid) := rmsNormReferenceBFloat16 vendoredInputs

  torch.data.saveTensor inputs.x (fixtureFile "x").toString
  torch.data.saveTensor inputs.residual (fixtureFile "residual").toString
  torch.data.saveTensor inputs.weight (fixtureFile "weight").toString
  torch.data.saveTensor expectedOut (fixtureFile "expected_out").toString
  torch.data.saveTensor expectedResid (fixtureFile "expected_resid").toString

  let outMean := torch.nn.item (torch.nn.meanAll expectedOut)
  let residMean := torch.nn.item (torch.nn.meanAll expectedResid)
  IO.println <|
    s!"Generated rmsnorm fixtures in {fixtureSpec.dir} " ++
    s!"input_dtype={inputs.x.dtype} ref_dtype={vendoredInputs.x.dtype} " ++
    s!"outMean={outMean} residMean={residMean}"

def runFloat32Once : IO Bool := do
  if !(← requireCuda suiteName) then
    return false
  seedFixtures s!"{suiteName}.f32" 0
  runInputs "random_f32" (← randomInputs .Float32)

def runBFloat16Once : IO Bool := do
  if !(← requireCuda suiteName) then
    return false
  seedFixtures s!"{suiteName}.bf16" 0
  runInputs "random_bf16" (← randomInputs .BFloat16)

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  if !(← fixturesPresent fixtureSpec) then
    generateFixtures

  let inputs : RMSNormInputs := {
    x := ← torch.data.loadTensor #[1, seqLen, hiddenDim] (fixtureFile "x").toString
    residual := ← torch.data.loadTensor #[1, seqLen, hiddenDim] (fixtureFile "residual").toString
    weight := ← torch.data.loadTensor #[hiddenDim] (fixtureFile "weight").toString
  }
  runInputs "fixtures" inputs

def main (args : List String) : IO UInt32 := do
  runWithFixtures args suiteName fixtureSpec generateFixtures runOnce

end Examples.GPU.RunRMSNorm
