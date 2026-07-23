/- End-to-end fused residual + RMSNorm validation. -/
import Tyr.Torch
import Tyr.GPU.Kernels.FusedRMSNorm
import Examples.GPU.Parity
import Examples.GPU.FixtureRunner
import Examples.GPU.Benchmark

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
      fusedRMSNormResidual64x1024F32Direct.launch
        inputs.x inputs.residual inputs.weight out outResid
        64 1 1 256 1 1 0 stream
      let _ ← torch.cuda_synchronize
      compareResults (if blackwell then "blackwell.f32" else "f32") expectedOut expectedResid out outResid 5e-3 5e-3
  | .ok .BFloat16 => do
      let (expectedOut, expectedResid) := rmsNormReferenceBFloat16 inputs
      let out := torch.zeros_like inputs.x
      let outResid := torch.zeros_like inputs.residual
      fusedRMSNormResidual64x1024Bf16Direct.launch
        inputs.x inputs.residual inputs.weight out outResid
        64 1 1 256 1 1 0 stream
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


private def benchmarkFloat32 (cfg : Benchmark.Config) (stream : UInt64)
    (_blackwell : Bool) : IO (String × Bool) := do
  let inputs ← randomInputs .Float32
  let (expectedOut, expectedResid) := rmsNormReference inputs
  let out := torch.zeros_like inputs.x
  let outResid := torch.zeros_like inputs.residual
  let launch := do
    fusedRMSNormResidual64x1024F32Direct.launch
      inputs.x inputs.residual inputs.weight out outResid
      64 1 1 256 1 1 0 stream
  launch; torch.cuda_synchronize
  let correct ← compareResults "bench.f32" expectedOut expectedResid out outResid 5e-3 5e-3
  let samples ← Benchmark.timeCudaEvents cfg stream launch
  let route := "generated_direct_row_cta_256x4"
  pure (Benchmark.summaryJson cfg "rmsnorm_residual_f32_64x1024" "tyr" route samples correct, correct)

private def benchmarkBFloat16 (cfg : Benchmark.Config) (stream : UInt64)
    (_blackwell : Bool) : IO (String × Bool) := do
  let inputs ← randomInputs .BFloat16
  let (expectedOut, expectedResid) := rmsNormReferenceBFloat16 inputs
  let out := torch.zeros_like inputs.x
  let outResid := torch.zeros_like inputs.residual
  let launch := do
    fusedRMSNormResidual64x1024Bf16Direct.launch
      inputs.x inputs.residual inputs.weight out outResid
      64 1 1 256 1 1 0 stream
  launch; torch.cuda_synchronize
  let correct ← compareResults "bench.bf16" expectedOut expectedResid out outResid 2e-2 2e-2
  let samples ← Benchmark.timeCudaEvents cfg stream launch
  let route := "generated_direct_row_cta_256x4"
  pure (Benchmark.summaryJson cfg "rmsnorm_residual_bf16_64x1024" "tyr" route samples correct, correct)

private inductive RMSNormBenchPayload where
  | micro
  | training (rows : UInt64)
  deriving Repr

private def rmsNormMatrix : LeanBenchmark.Matrix RMSNormBenchPayload := {
  defaultProfile := "training"
  cases := #[
    { id := "micro_64", payload := .training 64,
      profiles := #["micro"], tags := #["forward", "regression"] },
    { id := "b1_s768", payload := .training 768,
      profiles := #["training", "batch-sweep", "qwen3tts-talker"],
      tags := #["training", "latency"] },
    { id := "b2_s768", payload := .training 1536,
      profiles := #["training", "batch-sweep", "qwen3tts-talker"],
      tags := #["training", "throughput"] },
    { id := "b4_s768", payload := .training 3072,
      profiles := #["training", "batch-sweep", "qwen3tts-talker"],
      tags := #["training", "throughput", "primary"] },
    { id := "b8_s768", payload := .training 6144,
      profiles := #["training", "batch-sweep", "qwen3tts-talker"],
      tags := #["training", "throughput", "saturation"] }
  ]
}

private def benchmarkTrainingShape (cfg : Benchmark.Config) (stream : UInt64)
    (caseLabel : String) (rows : UInt64) : IO (Array String × Bool) := do
  let device := Device.CUDA 0
  let x : T #[rows, hiddenDim] :=
    torch.toBFloat16' (← torch.randn #[rows, hiddenDim] false device)
  let residual : T #[rows, hiddenDim] :=
    torch.toBFloat16' (← torch.randn #[rows, hiddenDim] false device)
  let weight : T #[hiddenDim] :=
    torch.toBFloat16' (← torch.randn #[hiddenDim] false device)
  let gradOut : T #[rows, hiddenDim] :=
    torch.toBFloat16' (← torch.randn #[rows, hiddenDim] false device)
  let gradOutResid : T #[rows, hiddenDim] :=
    torch.toBFloat16' (← torch.randn #[rows, hiddenDim] false device)

  let expectedResid : T #[rows, hiddenDim] := x + residual
  let residF : T #[rows, hiddenDim] := torch.toFloat' expectedResid
  let weightF : T #[hiddenDim] := torch.toFloat' weight
  let gradOutF : T #[rows, hiddenDim] := torch.toFloat' gradOut
  let gradOutResidF : T #[rows, hiddenDim] := torch.toFloat' gradOutResid
  let inv2d : T #[rows, 1] :=
    torch.rsqrt (torch.nn.meanDim (residF * residF) 1 true + rmsNormEps)
  let expectedInv : T #[rows] := torch.reshape inv2d #[rows]
  let invFull : T #[rows, hiddenDim] := torch.nn.expand inv2d #[rows, hiddenDim]
  let weight2d : T #[1, hiddenDim] := torch.reshape weightF #[1, hiddenDim]
  let weightFull : T #[rows, hiddenDim] :=
    torch.nn.expand weight2d #[rows, hiddenDim]
  let normalized : T #[rows, hiddenDim] := residF * invFull
  let expectedOut : T #[rows, hiddenDim] :=
    torch.toBFloat16' (normalized * weightFull)
  let weightedGrad : T #[rows, hiddenDim] := gradOutF * weightFull
  let dot : T #[rows, 1] :=
    torch.nn.sumDim (weightedGrad * residF) 1 true
  let correction : T #[rows, 1] :=
    torch.mul_scalar (dot * inv2d * inv2d * inv2d) 0.0009765625
  let correctionFull : T #[rows, hiddenDim] :=
    torch.nn.expand correction #[rows, hiddenDim]
  let expectedGradInput : T #[rows, hiddenDim] :=
    torch.toBFloat16' (weightedGrad * invFull - residF * correctionFull + gradOutResidF)
  let expectedGradWeight : T #[hiddenDim] :=
    torch.nn.sumDim (gradOutF * normalized) 0 false

  let out := torch.zeros_like x
  let outResid := torch.zeros_like residual
  let invRms : T #[rows] := torch.zeros #[rows] false device
  let gradInput := torch.zeros_like x
  let gradWeight : T #[hiddenDim] := torch.zeros #[hiddenDim] false device

  let launchForward : IO Unit :=
    fusedRMSNormResidualRows1024Bf16TrainFwd.launch
      x residual weight out outResid invRms rows
      rows 1 1 256 1 1 0 stream
  let launchBwdInput : IO Unit :=
    fusedRMSNormResidualRows1024Bf16BwdInput.launch
      gradOut gradOutResid outResid weight invRms gradInput rows
      rows 1 1 256 1 1 0 stream
  let launchBwdWeight : IO Unit :=
    fusedRMSNormResidualRows1024Bf16BwdWeight.launch
      gradOut outResid invRms gradWeight rows
      4 1 1 256 1 1 0 stream
  let launchTraining : IO Unit := do
    launchForward
    launchBwdInput
    launchBwdWeight

  launchTraining
  torch.cuda_synchronize
  let outCheck := compareTensors s!"rmsnorm.{caseLabel}.out" expectedOut out 3e-2 3e-2
  let residCheck := compareTensors s!"rmsnorm.{caseLabel}.resid" expectedResid outResid 0.0 0.0
  let invCheck := compareTensors s!"rmsnorm.{caseLabel}.inv" expectedInv invRms 2e-4 2e-4
  let dxCheck := compareTensors s!"rmsnorm.{caseLabel}.dx" expectedGradInput gradInput 3e-2 3e-2
  let dwCheck := compareTensors s!"rmsnorm.{caseLabel}.dw" expectedGradWeight gradWeight 3e-3 0.25
  for check in #[outCheck, residCheck, invCheck, dxCheck, dwCheck] do
    logTensorCheck check

  let fwdSamples ← Benchmark.timeCudaEvents cfg stream launchForward
  let dxSamples ← Benchmark.timeCudaEvents cfg stream launchBwdInput
  let dwSamples ← Benchmark.timeCudaEvents cfg stream launchBwdWeight
  let trainingSamples ← Benchmark.timeCudaEvents cfg stream launchTraining

  let postOut := compareTensors s!"rmsnorm.{caseLabel}.out.post" expectedOut out 3e-2 3e-2
  let postResid := compareTensors s!"rmsnorm.{caseLabel}.resid.post" expectedResid outResid 0.0 0.0
  let postInv := compareTensors s!"rmsnorm.{caseLabel}.inv.post" expectedInv invRms 2e-4 2e-4
  let postDx := compareTensors s!"rmsnorm.{caseLabel}.dx.post" expectedGradInput gradInput 3e-2 3e-2
  let postDw := compareTensors s!"rmsnorm.{caseLabel}.dw.post" expectedGradWeight gradWeight 3e-3 0.25
  for check in #[postOut, postResid, postInv, postDx, postDw] do
    logTensorCheck check

  let correct := outCheck.ok && residCheck.ok && invCheck.ok && dxCheck.ok && dwCheck.ok &&
    postOut.ok && postResid.ok && postInv.ok && postDx.ok && postDw.ok
  let elements := rows.toFloat * hiddenDim.toFloat
  let fwdRoute := "generated_bf16_rows1024_saved_inv_rowcta256"
  let dxRoute := "generated_bf16_rows1024_vjp_rowcta256"
  let dwRoute := "generated_bf16_rows1024_dw_colcta256"
  let lines := #[
    Benchmark.summaryJson cfg s!"rmsnorm_training_fwd_{caseLabel}" "tyr"
      fwdRoute fwdSamples correct "saved_state_forward" true
      (some elements) (some "elements"),
    Benchmark.summaryJson cfg s!"rmsnorm_training_bwd_input_{caseLabel}" "tyr"
      dxRoute dxSamples correct "input_and_residual_vjp" true
      (some elements) (some "elements"),
    Benchmark.summaryJson cfg s!"rmsnorm_training_bwd_weight_{caseLabel}" "tyr"
      dwRoute dwSamples correct "weight_gradient" true
      (some elements) (some "elements"),
    Benchmark.summaryJson cfg s!"rmsnorm_training_{caseLabel}" "tyr"
      s!"{fwdRoute}_plus_{dxRoute}_plus_{dwRoute}" trainingSamples correct
      "full_training_operation" true (some (3.0 * elements)) (some "elements")
  ]
  pure (lines, correct)

private def runBenchmark (args : List String) : IO UInt32 := do
  if !(← requireCuda suiteName) then return 1
  let cfg ← Benchmark.parseConfig args "rmsnorm_bench"
  seedFixtures s!"{suiteName}.benchmark" 0
  let stream ← torch.cuda_current_stream
  let blackwell ← isBlackwellFamily
  let selection := LeanBenchmark.MatrixSelection.parse args rmsNormMatrix.defaultProfile
  let selected ← match rmsNormMatrix.select selection with
    | .ok cases => pure cases
    | .error msg => throw <| IO.userError msg
  let mut lines : Array String := #[]
  let mut allOk := true
  for matrixCase in selected do
    match matrixCase.payload with
    | .micro => do
      let (f32Line, f32Ok) ← benchmarkFloat32 cfg stream blackwell
      let (bf16Line, bf16Ok) ← benchmarkBFloat16 cfg stream blackwell
      lines := lines.push f32Line |>.push bf16Line
      allOk := allOk && f32Ok && bf16Ok
    | .training rows => do
      let (caseLines, ok) ← benchmarkTrainingShape cfg stream matrixCase.id rows
      for line in caseLines do lines := lines.push line
      allOk := allOk && ok
  for line in lines do IO.println line
  match cfg.jsonlOut? with
  | some path => Benchmark.writeJsonl path lines
  | none => pure ()
  pure (if allOk then 0 else 1)

def main (args : List String) : IO UInt32 := do
  if args.contains "--benchmark" then runBenchmark args else
    runWithFixtures args suiteName fixtureSpec generateFixtures runOnce

end Examples.GPU.RunRMSNorm
