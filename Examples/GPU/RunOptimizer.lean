import Tyr.Torch
import Tyr.GPU.Kernels.Optimizer
import Examples.GPU.Parity
import Examples.GPU.Benchmark

namespace Examples.GPU.RunOptimizer

open torch
open Tyr.GPU.Kernels.Optimizer

private structure OptimizerShape where
  elements : UInt64

private def optimizerMatrix : LeanBenchmark.Matrix OptimizerShape := {
  defaultProfile := "qwen3tts-talker-primary"
  cases := #[
    { id := "qkv_weight_n1048576", payload := ⟨1048576⟩,
      profiles := #["qwen3tts-talker-primary", "training"],
      tags := #["attention", "matrix", "primary"] },
    { id := "kv_weight_n131072", payload := ⟨131072⟩,
      profiles := #["qwen3tts-talker-primary", "training"],
      tags := #["attention", "gqa", "small"] },
    { id := "mlp_weight_n2097152", payload := ⟨2097152⟩,
      profiles := #["qwen3tts-talker-primary", "training"],
      tags := #["mlp", "matrix", "primary"] },
    { id := "embedding_n3145728", payload := ⟨3145728⟩,
      profiles := #["qwen3tts-talker-primary", "training"],
      tags := #["embedding", "adam", "large"] },
    { id := "micro_n4096", payload := ⟨4096⟩,
      profiles := #["micro"], tags := #["regression"] }
  ]
}

private def benchmarkShape (cfg : Benchmark.Config) (stream : UInt64)
    (caseLabel : String) (elements : UInt64) : IO (String × Bool) := do
  let device := Device.CUDA 0
  let master : T #[elements] ← torch.randn #[elements] false device
  let masterOriginal : T #[elements] ← torch.randn #[elements] false device
  torch.copy_ masterOriginal master
  let gradF : T #[elements] ← torch.randn #[elements] false device
  let grad : T #[elements] := torch.toBFloat16' gradF
  let gradBackupF : T #[elements] ← torch.randn #[elements] false device
  let gradBackup : T #[elements] := torch.toBFloat16' gradBackupF
  torch.copy_ gradBackup grad
  let model : T #[elements] := torch.toBFloat16' master
  let moment1 : T #[elements] ← torch.randn #[elements] false device
  let moment2 : T #[elements] ← torch.randn #[elements] false device
  let zeroMoment : T #[elements] := torch.zeros_like master
  torch.copy_ moment1 zeroMoment
  torch.copy_ moment2 zeroMoment
  let learningRate : Float := 3.0e-4
  let beta1 : Float := 0.9
  let beta2 : Float := 0.95
  let epsilon : Float := 1.0e-8
  let weightDecay : Float := 0.1
  let invBias1 : Float := 1.0 / (1.0 - beta1)
  let invBias2 : Float := 1.0 / (1.0 - beta2)
  let grad32 := torch.toFloat' grad
  let expectedM : T #[elements] :=
    torch.add (moment1 * beta1) (grad32 * (1.0 - beta1))
  let expectedV : T #[elements] :=
    torch.add (moment2 * beta2) ((torch.mul grad32 grad32) * (1.0 - beta2))
  let mHat := expectedM * invBias1
  let vHat := expectedV * invBias2
  let invDenom := torch.nn.pow (torch.nn.sqrt vHat + epsilon) (-1.0)
  let update : T #[elements] :=
    torch.add (torch.mul mHat invDenom) (masterOriginal * weightDecay)
  let expectedMaster : T #[elements] := torch.sub masterOriginal (update * learningRate)
  let expectedModel := torch.toBFloat16' expectedMaster
  let blocks := (elements + 255) / 256
  let launch : IO Unit :=
    adamWMasterBf16.launch master model grad moment1 moment2 elements
      learningRate.toFloat32 beta1.toFloat32 beta2.toFloat32 epsilon.toFloat32
      weightDecay.toFloat32 invBias1.toFloat32 invBias2.toFloat32
      blocks 1 1 256 1 1 0 stream

  launch
  torch.cuda_synchronize
  let preMaster := compareTensors s!"optimizer.{caseLabel}.master" expectedMaster master 2e-5 2e-5
  let preModel := compareTensors s!"optimizer.{caseLabel}.model" expectedModel model 2e-3 2e-3
  let preM := compareTensors s!"optimizer.{caseLabel}.moment1" expectedM moment1 2e-5 2e-5
  let preV := compareTensors s!"optimizer.{caseLabel}.moment2" expectedV moment2 2e-5 2e-5
  for check in #[preMaster, preModel, preM, preV] do logTensorCheck check

  let zeroMaster : T #[elements] ← torch.randn #[elements] false device
  torch.copy_ master zeroMaster
  torch.copy_ model (torch.toBFloat16' zeroMaster)
  torch.copy_ grad (torch.zeros_like grad)
  torch.copy_ moment1 zeroMoment
  torch.copy_ moment2 zeroMoment
  launch
  torch.cuda_synchronize
  let zeroExpectedMaster : T #[elements] :=
    zeroMaster * (1.0 - learningRate * weightDecay)
  let zeroExpectedModel := torch.toBFloat16' zeroExpectedMaster
  let zeroMasterCheck :=
    compareTensors s!"optimizer.{caseLabel}.zero_grad.master" zeroExpectedMaster master 2e-5 2e-5
  let zeroModelCheck :=
    compareTensors s!"optimizer.{caseLabel}.zero_grad.model" zeroExpectedModel model 2e-3 2e-3
  let zeroMCheck :=
    compareTensors s!"optimizer.{caseLabel}.zero_grad.moment1" zeroMoment moment1 0 0
  let zeroVCheck :=
    compareTensors s!"optimizer.{caseLabel}.zero_grad.moment2" zeroMoment moment2 0 0
  for check in #[zeroMasterCheck, zeroModelCheck, zeroMCheck, zeroVCheck] do
    logTensorCheck check

  let resetMaster : T #[elements] ← torch.randn #[elements] false device
  torch.copy_ master resetMaster
  torch.copy_ model (torch.toBFloat16' resetMaster)
  torch.copy_ grad gradBackup
  torch.copy_ moment1 zeroMoment
  torch.copy_ moment2 zeroMoment
  let samples ← Benchmark.timeCudaEvents cfg stream launch
  let finiteMaster := compareTensors s!"optimizer.{caseLabel}.master.self" master master 0 0
  let finiteModel := compareTensors s!"optimizer.{caseLabel}.model.self" model model 0 0
  let correct := preMaster.ok && preModel.ok && preM.ok && preV.ok &&
    zeroMasterCheck.ok && zeroModelCheck.ok && zeroMCheck.ok && zeroVCheck.ok &&
    finiteMaster.ok && finiteModel.ok
  let line := Benchmark.summaryJson cfg s!"adamw_training_{caseLabel}" "tyr"
    "generated_fused_fp32_master_moments_bf16_grad_model" samples correct
    "in_place_full_optimizer_update" true (some elements.toFloat) (some "parameters")
  pure (line, correct)

private def runBenchmark (args : List String) : IO UInt32 := do
  if !(← requireCuda "optimizer") then return 1
  let cfg ← Benchmark.parseConfig args "optimizer_bench"
  torch.manualSeed 0
  let selection := LeanBenchmark.MatrixSelection.parse args optimizerMatrix.defaultProfile
  let selected ← match optimizerMatrix.select selection with
    | .ok cases => pure cases
    | .error msg => throw <| IO.userError msg
  let stream ← torch.cuda_current_stream
  let mut lines : Array String := #[]
  let mut allOk := true
  for matrixCase in selected do
    let (line, ok) ← benchmarkShape cfg stream matrixCase.id matrixCase.payload.elements
    lines := lines.push line
    allOk := allOk && ok
  for line in lines do IO.println line
  match cfg.jsonlOut? with
  | some path => Benchmark.writeJsonl path lines
  | none => pure ()
  pure (if allOk then 0 else 1)

def main (args : List String) : IO UInt32 := runBenchmark args

end Examples.GPU.RunOptimizer

def main (args : List String) : IO UInt32 :=
  Examples.GPU.RunOptimizer.main args
