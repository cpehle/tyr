import Tyr.Torch
import Tyr.GPU.Kernels.Loss
import Examples.GPU.Parity
import Examples.GPU.Benchmark

namespace Examples.GPU.RunLoss

open torch
open Tyr.GPU.Kernels.Loss

private def vocab : UInt64 := 3072

private structure LossShape where
  rows : UInt64

private def lossMatrix : LeanBenchmark.Matrix LossShape := {
  defaultProfile := "training"
  cases := #[
    { id := "b1_s768_v3072", payload := ⟨768⟩,
      profiles := #["training", "batch-sweep", "qwen3tts-talker"],
      tags := #["training", "latency"] },
    { id := "b2_s768_v3072", payload := ⟨1536⟩,
      profiles := #["training", "batch-sweep", "qwen3tts-talker"],
      tags := #["training", "throughput"] },
    { id := "b4_s768_v3072", payload := ⟨3072⟩,
      profiles := #["training", "batch-sweep", "qwen3tts-talker"],
      tags := #["training", "throughput", "primary"] },
    { id := "b8_s768_v3072", payload := ⟨6144⟩,
      profiles := #["training", "batch-sweep", "qwen3tts-talker"],
      tags := #["training", "throughput", "saturation"] },
    { id := "micro_r128_v3072", payload := ⟨128⟩,
      profiles := #["micro"], tags := #["training", "regression"] }
  ]
}

private def benchmarkShape (cfg : Benchmark.Config) (stream : UInt64)
    (caseLabel : String) (rows : UInt64) : IO (Array String × Bool) := do
  let device := Device.CUDA 0
  let logitsBase : T #[rows, vocab] :=
    torch.toBFloat16' (← torch.randn #[rows, vocab] false device)
  let logits : T #[rows, vocab] :=
    torch.autograd.set_requires_grad logitsBase true
  let targets : T #[rows] ← torch.randint 0 vocab.toInt64 #[rows] false device
  let logitsF : T #[rows, vocab] := torch.toFloat' logits
  let expectedLosses : T #[rows] :=
    torch.nn.cross_entropy_none logitsF targets
  let expectedMean : T #[] :=
    torch.nn.cross_entropy logitsF targets
  let gradSeed := torch.ones_like expectedMean
  let expectedGrad : T #[rows, vocab] :=
    torch.autograd.grad expectedMean logits gradSeed

  let losses : T #[rows] := torch.zeros #[rows] false device
  let meanLoss : T #[] := torch.zeros #[] false device
  let gradLogits := torch.zeros_like logitsBase
  let gradScale : Float32 := 1.0 / rows.toFloat32

  let launchRows : IO Unit :=
    crossEntropyRowsVocab3072Bf16Train.launch
      logitsBase targets losses gradLogits rows gradScale
      rows 1 1 256 1 1 0 stream
  let launchMean : IO Unit :=
    reduceMeanLossRowsF32.launch losses meanLoss rows
      1 1 1 256 1 1 0 stream
  let launchTraining : IO Unit := do
    launchRows
    launchMean

  launchTraining
  torch.cuda_synchronize
  let lossesCheck := compareTensors s!"loss.{caseLabel}.per_row" expectedLosses losses 2e-3 2e-3
  let meanCheck := compareTensors s!"loss.{caseLabel}.mean" expectedMean meanLoss 2e-3 2e-3
  let gradCheck := compareTensors s!"loss.{caseLabel}.grad_logits" expectedGrad gradLogits 3e-2 3e-2
  for check in #[lossesCheck, meanCheck, gradCheck] do logTensorCheck check

  let rowsSamples ← Benchmark.timeCudaEvents cfg stream launchRows
  let meanSamples ← Benchmark.timeCudaEvents cfg stream launchMean
  let trainingSamples ← Benchmark.timeCudaEvents cfg stream launchTraining

  let postLosses := compareTensors s!"loss.{caseLabel}.per_row.post" expectedLosses losses 2e-3 2e-3
  let postMean := compareTensors s!"loss.{caseLabel}.mean.post" expectedMean meanLoss 2e-3 2e-3
  let postGrad := compareTensors s!"loss.{caseLabel}.grad_logits.post" expectedGrad gradLogits 3e-2 3e-2
  for check in #[postLosses, postMean, postGrad] do logTensorCheck check
  let correct := lossesCheck.ok && meanCheck.ok && gradCheck.ok &&
    postLosses.ok && postMean.ok && postGrad.ok
  let elements := rows.toFloat * vocab.toFloat
  let trainRoute := "generated_bf16_v3072_rowcta256_fused_loss_vjp"
  let meanRoute := "generated_f32_loss_mean_block256"
  let lines := #[
    Benchmark.summaryJson cfg s!"cross_entropy_rows_{caseLabel}" "tyr"
      trainRoute rowsSamples correct "per_row_loss_plus_logits_vjp" true
      (some elements) (some "logits"),
    Benchmark.summaryJson cfg s!"cross_entropy_mean_{caseLabel}" "tyr"
      meanRoute meanSamples correct "mean_loss_reduction" true
      (some rows.toFloat) (some "losses"),
    Benchmark.summaryJson cfg s!"cross_entropy_training_{caseLabel}" "tyr"
      s!"{trainRoute}_plus_{meanRoute}" trainingSamples correct
      "full_training_operation" true (some elements) (some "logits")
  ]
  pure (lines, correct)

private def runBenchmark (args : List String) : IO UInt32 := do
  if !(← requireCuda "loss") then return 1
  let cfg ← Benchmark.parseConfig args "loss_bench"
  torch.manualSeed 0
  let selection := LeanBenchmark.MatrixSelection.parse args lossMatrix.defaultProfile
  let selected ← match lossMatrix.select selection with
    | .ok cases => pure cases
    | .error msg => throw <| IO.userError msg
  let stream ← torch.cuda_current_stream
  let mut lines : Array String := #[]
  let mut allOk := true
  for matrixCase in selected do
    let (caseLines, ok) ← benchmarkShape cfg stream matrixCase.id matrixCase.payload.rows
    for line in caseLines do lines := lines.push line
    allOk := allOk && ok
  for line in lines do IO.println line
  match cfg.jsonlOut? with
  | some path => Benchmark.writeJsonl path lines
  | none => pure ()
  pure (if allOk then 0 else 1)

def main (args : List String) : IO UInt32 :=
  runBenchmark args

end Examples.GPU.RunLoss

def main (args : List String) : IO UInt32 :=
  Examples.GPU.RunLoss.main args
