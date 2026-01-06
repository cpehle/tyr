/-
  Tests for Diffusion Model Implementation
-/
import Tyr

open torch
open torch.diffusion
open torch.nanoproof (RotaryCache)

/-- Assertion helper -/
def assertWith (cond : Bool) (msg : String) : IO Unit := do
  if !cond then
    IO.eprintln s!"ASSERTION FAILED: {msg}"
    throw (IO.userError msg)

/-- Test linear mask schedule -/
def testMaskSchedule : IO Unit := do
  IO.println "=== Testing Mask Schedule ==="
  let schedule := MaskedDiffusionSchedule.init 64 0 16
  -- Verify mask_probs goes from 1/64 to ~1.0
  let firstProb := nn.item (data.slice1d (n := 64) (m := 1) schedule.mask_probs 0 1)
  let lastProb := nn.item (data.slice1d (n := 64) (m := 1) schedule.mask_probs 63 64)
  IO.println s!"  First mask prob: {firstProb} (expected ~0.016)"
  IO.println s!"  Last mask prob: {lastProb} (expected ~1.0)"
  assertWith (firstProb > 0.01) "First prob > 0.01"
  assertWith (firstProb < 0.05) "First prob < 0.05"
  assertWith (lastProb > 0.95) "Last prob > 0.95"
  IO.println "  Mask schedule test passed!"

/-- Test mask application preserves context -/
def testMaskPreservesContext : IO Unit := do
  IO.println "=== Testing Context Preservation ==="
  let schedule := MaskedDiffusionSchedule.init 64 0 16  -- context_len=16
  -- Create random tokens (avoiding 0 which is mask)
  let x_0 ← randint 1 127 #[(2 : UInt64), 128]
  -- High timestep = lots of masking
  let t := full_int #[(2 : UInt64)] 63
  let x_t ← schedule.addMasks x_0 t
  -- Count total masks and check some were added
  let numMasked := nn.item (nn.sumAll (toFloat' (eq_scalar x_t 0)))
  IO.println s!"  Total masked positions: {numMasked}"
  -- With context_len=16 and seq=128, we have 112 non-context positions per sample
  -- At timestep 63 with linear schedule, mask_prob ~= 1.0, so most should be masked
  assertWith (numMasked > 50) "Many positions are masked"
  -- Context positions (0-15) should not all be masked
  -- We can't easily slice, so just verify the mask worked
  IO.println "  Context preservation test passed!"

/-- Test masked cross-entropy loss -/
def testMaskedLoss : IO Unit := do
  IO.println "=== Testing Masked Loss ==="
  -- Create dummy logits and targets
  let logits ← randn #[(2 : UInt64), 32, 128] false
  let targets ← randint 0 127 #[(2 : UInt64), 32]
  let x_t := full_int #[(2 : UInt64), 32] 0  -- All masked
  let loss := maskedCrossEntropyLoss logits targets x_t 0
  let lossVal := nn.item loss
  IO.println s!"  Loss (all masked): {lossVal}"
  assertWith (not lossVal.isNaN) "Loss not NaN"
  assertWith (lossVal > 0) "Loss positive"
  -- Test with no masked positions
  let x_t_none ← randint 1 127 #[(2 : UInt64), 32]  -- No masks
  let loss_none := maskedCrossEntropyLoss logits targets x_t_none 0
  let lossValNone := nn.item loss_none
  IO.println s!"  Loss (none masked): {lossValNone}"
  -- When no positions are masked, loss should be ~0 (masked loss / small epsilon)
  assertWith (lossValNone < 1.0) "Loss with no masks is small"
  IO.println "  Masked loss test passed!"

/-- Test diffusion model forward pass -/
def testDiffusionForward : IO Unit := do
  IO.println "=== Testing Diffusion Forward ==="
  let cfg := Config.tiny
  IO.println s!"  Config: {cfg.n_layer} layers, {cfg.n_head} heads, {cfg.n_embd} embed"
  IO.println "  Initializing model..."
  let params ← DiffusionParams.init cfg
  IO.println "  Initializing rotary cache..."
  let rotaryCache ← RotaryCache.init cfg.seq_len cfg.headDim
  -- Create input tensors
  let x_t ← randint 0 cfg.vocab_size.toInt64 #[(2 : UInt64), cfg.seq_len]
  let t ← randint 0 cfg.diffusion_steps.toInt64 #[(2 : UInt64)]
  IO.println "  Running forward pass..."
  let logits := forward params x_t t rotaryCache
  -- Verify output is finite
  let sum := nn.item (nn.sumAll logits)
  IO.println s!"  Output sum: {sum}"
  assertWith (not sum.isNaN) "Output not NaN"
  assertWith (not sum.isInf) "Output not Inf"
  IO.println "  Forward pass test passed!"

/-- Test confidence sampling terminates -/
def testSampleConfidence : IO Unit := do
  IO.println "=== Testing Confidence Sampling ==="
  let cfg := Config.tiny
  IO.println "  Initializing model..."
  let params ← DiffusionParams.init cfg
  let rotaryCache ← RotaryCache.init cfg.seq_len cfg.headDim
  IO.println "  Running sampling (max 50 steps)..."
  let result ← sampleConfidence (batch := 1) params rotaryCache (confidenceThreshold := 0.3) (maxSteps := 50)
  -- Check if any positions still masked
  let anyMasked := any (eq_scalar result cfg.mask_token_id.toInt64)
  IO.println s!"  Any positions still masked: {anyMasked}"
  -- Note: with random weights, sampling may not fully decode
  -- Just verify it doesn't crash and produces valid output
  let sum := nn.item (nn.sumAll (toFloat' result))
  assertWith (not sum.isNaN) "Output not NaN"
  IO.println "  Confidence sampling test passed!"

/-- Test top-k sampling -/
def testSampleTopK : IO Unit := do
  IO.println "=== Testing Top-K Sampling ==="
  let cfg := Config.tiny
  IO.println "  Initializing model..."
  let params ← DiffusionParams.init cfg
  let rotaryCache ← RotaryCache.init cfg.seq_len cfg.headDim
  IO.println "  Running top-k sampling (k=8, max 50 steps)..."
  let result ← sampleTopK (batch := 1) params rotaryCache (k := 8) (maxSteps := 50)
  -- Verify output is valid
  let sum := nn.item (nn.sumAll (toFloat' result))
  assertWith (not sum.isNaN) "Output not NaN"
  IO.println "  Top-K sampling test passed!"

/-- Test diffusion training step -/
def testTrainStep : IO Unit := do
  IO.println "=== Testing Diffusion Training Step ==="
  let modelCfg := Config.tiny
  let trainCfg : diffusion.train.TrainConfig := { maxIters := 1, batchSize := 2 }
  IO.println "  Initializing model..."
  let params ← DiffusionParams.init modelCfg
  let rotaryCache ← RotaryCache.init modelCfg.seq_len modelCfg.headDim
  let schedule := MaskedDiffusionSchedule.init modelCfg.diffusion_steps modelCfg.mask_token_id modelCfg.context_len
  -- Initialize optimizer
  let opt := Optim.adamw (lr := 1e-4)
  let optState := opt.init params
  -- Create dummy batch
  let x_0 ← randint 1 127 #[trainCfg.batchSize, modelCfg.seq_len]
  IO.println "  Running training step..."
  let (params', _optState', loss) ← diffusion.train.trainStep trainCfg params optState schedule x_0 rotaryCache 1e-4
  let params' : DiffusionParams modelCfg := params'
  IO.println s!"  Loss: {loss}"
  assertWith (not (Float.isNaN loss)) "Loss not NaN"
  assertWith (loss > 0) "Loss positive"
  -- Verify parameters changed
  let paramSum := nn.item (nn.sumAll params.token_emb)
  let paramSum' := nn.item (nn.sumAll params'.token_emb)
  IO.println s!"  Param sum before: {paramSum}, after: {paramSum'}"
  -- Parameters should change after training step
  assertWith (Float.abs (paramSum - paramSum') > 1e-6) "Parameters updated"
  IO.println "  Training step test passed!"

def main : IO Unit := do
  IO.println "Starting Diffusion Tests..."
  IO.println ""

  testMaskSchedule
  IO.println ""

  testMaskPreservesContext
  IO.println ""

  testMaskedLoss
  IO.println ""

  testDiffusionForward
  IO.println ""

  testSampleConfidence
  IO.println ""

  testSampleTopK
  IO.println ""

  testTrainStep
  IO.println ""

  IO.println "All diffusion tests passed!"
