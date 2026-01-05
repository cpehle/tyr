/-
  Training Infrastructure for Discrete Diffusion

  Adapted from GPT training for masked discrete diffusion.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Optim
import Tyr.NanoProof
import Tyr.Diffusion
import Tyr.DiffusionSchedule

namespace torch.diffusion.train

open torch
open torch.diffusion
open torch.nanoproof (RotaryCache BlockParams)

/-- Diffusion training configuration -/
structure TrainConfig where
  maxIters : Nat := 10000
  evalInterval : Nat := 500
  logInterval : Nat := 50
  learningRate : Float := 3e-4
  minLr : Float := 1e-5
  warmupIters : Nat := 200
  lrDecayIters : Nat := 10000
  gradClip : Float := 1.0
  batchSize : UInt64 := 32
  weightDecay : Float := 0.01
  deriving Repr, Inhabited

/-- Cosine learning rate schedule with warmup -/
def getLr (cfg : TrainConfig) (iterNum : Nat) : Float :=
  if iterNum < cfg.warmupIters then
    -- Linear warmup
    cfg.learningRate * (iterNum.toFloat / cfg.warmupIters.toFloat)
  else if iterNum > cfg.lrDecayIters then
    cfg.minLr
  else
    -- Cosine decay
    let decayRatio := (iterNum - cfg.warmupIters).toFloat /
                      (cfg.lrDecayIters - cfg.warmupIters).toFloat
    let pi : Float := 3.14159265358979323846
    let coeff := 0.5 * (1.0 + Float.cos (pi * decayRatio))
    cfg.minLr + coeff * (cfg.learningRate - cfg.minLr)

/-- Get a random batch of data from token array.
    For diffusion, we just need contiguous sequences (no shifting).
    Returns tokens of shape [batchSize, seqLen] -/
def getBatch {n : UInt64} (trainData : T #[n]) (batchSize seqLen : UInt64)
    : IO (T #[batchSize, seqLen]) := do
  -- Pick random starting positions for each sequence in the batch
  let totalTokens := batchSize * seqLen
  let maxStart := n - totalTokens
  let startTensor ← randint 0 maxStart.toInt64 #[1]
  let start := nn.itemInt startTensor

  -- Slice contiguous block from data
  let block : T #[totalTokens] := data.slice1d trainData start (start + totalTokens.toInt64)

  -- Reshape to [batchSize, seqLen]
  let batch := reshape block #[batchSize, seqLen]
  return batch

/-- Clip gradients for NanoProof-style block (used by diffusion) -/
def clipBlockGrads {n_embd n_head n_kv_head : UInt64}
    (params : BlockParams n_embd n_head n_kv_head)
    (maxNorm : Float) : IO Unit := do
  let _ ← nn.clip_grad_norm_ params.attn.c_q maxNorm
  let _ ← nn.clip_grad_norm_ params.attn.c_k maxNorm
  let _ ← nn.clip_grad_norm_ params.attn.c_v maxNorm
  let _ ← nn.clip_grad_norm_ params.attn.c_proj maxNorm
  let _ ← nn.clip_grad_norm_ params.mlp.c_fc maxNorm
  let _ ← nn.clip_grad_norm_ params.mlp.c_proj maxNorm

/-- Clip gradients for all diffusion model parameters -/
def clipDiffusionGrads {cfg : Config} (params : DiffusionParams cfg) (maxNorm : Float) : IO Unit := do
  let _ ← nn.clip_grad_norm_ params.token_emb maxNorm
  let _ ← nn.clip_grad_norm_ params.time_emb maxNorm
  for block in params.blocks do
    clipBlockGrads block maxNorm
  let _ ← nn.clip_grad_norm_ params.output_head maxNorm

/-- Single diffusion training step -/
def trainStep {modelCfg : Config} {batch seq num_timesteps : UInt64}
    (trainCfg : TrainConfig)
    (params : DiffusionParams modelCfg)
    (optState : Optim.AdamWState (DiffusionParams modelCfg))
    (schedule : MaskedDiffusionSchedule num_timesteps)
    (x_0 : T #[batch, seq])  -- Clean tokens
    (rotaryCache : RotaryCache rotaryLen modelCfg.headDim)
    (lr : Float)
    (debug : Bool := false)
    : IO (DiffusionParams modelCfg × Optim.AdamWState (DiffusionParams modelCfg) × Float) := do
  if debug then
    IO.println "    trainStep: zero grads..."
    (← IO.getStdout).flush

  -- Zero gradients BEFORE forward/backward
  let params := TensorStruct.zeroGrads params

  if debug then
    IO.println "    trainStep: sampling timesteps..."
    (← IO.getStdout).flush

  -- Sample random timesteps for each sample in batch
  let t ← randint 0 num_timesteps.toInt64 #[batch]

  if debug then
    IO.println "    trainStep: adding masks..."
    (← IO.getStdout).flush

  -- Add masks to get noisy input
  let x_t ← schedule.addMasks x_0 t

  if debug then
    IO.println "    trainStep: forward..."
    (← IO.getStdout).flush

  -- Forward pass
  let logits := forward params x_t t rotaryCache

  if debug then
    IO.println "    trainStep: loss..."
    (← IO.getStdout).flush

  -- Compute masked loss (only on masked positions)
  let lossT := maskedCrossEntropyLoss logits x_0 x_t schedule.mask_token_id

  if debug then
    IO.println "    trainStep: backward..."
    (← IO.getStdout).flush

  -- Backward pass
  autograd.backwardLoss lossT

  if debug then
    IO.println "    trainStep: grad clip..."
    (← IO.getStdout).flush

  -- Gradient clipping
  if trainCfg.gradClip > 0 then
    clipDiffusionGrads params trainCfg.gradClip

  if debug then
    IO.println "    trainStep: get loss val..."
    (← IO.getStdout).flush

  -- Get loss value for logging
  let lossVal := nn.item lossT

  if debug then
    IO.println "    trainStep: extract grads..."
    (← IO.getStdout).flush

  -- Extract gradients from parameters
  let grads := TensorStruct.grads params

  if debug then
    IO.println "    trainStep: optimizer..."
    (← IO.getStdout).flush

  -- Update parameters with AdamW
  let opt := Optim.adamw (lr := lr) (weight_decay := trainCfg.weightDecay)
  let (params', optState') := Optim.step opt params grads optState

  return (params', optState', lossVal)

/-- Evaluate model on validation data.
    Uses no_grad context to avoid building computation graphs. -/
def evalLoss {modelCfg : Config} {num_timesteps : UInt64}
    (params : DiffusionParams modelCfg)
    (schedule : MaskedDiffusionSchedule num_timesteps)
    (valData : T #[n])
    (batchSize seqLen : UInt64)
    (rotaryCache : RotaryCache rotaryLen modelCfg.headDim)
    (numBatches : Nat)
    : IO Float := autograd.no_grad do
  let mut totalLoss : Float := 0.0
  for _ in [:numBatches] do
    let x_0 ← getBatch valData batchSize seqLen
    let t ← randint 0 num_timesteps.toInt64 #[batchSize]
    let x_t ← schedule.addMasks x_0 t
    let logits := forward params x_t t rotaryCache
    let lossT := maskedCrossEntropyLoss logits x_0 x_t schedule.mask_token_id
    let lossVal := nn.item lossT
    totalLoss := totalLoss + lossVal
  return totalLoss / numBatches.toFloat

/-- Main diffusion training loop with validation -/
def trainLoop {modelCfg : Config} {num_timesteps : UInt64}
    (trainCfg : TrainConfig)
    (initParams : DiffusionParams modelCfg)
    (initOptState : Optim.AdamWState (DiffusionParams modelCfg))
    (schedule : MaskedDiffusionSchedule num_timesteps)
    (trainData : T #[nTrain])
    (valData : T #[nVal])
    (rotaryCache : RotaryCache rotaryLen modelCfg.headDim)
    (evalBatches : Nat := 10)
    : IO (DiffusionParams modelCfg × Float) := do
  let mut params := initParams
  let mut optState := initOptState
  let mut totalLoss : Float := 0.0
  let mut bestValLoss : Float := 1e10

  IO.println s!"Starting diffusion training for {trainCfg.maxIters} iterations..."
  IO.println s!"  batch_size={trainCfg.batchSize}, seq_len={modelCfg.seq_len}"
  IO.println s!"  lr={trainCfg.learningRate}, warmup={trainCfg.warmupIters}"
  IO.println s!"  diffusion_steps={num_timesteps}, context_len={schedule.context_len}"
  (← IO.getStdout).flush

  for iterNum in [:trainCfg.maxIters] do
    if iterNum == 0 then
      IO.println "Starting first iteration..."
      (← IO.getStdout).flush

    -- Get learning rate with schedule
    let lr := getLr trainCfg iterNum

    if iterNum == 0 then
      IO.println s!"  Got lr={lr}"
      (← IO.getStdout).flush

    -- Get batch of clean tokens
    let x_0 ← getBatch trainData trainCfg.batchSize modelCfg.seq_len

    if iterNum == 0 then
      IO.println "  Got batch"
      (← IO.getStdout).flush

    -- Training step
    let debug := iterNum == 0
    let (params', optState', lossVal) ← trainStep trainCfg params optState schedule x_0 rotaryCache lr debug

    if iterNum == 0 then
      IO.println s!"  Training step done, loss={lossVal}"
      (← IO.getStdout).flush

    params := params'
    optState := optState'
    totalLoss := totalLoss + lossVal

    -- Logging
    if iterNum % trainCfg.logInterval == 0 && iterNum > 0 then
      let avgLoss := totalLoss / trainCfg.logInterval.toFloat
      let liveTensors ← get_live_tensors
      IO.println s!"iter {iterNum}: loss={lossVal}, avg_loss={avgLoss}, lr={lr}, live_tensors={liveTensors}"
      (← IO.getStdout).flush
      totalLoss := 0.0

    -- Validation
    if iterNum % trainCfg.evalInterval == 0 && iterNum > 0 then
      let valLoss ← evalLoss params schedule valData trainCfg.batchSize modelCfg.seq_len rotaryCache evalBatches
      IO.println s!"  val_loss={valLoss}"
      if valLoss < bestValLoss then
        bestValLoss := valLoss
        IO.println s!"  [new best val_loss!]"
      (← IO.getStdout).flush

  -- Final validation
  let finalValLoss ← evalLoss params schedule valData trainCfg.batchSize modelCfg.seq_len rotaryCache evalBatches
  IO.println s!"Training complete! Final val_loss={finalValLoss}"
  IO.println s!"Best val_loss={bestValLoss}"
  return (params, bestValLoss)

end torch.diffusion.train
