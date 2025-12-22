/-
  Training Infrastructure for GPT

  Minimal viable training loop for nanoGPT-style training.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Optim
import Tyr.GPT

namespace torch.train

open torch
open torch.gpt

/-- Training configuration -/
structure TrainConfig where
  maxIters : Nat := 1000
  evalInterval : Nat := 100
  logInterval : Nat := 10
  learningRate : Float := 1e-3
  minLr : Float := 1e-4
  warmupIters : Nat := 100
  lrDecayIters : Nat := 1000
  gradClip : Float := 1.0
  batchSize : UInt64 := 4
  blockSize : UInt64 := 64
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

/-- Get a random batch of data from token array
    Returns (x, y) where y = x shifted by 1 position
    Uses a random starting position with batchSize consecutive sequences -/
def getBatch (trainData : T #[n]) (batchSize blockSize : UInt64) : IO (T #[batchSize, blockSize] × T #[batchSize, blockSize]) := do
  -- Pick a random starting position
  -- We need batchSize consecutive sequences of length blockSize+1
  let totalTokens := batchSize * (blockSize + 1)
  let maxStart := n - totalTokens
  let startTensor ← randint 0 maxStart.toInt64 #[1]
  let start := nn.itemInt startTensor

  -- Slice contiguous block from data using slice1d with type annotation
  let block : T #[totalTokens] := torch.data.slice1d trainData start (start + totalTokens.toInt64)

  -- Reshape to [batchSize, blockSize+1]
  let reshaped := reshape block #[batchSize, blockSize + 1]

  -- x = reshaped[:, :-1], y = reshaped[:, 1:]
  -- Using slice on dim 1 (second dimension)
  let xRaw := reshaped.slice 1 0 blockSize.toInt64 1
  let yRaw := reshaped.slice 1 1 (blockSize + 1).toInt64 1

  -- Reshape to ensure correct types
  let x := reshape xRaw #[batchSize, blockSize]
  let y := reshape yRaw #[batchSize, blockSize]

  return (x, y)

/-- Clip gradients for a single block's parameters -/
def clipBlockGrads {n_embd : UInt64} (params : BlockParams n_embd) (maxNorm : Float) : IO Unit := do
  let _ ← nn.clip_grad_norm_ params.ln1_weight maxNorm
  let _ ← nn.clip_grad_norm_ params.ln1_bias maxNorm
  let _ ← nn.clip_grad_norm_ params.q_proj maxNorm
  let _ ← nn.clip_grad_norm_ params.k_proj maxNorm
  let _ ← nn.clip_grad_norm_ params.v_proj maxNorm
  let _ ← nn.clip_grad_norm_ params.c_proj maxNorm
  let _ ← nn.clip_grad_norm_ params.c_proj_bias maxNorm
  let _ ← nn.clip_grad_norm_ params.ln2_weight maxNorm
  let _ ← nn.clip_grad_norm_ params.ln2_bias maxNorm
  let _ ← nn.clip_grad_norm_ params.mlp_fc maxNorm
  let _ ← nn.clip_grad_norm_ params.mlp_fc_bias maxNorm
  let _ ← nn.clip_grad_norm_ params.mlp_proj maxNorm
  let _ ← nn.clip_grad_norm_ params.mlp_proj_bias maxNorm

/-- Clip gradients for all GPT parameters -/
def clipGPTGrads {cfg : Config} (params : GPTParams cfg) (maxNorm : Float) : IO Unit := do
  let _ ← nn.clip_grad_norm_ params.wte maxNorm
  let _ ← nn.clip_grad_norm_ params.wpe maxNorm
  for block in params.blocks do
    clipBlockGrads block maxNorm
  let _ ← nn.clip_grad_norm_ params.ln_f_weight maxNorm
  let _ ← nn.clip_grad_norm_ params.ln_f_bias maxNorm

/-- Single training step using Optax-style optimizer -/
def trainStep {modelCfg : Config} {batch seq : UInt64}
    (trainCfg : TrainConfig)
    (params : GPTParams modelCfg)
    (optState : Optim.AdamWState (GPTParams modelCfg))
    (x : T #[batch, seq])
    (y : T #[batch, seq])
    (lr : Float)
    : IO (GPTParams modelCfg × Optim.AdamWState (GPTParams modelCfg) × Float) := do
  -- Zero gradients BEFORE forward/backward to prevent gradient accumulation
  let params := TensorStruct.zeroGrads params

  -- Forward pass: compute loss (training=true enables dropout)
  let lossT ← gpt.loss params x y true

  -- Backward pass
  autograd.backwardLoss lossT

  -- Gradient clipping (only if gradClip > 0)
  if trainCfg.gradClip > 0 then
    do
      let _ ← clipGPTGrads params trainCfg.gradClip
      pure ()
  else
    pure ()

  -- Get loss value for logging
  let lossVal := nn.item lossT

  -- Extract gradients from parameters
  let grads := TensorStruct.grads params

  -- Update parameters with AdamW using Optax-style step
  let opt := Optim.adamw (lr := lr)
  let (params', optState') := Optim.step opt params grads optState

  return (params', optState', lossVal)

/-- Compute perplexity from cross-entropy loss -/
def perplexity (loss : Float) : Float := Float.exp loss

/-- Evaluate model on validation data
    Returns average loss over numBatches batches.
    Uses no_grad context to avoid building computation graphs. -/
def evalLoss {modelCfg : Config}
    (params : GPTParams modelCfg)
    (valData : T #[n])
    (batchSize blockSize : UInt64)
    (numBatches : Nat)
    : IO Float := autograd.no_grad do
  let mut totalLoss : Float := 0.0
  for _ in [:numBatches] do
    let (x, y) ← getBatch valData batchSize blockSize
    -- Forward pass with training=false (disables dropout)
    let lossT ← gpt.loss params x y false
    let lossVal := nn.item lossT
    totalLoss := totalLoss + lossVal
  return totalLoss / numBatches.toFloat

/-- Main training loop -/
def trainLoop {modelCfg : Config}
    (trainCfg : TrainConfig)
    (initParams : GPTParams modelCfg)
    (initOptState : Optim.AdamWState (GPTParams modelCfg))
    (trainData : T #[n])
    : IO (GPTParams modelCfg) := do
  let mut params := initParams
  let mut optState := initOptState
  let mut totalLoss : Float := 0.0

  IO.println s!"Starting training for {trainCfg.maxIters} iterations..."
  IO.println s!"  batch_size={trainCfg.batchSize}, block_size={trainCfg.blockSize}"
  IO.println s!"  lr={trainCfg.learningRate}, warmup={trainCfg.warmupIters}"
  (← IO.getStdout).flush

  for iterNum in [:trainCfg.maxIters] do
    -- Get learning rate with schedule
    let lr := getLr trainCfg iterNum

    -- Get batch (for now, just use sequential data)
    let (x, y) ← getBatch trainData trainCfg.batchSize trainCfg.blockSize

    -- Training step
    let (params', optState', lossVal) ← trainStep trainCfg params optState x y lr

    params := params'
    optState := optState'
    totalLoss := totalLoss + lossVal

    -- Logging
    if iterNum % trainCfg.logInterval == 0 then
      let avgLoss := totalLoss / trainCfg.logInterval.toFloat
      let liveTensors ← get_live_tensors
      IO.println s!"iter {iterNum}: loss={lossVal}, avg_loss={avgLoss}, lr={lr}, live_tensors={liveTensors}"
      (← IO.getStdout).flush
      totalLoss := 0.0

  IO.println "Training complete!"
  return params

/-- Training loop with validation support
    Returns trained parameters and best validation loss -/
def trainLoopWithVal {modelCfg : Config}
    (trainCfg : TrainConfig)
    (initParams : GPTParams modelCfg)
    (initOptState : Optim.AdamWState (GPTParams modelCfg))
    (trainData : T #[nTrain])
    (valData : T #[nVal])
    (evalBatches : Nat := 10)
    : IO (GPTParams modelCfg × Float) := do
  let mut params := initParams
  let mut optState := initOptState
  let mut totalLoss : Float := 0.0
  let mut bestValLoss : Float := 1e10  -- Use large value instead of inf

  IO.println s!"Starting training for {trainCfg.maxIters} iterations..."
  IO.println s!"  batch_size={trainCfg.batchSize}, block_size={trainCfg.blockSize}"
  IO.println s!"  lr={trainCfg.learningRate}, warmup={trainCfg.warmupIters}"
  IO.println s!"  grad_clip={trainCfg.gradClip}, eval_interval={trainCfg.evalInterval}"
  (← IO.getStdout).flush

  for iterNum in [:trainCfg.maxIters] do
    -- Get learning rate with schedule
    let lr := getLr trainCfg iterNum

    -- Get batch
    let (x, y) ← getBatch trainData trainCfg.batchSize trainCfg.blockSize

    -- Training step
    let (params', optState', lossVal) ← trainStep trainCfg params optState x y lr

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

    -- Validation at evalInterval
    if iterNum % trainCfg.evalInterval == 0 && iterNum > 0 then
      let valLoss ← evalLoss params valData trainCfg.batchSize trainCfg.blockSize evalBatches
      let valPpl := perplexity valLoss
      IO.println s!"  val_loss={valLoss}, val_ppl={valPpl}"
      if valLoss < bestValLoss then
        bestValLoss := valLoss
        IO.println s!"  [new best val_loss!]"
      (← IO.getStdout).flush

  -- Final validation
  let finalValLoss ← evalLoss params valData trainCfg.batchSize trainCfg.blockSize evalBatches
  let finalValPpl := perplexity finalValLoss
  IO.println s!"Training complete! Final val_loss={finalValLoss}, val_ppl={finalValPpl}"
  IO.println s!"Best val_loss={bestValLoss}"
  return (params, bestValLoss)

end torch.train
