/-
  Training Infrastructure for GPT

  Minimal viable training loop for nanoGPT-style training.
-/
import Tyr.Torch
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

/-- Zero gradients for all block parameters -/
def zeroBlockGrads {n_embd : UInt64} (params : BlockParams n_embd) : BlockParams n_embd :=
  {
    ln1_weight := autograd.zero_grad params.ln1_weight
    ln1_bias := autograd.zero_grad params.ln1_bias
    q_proj := autograd.zero_grad params.q_proj
    k_proj := autograd.zero_grad params.k_proj
    v_proj := autograd.zero_grad params.v_proj
    c_proj := autograd.zero_grad params.c_proj
    c_proj_bias := autograd.zero_grad params.c_proj_bias
    ln2_weight := autograd.zero_grad params.ln2_weight
    ln2_bias := autograd.zero_grad params.ln2_bias
    mlp_fc := autograd.zero_grad params.mlp_fc
    mlp_fc_bias := autograd.zero_grad params.mlp_fc_bias
    mlp_proj := autograd.zero_grad params.mlp_proj
    mlp_proj_bias := autograd.zero_grad params.mlp_proj_bias
  }

/-- Zero gradients for all GPT parameters -/
def zeroGPTGrads {cfg : Config} (params : GPTParams cfg) : GPTParams cfg :=
  {
    wte := autograd.zero_grad params.wte
    wpe := autograd.zero_grad params.wpe
    blocks := params.blocks.map zeroBlockGrads
    ln_f_weight := autograd.zero_grad params.ln_f_weight
    ln_f_bias := autograd.zero_grad params.ln_f_bias
  }

/-- Update a single parameter with AdamW under `torch.no_grad`. -/
def updateParam {s : Shape} (cfg : optim.AdamWConfig)
    (param : T s) (state : optim.AdamWState s) : IO (T s × optim.AdamWState s) := do
  let grad := autograd.grad_of param
  optim.adamw cfg param grad state

/-- Update block parameters -/
def updateBlockParams {n_embd : UInt64} (cfg : optim.AdamWConfig)
    (params : BlockParams n_embd)
    (optState : BlockOptState n_embd)
    : IO (BlockParams n_embd × BlockOptState n_embd) := do
  let (ln1_weight', ln1_weight_opt') ← updateParam cfg params.ln1_weight optState.ln1_weight
  let (ln1_bias', ln1_bias_opt') ← updateParam cfg params.ln1_bias optState.ln1_bias
  let (q_proj', q_proj_opt') ← updateParam cfg params.q_proj optState.q_proj
  let (k_proj', k_proj_opt') ← updateParam cfg params.k_proj optState.k_proj
  let (v_proj', v_proj_opt') ← updateParam cfg params.v_proj optState.v_proj
  let (c_proj', c_proj_opt') ← updateParam cfg params.c_proj optState.c_proj
  let (c_proj_bias', c_proj_bias_opt') ← updateParam cfg params.c_proj_bias optState.c_proj_bias
  let (ln2_weight', ln2_weight_opt') ← updateParam cfg params.ln2_weight optState.ln2_weight
  let (ln2_bias', ln2_bias_opt') ← updateParam cfg params.ln2_bias optState.ln2_bias
  let (mlp_fc', mlp_fc_opt') ← updateParam cfg params.mlp_fc optState.mlp_fc
  let (mlp_fc_bias', mlp_fc_bias_opt') ← updateParam cfg params.mlp_fc_bias optState.mlp_fc_bias
  let (mlp_proj', mlp_proj_opt') ← updateParam cfg params.mlp_proj optState.mlp_proj
  let (mlp_proj_bias', mlp_proj_bias_opt') ← updateParam cfg params.mlp_proj_bias optState.mlp_proj_bias

  let params' : BlockParams n_embd := {
    ln1_weight := ln1_weight'
    ln1_bias := ln1_bias'
    q_proj := q_proj'
    k_proj := k_proj'
    v_proj := v_proj'
    c_proj := c_proj'
    c_proj_bias := c_proj_bias'
    ln2_weight := ln2_weight'
    ln2_bias := ln2_bias'
    mlp_fc := mlp_fc'
    mlp_fc_bias := mlp_fc_bias'
    mlp_proj := mlp_proj'
    mlp_proj_bias := mlp_proj_bias'
  }

  let optState' : BlockOptState n_embd := {
    ln1_weight := ln1_weight_opt'
    ln1_bias := ln1_bias_opt'
    q_proj := q_proj_opt'
    k_proj := k_proj_opt'
    v_proj := v_proj_opt'
    c_proj := c_proj_opt'
    c_proj_bias := c_proj_bias_opt'
    ln2_weight := ln2_weight_opt'
    ln2_bias := ln2_bias_opt'
    mlp_fc := mlp_fc_opt'
    mlp_fc_bias := mlp_fc_bias_opt'
    mlp_proj := mlp_proj_opt'
    mlp_proj_bias := mlp_proj_bias_opt'
  }

  return (params', optState')

/-- Update all GPT parameters with AdamW -/
def updateGPTParams {cfg : Config} (adamCfg : optim.AdamWConfig)
    (params : GPTParams cfg)
    (optState : GPTOptState cfg)
    : IO (GPTParams cfg × GPTOptState cfg) := do
  let (wte', wte_opt') ← updateParam adamCfg params.wte optState.wte
  let (wpe', wpe_opt') ← updateParam adamCfg params.wpe optState.wpe

  let mut blocks' : Array (BlockParams cfg.n_embd) := #[]
  let mut blockOpts' : Array (BlockOptState cfg.n_embd) := #[]
  for (bp, bo) in params.blocks.zip optState.blocks do
    let (block', blockOpt') ← updateBlockParams adamCfg bp bo
    blocks' := blocks'.push block'
    blockOpts' := blockOpts'.push blockOpt'

  let (ln_f_weight', ln_f_weight_opt') ← updateParam adamCfg params.ln_f_weight optState.ln_f_weight
  let (ln_f_bias', ln_f_bias_opt') ← updateParam adamCfg params.ln_f_bias optState.ln_f_bias

  let params' : GPTParams cfg := {
    wte := wte'
    wpe := wpe'
    blocks := blocks'
    ln_f_weight := ln_f_weight'
    ln_f_bias := ln_f_bias'
  }

  let optState' : GPTOptState cfg := {
    wte := wte_opt'
    wpe := wpe_opt'
    blocks := blockOpts'
    ln_f_weight := ln_f_weight_opt'
    ln_f_bias := ln_f_bias_opt'
  }

  return (params', optState')

/-- Single training step -/
def trainStep {modelCfg : Config} {batch seq : UInt64}
    (trainCfg : TrainConfig)
    (params : GPTParams modelCfg)
    (optState : GPTOptState modelCfg)
    (x : T #[batch, seq])
    (y : T #[batch, seq])
    (lr : Float)
    : IO (GPTParams modelCfg × GPTOptState modelCfg × Float) := do
  -- Zero gradients BEFORE forward/backward to prevent gradient accumulation
  let params := zeroGPTGrads params

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

  -- Update parameters with AdamW
  let adamCfg : optim.AdamWConfig := { optim.AdamWConfig.default with lr := lr }
  let (params', optState') ← updateGPTParams adamCfg params optState

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
    (initOptState : GPTOptState modelCfg)
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
    (initOptState : GPTOptState modelCfg)
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
