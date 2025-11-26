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

/-- Get a batch of data from token array
    Returns (x, y) where y = x shifted by 1 position -/
def getBatch (data : T #[n]) (batchSize blockSize : UInt64) : IO (T #[batchSize, blockSize] × T #[batchSize, blockSize]) := do
  -- For MVP: just take the first batch_size * block_size tokens sequentially
  -- A proper implementation would gather from random positions
  let startIdx : Int64 := 0
  let numTokens := batchSize * blockSize

  -- Slice input and target (offset by 1)
  let xFlat := torch.data.slice1d' data startIdx (startIdx + numTokens.toInt64)
  let yFlat := torch.data.slice1d' data (startIdx + 1) (startIdx + 1 + numTokens.toInt64)

  let x := reshape xFlat #[batchSize, blockSize]
  let y := reshape yFlat #[batchSize, blockSize]

  return (x, y)

/-- Update a single parameter with AdamW -/
def updateParam {s : Shape} (cfg : optim.AdamWConfig)
    (param : T s) (state : optim.AdamWState s) : T s × optim.AdamWState s :=
  let grad := autograd.grad_of param
  optim.adamw cfg param grad state

/-- Update block parameters -/
def updateBlockParams {n_embd : UInt64} (cfg : optim.AdamWConfig)
    (params : BlockParams n_embd)
    (optState : BlockOptState n_embd)
    : BlockParams n_embd × BlockOptState n_embd :=
  let (ln1_weight', ln1_weight_opt') := updateParam cfg params.ln1_weight optState.ln1_weight
  let (ln1_bias', ln1_bias_opt') := updateParam cfg params.ln1_bias optState.ln1_bias
  let (q_proj', q_proj_opt') := updateParam cfg params.q_proj optState.q_proj
  let (k_proj', k_proj_opt') := updateParam cfg params.k_proj optState.k_proj
  let (v_proj', v_proj_opt') := updateParam cfg params.v_proj optState.v_proj
  let (c_proj', c_proj_opt') := updateParam cfg params.c_proj optState.c_proj
  let (c_proj_bias', c_proj_bias_opt') := updateParam cfg params.c_proj_bias optState.c_proj_bias
  let (ln2_weight', ln2_weight_opt') := updateParam cfg params.ln2_weight optState.ln2_weight
  let (ln2_bias', ln2_bias_opt') := updateParam cfg params.ln2_bias optState.ln2_bias
  let (mlp_fc', mlp_fc_opt') := updateParam cfg params.mlp_fc optState.mlp_fc
  let (mlp_fc_bias', mlp_fc_bias_opt') := updateParam cfg params.mlp_fc_bias optState.mlp_fc_bias
  let (mlp_proj', mlp_proj_opt') := updateParam cfg params.mlp_proj optState.mlp_proj
  let (mlp_proj_bias', mlp_proj_bias_opt') := updateParam cfg params.mlp_proj_bias optState.mlp_proj_bias

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

  (params', optState')

/-- Update all GPT parameters with AdamW -/
def updateGPTParams {cfg : Config} (adamCfg : optim.AdamWConfig)
    (params : GPTParams cfg)
    (optState : GPTOptState cfg)
    : GPTParams cfg × GPTOptState cfg :=
  -- Update embeddings
  let (wte', wte_opt') := updateParam adamCfg params.wte optState.wte
  let (wpe', wpe_opt') := updateParam adamCfg params.wpe optState.wpe

  -- Update blocks
  let blocksAndOpts := params.blocks.zip optState.blocks
  let updatedBlocks := blocksAndOpts.map fun (bp, bo) => updateBlockParams adamCfg bp bo
  let blocks' := updatedBlocks.map Prod.fst
  let blockOpts' := updatedBlocks.map Prod.snd

  -- Update final layer norm
  let (ln_f_weight', ln_f_weight_opt') := updateParam adamCfg params.ln_f_weight optState.ln_f_weight
  let (ln_f_bias', ln_f_bias_opt') := updateParam adamCfg params.ln_f_bias optState.ln_f_bias

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

  (params', optState')

/-- Single training step -/
def trainStep {modelCfg : Config} {batch seq : UInt64}
    (trainCfg : TrainConfig)
    (params : GPTParams modelCfg)
    (optState : GPTOptState modelCfg)
    (x : T #[batch, seq])
    (y : T #[batch, seq])
    (lr : Float)
    : IO (GPTParams modelCfg × GPTOptState modelCfg × Float) := do
  -- Forward pass: compute loss (training=true enables dropout)
  let lossT ← gpt.loss params x y true

  -- Backward pass
  autograd.backwardLoss lossT

  -- Get loss value for logging
  let lossVal := nn.item lossT

  -- Update parameters with AdamW
  let adamCfg : optim.AdamWConfig := { optim.AdamWConfig.default with lr := lr }
  let (params', optState') := updateGPTParams adamCfg params optState

  return (params', optState', lossVal)

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
      IO.println s!"iter {iterNum}: loss={lossVal}, avg_loss={avgLoss}, lr={lr}"
      totalLoss := 0.0

  IO.println "Training complete!"
  return params

end torch.train
