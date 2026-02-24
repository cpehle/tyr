/-
  Tyr/ModdedTrain.lean

  Training infrastructure for modded-nanogpt style training.

  Key features:
  - Dynamic batch size and window size schedules
  - LR schedule with cosine cooldown
  - Muon momentum warmup/cooldown
  - Alternating optimizer steps (Muon only on even steps)
  - Validation with HellaSwag
  - Distributed training coordination

  Based on modded-nanogpt's training loop.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Distributed
import Examples.NanoChat.ModdedGPT
import Tyr.DataLoader
import Tyr.Checkpoint
import Examples.GPT.GPTDataLoader
import Tyr.Optim
import Tyr.Optim.NorMuon
import Tyr.Optim.DistAdam

/-!
# `Examples.NanoChat.ModdedTrain`

Training orchestration for ModdedGPT with dynamic schedules and distributed optimizer coordination.

## Overview
- Example entrypoint intended for runnable end-to-end workflows.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace torch.ModdedTrain

open torch
open torch.moddedGpt
open torch.DataLoader
open torch.Optim

private def moveToDevice [TensorStruct α] (x : α) (device : Device) : IO α := do
  let moved ← TensorStruct.mapM (fun t => pure (t.to device)) x
  pure (TensorStruct.makeLeafParams moved)

private def moveYarnToDevice {headDim maxSeqLen : UInt64}
    (yarn : YarnRotary headDim maxSeqLen) (device : Device) : YarnRotary headDim maxSeqLen :=
  { yarn with
    cos := yarn.cos.to device
    sin := yarn.sin.to device
    angularFreq := yarn.angularFreq.to device
  }

/-! ## Hyperparameters -/

/-- Training hyperparameters for nanochat-style loops. -/
structure Hyperparameters where
  /-- Total scheduled training iterations -/
  numIterations : UInt64 := 2050
  /-- Extension iterations beyond `numIterations` (usually 0). -/
  extensionIterations : UInt64 := 0
  /-- Per-rank micro-batch size. -/
  deviceBatchSize : UInt64 := 32
  /-- Target global batch size in tokens across all ranks/accumulation. -/
  totalBatchSizeTokens : UInt64 := 524288
  /-- Embedding learning rate (Adam). -/
  embeddingLr : Float := 0.3
  /-- Unembedding / lm-head learning rate (Adam). -/
  unembeddingLr : Float := 0.004
  /-- Matrix learning rate (Muon). -/
  matrixLr : Float := 0.02
  /-- Adam beta1 (nanochat default: 0.8). -/
  adamBeta1 : Float := 0.8
  /-- Adam beta2 (nanochat default: 0.95). -/
  adamBeta2 : Float := 0.95
  /-- Adam weight decay (nanochat default: 0.0). -/
  adamWeightDecay : Float := 0.0
  /-- Warmup fraction of total steps. -/
  warmupFrac : Float := 0.0
  /-- Final warmdown fraction of total steps. -/
  cooldownFrac : Float := 0.4
  /-- Final learning-rate multiplier at end of warmdown. -/
  finalLrFrac : Float := 0.0
  /-- Block size for windowed attention -/
  blockSize : UInt64 := 128
  /-- Maximum sequence length -/
  maxSeqLen : UInt64 := 2048
  /-- Validation interval (in iterations) -/
  valInterval : UInt64 := 50
  /-- Logging interval -/
  logInterval : UInt64 := 10
  /-- Checkpoint interval -/
  checkpointInterval : UInt64 := 100
  deriving Repr, Inhabited

/-- Default hyperparameters -/
def Hyperparameters.default : Hyperparameters := {}

/-! ## Batch Size Schedule -/

/-- Get batch size for a given iteration.

    modded-nanogpt schedule:
    - Iterations 0-199: batch_size = 8
    - Iterations 200-999: batch_size = 16
    - Iterations 1000+: batch_size = 24

    These correspond to tokens/iter:
    - 8 * 2048 * 8 = 131072
    - 16 * 2048 * 8 = 262144
    - 24 * 2048 * 8 = 393216
-/
def getBatchSize (step : UInt64) : UInt64 :=
  if step < 200 then 8
  else if step < 1000 then 16
  else 24

/-- Get tokens per batch for a given iteration -/
def getTokensPerBatch (step : UInt64) (seqLen gradAccum worldSize : UInt64) : UInt64 :=
  getBatchSize step * seqLen * gradAccum * worldSize

/-! ## Window Size Schedule -/

/-- Get window sizes (short, long) for a given iteration.

    modded-nanogpt schedule:
    - Iterations 0-199: (3, 3) blocks = 384 tokens
    - Iterations 200-999: (3, 7) blocks = 384, 896 tokens
    - Iterations 1000+: (3, 11) blocks = 384, 1408 tokens

    Short window is used for most attention heads,
    long window for a few "global" heads.
-/
def getWindowSizes (step : UInt64) (_blockSize : UInt64 := 128) : (UInt64 × UInt64) :=
  if step < 200 then (3, 3)
  else if step < 1000 then (3, 7)
  else (3, 11)

/-- Get sequence length for a given iteration -/
def getSeqLen (step : UInt64) (blockSize : UInt64 := 128) : UInt64 :=
  let (_, wsLong) := getWindowSizes step blockSize
  wsLong * blockSize

/-! ## Learning Rate Schedule -/

/-- Get learning rate for a given iteration.
    Matches nanochat base_train's warmup/plateau/linear-warmdown semantics. -/
def getLearningRate (step : UInt64) (hp : Hyperparameters)
    (baseLr : Float := 0.023) : Float :=
  let totalSteps := max 1 (hp.numIterations + hp.extensionIterations)
  let warmupSteps := (hp.warmupFrac * totalSteps.toFloat).toUInt64
  let warmdownSteps := (hp.cooldownFrac * totalSteps.toFloat).toUInt64
  let warmdownStart := totalSteps - warmdownSteps
  if warmupSteps > 0 && step < warmupSteps then
    baseLr * (step.toFloat + 1.0) / warmupSteps.toFloat
  else if warmdownSteps == 0 || step <= warmdownStart then
    baseLr
  else
    let progress := (totalSteps - step).toFloat / warmdownSteps.toFloat
    baseLr * (progress + (1.0 - progress) * hp.finalLrFrac)

/-- Batch-size LR scaling used by nanochat: sqrt(batch/reference_batch). -/
def batchLrScale (hp : Hyperparameters) (referenceBatch : Float := 524288.0) : Float :=
  if hp.totalBatchSizeTokens == 0 then
    1.0
  else
    Float.sqrt (hp.totalBatchSizeTokens.toFloat / referenceBatch)

/-- Scale Adam learning rates by model dimension: (d_model / 768)^(-0.5). -/
def dModelLrScale (cfg : moddedGpt.Config) : Float :=
  Float.pow (cfg.modelDim.toFloat / 768.0) (-0.5)

/-- Compute grad-accum steps from global token budget and world size.
    Expects divisibility to be validated up-front (nanochat parity). -/
def effectiveGradAccumSteps (hp : Hyperparameters) (worldSize : UInt64) : UInt64 :=
  let world := max 1 worldSize
  let perMicro := hp.deviceBatchSize * hp.maxSeqLen * world
  if perMicro == 0 then
    1
  else
    let target := max perMicro hp.totalBatchSizeTokens
    max 1 (target / perMicro)

/-- Enforce nanochat batch semantics:
    total_batch_size must be divisible by per-rank micro-batch tokens. -/
def validateGradAccumConfig (hp : Hyperparameters) (worldSize : UInt64) : IO Unit := do
  let world := max 1 worldSize
  let perMicro := hp.deviceBatchSize * hp.maxSeqLen * world
  if perMicro == 0 then
    throw <| IO.userError "Invalid grad-accum config: perMicro tokens is zero"
  if hp.totalBatchSizeTokens % perMicro != 0 then
    throw <| IO.userError s!"Invalid grad-accum config: total_batch_size={hp.totalBatchSizeTokens} must be divisible by device_batch_size*max_seq_len*world_size={perMicro}"

/-! ## Momentum Schedule -/

/-- Get Muon momentum for a given iteration (nanochat parity).

    Momentum schedule:
    - Warmup: 0.85 -> baseMomentum over 300 steps
    - Then constant at baseMomentum
-/
def getMuonMomentum (step : UInt64) (hp : Hyperparameters)
    (baseMomentum : Float := 0.95) (warmupSteps : UInt64 := 300)
    (cooldownSteps : UInt64 := 50) : Float :=
  let _ := hp
  let _ := cooldownSteps
  NorMuon.getMomentum step.toNat
    0
    baseMomentum warmupSteps.toNat 0

/-! ## Optimizer State -/

/-- Muon state for an attention module. -/
structure MuonAttnState (cfg : moddedGpt.Config) where
  wQ : NorMuon.ParamState #[cfg.nHead * cfg.headDim, cfg.modelDim]
  wK : NorMuon.ParamState #[cfg.nHead * cfg.headDim, cfg.modelDim]
  wV : NorMuon.ParamState #[cfg.nHead * cfg.headDim, cfg.modelDim]
  wO : NorMuon.ParamState #[cfg.modelDim, cfg.nHead * cfg.headDim]
  deriving Repr, TensorStruct

/-- Muon state for a transformer block. -/
structure MuonBlockState (cfg : moddedGpt.Config) where
  attn : Option (MuonAttnState cfg)
  cFc : NorMuon.ParamState #[4 * cfg.modelDim, cfg.modelDim]
  cProj : NorMuon.ParamState #[4 * cfg.modelDim, cfg.modelDim]
  deriving Repr, TensorStruct

/-- Full dual-optimizer parameter state (DistAdam + Muon). -/
structure DualParamState (cfg : moddedGpt.Config) where
  embed : DistAdam.ParamState #[cfg.vocabSize, cfg.modelDim]
  valueEmbeds : Array (DistAdam.ParamState #[cfg.vocabSize, cfg.modelDim])
  lmHead : DistAdam.ParamState #[cfg.vocabSize, cfg.modelDim]
  scalars : DistAdam.ParamState #[cfg.nLayer * 4 + 8]
  smearGate : NorMuon.ParamState #[1, 12]
  blocks : Array (MuonBlockState cfg)
  deriving Repr, TensorStruct

private def initDualParamState (cfg : moddedGpt.Config) (params : ModdedGPTParams cfg) : DualParamState cfg :=
  let attnState? (attn : Option (CausalSelfAttention cfg.modelDim cfg.headDim cfg.nHead)) :
      Option (MuonAttnState cfg) :=
    match attn with
    | none => none
    | some a =>
      some {
        wQ := NorMuon.initParamState a.wQ
        wK := NorMuon.initParamState a.wK
        wV := NorMuon.initParamState a.wV
        wO := NorMuon.initParamState a.wO
      }
  let blockStates : Array (MuonBlockState cfg) := params.blocks.map fun b =>
    {
      attn := attnState? b.attn
      cFc := NorMuon.initParamState b.mlp.cFc
      cProj := NorMuon.initParamState b.mlp.cProj
    }
  {
    embed := DistAdam.initParamState params.embed
    valueEmbeds := params.valueEmbeds.map DistAdam.initParamState
    lmHead := DistAdam.initParamState params.lmHead.weight
    scalars := DistAdam.initParamState params.scalars.values
    smearGate := NorMuon.initParamState params.smearGate.weight
    blocks := blockStates
  }

/-- Combined optimizer state for training.
    Keeps legacy Adam state for compatibility, and dual parameter states for updates. -/
structure OptimizerState (cfg : moddedGpt.Config) where
  /-- Legacy AdamW optimizer state (kept for checkpoint compatibility). -/
  adamState : Optim.AdamWState (ModdedGPTParams cfg)
  /-- DistAdam + Muon parameter states used by the training step. -/
  dualState : DualParamState cfg
  /-- Current step -/
  step : UInt64
  /-- Base learning rate -/
  baseLr : Float := 0.023
  /-- Weight decay -/
  weightDecay : Float := 0.01

/-- Initialize optimizer state from model parameters -/
def OptimizerState.init (cfg : moddedGpt.Config) (params : ModdedGPTParams cfg)
    (lr : Float := 0.023) (weightDecay : Float := 0.0) : OptimizerState cfg :=
  let opt := Optim.adamw (lr := lr) (weight_decay := weightDecay)
  {
    adamState := opt.init params
    dualState := initDualParamState cfg params
    step := 0
    baseLr := lr
    weightDecay := weightDecay
  }

/-- Get current learning rate from schedule -/
def OptimizerState.currentLr (state : OptimizerState cfg) (hp : Hyperparameters) : Float :=
  getLearningRate state.step hp state.baseLr

/-! ## Training Step -/

/-- Result of a single training step -/
structure StepResult where
  /-- Training loss -/
  loss : Float
  /-- Gradient norm (for monitoring) -/
  gradNorm : Float
  /-- Tokens processed -/
  tokensProcessed : UInt64
  /-- Time taken (ms) -/
  timeMs : Float
  deriving Repr

/-- Perform a single training step.

    Steps:
    1. Zero gradients
    2. Forward pass
    3. Backward pass
    4. Extract gradients
    5. Apply optimizer (AdamW)
    6. Return updated params and optimizer state
-/
def trainStep {cfg : moddedGpt.Config} {batch seq : UInt64}
    (params : ModdedGPTParams cfg)
    (yarn : YarnRotary cfg.headDim cfg.maxSeqLen)
    (input : T #[batch, seq])
    (target : T #[batch, seq])
    (optState : OptimizerState cfg)
    (hp : Hyperparameters)
    (gradClip : Float := 0.0)
    : IO (ModdedGPTParams cfg × OptimizerState cfg × StepResult) := do
  let startTime ← IO.monoMsNow

  -- Ensure parameters are trainable leaves before backprop.
  let params := TensorStruct.zeroGrads (TensorStruct.makeLeafParams params)

  -- Forward pass (window pattern is in cfg)
  let lossT ← moddedGpt.loss params yarn input target true

  -- Backward pass
  autograd.backwardLoss lossT

  -- Get loss value for logging
  let lossVal := nn.item lossT

  -- Gradient clipping (per-tensor norm clipping)
  if gradClip > 0 then
    let _ ← nn.clip_grad_norm_ params.embed gradClip
    let _ ← nn.clip_grad_norm_ params.smearGate.weight gradClip
    for ve in params.valueEmbeds do
      let _ ← nn.clip_grad_norm_ ve gradClip
    for block in params.blocks do
      -- Attention layer (may be None for some layers)
      match block.attn with
      | some attn =>
        let _ ← nn.clip_grad_norm_ attn.wQ gradClip
        let _ ← nn.clip_grad_norm_ attn.wK gradClip
        let _ ← nn.clip_grad_norm_ attn.wV gradClip
        let _ ← nn.clip_grad_norm_ attn.wO gradClip
      | none => pure ()
      -- MLP
      let _ ← nn.clip_grad_norm_ block.mlp.cFc gradClip
      let _ ← nn.clip_grad_norm_ block.mlp.cProj gradClip
    let _ ← nn.clip_grad_norm_ params.lmHead.weight gradClip
    let _ ← nn.clip_grad_norm_ params.scalars.values gradClip

  -- Extract gradients from parameters
  let grads := TensorStruct.grads params

  -- Get current learning rate from schedule
  let lr := getLearningRate optState.step hp optState.baseLr

  -- Create optimizer with current LR
  let opt := Optim.adamw (lr := lr) (weight_decay := optState.weightDecay)

  -- Apply optimizer step
  let (newParams, newAdamState) := Optim.step opt params grads optState.adamState

  let endTime ← IO.monoMsNow
  let timeMs := (endTime - startTime).toFloat

  let result : StepResult := {
    loss := lossVal
    gradNorm := 0.0  -- Could compute from grads if needed
    tokensProcessed := batch * seq
    timeMs := timeMs
  }

  let newOptState : OptimizerState cfg := {
    adamState := newAdamState
    dualState := optState.dualState
    step := optState.step + 1
    baseLr := optState.baseLr
    weightDecay := optState.weightDecay
  }

  return (newParams, newOptState, result)

/-! ## Gradient Accumulation -/

/-- Accumulate gradients over multiple micro-batches -/
def accumulateGradients {cfg : moddedGpt.Config} {batch seq : UInt64}
    (params : ModdedGPTParams cfg)
    (yarn : YarnRotary cfg.headDim cfg.maxSeqLen)
    (batches : Array (T #[batch, seq] × T #[batch, seq]))
    : IO Float := do
  let mut totalLoss := 0.0
  let numBatches := batches.size

  for (input, target) in batches do
    let lossT ← moddedGpt.loss params yarn input target true
    -- Scale loss for accumulation
    let scaledLoss := div_scalar lossT numBatches.toFloat
    autograd.backwardLoss scaledLoss
    totalLoss := totalLoss + nn.item lossT

  return totalLoss / numBatches.toFloat

/-! ## Validation -/

/-- Validation result -/
structure ValidationResult where
  /-- Validation loss -/
  loss : Float
  /-- HellaSwag accuracy (if evaluated) -/
  hellaswagAcc : Option Float
  /-- Time taken (ms) -/
  timeMs : Float
  deriving Repr

/-- Run validation on held-out data -/
def validate {cfg : moddedGpt.Config}
    (params : ModdedGPTParams cfg)
    (yarn : YarnRotary cfg.headDim cfg.maxSeqLen)
    (valData : DataShard)
    (batchSize seqLen : UInt64)
    (device : Device := Device.CPU)
    (numBatches : UInt64 := 10)
    : IO ValidationResult := do
  let startTime ← IO.monoMsNow
  let mut totalLoss := 0.0
  let mut numValid := 0

  -- Create iterator
  let mut iter := BatchIterator.new valData batchSize seqLen

  for _ in [:numBatches.toNat] do
    let (maybeBatch, newIter) ← iter.nextGPT
    iter := newIter
    match maybeBatch with
    | some (inputDyn, targetDyn) =>
      -- Reshape dynamic tensors to expected shape
      let input := (reshape inputDyn #[batchSize, seqLen]).to device
      let target := (reshape targetDyn #[batchSize, seqLen]).to device
      let lossT ← moddedGpt.loss params yarn input target false
      totalLoss := totalLoss + nn.item lossT
      numValid := numValid + 1
    | none => break

  let avgLoss := if numValid > 0 then totalLoss / numValid.toFloat else 0.0

  let endTime ← IO.monoMsNow
  let timeMs := (endTime - startTime).toFloat

  return {
    loss := avgLoss
    hellaswagAcc := none  -- Would run HellaSwag eval
    timeMs := timeMs
  }

/-! ## Checkpointing -/

/-- Checkpoint state -/
structure Checkpoint (cfg : moddedGpt.Config) where
  /-- Model parameters -/
  params : ModdedGPTParams cfg
  /-- Optimizer state -/
  optState : OptimizerState cfg
  /-- Current step -/
  step : UInt64
  /-- Best validation loss -/
  bestValLoss : Float

/-- Check whether a serialized checkpoint exists at `path`. -/
def checkpointExists (path : String) : IO Bool := do
  data.fileExists s!"{path}/step.pt"

private def saveScalarUInt64 (value : UInt64) (path : String) : IO Unit := do
  let t := data.fromInt64Array #[value.toInt64]
  data.saveTensor t path

private def loadScalarUInt64 (path : String) : IO UInt64 := do
  let t ← data.loadTensor #[1] path
  let vals ← data.tensorToUInt64Array t
  match vals[0]? with
  | some v => pure v
  | none => throw <| IO.userError s!"Missing scalar value in {path}"

private def saveScalarFloat (value : Float) (path : String) : IO Unit := do
  let t := full #[] value
  data.saveTensor t path

private def loadScalarFloat (path : String) : IO Float := do
  let t ← data.loadTensor #[] path
  pure (nn.item t)

/-- Save a checkpoint to disk -/
def saveCheckpoint {cfg : moddedGpt.Config} (ckpt : Checkpoint cfg) (path : String)
    : IO Unit := do
  IO.FS.createDirAll ⟨path⟩

  checkpoint.saveParams ckpt.params path "param"
  checkpoint.saveParams ckpt.optState.adamState path "optim_adam"
  checkpoint.saveParams ckpt.optState.dualState path "optim_dual"

  saveScalarUInt64 ckpt.step s!"{path}/step.pt"
  saveScalarUInt64 ckpt.optState.step s!"{path}/opt_step.pt"
  saveScalarUInt64 ckpt.optState.adamState.fst.count.toUInt64 s!"{path}/adam_count.pt"
  saveScalarFloat ckpt.bestValLoss s!"{path}/best_val_loss.pt"
  saveScalarFloat ckpt.optState.baseLr s!"{path}/base_lr.pt"
  saveScalarFloat ckpt.optState.weightDecay s!"{path}/weight_decay.pt"

  IO.println s!"Saving checkpoint to {path} at step {ckpt.step}"

  let numParams := TensorStruct.fold (fun {s} _t acc => acc + s.foldl (· * ·) 1) 0 ckpt.params
  let numTensors := TensorStruct.fold (fun {_s} _t acc => acc + 1) 0 ckpt.params

  IO.println s!"  Parameters: {numTensors} tensors, {numParams} elements"
  IO.println s!"  Best validation loss: {ckpt.bestValLoss}"

/-- Load a checkpoint from disk -/
def loadCheckpoint (cfg : moddedGpt.Config) (path : String)
    : IO (Option (Checkpoint cfg)) := do
  if !(← checkpointExists path) then
    return none

  try
    -- Templates supply the static structure and tensor shapes for deserialization.
    let templateParams ← ModdedGPTParams.init cfg
    let templateOpt := OptimizerState.init cfg templateParams

    let params ← checkpoint.loadParams templateParams path "param"
    let loadedAdamState ← checkpoint.loadParams templateOpt.adamState path "optim_adam"
    let loadedDualState ←
      try
        checkpoint.loadParams templateOpt.dualState path "optim_dual"
      catch _ =>
        -- Backward compatibility with checkpoints saved before dual-state serialization.
        pure (initDualParamState cfg params)

    let step ← loadScalarUInt64 s!"{path}/step.pt"
    let optStep ← loadScalarUInt64 s!"{path}/opt_step.pt"
    let adamCount := (← loadScalarUInt64 s!"{path}/adam_count.pt").toNat
    let bestValLoss ← loadScalarFloat s!"{path}/best_val_loss.pt"
    let baseLr ← loadScalarFloat s!"{path}/base_lr.pt"
    let weightDecay ← loadScalarFloat s!"{path}/weight_decay.pt"

    let adamState := {
      loadedAdamState with
      fst := { loadedAdamState.fst with count := adamCount }
    }
    let optState : OptimizerState cfg := {
      adamState := adamState
      dualState := loadedDualState
      step := optStep
      baseLr := baseLr
      weightDecay := weightDecay
    }

    return some {
      params := params
      optState := optState
      step := step
      bestValLoss := bestValLoss
    }
  catch e =>
    IO.eprintln s!"Failed to load checkpoint from {path}: {e}"
    return none

/-! ## Training Loop -/

/-- Training state for the main loop -/
structure TrainState (cfg : moddedGpt.Config) where
  /-- Model parameters -/
  params : ModdedGPTParams cfg
  /-- YaRN rotary embeddings -/
  yarn : YarnRotary cfg.headDim cfg.maxSeqLen
  /-- Optimizer state -/
  optState : OptimizerState cfg
  /-- Data generator -/
  dataGen : DistributedDataGenerator
  /-- Validation data -/
  valData : Option DataShard
  /-- Current step -/
  step : UInt64
  /-- Best validation loss -/
  bestValLoss : Float
  /-- Accumulated tokens -/
  totalTokens : UInt64
  /-- Training start time -/
  startTime : UInt64

/-- Initialize training state -/
def TrainState.init (cfg : moddedGpt.Config) (dataConfig : DataLoader.Config)
    (hp : Hyperparameters)
    (_distributed : Bool) (_worldSize : UInt64)
    (device : Device := Device.CPU)
    (lr : Float := 0.023) (weightDecay : Float := 0.0)
    : IO (TrainState cfg) := do
  let params ← ModdedGPTParams.init cfg
  let yarn ← YarnRotary.init cfg.headDim cfg.maxSeqLen cfg.ropeBase

  let params ← moveToDevice params device
  let yarn := moveYarnToDevice yarn device
  -- Initialize optimizer state from moved parameters so moment buffers match device
  let optState := OptimizerState.init cfg params lr weightDecay

  let initialBatchSize := hp.deviceBatchSize
  let initialSeqLen := hp.maxSeqLen
  let dataGen ← DistributedDataGenerator.init dataConfig initialBatchSize initialSeqLen
  let valData ←
    match dataConfig.valPath with
    | none => pure none
    | some valPath =>
      try
        let shard ← loadValidationData valPath cfg.maxSeqLen dataConfig.bosToken
        pure (some shard)
      catch e =>
        IO.eprintln s!"Warning: failed to load validation data from {valPath}: {e}"
        pure none

  let startTime ← IO.monoMsNow

  return {
    params := params
    yarn := yarn
    optState := optState
    dataGen := dataGen
    valData := valData
    step := 0
    bestValLoss := 1e30  -- Very large number instead of inf
    totalTokens := 0
    startTime := startTime.toUInt64
  }

/-- Log training progress -/
def logProgress (state : TrainState cfg) (result : StepResult)
    (hp : Hyperparameters) : IO Unit := do
  if state.step % hp.logInterval == 0 then
    let nowMs ← IO.monoMsNow
    let elapsed := nowMs - state.startTime.toNat
    let elapsedSec := elapsed.toFloat / 1000.0
    let tokensPerSec := state.totalTokens.toFloat / elapsedSec
    let batchSize := hp.deviceBatchSize
    let seqLen := hp.maxSeqLen
    let lr := getLearningRate state.step hp

    IO.println s!"Step {state.step}: loss={result.loss} lr={lr} batch={batchSize} seq={seqLen} tok/s={tokensPerSec} time={result.timeMs}ms"

/-- Run validation and log results -/
def runValidation (state : TrainState cfg) (hp : Hyperparameters)
    : IO (TrainState cfg) := do
  match state.valData with
  | none => return state
  | some valData =>
    let batchSize := hp.deviceBatchSize
    let seqLen := hp.maxSeqLen
    let device := state.params.embed.device

    let valResult ← validate state.params state.yarn valData batchSize seqLen device

    IO.println s!"Validation: loss={valResult.loss} time={valResult.timeMs}ms"

    let newBest := valResult.loss < state.bestValLoss
    if newBest then
      IO.println s!"  New best validation loss!"

    return { state with
      bestValLoss := if newBest then valResult.loss else state.bestValLoss
    }

/-- Distributed training step with gradient accumulation over micro-batches. -/
def trainStepDistributedAccum {cfg : moddedGpt.Config} {batch seq : UInt64}
    (params : ModdedGPTParams cfg)
    (yarn : YarnRotary cfg.headDim cfg.maxSeqLen)
    (microBatches : Array (T #[batch, seq] × T #[batch, seq]))
    (optState : OptimizerState cfg)
    (hp : Hyperparameters)
    (gradClip : Float := 0.0)
    : IO (ModdedGPTParams cfg × OptimizerState cfg × StepResult) := do
  if microBatches.isEmpty then
    return (params, optState, { loss := 0.0, gradNorm := 0.0, tokensProcessed := 0, timeMs := 0.0 })
  let startTime ← IO.monoMsNow
  let isDistributed ← dist.isInitialized
  let worldSize ← if isDistributed then dist.getWorldSize else pure 1
  let params := TensorStruct.zeroGrads (TensorStruct.makeLeafParams params)
  let microCount := microBatches.size.toUInt64
  let mut totalLoss := 0.0
  for (input, target) in microBatches do
    let lossT ← moddedGpt.loss params yarn input target true
    let scaledLoss := div_scalar lossT microCount.toFloat
    autograd.backwardLoss scaledLoss
    totalLoss := totalLoss + nn.item lossT
  if gradClip > 0 then
    let _ ← nn.clip_grad_norm_ params.embed gradClip
    let _ ← nn.clip_grad_norm_ params.smearGate.weight gradClip
    for ve in params.valueEmbeds do
      let _ ← nn.clip_grad_norm_ ve gradClip
    for block in params.blocks do
      match block.attn with
      | some attn =>
        let _ ← nn.clip_grad_norm_ attn.wQ gradClip
        let _ ← nn.clip_grad_norm_ attn.wK gradClip
        let _ ← nn.clip_grad_norm_ attn.wV gradClip
        let _ ← nn.clip_grad_norm_ attn.wO gradClip
      | none => pure ()
      let _ ← nn.clip_grad_norm_ block.mlp.cFc gradClip
      let _ ← nn.clip_grad_norm_ block.mlp.cProj gradClip
    let _ ← nn.clip_grad_norm_ params.lmHead.weight gradClip
    let _ ← nn.clip_grad_norm_ params.scalars.values gradClip
  let grads := TensorStruct.grads params
  let batchScale := batchLrScale hp
  let dScale := dModelLrScale cfg
  let lrEmbed := getLearningRate optState.step hp (hp.embeddingLr * batchScale * dScale)
  let lrUnembed := getLearningRate optState.step hp (hp.unembeddingLr * batchScale * dScale)
  let lrMatrix := getLearningRate optState.step hp (hp.matrixLr * batchScale)
  let muMomentum := getMuonMomentum optState.step hp

  let adamCfgBase : DistAdam.Config := {
    lr := 1.0
    beta1 := hp.adamBeta1
    beta2 := hp.adamBeta2
    eps := 1e-10
    weightDecay := hp.adamWeightDecay
    distributed := isDistributed
  }
  let muonCfg : NorMuon.Config := {
    lr := lrMatrix
    weightDecay := 0.0
    momentum := muMomentum
    beta2 := 0.95
    numIters := 5
    distributed := isDistributed
    worldSize := worldSize
  }

  let runAdamStep : {s : Shape} → (p : T s) → (g : T s) → DistAdam.ParamState s → Float →
      IO (T s × DistAdam.ParamState s) :=
    fun {_} p g st lr => do
      DistAdam.stepDistributed p g st { adamCfgBase with lr := lr } 1.0 1.0

  let (embed', embedState') ← runAdamStep
    params.embed grads.embed optState.dualState.embed lrEmbed

  let mut newValueEmbeds : Array (T #[cfg.vocabSize, cfg.modelDim]) := #[]
  let mut newValueEmbedStates : Array (DistAdam.ParamState #[cfg.vocabSize, cfg.modelDim]) := #[]
  for i in [:params.valueEmbeds.size] do
    let ve := params.valueEmbeds[i]!
    let veGrad := grads.valueEmbeds[i]!
    let veState := optState.dualState.valueEmbeds[i]?.getD (DistAdam.initParamState ve)
    let (ve', veState') ← runAdamStep ve veGrad veState lrEmbed
    newValueEmbeds := newValueEmbeds.push ve'
    newValueEmbedStates := newValueEmbedStates.push veState'

  let (lmHeadW', lmHeadState') ← runAdamStep
    params.lmHead.weight grads.lmHead.weight optState.dualState.lmHead lrUnembed
  let (scalars', scalarState') ← runAdamStep
    params.scalars.values grads.scalars.values optState.dualState.scalars lrUnembed

  let (smearGateWs, smearGateStates) ← NorMuon.stepDistributedGroup
    #[params.smearGate.weight]
    #[grads.smearGate.weight]
    #[optState.dualState.smearGate]
    muonCfg
  let smearGateW' := smearGateWs[0]?.getD params.smearGate.weight
  let smearGateState' := smearGateStates[0]?.getD optState.dualState.smearGate

  let mut blockStatesIn : Array (MuonBlockState cfg) := #[]
  let mut cFcParams : Array (T #[4 * cfg.modelDim, cfg.modelDim]) := #[]
  let mut cFcGrads : Array (T #[4 * cfg.modelDim, cfg.modelDim]) := #[]
  let mut cFcStates : Array (NorMuon.ParamState #[4 * cfg.modelDim, cfg.modelDim]) := #[]
  let mut cProjParams : Array (T #[4 * cfg.modelDim, cfg.modelDim]) := #[]
  let mut cProjGrads : Array (T #[4 * cfg.modelDim, cfg.modelDim]) := #[]
  let mut cProjStates : Array (NorMuon.ParamState #[4 * cfg.modelDim, cfg.modelDim]) := #[]

  let mut qParams : Array (T #[cfg.nHead * cfg.headDim, cfg.modelDim]) := #[]
  let mut qGrads : Array (T #[cfg.nHead * cfg.headDim, cfg.modelDim]) := #[]
  let mut qStates : Array (NorMuon.ParamState #[cfg.nHead * cfg.headDim, cfg.modelDim]) := #[]
  let mut kParams : Array (T #[cfg.nHead * cfg.headDim, cfg.modelDim]) := #[]
  let mut kGrads : Array (T #[cfg.nHead * cfg.headDim, cfg.modelDim]) := #[]
  let mut kStates : Array (NorMuon.ParamState #[cfg.nHead * cfg.headDim, cfg.modelDim]) := #[]
  let mut vParams : Array (T #[cfg.nHead * cfg.headDim, cfg.modelDim]) := #[]
  let mut vGrads : Array (T #[cfg.nHead * cfg.headDim, cfg.modelDim]) := #[]
  let mut vStates : Array (NorMuon.ParamState #[cfg.nHead * cfg.headDim, cfg.modelDim]) := #[]
  let mut oParams : Array (T #[cfg.modelDim, cfg.nHead * cfg.headDim]) := #[]
  let mut oGrads : Array (T #[cfg.modelDim, cfg.nHead * cfg.headDim]) := #[]
  let mut oStates : Array (NorMuon.ParamState #[cfg.modelDim, cfg.nHead * cfg.headDim]) := #[]

  for i in [:params.blocks.size] do
    let block := params.blocks[i]!
    let blockGrad := grads.blocks[i]!
    let defaultBlockState : MuonBlockState cfg := {
      attn := block.attn.map fun attn =>
        {
          wQ := NorMuon.initParamState attn.wQ
          wK := NorMuon.initParamState attn.wK
          wV := NorMuon.initParamState attn.wV
          wO := NorMuon.initParamState attn.wO
        }
      cFc := NorMuon.initParamState block.mlp.cFc
      cProj := NorMuon.initParamState block.mlp.cProj
    }
    let blockState := optState.dualState.blocks[i]?.getD defaultBlockState
    blockStatesIn := blockStatesIn.push blockState

    cFcParams := cFcParams.push block.mlp.cFc
    cFcGrads := cFcGrads.push blockGrad.mlp.cFc
    cFcStates := cFcStates.push blockState.cFc
    cProjParams := cProjParams.push block.mlp.cProj
    cProjGrads := cProjGrads.push blockGrad.mlp.cProj
    cProjStates := cProjStates.push blockState.cProj

    match block.attn, blockGrad.attn with
    | some attn, some attnGrad =>
      let attnState := blockState.attn.getD {
        wQ := NorMuon.initParamState attn.wQ
        wK := NorMuon.initParamState attn.wK
        wV := NorMuon.initParamState attn.wV
        wO := NorMuon.initParamState attn.wO
      }
      qParams := qParams.push attn.wQ
      qGrads := qGrads.push attnGrad.wQ
      qStates := qStates.push attnState.wQ
      kParams := kParams.push attn.wK
      kGrads := kGrads.push attnGrad.wK
      kStates := kStates.push attnState.wK
      vParams := vParams.push attn.wV
      vGrads := vGrads.push attnGrad.wV
      vStates := vStates.push attnState.wV
      oParams := oParams.push attn.wO
      oGrads := oGrads.push attnGrad.wO
      oStates := oStates.push attnState.wO
    | _, _ => pure ()

  let (qParams', qStates') ← NorMuon.stepDistributedGroup qParams qGrads qStates muonCfg
  let (kParams', kStates') ← NorMuon.stepDistributedGroup kParams kGrads kStates muonCfg
  let (vParams', vStates') ← NorMuon.stepDistributedGroup vParams vGrads vStates muonCfg
  let (oParams', oStates') ← NorMuon.stepDistributedGroup oParams oGrads oStates muonCfg
  let (cFcParams', cFcStates') ← NorMuon.stepDistributedGroup cFcParams cFcGrads cFcStates muonCfg
  let (cProjParams', cProjStates') ← NorMuon.stepDistributedGroup cProjParams cProjGrads cProjStates muonCfg

  let mut newBlocks : Array (Block cfg.modelDim cfg.headDim cfg.nHead) := #[]
  let mut newBlockStates : Array (MuonBlockState cfg) := #[]
  let mut attnCursor : Nat := 0
  for i in [:params.blocks.size] do
    let block := params.blocks[i]!
    let blockGrad := grads.blocks[i]!
    let defaultBlockState : MuonBlockState cfg := {
      attn := block.attn.map fun attn =>
        {
          wQ := NorMuon.initParamState attn.wQ
          wK := NorMuon.initParamState attn.wK
          wV := NorMuon.initParamState attn.wV
          wO := NorMuon.initParamState attn.wO
        }
      cFc := NorMuon.initParamState block.mlp.cFc
      cProj := NorMuon.initParamState block.mlp.cProj
    }
    let blockState := blockStatesIn[i]?.getD defaultBlockState
    let cFc' := cFcParams'[i]?.getD block.mlp.cFc
    let cProj' := cProjParams'[i]?.getD block.mlp.cProj
    let cFcState' := cFcStates'[i]?.getD (NorMuon.initParamState cFc')
    let cProjState' := cProjStates'[i]?.getD (NorMuon.initParamState cProj')
    let (attn', attnState', nextAttnCursor) :=
      match block.attn, blockGrad.attn with
      | some attn, some _ =>
        if attnCursor < qParams'.size then
          let wQ' := qParams'[attnCursor]?.getD attn.wQ
          let wK' := kParams'[attnCursor]?.getD attn.wK
          let wV' := vParams'[attnCursor]?.getD attn.wV
          let wO' := oParams'[attnCursor]?.getD attn.wO
          let wQState' := qStates'[attnCursor]?.getD (NorMuon.initParamState wQ')
          let wKState' := kStates'[attnCursor]?.getD (NorMuon.initParamState wK')
          let wVState' := vStates'[attnCursor]?.getD (NorMuon.initParamState wV')
          let wOState' := oStates'[attnCursor]?.getD (NorMuon.initParamState wO')
          let newAttn : CausalSelfAttention cfg.modelDim cfg.headDim cfg.nHead := {
            wQ := wQ'
            wK := wK'
            wV := wV'
            wO := wO'
          }
          let newAttnState : MuonAttnState cfg := {
            wQ := wQState'
            wK := wKState'
            wV := wVState'
            wO := wOState'
          }
          (some newAttn, some newAttnState, attnCursor + 1)
        else
          (block.attn, blockState.attn, attnCursor)
      | none, _ => (none, none, attnCursor)
      | _, none => (block.attn, blockState.attn, attnCursor)
    attnCursor := nextAttnCursor
    let newBlock : Block cfg.modelDim cfg.headDim cfg.nHead := {
      attn := attn'
      mlp := { cFc := cFc', cProj := cProj' }
    }
    let newBlockState : MuonBlockState cfg := {
      attn := attnState'
      cFc := cFcState'
      cProj := cProjState'
    }
    newBlocks := newBlocks.push newBlock
    newBlockStates := newBlockStates.push newBlockState

  let newParams : ModdedGPTParams cfg := {
    params with
    embed := embed'
    valueEmbeds := newValueEmbeds
    lmHead := { params.lmHead with weight := lmHeadW' }
    scalars := { params.scalars with values := scalars' }
    smearGate := { params.smearGate with weight := smearGateW' }
    blocks := newBlocks
  }
  let legacyAdamState := {
    optState.adamState with
    fst := { optState.adamState.fst with count := optState.adamState.fst.count + 1 }
  }
  let newDualState : DualParamState cfg := {
    embed := embedState'
    valueEmbeds := newValueEmbedStates
    lmHead := lmHeadState'
    scalars := scalarState'
    smearGate := smearGateState'
    blocks := newBlockStates
  }
  let newParams := TensorStruct.makeLeafParams newParams
  let endTime ← IO.monoMsNow
  let timeMs := (endTime - startTime).toFloat
  let result : StepResult := {
    loss := totalLoss / microCount.toFloat
    gradNorm := 0.0
    tokensProcessed := batch * seq * worldSize * microCount
    timeMs := timeMs
  }
  let newOptState : OptimizerState cfg := {
    adamState := legacyAdamState
    dualState := newDualState
    step := optState.step + 1
    baseLr := optState.baseLr
    weightDecay := optState.weightDecay
  }
  return (newParams, newOptState, result)

/-- Main training loop -/
def trainLoop (cfg : moddedGpt.Config) (hp : Hyperparameters)
    (state : TrainState cfg) (checkpointDir : String := "checkpoints/modded")
    : IO (TrainState cfg) := do
  let totalSteps := hp.numIterations + hp.extensionIterations
  let mut state := state
  let startStep := state.optState.step

  IO.FS.createDirAll ⟨checkpointDir⟩

  if startStep >= totalSteps then
    IO.println s!"Training already complete at step {state.step} (target {totalSteps})"
    return state

  for step in [startStep.toNat:totalSteps.toNat] do
    let stepU := step.toUInt64
    let batchSize := hp.deviceBatchSize
    let seqLen := hp.maxSeqLen
    let gradAccum := effectiveGradAccumSteps hp 1
    let dataGen := {
      state.dataGen with
      iterator := state.dataGen.iterator.updateParams batchSize seqLen
    }
    let mut microBatches : Array (T #[batchSize, seqLen] × T #[batchSize, seqLen]) := #[]
    let mut newDataGen := dataGen
    let device := state.params.embed.device
    for _ in [:gradAccum.toNat] do
      let (maybeBatch, newerDataGen) ← newDataGen.nextBatchGPT
      newDataGen := newerDataGen
      match maybeBatch with
      | none => pure ()
      | some (inputDyn, targetDyn) =>
        let input := (reshape inputDyn #[batchSize, seqLen]).to device
        let target := (reshape targetDyn #[batchSize, seqLen]).to device
        microBatches := microBatches.push (input, target)
    if microBatches.isEmpty then
      state := { state with dataGen := newDataGen }
      continue
    let (newParams, newOptState, result) ← trainStepDistributedAccum state.params state.yarn
      microBatches state.optState hp
    state := { state with
      params := newParams
      optState := newOptState
      dataGen := newDataGen
      step := stepU
      totalTokens := state.totalTokens + result.tokensProcessed
    }
    logProgress state result hp
    if stepU % hp.valInterval == 0 && stepU > 0 then
      state ← runValidation state hp
    if stepU % hp.checkpointInterval == 0 && stepU > 0 then
      let ckpt : Checkpoint cfg := {
        params := state.params
        optState := state.optState
        step := stepU
        bestValLoss := state.bestValLoss
      }
      let stepPath := s!"{checkpointDir}/step_{stepU}.ckpt"
      let latestPath := s!"{checkpointDir}/latest.ckpt"
      saveCheckpoint ckpt stepPath
      saveCheckpoint ckpt latestPath

  let finalCkpt : Checkpoint cfg := {
    params := state.params
    optState := state.optState
    step := state.step
    bestValLoss := state.bestValLoss
  }
  saveCheckpoint finalCkpt s!"{checkpointDir}/latest.ckpt"

  IO.println s!"Training complete! Total tokens: {state.totalTokens}"
  return state

/-! ## Distributed Training -/

/-- Synchronize model parameters across ranks -/
def syncParameters {cfg : moddedGpt.Config} (params : ModdedGPTParams cfg) : IO (ModdedGPTParams cfg) := do
  let isDistributed ← dist.isInitialized
  if isDistributed then
    let syncedParams ← dist.broadcastParams params
    dist.barrier
    return TensorStruct.makeLeafParams syncedParams
  else
    return TensorStruct.makeLeafParams params

/-- All-reduce gradients across all ranks.
    Call this after backward() and before optimizer.step(). -/
def syncGradients {cfg : moddedGpt.Config} (params : ModdedGPTParams cfg) : IO Unit := do
  let isDistributed ← dist.isInitialized
  if isDistributed then
    -- Extract gradients from all parameters
    let grads := TensorStruct.grads params
    -- All-reduce to average gradients across ranks
    let _ ← dist.allReduceGrads grads .avg
    pure ()

/-- Distributed training step with gradient synchronization.

    This is the key integration point for multi-GPU training:
    1. Each rank computes loss and gradients on its local batch
    2. Gradients are all-reduced (averaged) across ranks
    3. Each rank applies the same optimizer update
    4. All ranks end up with identical parameters

    This pattern is called Data Parallel (DP) training.
-/
def trainStepDistributed {cfg : moddedGpt.Config} {batch seq : UInt64}
    (params : ModdedGPTParams cfg)
    (yarn : YarnRotary cfg.headDim cfg.maxSeqLen)
    (input : T #[batch, seq])
    (target : T #[batch, seq])
    (optState : OptimizerState cfg)
    (hp : Hyperparameters)
    (gradClip : Float := 0.0)
    : IO (ModdedGPTParams cfg × OptimizerState cfg × StepResult) := do
  let startTime ← IO.monoMsNow

  -- Check if distributed
  let isDistributed ← dist.isInitialized
  let worldSize ← if isDistributed then dist.getWorldSize else pure 1

  -- Ensure parameters are trainable leaves before backprop.
  let params := TensorStruct.zeroGrads (TensorStruct.makeLeafParams params)

  -- Forward pass (window pattern is in cfg)
  let lossT ← moddedGpt.loss params yarn input target true

  -- Backward pass
  autograd.backwardLoss lossT

  -- Get loss value for logging (before sync to avoid extra comm)
  let lossVal := nn.item lossT

  -- *** DISTRIBUTED GRADIENT SYNCHRONIZATION ***
  -- All-reduce gradients across all ranks
  if isDistributed then
    syncGradients params

  -- Gradient clipping (per-tensor norm clipping)
  if gradClip > 0 then
    let _ ← nn.clip_grad_norm_ params.embed gradClip
    let _ ← nn.clip_grad_norm_ params.smearGate.weight gradClip
    for ve in params.valueEmbeds do
      let _ ← nn.clip_grad_norm_ ve gradClip
    for block in params.blocks do
      -- Attention layer (may be None for some layers)
      match block.attn with
      | some attn =>
        let _ ← nn.clip_grad_norm_ attn.wQ gradClip
        let _ ← nn.clip_grad_norm_ attn.wK gradClip
        let _ ← nn.clip_grad_norm_ attn.wV gradClip
        let _ ← nn.clip_grad_norm_ attn.wO gradClip
      | none => pure ()
      -- MLP
      let _ ← nn.clip_grad_norm_ block.mlp.cFc gradClip
      let _ ← nn.clip_grad_norm_ block.mlp.cProj gradClip
    let _ ← nn.clip_grad_norm_ params.lmHead.weight gradClip
    let _ ← nn.clip_grad_norm_ params.scalars.values gradClip

  -- Extract gradients from parameters
  let grads := TensorStruct.grads params

  -- Get current learning rate from schedule
  let lr := getLearningRate optState.step hp optState.baseLr

  -- Create optimizer with current LR
  let opt := Optim.adamw (lr := lr) (weight_decay := optState.weightDecay)

  -- Apply optimizer step
  let (newParams, newAdamState) := Optim.step opt params grads optState.adamState

  let endTime ← IO.monoMsNow
  let timeMs := (endTime - startTime).toFloat

  -- Tokens processed = local batch * worldSize
  let result : StepResult := {
    loss := lossVal
    gradNorm := 0.0  -- Could compute from grads if needed
    tokensProcessed := batch * seq * worldSize
    timeMs := timeMs
  }

  let newOptState : OptimizerState cfg := {
    adamState := newAdamState
    dualState := optState.dualState
    step := optState.step + 1
    baseLr := optState.baseLr
    weightDecay := optState.weightDecay
  }

  return (newParams, newOptState, result)

/-- Distributed training state with sampler -/
structure DistributedTrainState (cfg : moddedGpt.Config) extends TrainState cfg where
  /-- Distributed sampler for data sharding -/
  sampler : Option dist.DistributedSampler := none
  /-- World size -/
  worldSize : UInt64 := 1
  /-- This rank -/
  rank : UInt64 := 0

/-- Initialize distributed training state -/
def DistributedTrainState.init (cfg : moddedGpt.Config) (dataConfig : DataLoader.Config)
    (hp : Hyperparameters)
    (device : Device := Device.CPU)
    (lr : Float := 0.023) (weightDecay : Float := 0.0)
    : IO (DistributedTrainState cfg) := do
  -- Check distributed status
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then
      dist.getRankAndWorldSize
    else
      pure (0, 1)

  -- Initialize base training state
  let params ← ModdedGPTParams.init cfg
  let yarn ← YarnRotary.init cfg.headDim cfg.maxSeqLen cfg.ropeBase

  let params ← moveToDevice params device
  let yarn := moveYarnToDevice yarn device
  -- Initialize optimizer state from moved parameters so moment buffers match device
  let optState := OptimizerState.init cfg params lr weightDecay

  let initialBatchSize := hp.deviceBatchSize
  let initialSeqLen := hp.maxSeqLen
  let dataGen ← DistributedDataGenerator.init dataConfig initialBatchSize initialSeqLen
  let valData ←
    match dataConfig.valPath with
    | none => pure none
    | some valPath =>
      try
        let shard ← loadValidationData valPath cfg.maxSeqLen dataConfig.bosToken
        pure (some shard)
      catch e =>
        IO.eprintln s!"Warning: failed to load validation data from {valPath}: {e}"
        pure none

  -- Create distributed sampler if in distributed mode
  let sampler := if isDistributed then
      some (dist.DistributedSampler.create {
        datasetSize := 1000000  -- Will be updated based on actual data
        rank := rank
        worldSize := worldSize
        seed := 42
        shuffle := true
      })
    else
      none

  let startTime ← IO.monoMsNow

  return {
    params := params
    yarn := yarn
    optState := optState
    dataGen := dataGen
    valData := valData
    step := 0
    bestValLoss := 1e30
    totalTokens := 0
    startTime := startTime.toUInt64
    sampler := sampler
    worldSize := worldSize
    rank := rank
  }

/-- Distributed training loop with proper gradient sync -/
def trainLoopDistributed (cfg : moddedGpt.Config) (hp : Hyperparameters)
    (state : DistributedTrainState cfg) (checkpointDir : String := "checkpoints/modded")
    : IO (DistributedTrainState cfg) := do
  let totalSteps := hp.numIterations + hp.extensionIterations
  let mut state := state
  let isMaster := state.rank == 0
  let startStep := state.optState.step

  if isMaster then
    IO.FS.createDirAll ⟨checkpointDir⟩

  if startStep >= totalSteps then
    if isMaster then
      IO.println s!"Training already complete at step {state.step} (target {totalSteps})"
    return state

  for step in [startStep.toNat:totalSteps.toNat] do
    let stepU := step.toUInt64
    let batchSize := hp.deviceBatchSize
    let seqLen := hp.maxSeqLen
    let gradAccum := effectiveGradAccumSteps hp state.worldSize
    let dataGen := {
      state.dataGen with
      iterator := state.dataGen.iterator.updateParams batchSize seqLen
    }
    let mut microBatches : Array (T #[batchSize, seqLen] × T #[batchSize, seqLen]) := #[]
    let mut newDataGen := dataGen
    let device := state.params.embed.device
    for _ in [:gradAccum.toNat] do
      let (maybeBatch, newerDataGen) ← newDataGen.nextBatchGPT
      newDataGen := newerDataGen
      match maybeBatch with
      | none => pure ()
      | some (inputDyn, targetDyn) =>
        let input := (reshape inputDyn #[batchSize, seqLen]).to device
        let target := (reshape targetDyn #[batchSize, seqLen]).to device
        microBatches := microBatches.push (input, target)
    if microBatches.isEmpty then
      state := { state with dataGen := newDataGen }
      continue
    let (newParams, newOptState, result) ← trainStepDistributedAccum state.params state.yarn
      microBatches state.optState hp
    state := { state with
      params := newParams
      optState := newOptState
      dataGen := newDataGen
      step := stepU
      totalTokens := state.totalTokens + result.tokensProcessed
    }
    if isMaster then
      logProgress state.toTrainState result hp
    if stepU % hp.valInterval == 0 && stepU > 0 && isMaster then
      let baseState ← runValidation state.toTrainState hp
      state := { state with
        bestValLoss := baseState.bestValLoss
        valData := baseState.valData
      }
    if stepU % hp.checkpointInterval == 0 && stepU > 0 && isMaster then
      let ckpt : Checkpoint cfg := {
        params := state.params
        optState := state.optState
        step := stepU
        bestValLoss := state.bestValLoss
      }
      let stepPath := s!"{checkpointDir}/step_{stepU}.ckpt"
      let latestPath := s!"{checkpointDir}/latest.ckpt"
      saveCheckpoint ckpt stepPath
      saveCheckpoint ckpt latestPath

  if isMaster then
    let finalCkpt : Checkpoint cfg := {
      params := state.params
      optState := state.optState
      step := state.step
      bestValLoss := state.bestValLoss
    }
    saveCheckpoint finalCkpt s!"{checkpointDir}/latest.ckpt"
    IO.println s!"Training complete! Total tokens: {state.totalTokens}"
  return state

/-- Distributed training loop wrapper -/
def trainDistributed (cfg : moddedGpt.Config) (hp : Hyperparameters)
    (dataConfig : DataLoader.Config) (device : Device)
    (checkpointDir : String := "checkpoints/modded")
    (resume : Option String := none)
    : IO (DistributedTrainState cfg) := do
  -- Check if distributed
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then
      dist.getRankAndWorldSize
    else
      pure (0, 1)

  let isMaster := rank == 0

  if isMaster then
    IO.println s!"Starting training with {worldSize} GPUs"
    IO.println s!"Config: {repr cfg}"
    IO.println s!"Hyperparameters: {repr hp}"

  -- Initialize distributed training state
  let initState ← DistributedTrainState.init cfg dataConfig hp device

  -- Resolve resume path (explicit --resume takes precedence over latest in checkpointDir).
  let latestPath := s!"{checkpointDir}/latest.ckpt"
  let resumePath? ←
    match resume with
    | some path => pure (some path)
    | none =>
      if ← checkpointExists latestPath then
        pure (some latestPath)
      else
        pure none

  let state ←
    match resumePath? with
    | none => pure initState
    | some resumePath =>
      if isMaster then
        IO.println s!"Resuming from checkpoint: {resumePath}"
      match ← loadCheckpoint cfg resumePath with
      | some ckpt =>
        let resumedParams ← moveToDevice ckpt.params device
        let resumedOptState : OptimizerState cfg := {
          ckpt.optState with
          adamState := TensorStruct.map (fun t => t.to device) ckpt.optState.adamState
          dualState := TensorStruct.map (fun t => t.to device) ckpt.optState.dualState
        }
        pure {
          initState with
          params := resumedParams
          optState := resumedOptState
          step := ckpt.step
          bestValLoss := ckpt.bestValLoss
        }
      | none =>
        if resume.isSome then
          throw <| IO.userError s!"Resume checkpoint not found or invalid: {resumePath}"
        else
          if isMaster then
            IO.println s!"Warning: failed to load auto-resume checkpoint {resumePath}; starting fresh"
          pure initState

  -- Normalize all train state tensors onto the selected training device
  -- before any NCCL collectives.
  let paramsOnDevice ← moveToDevice state.params device
  let adamOnDevice := TensorStruct.map (fun t => t.to device) state.optState.adamState
  let dualOnDevice := TensorStruct.map (fun t => t.to device) state.optState.dualState
  let state := {
    state with
    params := paramsOnDevice
    optState := {
      state.optState with
      adamState := adamOnDevice
      dualState := dualOnDevice
    }
  }

  -- Synchronize parameters from rank 0
  let syncedParams ← syncParameters state.params
  let state := {
    state with
    params := syncedParams
  }

  -- Match nanochat's strict grad-accum divisibility contract.
  validateGradAccumConfig hp worldSize

  -- Barrier to ensure all ranks are ready
  if isDistributed then
    dist.barrier

  -- Run distributed training loop
  let finalState ← trainLoopDistributed cfg hp state checkpointDir

  if isMaster then
    IO.println s!"Training finished!"
    IO.println s!"Best validation loss: {finalState.bestValLoss}"

  return finalState

/-- Dynamically-shaped GPT micro-batch `(inputs, targets)`. -/
abbrev DynamicGPTBatch := T #[] × T #[]

/-- Provider callback for GPT micro-batches. -/
abbrev DynamicGPTBatchProvider := IO (Option DynamicGPTBatch)

/-- Distributed streaming training state (no shard-backed data generator). -/
structure StreamTrainState (cfg : moddedGpt.Config) where
  params : ModdedGPTParams cfg
  yarn : YarnRotary cfg.headDim cfg.maxSeqLen
  optState : OptimizerState cfg
  step : UInt64 := 0
  bestValLoss : Float := 1e30
  totalTokens : UInt64 := 0
  startTime : UInt64 := 0
  rank : UInt64 := 0
  worldSize : UInt64 := 1

private def logStreamProgress (state : StreamTrainState cfg) (result : StepResult)
    (hp : Hyperparameters) : IO Unit := do
  if state.step % hp.logInterval == 0 then
    let nowMs ← IO.monoMsNow
    let elapsed := nowMs - state.startTime.toNat
    let elapsedSec := elapsed.toFloat / 1000.0
    let tokensPerSec :=
      if elapsedSec <= 0.0 then 0.0 else state.totalTokens.toFloat / elapsedSec
    let lr := getLearningRate state.step hp
    IO.println s!"Step {state.step}: loss={result.loss} lr={lr} batch={hp.deviceBatchSize} seq={hp.maxSeqLen} tok/s={tokensPerSec} time={result.timeMs}ms"

private def initStreamTrainState (cfg : moddedGpt.Config) (device : Device)
    (lr : Float := 0.023) (weightDecay : Float := 0.0)
    : IO (StreamTrainState cfg) := do
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then
      dist.getRankAndWorldSize
    else
      pure (0, 1)

  let params ← ModdedGPTParams.init cfg
  let yarn ← YarnRotary.init cfg.headDim cfg.maxSeqLen cfg.ropeBase
  let params ← moveToDevice params device
  let yarn := moveYarnToDevice yarn device
  let optState := OptimizerState.init cfg params lr weightDecay
  let startTime ← IO.monoMsNow

  return {
    params := params
    yarn := yarn
    optState := optState
    startTime := startTime.toUInt64
    rank := rank
    worldSize := worldSize
  }

private def collectMicroBatches {batch seq : UInt64}
    (provider : DynamicGPTBatchProvider)
    (gradAccum : Nat)
    (device : Device)
    : IO (Array (T #[batch, seq] × T #[batch, seq])) := do
  let mut microBatches : Array (T #[batch, seq] × T #[batch, seq]) := #[]
  for _ in [:gradAccum] do
    match ← provider with
    | none => pure ()
    | some (inputDyn, targetDyn) =>
      let input := (reshape inputDyn #[batch, seq]).to device
      let target := (reshape targetDyn #[batch, seq]).to device
      microBatches := microBatches.push (input, target)
  return microBatches

private def validateWithProvider {cfg : moddedGpt.Config} {batch seq : UInt64}
    (params : ModdedGPTParams cfg)
    (yarn : YarnRotary cfg.headDim cfg.maxSeqLen)
    (provider : DynamicGPTBatchProvider)
    (numBatches : Nat)
    (device : Device)
    : IO (Option Float) := do
  if numBatches == 0 then
    return none

  let mut totalLoss := 0.0
  let mut seen := 0

  for _ in [:numBatches] do
    match ← provider with
    | none => pure ()
    | some (inputDyn, targetDyn) =>
      let input := (reshape inputDyn #[batch, seq]).to device
      let target := (reshape targetDyn #[batch, seq]).to device
      let lossT ← moddedGpt.loss params yarn input target false
      totalLoss := totalLoss + nn.item lossT
      seen := seen + 1

  if seen == 0 then
    return none
  else
    return some (totalLoss / seen.toFloat)

private def trainLoopDistributedStream (cfg : moddedGpt.Config) (hp : Hyperparameters)
    (state : StreamTrainState cfg)
    (trainProvider : DynamicGPTBatchProvider)
    (valProvider? : Option DynamicGPTBatchProvider)
    (valBatches : Nat)
    (checkpointDir : String)
    : IO (StreamTrainState cfg) := do
  let totalSteps := hp.numIterations + hp.extensionIterations
  let mut state := state
  let isMaster := state.rank == 0
  let startStep := state.optState.step
  if isMaster then
    IO.FS.createDirAll ⟨checkpointDir⟩

  if startStep >= totalSteps then
    if isMaster then
      IO.println s!"Training already complete at step {state.step} (target {totalSteps})"
    return state

  for step in [startStep.toNat:totalSteps.toNat] do
    let stepU := step.toUInt64
    let batchSize := hp.deviceBatchSize
    let seqLen := hp.maxSeqLen
    let gradAccum := effectiveGradAccumSteps hp state.worldSize
    let device := state.params.embed.device

    let microBatches ← collectMicroBatches
      (batch := batchSize) (seq := seqLen)
      trainProvider gradAccum.toNat device

    if microBatches.isEmpty then
      if isMaster then
        IO.println "No micro-batches available; ending stream training early."
      break

    let (newParams, newOptState, result) ← trainStepDistributedAccum state.params state.yarn
      microBatches state.optState hp

    state := { state with
      params := newParams
      optState := newOptState
      step := stepU
      totalTokens := state.totalTokens + result.tokensProcessed
    }

    if isMaster then
      logStreamProgress state result hp

    if stepU % hp.valInterval == 0 && stepU > 0 && isMaster then
      match valProvider? with
      | none => pure ()
      | some valProvider =>
        match ← validateWithProvider
            (cfg := cfg) (batch := batchSize) (seq := seqLen)
            state.params state.yarn valProvider valBatches device with
        | none => pure ()
        | some valLoss =>
          IO.println s!"Validation: loss={valLoss}"
          if valLoss < state.bestValLoss then
            IO.println "  New best validation loss!"
            state := { state with bestValLoss := valLoss }

    if stepU % hp.checkpointInterval == 0 && stepU > 0 && isMaster then
      let ckpt : Checkpoint cfg := {
        params := state.params
        optState := state.optState
        step := stepU
        bestValLoss := state.bestValLoss
      }
      let stepPath := s!"{checkpointDir}/step_{stepU}.ckpt"
      let latestPath := s!"{checkpointDir}/latest.ckpt"
      saveCheckpoint ckpt stepPath
      saveCheckpoint ckpt latestPath

  if isMaster then
    let finalCkpt : Checkpoint cfg := {
      params := state.params
      optState := state.optState
      step := state.step
      bestValLoss := state.bestValLoss
    }
    saveCheckpoint finalCkpt s!"{checkpointDir}/latest.ckpt"
    IO.println s!"Training complete! Total tokens: {state.totalTokens}"

  return state

/-- Train from streaming GPT batch providers instead of parquet shards.
    This keeps the distributed optimizer/checkpoint path unchanged while
    allowing task-mixture token-buffer data feeding. -/
def trainDistributedWithBatchProvider (cfg : moddedGpt.Config) (hp : Hyperparameters)
    (device : Device)
    (trainProvider : DynamicGPTBatchProvider)
    (valProvider? : Option DynamicGPTBatchProvider := none)
    (valBatches : Nat := 0)
    (checkpointDir : String := "checkpoints/modded")
    (resume : Option String := none)
    : IO (StreamTrainState cfg) := do
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then
      dist.getRankAndWorldSize
    else
      pure (0, 1)

  let isMaster := rank == 0

  if isMaster then
    IO.println s!"Starting streaming training with {worldSize} GPUs"
    IO.println s!"Config: {repr cfg}"
    IO.println s!"Hyperparameters: {repr hp}"

  let initState ← initStreamTrainState cfg device

  let latestPath := s!"{checkpointDir}/latest.ckpt"
  let resumePath? ←
    match resume with
    | some path => pure (some path)
    | none =>
      if ← checkpointExists latestPath then
        pure (some latestPath)
      else
        pure none

  let state ←
    match resumePath? with
    | none => pure initState
    | some resumePath =>
      if isMaster then
        IO.println s!"Resuming from checkpoint: {resumePath}"
      match ← loadCheckpoint cfg resumePath with
      | some ckpt =>
        let resumedParams ← moveToDevice ckpt.params device
        let resumedOptState : OptimizerState cfg := {
          ckpt.optState with
          adamState := TensorStruct.map (fun t => t.to device) ckpt.optState.adamState
          dualState := TensorStruct.map (fun t => t.to device) ckpt.optState.dualState
        }
        pure {
          initState with
          params := resumedParams
          optState := resumedOptState
          step := ckpt.step
          bestValLoss := ckpt.bestValLoss
        }
      | none =>
        if resume.isSome then
          throw <| IO.userError s!"Resume checkpoint not found or invalid: {resumePath}"
        else
          if isMaster then
            IO.println s!"Warning: failed to load auto-resume checkpoint {resumePath}; starting fresh"
          pure initState

  -- Normalize all train state tensors onto the selected training device
  -- before any NCCL collectives.
  let paramsOnDevice ← moveToDevice state.params device
  let adamOnDevice := TensorStruct.map (fun t => t.to device) state.optState.adamState
  let dualOnDevice := TensorStruct.map (fun t => t.to device) state.optState.dualState
  let state := {
    state with
    params := paramsOnDevice
    optState := {
      state.optState with
      adamState := adamOnDevice
      dualState := dualOnDevice
    }
  }

  let syncedParams ← syncParameters state.params
  let state := {
    state with
    params := syncedParams
  }

  -- Match nanochat's strict grad-accum divisibility contract.
  validateGradAccumConfig hp worldSize

  if isDistributed then
    dist.barrier

  let finalState ← trainLoopDistributedStream cfg hp state trainProvider valProvider? valBatches checkpointDir

  if isMaster then
    IO.println "Streaming training finished!"
    IO.println s!"Best validation loss: {finalState.bestValLoss}"

  return finalState

end torch.ModdedTrain
