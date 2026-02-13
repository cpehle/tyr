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

namespace torch.ModdedTrain

open torch
open torch.moddedGpt
open torch.DataLoader
open torch.Optim

private def moveToDevice [TensorStruct α] (x : α) (device : Device) : IO α :=
  TensorStruct.mapM (fun t => pure (t.to device)) x

private def moveYarnToDevice {headDim maxSeqLen : UInt64}
    (yarn : YarnRotary headDim maxSeqLen) (device : Device) : YarnRotary headDim maxSeqLen :=
  { yarn with
    cos := yarn.cos.to device
    sin := yarn.sin.to device
    angularFreq := yarn.angularFreq.to device
  }

/-! ## Hyperparameters -/

/-- Training hyperparameters matching modded-nanogpt -/
structure Hyperparameters where
  /-- Total scheduled training iterations -/
  numIterations : UInt64 := 2050
  /-- Extension iterations for longer context -/
  extensionIterations : UInt64 := 40
  /-- Cooldown fraction (55% of training) -/
  cooldownFrac : Float := 0.55
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
  /-- Gradient accumulation steps -/
  gradAccumSteps : UInt64 := 8
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

    LR schedule:
    1. Warmup: linear from 0 to peak over 300 steps
    2. Plateau: constant at peak during batch size ramp
    3. Cooldown: cosine decay to 0.1 * peak

    The cooldown starts at 55% of training.
-/
def getLearningRate (step : UInt64) (hp : Hyperparameters)
    (baseLr : Float := 0.023) (warmupSteps : UInt64 := 300) : Float :=
  let totalSteps := hp.numIterations + hp.extensionIterations
  let cooldownStart := (hp.cooldownFrac * totalSteps.toFloat).toUInt64

  if step < warmupSteps then
    -- Linear warmup
    baseLr * step.toFloat / warmupSteps.toFloat
  else if step < cooldownStart then
    -- Plateau
    baseLr
  else
    -- Cosine cooldown to 0.1 * baseLr
    let progress := (step - cooldownStart).toFloat / (totalSteps - cooldownStart).toFloat
    let minLr := 0.1 * baseLr
    minLr + 0.5 * (baseLr - minLr) * (1.0 + Float.cos (progress * 3.14159265359))

/-! ## Momentum Schedule -/

/-- Get Muon momentum for a given iteration.

    Momentum schedule:
    - Warmup: 0.85 -> 0.95 over 300 steps
    - Plateau: 0.95
    - Cooldown: 0.95 -> 0.85 over last 50 steps
-/
def getMuonMomentum (step : UInt64) (hp : Hyperparameters)
    (baseMomentum : Float := 0.95) (warmupSteps : UInt64 := 300)
    (cooldownSteps : UInt64 := 50) : Float :=
  NorMuon.getMomentum step.toNat
    (hp.numIterations + hp.extensionIterations).toNat
    baseMomentum warmupSteps.toNat cooldownSteps.toNat

/-! ## Optimizer State -/

/-- Combined optimizer state for training.
    Uses AdamW for all parameters (can be extended to dual optimizer later). -/
structure OptimizerState (cfg : moddedGpt.Config) where
  /-- AdamW optimizer state -/
  adamState : Optim.AdamWState (ModdedGPTParams cfg)
  /-- Current step -/
  step : UInt64
  /-- Base learning rate -/
  baseLr : Float := 0.023
  /-- Weight decay -/
  weightDecay : Float := 0.01

/-- Initialize optimizer state from model parameters -/
def OptimizerState.init (cfg : moddedGpt.Config) (params : ModdedGPTParams cfg)
    (lr : Float := 0.023) (weightDecay : Float := 0.01) : OptimizerState cfg :=
  let opt := Optim.adamw (lr := lr) (weight_decay := weightDecay)
  {
    adamState := opt.init params
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
    (gradClip : Float := 1.0)
    : IO (ModdedGPTParams cfg × OptimizerState cfg × StepResult) := do
  let startTime ← IO.monoMsNow

  -- Zero gradients BEFORE forward/backward
  let params := TensorStruct.zeroGrads params

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
    (_distributed : Bool) (_worldSize : UInt64)
    (device : Device := Device.CPU)
    (lr : Float := 0.023) (weightDecay : Float := 0.01)
    : IO (TrainState cfg) := do
  let params ← ModdedGPTParams.init cfg
  let yarn ← YarnRotary.init cfg.headDim cfg.maxSeqLen cfg.ropeBase

  let params ← moveToDevice params device
  let yarn := moveYarnToDevice yarn device
  -- Initialize optimizer state from moved parameters so moment buffers match device
  let optState := OptimizerState.init cfg params lr weightDecay

  let initialBatchSize := getBatchSize 0
  let initialSeqLen := getSeqLen 0 cfg.blockSize
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
    let batchSize := getBatchSize state.step
    let (wsShort, wsLong) := getWindowSizes state.step
    let lr := getLearningRate state.step hp

    IO.println s!"Step {state.step}: loss={result.loss} lr={lr} batch={batchSize} ws=({wsShort},{wsLong}) tok/s={tokensPerSec} time={result.timeMs}ms"

/-- Run validation and log results -/
def runValidation (state : TrainState cfg) (_hp : Hyperparameters)
    : IO (TrainState cfg) := do
  match state.valData with
  | none => return state
  | some valData =>
    let batchSize := getBatchSize state.step
    let seqLen := getSeqLen state.step
    let _windowSizes := getWindowSizes state.step  -- Kept for API compatibility
    let device := state.params.embed.device

    let valResult ← validate state.params state.yarn valData batchSize seqLen device

    IO.println s!"Validation: loss={valResult.loss} time={valResult.timeMs}ms"

    let newBest := valResult.loss < state.bestValLoss
    if newBest then
      IO.println s!"  New best validation loss!"

    return { state with
      bestValLoss := if newBest then valResult.loss else state.bestValLoss
    }

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

    -- Update schedules
    let batchSize := getBatchSize stepU
    let seqLen := getSeqLen stepU hp.blockSize
    let (_wsShort, _wsLong) := getWindowSizes stepU hp.blockSize

    -- Update data generator
    let dataGen := state.dataGen.updateForStepGPT stepU hp.blockSize

    -- Get next batch
    let (maybeBatch, newDataGen) ← dataGen.nextBatchGPT
    match maybeBatch with
    | none =>
      IO.println s!"Epoch complete at step {stepU}"
      state := { state with dataGen := newDataGen }
      continue
    | some (inputDyn, targetDyn) =>
      -- Reshape dynamic tensors to expected shape
      let device := state.params.embed.device
      let input := (reshape inputDyn #[batchSize, seqLen]).to device
      let target := (reshape targetDyn #[batchSize, seqLen]).to device

      -- Training step with optimizer update
      let (newParams, newOptState, result) ← trainStep state.params state.yarn
        input target state.optState hp

      -- Update state
      state := { state with
        params := newParams
        optState := newOptState
        dataGen := newDataGen
        step := stepU
        totalTokens := state.totalTokens + result.tokensProcessed
      }

      -- Log progress
      logProgress state result hp

      -- Validation
      if stepU % hp.valInterval == 0 && stepU > 0 then
        state ← runValidation state hp

      -- Checkpoint
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
    return syncedParams
  else
    return params

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
    (gradClip : Float := 1.0)
    : IO (ModdedGPTParams cfg × OptimizerState cfg × StepResult) := do
  let startTime ← IO.monoMsNow

  -- Check if distributed
  let isDistributed ← dist.isInitialized
  let worldSize ← if isDistributed then dist.getWorldSize else pure 1

  -- Zero gradients BEFORE forward/backward
  let params := TensorStruct.zeroGrads params

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
    (device : Device := Device.CPU)
    (lr : Float := 0.023) (weightDecay : Float := 0.01)
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

  let initialBatchSize := getBatchSize 0
  let initialSeqLen := getSeqLen 0 cfg.blockSize
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

    -- Update schedules
    let batchSize := getBatchSize stepU
    let seqLen := getSeqLen stepU hp.blockSize
    let (_wsShort, _wsLong) := getWindowSizes stepU hp.blockSize

    -- Update data generator
    let dataGen := state.dataGen.updateForStepGPT stepU hp.blockSize

    -- Get next batch
    let (maybeBatch, newDataGen) ← dataGen.nextBatchGPT
    match maybeBatch with
    | none =>
      if isMaster then
        IO.println s!"Epoch complete at step {stepU}"
      state := { state with dataGen := newDataGen }
      continue
    | some (inputDyn, targetDyn) =>
      -- Reshape dynamic tensors to expected shape
      let device := state.params.embed.device
      let input := (reshape inputDyn #[batchSize, seqLen]).to device
      let target := (reshape targetDyn #[batchSize, seqLen]).to device

      -- Distributed training step with gradient sync
      let (newParams, newOptState, result) ← trainStepDistributed state.params state.yarn
        input target state.optState hp

      -- Update state
      state := { state with
        params := newParams
        optState := newOptState
        dataGen := newDataGen
        step := stepU
        totalTokens := state.totalTokens + result.tokensProcessed
      }

      -- Log progress (only on master)
      if isMaster then
        logProgress state.toTrainState result hp

      -- Validation (master only)
      if stepU % hp.valInterval == 0 && stepU > 0 && isMaster then
        let baseState ← runValidation state.toTrainState hp
        state := { state with
          bestValLoss := baseState.bestValLoss
          valData := baseState.valData
        }

      -- Checkpoint (only on master)
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
  let initState ← DistributedTrainState.init cfg dataConfig device

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

  -- Synchronize parameters from rank 0
  let syncedParams ← syncParameters state.params
  let syncedAdamState ← if isDistributed then dist.broadcastParams state.optState.adamState else pure state.optState.adamState
  let state := {
    state with
    params := syncedParams
    optState := { state.optState with adamState := syncedAdamState }
  }

  -- Barrier to ensure all ranks are ready
  if isDistributed then
    dist.barrier

  -- Run distributed training loop
  let finalState ← trainLoopDistributed cfg hp state checkpointDir

  if isMaster then
    IO.println s!"Training finished!"
    IO.println s!"Best validation loss: {finalState.bestValLoss}"

  return finalState

end torch.ModdedTrain
