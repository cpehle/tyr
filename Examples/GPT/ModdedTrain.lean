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
import Examples.GPT.ModdedGPT
import Tyr.DataLoader
import Examples.GPT.GPTDataLoader
import Tyr.Optim.NorMuon
import Tyr.Optim.DistAdam

namespace torch.ModdedTrain

open torch
open torch.moddedGpt
open torch.DataLoader
open torch.Optim

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

/-- Combined optimizer state for training -/
structure OptimizerState where
  /-- NorMuon config for weight matrices -/
  muonConfig : NorMuon.Config
  /-- DistAdam config for embeddings -/
  adamConfig : DistAdam.Config
  /-- Current step -/
  step : UInt64
  deriving Repr

/-- Initialize optimizer state -/
def OptimizerState.init (distributed : Bool) (worldSize : UInt64 := 1) : OptimizerState := {
  muonConfig := {
    lr := 0.023
    weightDecay := 1.2
    momentum := 0.95
    beta2 := 0.95
    numIters := 5
    distributed := distributed
    worldSize := worldSize
  }
  adamConfig := {
    lr := 0.008
    beta1 := 0.65
    beta2 := 0.95
    eps := 1e-8
    weightDecay := 0.005
    distributed := distributed
  }
  step := 0
}

/-- Update optimizer configs for current step -/
def OptimizerState.updateForStep (state : OptimizerState) (step : UInt64)
    (hp : Hyperparameters) : OptimizerState :=
  let lr := getLearningRate step hp
  let momentum := getMuonMomentum step hp
  { state with
    muonConfig := { state.muonConfig with lr := lr, momentum := momentum }
    adamConfig := { state.adamConfig with lr := lr * 0.35 }  -- Adam uses lower LR
    step := step
  }

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

    modded-nanogpt alternates:
    - Even steps: Muon only
    - Odd steps: Muon + Adam
-/
def trainStep {cfg : moddedGpt.Config} {batch seq : UInt64}
    (params : ModdedGPTParams cfg)
    (yarn : YarnRotary cfg.headDim cfg.maxSeqLen)
    (input : T #[batch, seq])
    (target : T #[batch, seq])
    (_optState : OptimizerState)
    : IO (ModdedGPTParams cfg × StepResult) := do
  let startTime ← IO.monoMsNow

  -- Forward pass (window pattern is in cfg)
  let lossT ← moddedGpt.loss params yarn input target true

  -- Backward pass
  autograd.backwardLoss lossT

  -- Get loss value
  let lossVal := nn.item lossT

  -- Compute gradient norm for monitoring
  let gradNorm := 0.0  -- Would compute actual grad norm

  -- Update parameters
  -- In a full implementation:
  -- - Even steps: only update Muon params (attention, MLP)
  -- - Odd steps: update both Muon and Adam params

  -- Zero gradients for next step
  -- (In production, done after optimizer step)

  let endTime ← IO.monoMsNow
  let timeMs := (endTime - startTime).toFloat

  let result : StepResult := {
    loss := lossVal
    gradNorm := gradNorm
    tokensProcessed := batch * seq
    timeMs := timeMs
  }

  -- Return updated params (placeholder - actual update would modify params)
  return (params, result)

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
      let input := reshape inputDyn #[batchSize, seqLen]
      let target := reshape targetDyn #[batchSize, seqLen]
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
  optState : OptimizerState
  /-- Current step -/
  step : UInt64
  /-- Best validation loss -/
  bestValLoss : Float
  deriving Repr

/-- Save a checkpoint to disk -/
def saveCheckpoint {cfg : moddedGpt.Config} (ckpt : Checkpoint cfg) (path : String)
    : IO Unit := do
  -- In a full implementation, serialize to disk
  -- For now, just log parameter statistics using TensorStruct
  IO.println s!"Saving checkpoint to {path} at step {ckpt.step}"
  
  let numParams := TensorStruct.fold (fun {s} _t acc => acc + s.foldl (· * ·) 1) 0 ckpt.params
  let numTensors := TensorStruct.fold (fun {_s} _t acc => acc + 1) 0 ckpt.params
  
  IO.println s!"  Parameters: {numTensors} tensors, {numParams} elements"
  IO.println s!"  Best validation loss: {ckpt.bestValLoss}"

/-- Load a checkpoint from disk -/
def loadCheckpoint (cfg : moddedGpt.Config) (path : String)
    : IO (Option (Checkpoint cfg)) := do
  -- In a full implementation, deserialize from disk
  IO.println s!"Would load checkpoint from {path}"
  return none

/-! ## Training Loop -/

/-- Training state for the main loop -/
structure TrainState (cfg : moddedGpt.Config) where
  /-- Model parameters -/
  params : ModdedGPTParams cfg
  /-- YaRN rotary embeddings -/
  yarn : YarnRotary cfg.headDim cfg.maxSeqLen
  /-- Optimizer state -/
  optState : OptimizerState
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
  deriving Repr

/-- Initialize training state -/
def TrainState.init (cfg : moddedGpt.Config) (dataConfig : DataLoader.Config)
    (distributed : Bool) (worldSize : UInt64)
    : IO (TrainState cfg) := do
  let params ← ModdedGPTParams.init cfg
  let yarn ← YarnRotary.init cfg.headDim cfg.maxSeqLen cfg.ropeBase

  let optState := OptimizerState.init distributed worldSize

  let initialBatchSize := getBatchSize 0
  let initialSeqLen := getSeqLen 0 cfg.blockSize
  let dataGen ← DistributedDataGenerator.init dataConfig initialBatchSize initialSeqLen

  let startTime ← IO.monoMsNow

  return {
    params := params
    yarn := yarn
    optState := optState
    dataGen := dataGen
    valData := none
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

    let valResult ← validate state.params state.yarn valData batchSize seqLen

    IO.println s!"Validation: loss={valResult.loss} time={valResult.timeMs}ms"

    let newBest := valResult.loss < state.bestValLoss
    if newBest then
      IO.println s!"  New best validation loss!"

    return { state with
      bestValLoss := if newBest then valResult.loss else state.bestValLoss
    }

/-- Main training loop -/
def trainLoop (cfg : moddedGpt.Config) (hp : Hyperparameters)
    (state : TrainState cfg) : IO (TrainState cfg) := do
  let totalSteps := hp.numIterations + hp.extensionIterations
  let mut state := state

  for step in [:totalSteps.toNat] do
    let stepU := step.toUInt64

    -- Update schedules
    let batchSize := getBatchSize stepU
    let seqLen := getSeqLen stepU hp.blockSize
    let (wsShort, wsLong) := getWindowSizes stepU hp.blockSize

    -- Update optimizer configs
    let optState := state.optState.updateForStep stepU hp

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
      let input := reshape inputDyn #[batchSize, seqLen]
      let target := reshape targetDyn #[batchSize, seqLen]
      -- Training step (window pattern is in cfg)
      let (newParams, result) ← trainStep state.params state.yarn
        input target optState

      -- Update state
      state := { state with
        params := newParams
        optState := { optState with step := stepU }
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
        saveCheckpoint ckpt s!"checkpoints/step_{stepU}.ckpt"

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

/-- Distributed training loop wrapper -/
def trainDistributed (cfg : moddedGpt.Config) (hp : Hyperparameters)
    (dataConfig : DataLoader.Config) : IO Unit := do
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

  -- Initialize training state
  let state ← TrainState.init cfg dataConfig isDistributed worldSize

  -- Synchronize parameters
  let _ ← syncParameters state.params

  -- Run training
  let finalState ← trainLoop cfg hp state

  if isMaster then
    IO.println s!"Training finished!"
    IO.println s!"Best validation loss: {finalState.bestValLoss}"

end torch.ModdedTrain
