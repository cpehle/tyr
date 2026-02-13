/-
  ModdedTrainGPT.lean

  Main training script for modded-nanogpt style GPT training.

  This is the entry point that ties together:
  - Model architecture (ModdedGPT)
  - NorMuon optimizer (for weight matrices)
  - DistAdam optimizer (for embeddings)
  - Distributed training
  - Document-aware data loading

  Usage:
    ninja modded
    export DYLD_LIBRARY_PATH=external/libtorch/lib:/opt/homebrew/opt/libomp/lib
    out/exe/ModdedTrainGPT

  For distributed training (multi-GPU):
    torchrun --nproc_per_node=8 out/exe/ModdedTrainGPT
-/
import Tyr.Torch
import Tyr.Distributed
import Examples.NanoChat.ModdedGPT
import Tyr.DataLoader
import Examples.NanoChat.ModdedTrain
import Tyr.Optim.NorMuon
import Tyr.Optim.DistAdam

open torch
open torch.moddedGpt
open torch.DataLoader
open torch.ModdedTrain
open torch.dist

/-- Parse command line arguments -/
structure Args where
  /-- Path to training data -/
  dataPath : String := "data/fineweb10B"
  /-- Path to validation data -/
  valPath : String := "data/fineweb_val"
  /-- Checkpoint directory -/
  checkpointDir : String := "checkpoints/modded"
  /-- Resume from checkpoint -/
  resume : Option String := none
  /-- Override number of iterations -/
  numIterations : Option UInt64 := none
  /-- Local rank for distributed training (from env) -/
  localRank : UInt64 := 0
  /-- Run in debug mode with small model -/
  debug : Bool := false
  deriving Repr, Inhabited

/-- Parse args from command line -/
def parseArgs (args : List String) : Args :=
  let rec go : List String → Args → Args
    | [], acc => acc
    | "--data" :: path :: rest, acc => go rest { acc with dataPath := path }
    | "--val" :: path :: rest, acc => go rest { acc with valPath := path }
    | "--checkpoint-dir" :: dir :: rest, acc => go rest { acc with checkpointDir := dir }
    | "--resume" :: path :: rest, acc => go rest { acc with resume := some path }
    | "--iterations" :: n :: rest, acc =>
        go rest { acc with numIterations := n.toNat?.map (·.toUInt64) }
    | "--debug" :: rest, acc => go rest { acc with debug := true }
    | _ :: rest, acc => go rest acc
  go args default

/-- Get local rank from environment (for torchrun) -/
def getLocalRankFromEnv : IO UInt64 := do
  let envVar ← IO.getEnv "LOCAL_RANK"
  match envVar with
  | some rank => return rank.toNat!.toUInt64
  | none => return 0

/-- Get world size from environment -/
def getWorldSizeFromEnv : IO UInt64 := do
  let envVar ← IO.getEnv "WORLD_SIZE"
  match envVar with
  | some ws => return ws.toNat!.toUInt64
  | none => return 1

/-- Get global rank from environment (for distributed training). -/
def getRankFromEnv : IO UInt64 := do
  let envVar ← IO.getEnv "RANK"
  match envVar with
  | some rank => return rank.toNat!.toUInt64
  | none => getLocalRankFromEnv

/-- Render a device for logging. -/
def deviceToString : Device → String
  | Device.MPS => "MPS"
  | Device.CPU => "CPU"
  | Device.CUDA n => s!"CUDA:{n}"

/-- Resolve training device from TYR_DEVICE and distributed rank. -/
def resolveTrainingDevice (rank : UInt64) (isDistributed : Bool) : IO Device := do
  let requested? := (← IO.getEnv "TYR_DEVICE").map String.toLower
  match requested? with
  | some "cpu" => pure Device.CPU
  | some "mps" => pure Device.MPS
  | some "cuda" => pure (Device.CUDA (if isDistributed then rank else 0))
  | some "auto" | none =>
    if isDistributed then
      if ← cuda_is_available then pure (Device.CUDA rank) else pure Device.CPU
    else
      getBestDevice
  | some _ =>
    if isDistributed then
      if ← cuda_is_available then pure (Device.CUDA rank) else pure Device.CPU
    else
      getBestDevice

/-- Initialize distributed training if needed -/
def initDistributedIfNeeded : IO Bool := do
  let worldSize ← getWorldSizeFromEnv
  if worldSize > 1 then
    let rank ← getRankFromEnv
    let localRank ← getLocalRankFromEnv
    let masterAddrEnv ← IO.getEnv "MASTER_ADDR"
    let masterAddr := masterAddrEnv.getD "localhost"
    let masterPortEnv ← IO.getEnv "MASTER_PORT"
    let masterPort := (masterPortEnv.getD "29500").toNat!.toUInt64
    IO.println s!"Initializing distributed: rank {rank} (local {localRank})/{worldSize} master={masterAddr}:{masterPort}"
    dist.initProcessGroup "nccl" masterAddr masterPort rank worldSize
    return true
  else
    return false

/-- Main training function -/
def runTraining (args : Args) : IO Unit := do
  IO.println "=== Modded NanoGPT Training (Tyr Port) ==="
  IO.println ""

  -- Initialize distributed if needed
  let isDistributed ← initDistributedIfNeeded
  let (rank, worldSize) ← if isDistributed then
      getRankAndWorldSize
    else
      pure (0, 1)
  let trainDevice ← resolveTrainingDevice rank isDistributed

  let isMaster := rank == 0

  if isMaster then
    IO.println s!"Configuration:"
    IO.println s!"  Data path: {args.dataPath}"
    IO.println s!"  Validation path: {args.valPath}"
    IO.println s!"  Checkpoint dir: {args.checkpointDir}"
    IO.println s!"  Distributed: {isDistributed} (world size: {worldSize})"
    IO.println s!"  Device: {deviceToString trainDevice}"
    IO.println s!"  Debug mode: {args.debug}"
    IO.println ""

  -- Model config
  let cfg := if args.debug then
      -- Smaller model for debugging
      { Config.default with
        nLayer := 4
        nHead := 4
        headDim := 64
        modelDim := 256
      }
    else
      Config.default

  if isMaster then
    IO.println s!"Model config:"
    IO.println s!"  Vocab size: {cfg.vocabSize}"
    IO.println s!"  Layers: {cfg.nLayer}"
    IO.println s!"  Heads: {cfg.nHead}"
    IO.println s!"  Head dim: {cfg.headDim}"
    IO.println s!"  Model dim: {cfg.modelDim}"
    IO.println ""

  -- Hyperparameters
  let hp : Hyperparameters := {
    Hyperparameters.default with
    numIterations := args.numIterations.getD Hyperparameters.default.numIterations
  }

  -- Data config
  let dataConfig : DataLoader.Config := {
    dataPath := args.dataPath
    valPath := some args.valPath
    seqLen := cfg.maxSeqLen
    bosToken := 50256  -- GPT-2 BOS
    numWorkers := 4
    bufferSize := 8
    seed := 42 + rank  -- Different seed per rank
  }

  if isMaster then
    IO.println s!"Training for {hp.numIterations + hp.extensionIterations} iterations"
    IO.println s!"  Cooldown fraction: {hp.cooldownFrac}"
    IO.println s!"  Gradient accumulation: {hp.gradAccumSteps}"
    IO.println s!"  Validation interval: {hp.valInterval}"
    IO.println ""

  if isDistributed then
    if args.resume.isSome && isMaster then
      IO.println "Warning: --resume is currently only supported in single-process mode"
    trainDistributed cfg hp dataConfig trainDevice
  else
    -- Initialize training state
    if isMaster then IO.println "Initializing model..."
    let state ← TrainState.init cfg dataConfig isDistributed worldSize trainDevice

    -- Check for resume
    let state ← match args.resume with
      | some path =>
        if isMaster then IO.println s!"Resuming from checkpoint: {path}"
        let ckpt ← loadCheckpoint cfg path
        match ckpt with
        | some ckpt => pure { state with
            params := ckpt.params
            optState := ckpt.optState
            step := ckpt.step
            bestValLoss := ckpt.bestValLoss
          }
        | none =>
          if isMaster then IO.println "  Warning: checkpoint not found, starting fresh"
          pure state
      | none => pure state

    -- Run training
    if isMaster then
      IO.println ""
      IO.println "Starting training..."
      IO.println "=============================================================="

    let finalState ← trainLoop cfg hp state

    -- Final summary
    if isMaster then
      IO.println ""
      IO.println "=============================================================="
      IO.println "Training complete!"
      IO.println s!"  Total steps: {finalState.step}"
      IO.println s!"  Total tokens: {finalState.totalTokens}"
      IO.println s!"  Best validation loss: {finalState.bestValLoss}"

  -- Cleanup distributed
  if isDistributed then
    dist.barrier
    dist.destroyProcessGroup

  if isMaster then
    IO.println ""
    IO.println "Done!"

/-- Main entry point -/
def main (args : List String) : IO Unit := do
  -- Parse arguments
  let parsedArgs := parseArgs args

  -- Set random seed for reproducibility
  -- (would need FFI for torch.manual_seed)

  -- Run training with error handling
  try
    runTraining parsedArgs
  catch e =>
    IO.eprintln s!"Error: {e}"
    -- Cleanup if distributed
    let isInit ← dist.isInitialized
    if isInit then
      dist.destroyProcessGroup
    IO.Process.exit 1
