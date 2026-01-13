/-
  Tyr/Data/Pipeline.lean

  Multi-stage training pipeline configuration.

  Based on nanochat's 4-stage training:
  1. Tokenizer training (separate)
  2. Base pretraining on raw text
  3. Midtraining on task conversations
  4. SFT on curated instruction data
  5. (Optional) RL fine-tuning

  Each stage has different:
  - Data sources and loading strategies
  - Learning rate schedules
  - Optimizer configurations
  - Evaluation metrics
-/
import Tyr.Optim.Scheduler
import Tyr.Optim.DualOptimizer

namespace torch.Data.Pipeline

open torch.Optim.Scheduler
open torch.Optim.DualOptimizer

/-! ## Training Stage Types -/

/-- Training stage enumeration -/
inductive Stage where
  | pretraining    -- Raw text next-token prediction
  | midtraining    -- Task-based instruction tuning
  | sft            -- Supervised fine-tuning on curated data
  | rl             -- Reinforcement learning
  deriving Repr, BEq, Inhabited

/-! ## LR Schedule Configurations per Stage -/

/-- Learning rate configuration for a training stage -/
structure StageLRConfig where
  /-- Base learning rates (scaled by batch size) -/
  embeddingLr : Float := 0.3
  unembeddingLr : Float := 0.004
  matrixLr : Float := 0.02
  scalarLr : Float := 0.5
  /-- Initial LR fraction (1.0 = use full base LRs, 0.02 = start at 2%) -/
  initLrFrac : Float := 1.0
  /-- Warmup ratio (0.0-1.0, fraction of total steps) -/
  warmupRatio : Float := 0.0
  /-- Warmdown/decay ratio (fraction of steps at end for decay) -/
  warmdownRatio : Float := 0.4
  /-- Minimum LR at end of warmdown -/
  minLrFrac : Float := 0.0
  deriving Repr, Inhabited

/-- Default LR configs for each stage (from nanochat) -/
def pretrainingLRConfig : StageLRConfig := {
  embeddingLr := 0.3
  unembeddingLr := 0.004
  matrixLr := 0.02
  scalarLr := 0.5
  initLrFrac := 1.0
  warmupRatio := 0.0
  warmdownRatio := 0.4
}

def midtrainingLRConfig : StageLRConfig := {
  -- Same base LRs as pretraining
  embeddingLr := 0.3
  unembeddingLr := 0.004
  matrixLr := 0.02
  scalarLr := 0.5
  initLrFrac := 1.0
  warmupRatio := 0.0
  warmdownRatio := 0.2  -- Last 20% decay
}

def sftLRConfig : StageLRConfig := {
  embeddingLr := 0.3
  unembeddingLr := 0.004
  matrixLr := 0.02
  scalarLr := 0.5
  initLrFrac := 0.02    -- Start at 2% of base LRs
  warmupRatio := 0.0
  warmdownRatio := 1.0  -- Linear decay throughout
  minLrFrac := 0.0
}

def rlLRConfig : StageLRConfig := {
  embeddingLr := 0.3
  unembeddingLr := 0.004
  matrixLr := 0.02
  scalarLr := 0.5
  initLrFrac := 0.05    -- Start at 5% of base LRs
  warmupRatio := 0.0
  warmdownRatio := 1.0
  minLrFrac := 0.0
}

/-! ## Stage Configuration -/

/-- Full configuration for a training stage -/
structure StageConfig where
  /-- Stage type -/
  stage : Stage
  /-- Learning rate configuration -/
  lrConfig : StageLRConfig
  /-- Total training steps (or 0 for epoch-based) -/
  totalSteps : Nat := 0
  /-- Number of epochs (used if totalSteps = 0) -/
  numEpochs : Nat := 1
  /-- Batch size in tokens (for pretraining) or examples (for SFT) -/
  batchSize : Nat := 524288  -- 2^19 tokens default
  /-- Maximum sequence length -/
  maxSeqLen : Nat := 2048
  /-- Gradient accumulation steps -/
  gradAccumSteps : Nat := 1
  /-- Evaluation interval (steps) -/
  evalInterval : Nat := 100
  /-- Checkpoint interval (steps) -/
  checkpointInterval : Nat := 1000
  /-- Weight decay (base value before scheduling) -/
  weightDecay : Float := 0.01
  /-- Weight decay depth scaling factor -/
  weightDecayDepthScale : Float := 12.0
  /-- Window pattern for attention (e.g., "SSSL") -/
  windowPattern : String := ""
  deriving Repr, Inhabited

/-- Default configs for each stage -/
def pretrainingConfig : StageConfig := {
  stage := .pretraining
  lrConfig := pretrainingLRConfig
  batchSize := 524288     -- 2^19 tokens
  maxSeqLen := 2048
  evalInterval := 100
  checkpointInterval := 2000
  windowPattern := "SSSL" -- Mix of full and sliding window
}

def midtrainingConfig : StageConfig := {
  stage := .midtraining
  lrConfig := midtrainingLRConfig
  numEpochs := 1
  batchSize := 524288
  maxSeqLen := 2048
  evalInterval := 100
  checkpointInterval := 1000
}

def sftConfig : StageConfig := {
  stage := .sft
  lrConfig := sftLRConfig
  numEpochs := 1
  batchSize := 32         -- Examples, not tokens
  maxSeqLen := 2048
  evalInterval := 100
  checkpointInterval := 500
}

def rlConfig : StageConfig := {
  stage := .rl
  lrConfig := rlLRConfig
  numEpochs := 1
  batchSize := 16
  maxSeqLen := 2048
  evalInterval := 50
  checkpointInterval := 200
}

/-! ## LR Computation -/

/-- Compute batch size scaling factor.
    LRs are tuned at batch size 2^19, scale by sqrt(actual/reference). -/
def batchSizeScale (actualBatchSize : Nat) (referenceBatchSize : Nat := 524288) : Float :=
  Float.sqrt (actualBatchSize.toFloat / referenceBatchSize.toFloat)

/-- Compute weight decay scaling based on model depth.
    wd_scaled = wd * (reference_depth / actual_depth)^2
    Reference depth is 12 (where hyperparameters were tuned). -/
def weightDecayScale (modelDepth : Nat) (referenceDepth : Float := 12.0) : Float :=
  let ratio := referenceDepth / modelDepth.toFloat
  ratio * ratio

/-- Get LR multiplier for current step in a stage.
    Handles warmup and warmdown phases. -/
def getStageLRMultiplier (cfg : StageLRConfig) (step totalSteps : Nat) : Float :=
  if totalSteps == 0 then
    cfg.initLrFrac
  else
    let progress := step.toFloat / totalSteps.toFloat
    let warmupEnd := cfg.warmupRatio
    let warmdownStart := 1.0 - cfg.warmdownRatio

    if progress < warmupEnd then
      -- Warmup phase: linear increase from 0 to initLrFrac
      let warmupProgress := progress / warmupEnd
      cfg.initLrFrac * warmupProgress
    else if progress >= warmdownStart then
      -- Warmdown phase: linear decay from initLrFrac to minLrFrac
      let warmdownProgress := (progress - warmdownStart) / cfg.warmdownRatio
      let decay := 1.0 - warmdownProgress
      cfg.minLrFrac + decay * (cfg.initLrFrac - cfg.minLrFrac)
    else
      -- Plateau phase: constant at initLrFrac
      cfg.initLrFrac

/-- Compute actual learning rates for a step -/
structure StepLRs where
  embeddingLr : Float
  unembeddingLr : Float
  matrixLr : Float
  scalarLr : Float
  deriving Repr

def computeStepLRs (cfg : StageConfig) (step : Nat) (batchSize : Nat) : StepLRs :=
  let mult := getStageLRMultiplier cfg.lrConfig step cfg.totalSteps
  let scale := batchSizeScale batchSize
  {
    embeddingLr := cfg.lrConfig.embeddingLr * mult * scale
    unembeddingLr := cfg.lrConfig.unembeddingLr * mult * scale
    matrixLr := cfg.lrConfig.matrixLr * mult * scale
    scalarLr := cfg.lrConfig.scalarLr * mult * scale
  }

/-! ## Training Duration Calculation -/

/-- Ways to specify training duration -/
inductive TrainingDuration where
  /-- Fixed number of iterations -/
  | iterations (n : Nat)
  /-- Target total FLOPs -/
  | flops (totalFlops : Float)
  /-- Chinchilla-style parameter/data ratio -/
  | paramDataRatio (ratio : Float)
  /-- Number of epochs over dataset -/
  | epochs (n : Nat)
  deriving Repr

/-- Estimate FLOPs per iteration.
    Approximation: 6 * params * tokens_per_batch (forward + backward) -/
def flopsPerIteration (numParams : Nat) (tokensPerBatch : Nat) : Float :=
  6.0 * numParams.toFloat * tokensPerBatch.toFloat

/-- Calculate number of iterations from duration specification -/
def calculateIterations
    (duration : TrainingDuration)
    (numParams : Nat)
    (tokensPerBatch : Nat)
    (datasetTokens : Nat)
    : Nat :=
  match duration with
  | .iterations n => n
  | .flops totalFlops =>
    let perIter := flopsPerIteration numParams tokensPerBatch
    (totalFlops / perIter).toUInt64.toNat
  | .paramDataRatio ratio =>
    -- Total tokens = ratio * params, iterations = total / batch
    let totalTokens := ratio * numParams.toFloat
    (totalTokens / tokensPerBatch.toFloat).toUInt64.toNat
  | .epochs n =>
    -- iterations = epochs * (dataset_tokens / tokens_per_batch)
    n * (datasetTokens / tokensPerBatch)

/-! ## Pipeline State -/

/-- State for tracking progress through training pipeline -/
structure PipelineState where
  /-- Current stage -/
  currentStage : Stage
  /-- Current step within stage -/
  step : Nat
  /-- Total steps for current stage -/
  totalSteps : Nat
  /-- Global step across all stages -/
  globalStep : Nat
  /-- Current epoch -/
  epoch : Nat
  /-- Best validation metric seen so far -/
  bestMetric : Float := 1e30
  /-- Steps since last improvement -/
  stepsSinceImprovement : Nat := 0
  deriving Repr, Inhabited

def PipelineState.progress (state : PipelineState) : Float :=
  if state.totalSteps == 0 then 0.0
  else state.step.toFloat / state.totalSteps.toFloat

def PipelineState.isComplete (state : PipelineState) : Bool :=
  state.step >= state.totalSteps

def PipelineState.advance (state : PipelineState) : PipelineState :=
  { state with
    step := state.step + 1
    globalStep := state.globalStep + 1
  }

def PipelineState.updateBest (state : PipelineState) (metric : Float) : PipelineState :=
  if metric < state.bestMetric then
    { state with bestMetric := metric, stepsSinceImprovement := 0 }
  else
    { state with stepsSinceImprovement := state.stepsSinceImprovement + 1 }

/-! ## Full Pipeline Configuration -/

/-- Complete multi-stage training pipeline -/
structure Pipeline where
  /-- Ordered list of stages to run -/
  stages : Array StageConfig
  /-- Current stage index -/
  currentStageIdx : Nat := 0
  /-- Pipeline state -/
  state : PipelineState
  deriving Repr

/-- Create a standard pretraining → midtraining → SFT pipeline -/
def Pipeline.standard (pretrainSteps midtrainSteps sftSteps : Nat) : Pipeline :=
  let pretrain := { pretrainingConfig with totalSteps := pretrainSteps }
  let midtrain := { midtrainingConfig with totalSteps := midtrainSteps }
  let sft := { sftConfig with totalSteps := sftSteps }
  {
    stages := #[pretrain, midtrain, sft]
    currentStageIdx := 0
    state := {
      currentStage := .pretraining
      step := 0
      totalSteps := pretrainSteps
      globalStep := 0
      epoch := 0
    }
  }

/-- Get current stage config -/
def Pipeline.currentConfig (p : Pipeline) : Option StageConfig :=
  p.stages[p.currentStageIdx]?

/-- Advance to next stage -/
def Pipeline.nextStage (p : Pipeline) : Pipeline :=
  let nextIdx := p.currentStageIdx + 1
  if nextIdx >= p.stages.size then p
  else
    let nextConfig := p.stages[nextIdx]!
    { p with
      currentStageIdx := nextIdx
      state := {
        p.state with
        currentStage := nextConfig.stage
        step := 0
        totalSteps := nextConfig.totalSteps
        epoch := 0
      }
    }

/-- Check if pipeline is complete -/
def Pipeline.isComplete (p : Pipeline) : Bool :=
  p.currentStageIdx >= p.stages.size ||
  (p.currentStageIdx == p.stages.size - 1 && p.state.isComplete)

end torch.Data.Pipeline
