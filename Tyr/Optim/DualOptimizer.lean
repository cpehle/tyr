/-
  Tyr/Optim/DualOptimizer.lean

  Dual Optimizer Setup following nanochat's approach:
  - Muon (with Polar Express) for weight matrices (attention, MLP)
  - AdamW for embeddings, unembeddings, and scalar parameters

  This strategy reportedly improves training efficiency by 2x+ over pure AdamW.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Optim.NorMuon
import Tyr.Optim.DistAdam
import Tyr.Optim.Scheduler

/-!
# `Tyr.Optim.DualOptimizer`

Optimizer submodule for Dual Optimizer, used in training-time parameter updates.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace torch.Optim.DualOptimizer

open torch
open NorMuon (ParamLabel)

/-- Configuration for dual optimizer setup -/
structure Config where
  /-- Base learning rate for matrix parameters (Muon) -/
  matrixLr : Float := 0.02
  /-- Learning rate for token embeddings (AdamW) -/
  embeddingLr : Float := 0.3
  /-- Learning rate for language model head (AdamW) -/
  lmHeadLr : Float := 0.3
  /-- Learning rate for scalar parameters (AdamW) -/
  scalarLr : Float := 0.02
  /-- Muon momentum coefficient -/
  muonMomentum : Float := 0.95
  /-- Muon weight decay -/
  muonWeightDecay : Float := 0.01
  /-- AdamW beta1 -/
  adamBeta1 : Float := 0.9
  /-- AdamW beta2 -/
  adamBeta2 : Float := 0.95
  /-- AdamW epsilon -/
  adamEps : Float := 1e-8
  /-- Model dimension (for LR scaling) -/
  modelDim : UInt64 := 768
  /-- Reference dimension for LR scaling (typically 768 for GPT-2 base) -/
  refDim : UInt64 := 768
  /-- Total training steps (for scheduling) -/
  totalSteps : Nat := 10000
  /-- Warmup steps for momentum -/
  warmupSteps : Nat := 300
  deriving Repr, Inhabited

/-- Compute dimension-based learning rate scaling.
    scale = (modelDim / refDim)^(-0.5)
    Larger models get smaller learning rates. -/
def dimLrScale (cfg : Config) : Float :=
  Float.pow (cfg.modelDim.toFloat / cfg.refDim.toFloat) (-0.5)

/-- Get Muon config from dual optimizer config -/
def toMuonConfig (cfg : Config) : NorMuon.Config := {
  lr := cfg.matrixLr
  weightDecay := cfg.muonWeightDecay
  momentum := cfg.muonMomentum
  beta2 := 0.95
  numIters := 5
  distributed := false
  worldSize := 1
}

/-- Get AdamW config from dual optimizer config -/
def toAdamConfig (cfg : Config) : DistAdam.Config := {
  lr := cfg.embeddingLr * dimLrScale cfg
  beta1 := cfg.adamBeta1
  beta2 := cfg.adamBeta2
  eps := cfg.adamEps
  weightDecay := 0.0  -- AdamW typically has no weight decay in nanochat
}

/-- Parameter group classification for dual optimizer -/
inductive ParamGroup where
  | matrix     : ParamGroup  -- Use Muon (attention, MLP weights)
  | embedding  : ParamGroup  -- Use AdamW (token embeddings)
  | lmHead     : ParamGroup  -- Use AdamW (output layer)
  | scalar     : ParamGroup  -- Use AdamW (per-layer scalars)
  deriving Repr, BEq, Inhabited

/-- Map NorMuon's ParamLabel to ParamGroup -/
def labelToGroup (label : ParamLabel) : ParamGroup :=
  match label with
  | .attn => .matrix
  | .mlp => .matrix
  | .smearGate => .matrix
  | .attnGate => .matrix
  | .embed => .embedding
  | .valueEmbed => .embedding
  | .lmHead => .lmHead
  | .scalars => .scalar

/-- Get learning rate multiplier for a parameter group -/
def groupLrMul (cfg : Config) (group : ParamGroup) : Float :=
  let scale := dimLrScale cfg
  match group with
  | .matrix => 1.0  -- Base LR for Muon
  | .embedding => cfg.embeddingLr / cfg.matrixLr * scale
  | .lmHead => cfg.lmHeadLr / cfg.matrixLr * scale
  | .scalar => cfg.scalarLr / cfg.matrixLr

/-- Get weight decay multiplier for a parameter group -/
def groupWdMul (_cfg : Config) (group : ParamGroup) : Float :=
  match group with
  | .matrix => 1.0
  | .embedding => 0.0  -- No weight decay on embeddings
  | .lmHead => 0.0     -- No weight decay on lm_head
  | .scalar => 0.0     -- No weight decay on scalars

/-- Determine if a parameter group should use orthogonalized updates (Muon) -/
def groupUsesMuon (group : ParamGroup) : Bool :=
  match group with
  | .matrix => true
  | _ => false

/-- Get scheduled momentum for a given step (with warmup/cooldown) -/
def getScheduledMomentum (cfg : Config) (step : Nat) : Float :=
  NorMuon.getMomentum step cfg.totalSteps cfg.muonMomentum cfg.warmupSteps 50

/-- Get scheduled weight decay for a given step (linear decay to zero) -/
def getScheduledWeightDecay (cfg : Config) (step : Nat) : Float :=
  Scheduler.linearWeightDecay { baseWd := cfg.muonWeightDecay, totalSteps := cfg.totalSteps } step

/-- Simple training step tracker -/
structure TrainingStep where
  /-- Current step -/
  step : Nat := 0
  deriving Repr, Inhabited

/-- Initialize training step tracker -/
def TrainingStep.init : TrainingStep := { step := 0 }

/-- Increment step counter -/
def TrainingStep.next (ts : TrainingStep) : TrainingStep :=
  { ts with step := ts.step + 1 }

/-- Get current Muon config with scheduled values -/
def getMuonConfigAtStep (cfg : Config) (step : Nat) : NorMuon.Config :=
  let momentum := getScheduledMomentum cfg step
  let weightDecay := getScheduledWeightDecay cfg step
  { toMuonConfig cfg with momentum, weightDecay }

/-- Initialize Muon optimizer state -/
def initMuonState (cfg : Config) : IO NorMuon.State :=
  NorMuon.State.init (toMuonConfig cfg)

end torch.Optim.DualOptimizer
