/-
  Tyr/Optim/NorMuon.lean

  NorMuon Optimizer implementation for Tyr.

  NorMuon is an advanced optimizer combining:
  - Muon's orthogonalized momentum (via Polar Express)
  - Low-rank variance reduction (Adafactor-style)
  - Cautious weight decay (only when gradient and param have same sign)
  - Distributed training with reduce-scatter/all-gather

  Based on modded-nanogpt's NorMuon implementation.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Distributed
import Tyr.Optim.PolarExpress

namespace torch.Optim.NorMuon

open torch

/-- Parameter labels for grouping parameters with different update rules -/
inductive ParamLabel where
  | smearGate   : ParamLabel  -- Smear gate parameters
  | attnGate    : ParamLabel  -- Attention gate parameters
  | attn        : ParamLabel  -- Attention weight matrices
  | mlp         : ParamLabel  -- MLP weight matrices
  | embed       : ParamLabel  -- Token embeddings
  | lmHead      : ParamLabel  -- Language model head
  | scalars     : ParamLabel  -- Learnable scalar parameters
  | valueEmbed  : ParamLabel  -- Value embeddings
  deriving Repr, BEq, Hashable, Inhabited

/-- Labeled parameter with metadata for optimization -/
structure LabeledParam (s : Shape) where
  /-- The parameter tensor -/
  tensor : T s
  /-- Parameter label for grouping -/
  label : ParamLabel
  /-- Learning rate multiplier (1.0 = base LR) -/
  lrMul : Float := 1.0
  /-- Weight decay multiplier (1.0 = base WD, 0.0 = no WD) -/
  wdMul : Float := 1.0
  deriving Repr

/-- NorMuon configuration -/
structure Config where
  /-- Base learning rate -/
  lr : Float := 0.023
  /-- Weight decay -/
  weightDecay : Float := 1.2
  /-- Momentum coefficient -/
  momentum : Float := 0.95
  /-- Second moment coefficient for variance reduction -/
  beta2 : Float := 0.95
  /-- Number of Newton-Schulz iterations for Polar Express -/
  numIters : UInt64 := 5
  /-- Whether to use distributed training -/
  distributed : Bool := false
  /-- World size for distributed (set automatically if distributed=true) -/
  worldSize : UInt64 := 1
  deriving Repr

instance : Inhabited Config where
  default := { lr := 0.023, weightDecay := 1.2, momentum := 0.95, beta2 := 0.95,
               numIters := 5, distributed := false, worldSize := 1 }

/-- State for a single parameter in NorMuon -/
structure ParamState (s : Shape) where
  /-- Momentum buffer (EMA of orthogonalized gradients) -/
  momentumBuffer : Option (T s) := none
  /-- Second moment for variance reduction (low-rank approximation) -/
  secondMoment : Option (T #[]) := none  -- Reduced dimension
  /-- Step count for bias correction -/
  step : Nat := 0
  deriving Repr

instance {s : Shape} : TensorStruct (ParamState s) where
  map f ps := { ps with
    momentumBuffer := ps.momentumBuffer.map f,
    secondMoment := ps.secondMoment.map f
  }
  mapM f ps := do
    let momentumBuffer ← match ps.momentumBuffer with
      | some t => some <$> f t
      | none => pure none
    let secondMoment ← match ps.secondMoment with
      | some t => some <$> f t
      | none => pure none
    return { ps with momentumBuffer, secondMoment }
  zipWith f ps1 ps2 := {
    momentumBuffer := match ps1.momentumBuffer, ps2.momentumBuffer with
      | some t1, some t2 => some (f t1 t2)
      | _, _ => none
    secondMoment := match ps1.secondMoment, ps2.secondMoment with
      | some t1, some t2 => some (f t1 t2)
      | _, _ => none
    step := ps1.step
  }
  fold f acc ps :=
    let acc := match ps.momentumBuffer with
      | some t => f t acc
      | none => acc
    match ps.secondMoment with
      | some t => f t acc
      | none => acc

/-- Full NorMuon optimizer state -/
structure State where
  /-- Configuration -/
  config : Config
  /-- Per-parameter states stored by a hash of their identity -/
  paramStates : Array (ParamState #[])  -- Type-erased states
  /-- Global step count -/
  globalStep : Nat := 0
  deriving Repr

/-- Initialize NorMuon state from configuration -/
def State.init (cfg : Config) : IO State := do
  return {
    config := cfg
    paramStates := #[]
    globalStep := 0
  }

/-- Apply variance reduction to orthogonalized gradient.

    Uses low-rank second moment estimation (Adafactor-style):
    v_t = beta2 * v_{t-1} + (1 - beta2) * mean(g^2, dim=reduce_dim)
    g_normalized = g / sqrt(v_t + eps)

    This reduces variance of the update while maintaining direction.
-/
def applyVarianceReduction {s : Shape} (grad : T s) (secondMoment : Option (T #[]))
    (beta2 : Float) (step : Nat) : IO (T s × T #[]) := do
  let eps : Float := 1e-8

  -- Compute mean of squared gradients along last dimension
  let gradSquared := grad * grad
  let newSecondMoment := nn.meanAll gradSquared  -- Simplified: mean over all

  -- Update EMA of second moment
  let updatedSecondMoment ← match secondMoment with
    | some sm =>
      let b2 := mul_scalar sm beta2
      let g2 := mul_scalar newSecondMoment (1.0 - beta2)
      pure (b2 + g2)
    | none => pure newSecondMoment

  -- Bias correction
  let biasCorrection := 1.0 - Float.pow beta2 (step + 1).toFloat
  let correctedMoment := div_scalar updatedSecondMoment biasCorrection

  -- Normalize gradient by sqrt(second moment)
  let scale := 1.0 / Float.sqrt (nn.item correctedMoment + eps)
  let normalizedGrad := mul_scalar grad scale

  return (normalizedGrad, updatedSecondMoment)

/-- Apply cautious weight decay.

    Only applies weight decay when gradient and parameter have the same sign:
    mask = sign(grad) == sign(param)
    update = grad + wd * mask * param

    This prevents weight decay from fighting the gradient.
-/
def applyCautiousWeightDecay {s : Shape} (param grad : T s) (wd : Float) : IO (T s) := do
  if wd == 0.0 then
    return grad
  else
    dist.cautiousUpdate param grad 1.0 wd

/-- Update momentum buffer with orthogonalized gradient.

    momentum_t = beta * momentum_{t-1} + (1 - beta) * g_orth
-/
def updateMomentum {s : Shape} (orthGrad : T s) (momentumBuffer : Option (T s))
    (momentum : Float) : T s :=
  match momentumBuffer with
  | some buf =>
    let decayed := mul_scalar buf momentum
    let scaled := mul_scalar orthGrad (1.0 - momentum)
    decayed + scaled
  | none => orthGrad

/-- Single NorMuon update step for one parameter.

    1. Orthogonalize gradient via Polar Express
    2. Apply variance reduction
    3. Apply cautious weight decay
    4. Update momentum buffer
    5. Apply update to parameter

    Returns: (new_param, new_state)
-/
def stepSingle {s : Shape} (param : T s) (grad : T s) (state : ParamState s)
    (cfg : Config) (lrMul wdMul : Float) : IO (T s × ParamState s) := do
  -- 1. Orthogonalize gradient
  let orthGrad ← PolarExpress.muonOrthogonalize grad cfg.numIters

  -- 2. Apply variance reduction
  let (normalizedGrad, newSecondMoment) ←
    applyVarianceReduction orthGrad state.secondMoment cfg.beta2 state.step

  -- 3. Apply cautious weight decay
  let effectiveWd := cfg.weightDecay * wdMul
  let gradWithWd ← applyCautiousWeightDecay param normalizedGrad effectiveWd

  -- 4. Update momentum
  let newMomentum := updateMomentum gradWithWd state.momentumBuffer cfg.momentum

  -- 5. Apply update
  let effectiveLr := cfg.lr * lrMul
  let update := mul_scalar newMomentum effectiveLr
  let newParam := param - update
  let newParam := autograd.set_requires_grad (autograd.detach newParam) true

  let newState : ParamState s := {
    momentumBuffer := some newMomentum
    secondMoment := some newSecondMoment
    step := state.step + 1
  }

  return (newParam, newState)

/-- Step function for parameters using AdamW-like updates (non-orthogonalized).

    Used for embeddings, scalars, and other non-matrix parameters where
    orthogonalization doesn't apply.
-/
def stepAdamLike {s : Shape} (param : T s) (grad : T s) (state : ParamState s)
    (cfg : Config) (lrMul wdMul : Float) : IO (T s × ParamState s) := do
  -- Simple SGD with momentum for non-matrix params
  let effectiveWd := cfg.weightDecay * wdMul
  let gradWithWd := if effectiveWd > 0 then
      grad + mul_scalar param effectiveWd
    else
      grad

  -- Update momentum
  let newMomentum := updateMomentum gradWithWd state.momentumBuffer cfg.momentum

  -- Apply update
  let effectiveLr := cfg.lr * lrMul
  let update := mul_scalar newMomentum effectiveLr
  let newParam := param - update
  let newParam := autograd.set_requires_grad (autograd.detach newParam) true

  let newState : ParamState s := {
    momentumBuffer := some newMomentum
    secondMoment := state.secondMoment
    step := state.step + 1
  }

  return (newParam, newState)

/-- Determine if a parameter should use orthogonalized updates.

    Matrix parameters (attention, MLP weights) use orthogonalization.
    Embeddings, scalars, and biases use standard momentum updates.
-/
def shouldOrthogonalize (label : ParamLabel) : Bool :=
  match label with
  | .smearGate => true
  | .attnGate => true
  | .attn => true
  | .mlp => true
  | .embed => false
  | .lmHead => false
  | .scalars => false
  | .valueEmbed => false

/-- Get default learning rate multiplier for a parameter label.

    Based on modded-nanogpt:
    - Embeddings: 75x (need larger updates)
    - Scalars: 5x
    - Others: 1x
-/
def defaultLrMul (label : ParamLabel) : Float :=
  match label with
  | .embed => 75.0
  | .valueEmbed => 75.0
  | .lmHead => 1.0
  | .scalars => 5.0
  | _ => 1.0

/-- Get default weight decay multiplier for a parameter label.

    Based on modded-nanogpt:
    - Scalars: 0x (no weight decay)
    - Others: 1x
-/
def defaultWdMul (label : ParamLabel) : Float :=
  match label with
  | .scalars => 0.0
  | _ => 1.0

/-- Initialize state for a parameter -/
def initParamState {s : Shape} (_param : T s) : ParamState s := {
  momentumBuffer := none
  secondMoment := none
  step := 0
}

/-- Warmup/cooldown schedule for momentum.

    modded-nanogpt uses:
    - Warmup: 300 steps (0.85 -> 0.95)
    - Cooldown: 50 steps before end (0.95 -> 0.85)
-/
def getMomentum (step totalSteps : Nat) (baseMomentum : Float := 0.95)
    (warmupSteps : Nat := 300) (cooldownSteps : Nat := 50) : Float :=
  let minMomentum := 0.85
  if step < warmupSteps then
    -- Linear warmup
    let progress := step.toFloat / warmupSteps.toFloat
    minMomentum + progress * (baseMomentum - minMomentum)
  else if step > totalSteps - cooldownSteps then
    -- Linear cooldown
    let stepsFromEnd := totalSteps - step
    let progress := stepsFromEnd.toFloat / cooldownSteps.toFloat
    minMomentum + progress * (baseMomentum - minMomentum)
  else
    baseMomentum

end torch.Optim.NorMuon
