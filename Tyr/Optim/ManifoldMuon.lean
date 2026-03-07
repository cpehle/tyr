/-
  Tyr/Optim/ManifoldMuon.lean

  Experimental Stiefel manifold variant of Muon.

  Design goals:
  - Keep matrix parameters on/near Stiefel via tangent projection + QR retraction.
  - Support an optional lightweight dual-ascent loop for tangent-constrained
    spectral-style updates.
  - Keep API shape close to NorMuon for easy experimentation.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Distributed
import Tyr.Optim.PolarExpress
import Tyr.Optim.NorMuon
import Tyr.Manifolds.Optimizer

namespace torch.Optim.ManifoldMuon

open torch

/-- Manifold-Muon configuration (experimental). -/
structure Config where
  /-- Base learning rate. -/
  lr : Float := 0.02
  /-- Momentum coefficient for Nesterov-style blending. -/
  momentum : Float := 0.95
  /-- Newton-Schulz iterations used by matrix-sign approximation. -/
  numIters : UInt64 := 5
  /-- Number of dual-ascent iterations for tangent-constrained solve. -/
  dualAscentSteps : Nat := 2
  /-- Step size for dual-ascent variable update. -/
  dualAscentLr : Float := 0.1
  /-- Whether to run distributed gradient averaging before local step. -/
  distributed : Bool := false
  deriving Repr, Inhabited

/-- Per-parameter state for Stiefel Manifold-Muon. -/
structure ParamState (m n : UInt64) where
  /-- Momentum buffer in ambient space. -/
  momentumBuffer : Option (T #[m, n]) := none
  /-- Dual variable for tangent-constrained solve. -/
  dualVar : Option (T #[n, n]) := none
  /-- Step counter. -/
  step : Nat := 0
  deriving Repr

/-- Initialize per-parameter state. -/
def initParamState {m n : UInt64} (_param : T #[m, n]) : ParamState m n := {
  momentumBuffer := none
  dualVar := none
  step := 0
}

instance {m n : UInt64} : TensorStruct (ParamState m n) where
  map f s := {
    momentumBuffer := s.momentumBuffer.map f
    dualVar := s.dualVar.map f
    step := s.step
  }
  mapM f s := do
    let momentumBuffer ← match s.momentumBuffer with
      | some t => some <$> f t
      | none => pure none
    let dualVar ← match s.dualVar with
      | some t => some <$> f t
      | none => pure none
    pure { s with momentumBuffer, dualVar }
  zipWith f a b := {
    momentumBuffer := match a.momentumBuffer, b.momentumBuffer with
      | some t1, some t2 => some (f t1 t2)
      | _, _ => none
    dualVar := match a.dualVar, b.dualVar with
      | some t1, some t2 => some (f t1 t2)
      | _, _ => none
    step := a.step
  }
  fold f init s :=
    let acc := match s.momentumBuffer with
      | some t => f t init
      | none => init
    match s.dualVar with
    | some t => f t acc
    | none => acc

/-- Symmetric part of a square matrix. -/
def sym {n : UInt64} (A : T #[n, n]) : T #[n, n] :=
  mul_scalar (A + nn.transpose2d A) 0.5

/-- Project an ambient matrix to the Stiefel tangent space at W. -/
def tangentProject {m n : UInt64} (W : T #[m, n]) (G : T #[m, n]) : T #[m, n] :=
  let XtG := nn.mm (nn.transpose2d W) G
  let correction := nn.mm W (sym XtG)
  G - correction

/--
Run lightweight dual ascent and return matrix-sign direction plus updated dual var.
-/
def solveDirectionDualAscent {m n : UInt64}
    (W : T #[m, n]) (G : T #[m, n]) (dual0 : T #[n, n]) (cfg : Config)
    : IO (T #[m, n] × T #[n, n]) := do
  let mut lambda := dual0
  let mut signedDir := G
  for _ in [:cfg.dualAscentSteps] do
    let lambdaPlus := lambda + nn.transpose2d lambda
    let adjusted := G + mul_scalar (nn.mm W lambdaPlus) 2.0
    signedDir ← PolarExpress.apply adjusted { numIters := cfg.numIters }
    let h := nn.mm (nn.transpose2d W) signedDir + nn.mm (nn.transpose2d signedDir) W
    lambda := lambda + mul_scalar h cfg.dualAscentLr
  return (signedDir, lambda)

/-- Retraction onto Stiefel using reduced QR. -/
def retractStiefel {m n : UInt64} (W : T #[m, n]) : T #[m, n] :=
  (Tyr.AD.Stiefel.project m n W).matrix

/-- Single local update step for a Stiefel-constrained matrix parameter. -/
def stepSingle {m n : UInt64}
    (param : T #[m, n])
    (grad : T #[m, n])
    (state : ParamState m n)
    (cfg : Config)
    (lrMul : Float := 1.0)
    : IO (T #[m, n] × ParamState m n) := do
  let gRaw := autograd.detach grad

  -- Nesterov-style blend (aligned with NorMuon's momentum semantics).
  let newMomentum := NorMuon.updateMomentum gRaw state.momentumBuffer cfg.momentum
  let nesterovGrad := mul_scalar gRaw (1.0 - cfg.momentum) + mul_scalar newMomentum cfg.momentum

  -- Optional dual-ascent solve in ambient space.
  let dual0 := state.dualVar.getD (zeros #[n, n])
  let (signedDir, dualVar') ←
    if cfg.dualAscentSteps == 0 then
      let signed ← PolarExpress.apply nesterovGrad { numIters := cfg.numIters }
      pure (signed, dual0)
    else
      solveDirectionDualAscent param nesterovGrad dual0 cfg

  -- Enforce tangent constraint explicitly.
  let tangentDir := tangentProject param signedDir

  -- Keep step magnitude predictable.
  let fnorm := linalg.frobeniusNorm tangentDir
  let tangentUnit :=
    if fnorm == 0.0 then
      tangentDir
    else
      mul_scalar tangentDir (1.0 / fnorm)

  let aspectScale := NorMuon.aspectRatioScale #[m, n]
  let effectiveLr := cfg.lr * lrMul * aspectScale

  -- Ambient update + Stiefel retraction.
  let updated := param - mul_scalar tangentUnit effectiveLr
  let retracted := retractStiefel updated
  let newParam := autograd.set_requires_grad (autograd.detach retracted) true

  let newState : ParamState m n := {
    momentumBuffer := some newMomentum
    dualVar := some dualVar'
    step := state.step + 1
  }
  return (newParam, newState)

/-- Local group update for homogeneous matrix shapes. -/
def stepGroupLocal {m n : UInt64}
    (params : Array (T #[m, n]))
    (grads : Array (T #[m, n]))
    (states : Array (ParamState m n))
    (cfg : Config)
    (lrMul : Float := 1.0)
    : IO (Array (T #[m, n]) × Array (ParamState m n)) := do
  let mut outParams : Array (T #[m, n]) := #[]
  let mut outStates : Array (ParamState m n) := #[]
  for i in [:params.size] do
    let p := params[i]!
    let g := grads[i]?.getD (zeros_like p)
    let st := states[i]?.getD (initParamState p)
    let (p', st') ← stepSingle p g st cfg lrMul
    outParams := outParams.push p'
    outStates := outStates.push st'
  return (outParams, outStates)

/--
Distributed group update (prototype): all-reduce gradients then local manifold step.
-/
def stepDistributedGroup {m n : UInt64}
    (params : Array (T #[m, n]))
    (grads : Array (T #[m, n]))
    (states : Array (ParamState m n))
    (cfg : Config)
    (lrMul : Float := 1.0)
    : IO (Array (T #[m, n]) × Array (ParamState m n)) := do
  if params.isEmpty then
    return (params, states)

  let isDist ← dist.isInitialized
  if !cfg.distributed || !isDist then
    return (← stepGroupLocal params grads states cfg lrMul)

  let worldSize ← dist.getWorldSize
  if worldSize <= 1 then
    return (← stepGroupLocal params grads states cfg lrMul)

  let mut reducedGrads : Array (T #[m, n]) := #[]
  for i in [:params.size] do
    let p := params[i]!
    let g := autograd.detach (grads[i]?.getD (zeros_like p))
    dist.allReduce g .avg
    reducedGrads := reducedGrads.push g

  stepGroupLocal params reducedGrads states cfg lrMul

/-- Measure average local step latency for one parameter (benchmark helper). -/
def benchmarkLocalStep {m n : UInt64}
    (numSteps : Nat := 5)
    (cfg : Config := {})
    : IO Float := do
  let p0 ← randn #[m, n] false
  let p0 := autograd.set_requires_grad p0 true
  let mut p := retractStiefel p0
  let mut st := initParamState p

  let t0 ← IO.monoMsNow
  for _ in [:numSteps] do
    let g ← randn #[m, n] false
    let (p', st') ← stepSingle p g st cfg
    p := p'
    st := st'
  let t1 ← IO.monoMsNow

  let elapsed := (t1 - t0).toFloat
  if numSteps == 0 then
    return elapsed
  else
    return elapsed / numSteps.toFloat

end torch.Optim.ManifoldMuon
