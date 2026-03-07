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

/-- Dual solve backend used inside manifold-Muon. -/
inductive SolverKind where
  | dualAscent
  | fixedPoint
  deriving Repr, BEq, Inhabited

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
  /-- Solver backend for the tangent-constrained inner problem. -/
  solver : SolverKind := .dualAscent
  /-- Minimum inner iterations before checking stopping criteria. -/
  minSolveSteps : Nat := 1
  /-- Residual tolerance for tangent-constraint satisfaction. -/
  solveResidualTol : Float := 1e-4
  /-- Dual-variable delta tolerance for solver convergence. -/
  solveDualDeltaTol : Float := 1e-5
  /-- Damping used by fixed-point updates. -/
  fixedPointDamping : Float := 0.5
  /-- Whether to run distributed gradient averaging before local step. -/
  distributed : Bool := false
  deriving Repr, Inhabited

/-- Diagnostics for the inner tangent-constrained solve. -/
structure SolveDiagnostics where
  iterations : Nat := 0
  residual : Float := 0.0
  dualDelta : Float := 0.0
  dualObjective : Float := 0.0
  converged : Bool := false
  solver : SolverKind := .dualAscent
  deriving Repr, Inhabited

/-- Result of solving for a constrained manifold update direction. -/
structure SolveResult (m n : UInt64) where
  direction : T #[m, n]
  dualVar : T #[n, n]
  diagnostics : SolveDiagnostics
  deriving Repr

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

/-- Tangent-constraint residual matrix: `WᵀA + AᵀW`. -/
def tangentResidual {m n : UInt64} (W : T #[m, n]) (A : T #[m, n]) : T #[n, n] :=
  nn.mm (nn.transpose2d W) A + nn.mm (nn.transpose2d A) W

private def solveConverged (cfg : Config) (iter residual dualDelta : Float) : Bool :=
  iter >= cfg.minSolveSteps.toFloat &&
    residual <= cfg.solveResidualTol &&
    dualDelta <= cfg.solveDualDeltaTol

/-- Dual objective estimate used for diagnostics. -/
private def dualObjectiveEstimate {m n : UInt64} (W G : T #[m, n]) (lambda : T #[n, n]) : Float :=
  let lambdaPlus := lambda + nn.transpose2d lambda
  let adjusted := G + mul_scalar (nn.mm W lambdaPlus) 2.0
  0.0 - linalg.nuclearNorm adjusted

/--
Run lightweight dual ascent and return matrix-sign direction plus updated dual var.
-/
def solveDirectionDualAscent {m n : UInt64}
    (W : T #[m, n]) (G : T #[m, n]) (dual0 : T #[n, n]) (cfg : Config)
    : IO (SolveResult m n) := do
  let mut lambda := dual0
  let mut signedDir := G
  let mut residual := 1e30
  let mut dualDelta := 1e30
  let mut iterations : Nat := 0
  let mut converged := false
  for i in [:cfg.dualAscentSteps] do
    if !converged then
      let lambdaPrev := lambda
      let lambdaPlus := lambda + nn.transpose2d lambda
      let adjusted := G + mul_scalar (nn.mm W lambdaPlus) 2.0
      signedDir ← PolarExpress.apply adjusted { numIters := cfg.numIters }
      let h := tangentResidual W signedDir
      lambda := lambda + mul_scalar h cfg.dualAscentLr
      residual := linalg.frobeniusNorm h
      dualDelta := linalg.frobeniusNorm (lambda - lambdaPrev)
      iterations := i + 1
      converged := solveConverged cfg iterations.toFloat residual dualDelta
  let diag : SolveDiagnostics := {
    iterations := iterations
    residual := residual
    dualDelta := dualDelta
    dualObjective := dualObjectiveEstimate W G lambda
    converged := converged
    solver := .dualAscent
  }
  return { direction := signedDir, dualVar := lambda, diagnostics := diag }

/--
Fixed-point solver variant for `H(Λ)=0` style updates with damping.
-/
def solveDirectionFixedPoint {m n : UInt64}
    (W : T #[m, n]) (G : T #[m, n]) (dual0 : T #[n, n]) (cfg : Config)
    : IO (SolveResult m n) := do
  let mut lambda := dual0
  let mut signedDir := G
  let mut residual := 1e30
  let mut dualDelta := 1e30
  let mut iterations : Nat := 0
  let mut converged := false
  let damp := if cfg.fixedPointDamping < 0.0 then 0.0 else if cfg.fixedPointDamping > 1.0 then 1.0 else cfg.fixedPointDamping
  for i in [:cfg.dualAscentSteps] do
    if !converged then
      let lambdaPrev := lambda
      let lambdaPlus := lambda + nn.transpose2d lambda
      let adjusted := G + mul_scalar (nn.mm W lambdaPlus) 2.0
      signedDir ← PolarExpress.apply adjusted { numIters := cfg.numIters }
      let h := tangentResidual W signedDir
      let candidate := lambda - mul_scalar h cfg.dualAscentLr
      lambda := mul_scalar lambda (1.0 - damp) + mul_scalar candidate damp
      residual := linalg.frobeniusNorm h
      dualDelta := linalg.frobeniusNorm (lambda - lambdaPrev)
      iterations := i + 1
      converged := solveConverged cfg iterations.toFloat residual dualDelta
  let diag : SolveDiagnostics := {
    iterations := iterations
    residual := residual
    dualDelta := dualDelta
    dualObjective := dualObjectiveEstimate W G lambda
    converged := converged
    solver := .fixedPoint
  }
  return { direction := signedDir, dualVar := lambda, diagnostics := diag }

/-- Solve for direction using the configured inner solver backend. -/
def solveDirection {m n : UInt64}
    (W : T #[m, n]) (G : T #[m, n]) (dual0 : T #[n, n]) (cfg : Config)
    : IO (SolveResult m n) := do
  match cfg.solver with
  | .dualAscent => solveDirectionDualAscent W G dual0 cfg
  | .fixedPoint => solveDirectionFixedPoint W G dual0 cfg

/-- Retraction onto Stiefel using reduced QR. -/
def retractStiefel {m n : UInt64} (W : T #[m, n]) : T #[m, n] :=
  (Tyr.AD.Stiefel.project m n W).matrix

/-- Single update step with inner-solver diagnostics. -/
def stepSingleWithDiagnostics {m n : UInt64}
    (param : T #[m, n])
    (grad : T #[m, n])
    (state : ParamState m n)
    (cfg : Config)
    (lrMul : Float := 1.0)
    : IO (T #[m, n] × ParamState m n × SolveDiagnostics) := do
  let gRaw := autograd.detach grad

  -- Nesterov-style blend (aligned with NorMuon's momentum semantics).
  let newMomentum := NorMuon.updateMomentum gRaw state.momentumBuffer cfg.momentum
  let nesterovGrad := mul_scalar gRaw (1.0 - cfg.momentum) + mul_scalar newMomentum cfg.momentum

  -- Optional dual-ascent solve in ambient space.
  let dual0 := state.dualVar.getD (zeros #[n, n])
  let (signedDir, dualVar', solveDiag) ←
    if cfg.dualAscentSteps == 0 then
      let signed ← PolarExpress.apply nesterovGrad { numIters := cfg.numIters }
      let diag : SolveDiagnostics := {
        iterations := 0
        residual := linalg.frobeniusNorm (tangentResidual param signed)
        dualDelta := 0.0
        dualObjective := dualObjectiveEstimate param nesterovGrad dual0
        converged := true
        solver := cfg.solver
      }
      pure (signed, dual0, diag)
    else
      let solveRes ← solveDirection param nesterovGrad dual0 cfg
      pure (solveRes.direction, solveRes.dualVar, solveRes.diagnostics)

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
  return (newParam, newState, solveDiag)

/-- Single local update step for a Stiefel-constrained matrix parameter. -/
def stepSingle {m n : UInt64}
    (param : T #[m, n])
    (grad : T #[m, n])
    (state : ParamState m n)
    (cfg : Config)
    (lrMul : Float := 1.0)
    : IO (T #[m, n] × ParamState m n) := do
  let (param', state', _) ← stepSingleWithDiagnostics param grad state cfg lrMul
  return (param', state')

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
