import Tyr.DiffEq.Solver.Base

namespace torch
namespace DiffEq

/-! ## ALIGN Solver (underdamped Langevin SRK, adaptive)

Solver-faithful staged implementation for
`MultiTerm(UnderdampedLangevinDriftTerm, UnderdampedLangevinDiffusionTerm)`,
including a local velocity error estimate.
-/

attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hAddInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hSubInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hMulInst

structure ALIGN where
  deriving Inhabited

private structure ALIGNCoeffs where
  beta : Float
  a1 : Float
  b1 : Float
  aa : Float
  deriving Inhabited

private def alignCoeffs (h gamma : Float) : ALIGNCoeffs :=
  let gh := gamma * h
  if Float.abs gh < 1.0e-4 then
    let gh2 := gh * gh
    {
      beta := 1.0 - gh + 0.5 * gh2
      a1 := h * (1.0 - 0.5 * gh + gh2 / 6.0)
      b1 := h * (0.5 - gh / 6.0 + gh2 / 24.0)
      aa := 1.0 - 0.5 * gh + gh2 / 6.0
    }
  else
    let beta := Float.exp (-gh)
    let a1 := (1.0 - beta) / gamma
    let b1 := (beta + gh - 1.0) / (gamma * gh)
    {
      beta := beta
      a1 := a1
      b1 := b1
      aa := a1 / h
    }

def ALIGN.solver {X Args : Type}
    [DiffEqSpace X] :
    AbstractSolver
      (MultiTerm (UnderdampedLangevinDriftTerm X Args) (UnderdampedLangevinDiffusionTerm X Args))
      (X × X)
      ((X × X) × Scalar)
      (Time × X)
      Args := {
  SolverState := Unit
  DenseInfo := LocalLinearDenseInfo (X × X)
  termStructure := TermStructure.multi
  order := fun _ => 2
  strongOrder := fun _ => 2.0
  init := fun _ _ _ _ _ => ()
  step := fun terms t0 t1 y0 args state _madeJump =>
    let drift := terms.term1
    let diffusion := terms.term2
    let driftInst := (inferInstance :
      TermLike (UnderdampedLangevinDriftTerm X Args) (X × X) (X × X) Time Args)
    let diffInst := (inferInstance :
      TermLike (UnderdampedLangevinDiffusionTerm X Args) (X × X) Scalar X Args)
    -- Preserve runtime validation performed by the term instances.
    let _ := driftInst.vf drift t0 y0 args
    let _ := diffInst.vf diffusion t0 y0 args

    let x0 := y0.1
    let v0 := y0.2
    let h := driftInst.contr drift t0 t1
    let dW := diffInst.contr diffusion t0 t1
    let gamma := drift.gamma t0 x0 v0 args
    let u := drift.u t0 x0 v0 args
    let rho := Float.sqrt (2.0 * gamma * u)
    let coeffs := alignCoeffs h gamma
    let f0 := drift.gradPotential t0 x0 args

    let xDrift := coeffs.a1 * v0 - coeffs.b1 * ((u * h) * f0)
    let xDiff := (rho * coeffs.b1) * dW
    let x1 := x0 + (xDrift + xDiff)

    let f1 := drift.gradPotential t1 x1 args
    let vDrift :=
      coeffs.beta * v0 -
        u * (((coeffs.a1 - coeffs.b1) * f0) + coeffs.b1 * f1)
    let vDiff := (rho * coeffs.aa) * dW
    let v1 := vDrift + vDiff

    let y1 : X × X := (x1, v1)
    let yErr : X × X := (0.0 * x0, (-u * coeffs.b1) * (f1 - f0))
    let dense := { t0 := t0, t1 := t1, y0 := y0, y1 := y1 }
    {
      y1 := y1
      yError := some yErr
      denseInfo := dense
      solverState := state
      result := Result.successful
    }
  func := fun terms t y args =>
    let driftInst := (inferInstance :
      TermLike (UnderdampedLangevinDriftTerm X Args) (X × X) (X × X) Time Args)
    let diffInst := (inferInstance :
      TermLike (UnderdampedLangevinDiffusionTerm X Args) (X × X) Scalar X Args)
    (driftInst.vf terms.term1 t y args, diffInst.vf terms.term2 t y args)
  interpolation := fun info => info.toInterpolation
}

instance : ExplicitSolver ALIGN := ⟨True.intro⟩
instance : StratonovichSolver ALIGN := ⟨True.intro⟩
instance : AdaptiveSolver ALIGN := ⟨True.intro⟩

end DiffEq
end torch
