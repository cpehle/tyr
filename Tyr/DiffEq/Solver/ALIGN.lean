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
  taylorThreshold : Float := 0.1
  deriving Inhabited

private structure ALIGNCoeffs where
  beta : Float
  a1 : Float
  b1 : Float
  aa : Float
  chh : Float
  deriving Inhabited

private def alignCoeffs (h gamma taylorThreshold : Float) : ALIGNCoeffs :=
  let gh := gamma * h
  if Float.abs gamma <= 1.0e-12 || Float.abs gh < taylorThreshold then
    let gh2 := gh * gh
    let gh3 := gh2 * gh
    let gh4 := gh3 * gh
    {
      beta := 1.0 - gh + 0.5 * gh2
      a1 := h * (1.0 - 0.5 * gh + gh2 / 6.0)
      b1 := h * (0.5 - gh / 6.0 + gh2 / 24.0)
      aa := 1.0 - 0.5 * gh + gh2 / 6.0
      chh := h * (1.0 - 0.5 * gh + (3.0 / 20.0) * gh2 - gh3 / 30.0 + gh4 / 168.0)
    }
  else
    let beta := Float.exp (-gh)
    let a1 := (1.0 - beta) / gamma
    let b1 := (beta + gh - 1.0) / (gamma * gh)
    let chh := 6.0 * (beta * (gh + 2.0) + gh - 2.0) / ((gh * gh) * gamma)
    {
      beta := beta
      a1 := a1
      b1 := b1
      aa := if Float.abs h <= 1.0e-12 then 1.0 else a1 / h
      chh := chh
    }

def ALIGN.solver (cfg : ALIGN := {}) {X Args : Type}
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
    let denseFail := { t0 := t0, t1 := t1, y0 := y0, y1 := y0 }
    match UnderdampedLangevinDriftTerm.validate? drift t0 y0 args with
    | some _ =>
        {
          y1 := y0
          yError := none
          denseInfo := denseFail
          solverState := state
          result := Result.internalError
        }
    | none =>
        match UnderdampedLangevinDiffusionTerm.validate? diffusion t0 y0 args with
        | some _ =>
            {
              y1 := y0
              yError := none
              denseInfo := denseFail
              solverState := state
              result := Result.internalError
            }
        | none =>
            let driftInst := (inferInstance :
              TermLike (UnderdampedLangevinDriftTerm X Args) (X × X) (X × X) Time Args)
            let diffInst := (inferInstance :
              TermLike (UnderdampedLangevinDiffusionTerm X Args) (X × X) Scalar X Args)
            let x0 := y0.1
            let v0 := y0.2
            let h := driftInst.contr drift t0 t1
            let dW := diffInst.contr diffusion t0 t1
            let zeroControl : X := 0.0 * dW
            let dHOpt := diffusion.controlH?.map (fun controlH => controlH t0 t1)
            let dH : X :=
              match dHOpt with
              | some hCtrl => hCtrl
              | none => zeroControl
            let hasHControl := dHOpt.isSome
            let gamma := drift.gamma t0 x0 v0 args
            let u := drift.u t0 x0 v0 args
            let rho := Float.sqrt (2.0 * gamma * u)
            let coeffs := alignCoeffs h gamma cfg.taylorThreshold
            let f0 := drift.gradPotential t0 x0 args

            let xDrift := coeffs.a1 * v0 - coeffs.b1 * ((u * h) * f0)
            let xDiff :=
              if hasHControl then
                rho * (coeffs.b1 * dW + coeffs.chh * dH)
              else
                rho * (coeffs.b1 * dW)
            let x1 := x0 + (xDrift + xDiff)

            let f1 := drift.gradPotential t1 x1 args
            let vDrift :=
              coeffs.beta * v0 -
                u * (((coeffs.a1 - coeffs.b1) * f0) + coeffs.b1 * f1)
            let vDiff :=
              if hasHControl then
                rho * (coeffs.aa * dW - (gamma * coeffs.chh) * dH)
              else
                rho * (coeffs.aa * dW)
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
