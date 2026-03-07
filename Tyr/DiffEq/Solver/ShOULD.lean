import Tyr.DiffEq.Solver.Base

namespace torch
namespace DiffEq

/-! ## ShOULD Solver (underdamped Langevin SRK, staged)

Staged underdamped implementation inspired by the diffrax ShOULD update.
This version uses the Brownian increment channel and omits high-order
space-time-time Levy corrections.
-/

attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hAddInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hSubInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hMulInst

structure ShOULD where
  deriving Inhabited

private structure ShOULDCoeffs where
  betaHalf : Float
  aHalf : Float
  bHalf : Float
  beta1 : Float
  a1 : Float
  b1 : Float
  aa : Float
  deriving Inhabited

private def shouldCoeffs (h gamma : Float) : ShOULDCoeffs :=
  let gh := gamma * h
  if Float.abs gamma <= 1.0e-12 || Float.abs gh <= 1.0e-6 then
    let gh2 := gh * gh
    {
      betaHalf := 1.0 - 0.5 * gh + 0.125 * gh2
      aHalf := h * (0.5 - 0.125 * gh + (1.0 / 48.0) * gh2)
      bHalf := h * (0.125 - (1.0 / 48.0) * gh + (1.0 / 384.0) * gh2)
      beta1 := 1.0 - gh + 0.5 * gh2
      a1 := h * (1.0 - 0.5 * gh + (1.0 / 6.0) * gh2)
      b1 := h * (0.5 - (1.0 / 6.0) * gh + (1.0 / 24.0) * gh2)
      aa := 1.0 - 0.5 * gh + (1.0 / 6.0) * gh2
    }
  else
    let betaHalf := Float.exp (-0.5 * gh)
    let beta1 := Float.exp (-gh)
    let aHalf := (1.0 - betaHalf) / gamma
    let a1 := (1.0 - beta1) / gamma
    let bHalf := (betaHalf + 0.5 * gh - 1.0) / (gamma * gh)
    let b1 := (beta1 + gh - 1.0) / (gamma * gh)
    {
      betaHalf := betaHalf
      aHalf := aHalf
      bHalf := bHalf
      beta1 := beta1
      a1 := a1
      b1 := b1
      aa := if Float.abs h <= 1.0e-12 then 1.0 else a1 / h
    }

def ShOULD.solver {X Args : Type}
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
  order := fun _ => 3
  strongOrder := fun _ => 3.0
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
            let gamma := drift.gamma t0 x0 v0 args
            let u := drift.u t0 x0 v0 args
            let rho := Float.sqrt (2.0 * gamma * u)
            let coeffs := shouldCoeffs h gamma
            let uh := u * h

            let f0 := drift.gradPotential t0 x0 args
            let rhoW : X := rho * dW

            let xHalf : X :=
              x0 + coeffs.aHalf * v0 + coeffs.bHalf * ((-uh) * f0 + rhoW)
            let fHalf := drift.gradPotential (t0 + 0.5 * h) xHalf args

            let fBlend : X := (1.0 / 3.0) * f0 + (2.0 / 3.0) * fHalf
            let x1 : X :=
              x0 + coeffs.a1 * v0 + coeffs.b1 * ((-uh) * fBlend + rhoW)
            let f1 := drift.gradPotential t1 x1 args

            let forceBlend : X :=
              (coeffs.beta1 / 6.0) * f0 + ((2.0 / 3.0) * coeffs.betaHalf * fHalf) + (1.0 / 6.0) * f1
            let v1 : X := (coeffs.beta1 * v0) - (uh * forceBlend) + (coeffs.aa * rhoW)

            let y1 : X × X := (x1, v1)
            let dense := { t0 := t0, t1 := t1, y0 := y0, y1 := y1 }
            {
              y1 := y1
              yError := none
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

instance : ExplicitSolver ShOULD := ⟨True.intro⟩
instance : StratonovichSolver ShOULD := ⟨True.intro⟩

end DiffEq
end torch
