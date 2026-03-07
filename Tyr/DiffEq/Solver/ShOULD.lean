import Tyr.DiffEq.Solver.Base

namespace torch
namespace DiffEq

/-! ## ShOULD Solver (underdamped Langevin SRK, staged)

Staged underdamped implementation inspired by the diffrax ShOULD update.
Includes explicit handling for richer Brownian controls (`W/H/K`) when
available, with `W`-only fallback behavior.
-/

attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hAddInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hSubInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hMulInst

structure ShOULD where
  taylorThreshold : Float := 0.1
  deriving Inhabited

private structure ShOULDCoeffs where
  betaHalf : Float
  aHalf : Float
  bHalf : Float
  beta1 : Float
  a1 : Float
  b1 : Float
  aa : Float
  chh : Float
  ckk : Float
  deriving Inhabited

private def shouldCoeffs (h gamma taylorThreshold : Float) : ShOULDCoeffs :=
  let gh := gamma * h
  if Float.abs gamma <= 1.0e-12 || Float.abs gh < taylorThreshold then
    let gh2 := gh * gh
    let gh3 := gh2 * gh
    let gh4 := gh3 * gh
    {
      betaHalf := 1.0 - 0.5 * gh + 0.125 * gh2
      aHalf := h * (0.5 - 0.125 * gh + (1.0 / 48.0) * gh2)
      bHalf := h * (0.125 - (1.0 / 48.0) * gh + (1.0 / 384.0) * gh2)
      beta1 := 1.0 - gh + 0.5 * gh2
      a1 := h * (1.0 - 0.5 * gh + (1.0 / 6.0) * gh2)
      b1 := h * (0.5 - (1.0 / 6.0) * gh + (1.0 / 24.0) * gh2)
      aa := 1.0 - 0.5 * gh + (1.0 / 6.0) * gh2
      chh := h * (1.0 - 0.5 * gh + (3.0 / 20.0) * gh2 - gh3 / 30.0 + gh4 / 168.0)
      ckk := (gamma * h * h) * (-1.0 + 0.5 * gh - gh2 / 7.0 + (5.0 / 168.0) * gh3)
    }
  else
    let betaHalf := Float.exp (-0.5 * gh)
    let beta1 := Float.exp (-gh)
    let aHalf := (1.0 - betaHalf) / gamma
    let a1 := (1.0 - beta1) / gamma
    let bHalf := (betaHalf + 0.5 * gh - 1.0) / (gamma * gh)
    let b1 := (beta1 + gh - 1.0) / (gamma * gh)
    let chh := 6.0 * (beta1 * (gh + 2.0) + gh - 2.0) / ((gh * gh) * gamma)
    let ckk :=
      60.0 * (beta1 * (gh * (gh + 6.0) + 12.0) - gh * (gh - 6.0) - 12.0) /
        ((gh * gh * gh) * gamma)
    {
      betaHalf := betaHalf
      aHalf := aHalf
      bHalf := bHalf
      beta1 := beta1
      a1 := a1
      b1 := b1
      aa := if Float.abs h <= 1.0e-12 then 1.0 else a1 / h
      chh := chh
      ckk := ckk
    }

def ShOULD.solver (cfg : ShOULD := {}) {X Args : Type}
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
            let zeroControl : X := 0.0 * dW
            let dHOpt := diffusion.controlH?.map (fun controlH => controlH t0 t1)
            let dKOpt := diffusion.controlK?.map (fun controlK => controlK t0 t1)
            let dH : X :=
              match dHOpt with
              | some hCtrl => hCtrl
              | none => zeroControl
            let dK : X :=
              match dKOpt with
              | some kCtrl => kCtrl
              | none => zeroControl
            let gamma := drift.gamma t0 x0 v0 args
            let u := drift.u t0 x0 v0 args
            let rho := Float.sqrt (2.0 * gamma * u)
            let coeffs := shouldCoeffs h gamma cfg.taylorThreshold
            let uh := u * h

            let f0 := drift.gradPotential t0 x0 args
            let chhHPlusCkkK : X := coeffs.chh * dH + coeffs.ckk * dK
            let rhoWK : X := rho * (dW - 12.0 * dK)
            let vTilde : X := v0 + rho * (dH + 6.0 * dK)

            let xHalf : X :=
              x0 + coeffs.aHalf * vTilde + coeffs.bHalf * ((-uh) * f0 + rhoWK)
            let fHalf := drift.gradPotential (t0 + 0.5 * h) xHalf args

            let x1 : X :=
              x0 + coeffs.a1 * v0 - (uh * coeffs.b1) * ((1.0 / 3.0) * f0 + (2.0 / 3.0) * fHalf) +
                rho * (coeffs.b1 * dW + chhHPlusCkkK)
            let f1 := drift.gradPotential t1 x1 args

            let forceBlend : X :=
              (coeffs.beta1 / 6.0) * f0 + ((2.0 / 3.0) * coeffs.betaHalf * fHalf) + (1.0 / 6.0) * f1
            let v1 : X :=
              (coeffs.beta1 * v0) - (uh * forceBlend) +
                rho * (coeffs.aa * dW - gamma * chhHPlusCkkK)

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
