import Tyr.DiffEq.Solver.Base

namespace torch
namespace DiffEq

/-! ## QUICSORT Solver (underdamped Langevin SRK, staged)

Staged underdamped implementation inspired by diffrax QUICSORT updates.
This version uses the Brownian increment channel and omits high-order
space-time-time Levy corrections.
-/

attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hAddInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hSubInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hMulInst

structure QUICSORT where
  deriving Inhabited

private structure QUICCoeffs where
  betaL : Float
  betaR : Float
  beta1 : Float
  aL : Float
  aR : Float
  a1 : Float
  bL : Float
  bR : Float
  b1 : Float
  aThird : Float
  aDivH : Float
  deriving Inhabited

private def lCoeff : Float := 0.5 - Float.sqrt 3.0 / 6.0
private def rCoeff : Float := 0.5 + Float.sqrt 3.0 / 6.0

private def betaTau (gh tau : Float) : Float :=
  Float.exp (-(tau * gh))

private def aTau (h gh gamma tau : Float) : Float :=
  if Float.abs gamma <= 1.0e-12 || Float.abs gh <= 1.0e-6 then
    h * (tau - 0.5 * tau * tau * gh + (1.0 / 6.0) * tau * tau * tau * gh * gh)
  else
    (1.0 - betaTau gh tau) / gamma

private def bTau (h gh gamma tau : Float) : Float :=
  if Float.abs gamma <= 1.0e-12 || Float.abs gh <= 1.0e-6 then
    h * (0.5 * tau * tau - (1.0 / 6.0) * tau * tau * tau * gh +
      (1.0 / 24.0) * tau * tau * tau * tau * gh * gh)
  else
    (betaTau gh tau + tau * gh - 1.0) / (gamma * gh)

private def quicsortCoeffs (h gamma : Float) : QUICCoeffs :=
  let gh := gamma * h
  let betaL := betaTau gh lCoeff
  let betaR := betaTau gh rCoeff
  let beta1 := betaTau gh 1.0
  let aL := aTau h gh gamma lCoeff
  let aR := aTau h gh gamma rCoeff
  let a1 := aTau h gh gamma 1.0
  let bL := bTau h gh gamma lCoeff
  let bR := bTau h gh gamma rCoeff
  let b1 := bTau h gh gamma 1.0
  let aThird :=
    if Float.abs gamma <= 1.0e-12 || Float.abs gh <= 1.0e-6 then
      h * ((1.0 / 3.0) - (1.0 / 18.0) * gh + (1.0 / 162.0) * gh * gh)
    else
      (1.0 - betaTau gh (1.0 / 3.0)) / gamma
  let aDivH :=
    if Float.abs gh <= 1.0e-12 then
      1.0
    else
      (1.0 - betaTau gh 1.0) / gh
  {
    betaL := betaL
    betaR := betaR
    beta1 := beta1
    aL := aL
    aR := aR
    a1 := a1
    bL := bL
    bR := bR
    b1 := b1
    aThird := aThird
    aDivH := aDivH
  }

def QUICSORT.solver {X Args : Type}
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
            let uh := u * h
            let coeffs := quicsortCoeffs h gamma
            let rhoW := rho * dW

            let xL := x0 + coeffs.aL * v0 + coeffs.bL * rhoW
            let fLUh := uh * drift.gradPotential (t0 + lCoeff * h) xL args

            let xR := x0 + coeffs.aR * v0 + coeffs.bR * rhoW - coeffs.aThird * fLUh
            let fRUh := uh * drift.gradPotential (t0 + rCoeff * h) xR args

            let x1 :=
              x0 + coeffs.a1 * v0 + coeffs.b1 * rhoW -
              0.5 * (coeffs.aR * fLUh + coeffs.aL * fRUh)

            let v1 :=
              coeffs.beta1 * v0 -
              0.5 * (coeffs.betaR * fLUh + coeffs.betaL * fRUh) +
              coeffs.aDivH * rhoW

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

instance : ExplicitSolver QUICSORT := ⟨True.intro⟩
instance : StratonovichSolver QUICSORT := ⟨True.intro⟩

end DiffEq
end torch
