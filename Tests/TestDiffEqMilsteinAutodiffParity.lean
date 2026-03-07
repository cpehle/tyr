import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqMilsteinAutodiffParity

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def safeQvDenom (qv : Float) : Float :=
  if Float.abs qv <= 1.0e-12 then
    if qv < 0.0 then -1.0e-12 else 1.0e-12
  else
    qv

private def finalSaved {S C : Type}
    (label : String) (sol : Solution Float S C) : IO Float := do
  match sol.ys with
  | some ys =>
      if ys.size == 0 then
        LeanTest.fail s!"Expected saved state for {label}"
        return 0.0
      else
        return ys[ys.size - 1]!
  | none =>
      LeanTest.fail s!"Expected ys for {label}"
      return 0.0

@[test] def testMilsteinFiniteDiffFallbackScalarParity : IO Unit := do
  let drift : ODETerm Float Unit := { vectorField := fun _t _y _ => 0.0 }
  let path := AbstractPath.linearInterpolation 0.0 1.0 (0.0 : Float) (2.0 : Float)
  let diffusionBase : ControlTerm Float Float Float Unit :=
    ControlTerm.ofPath
      (fun _t y _ => y * y)
      path
      (fun vf control => vf * control)
  let diffusionFinite := withFiniteDiffJacobianProd diffusionBase
  let diffusionExact :
      JacobianProdDiffusion (ControlTerm Float Float Float Unit) Float Float Unit :=
    withJacobianProd diffusionBase (fun _t y _args _control => 2.0 * y * y * y)

  let termsBase : MultiTerm (ODETerm Float Unit) (ControlTerm Float Float Float Unit) := {
    term1 := drift
    term2 := diffusionBase
  }
  let termsFinite :
      MultiTerm (ODETerm Float Unit) (FiniteDiffProdJacobianDiffusion (ControlTerm Float Float Float Unit)) := {
    term1 := drift
    term2 := diffusionFinite
  }
  let termsExact :
      MultiTerm (ODETerm Float Unit) (JacobianProdDiffusion (ControlTerm Float Float Float Unit) Float Float Unit) := {
    term1 := drift
    term2 := diffusionExact
  }

  let solverBase :=
    Milstein.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := ControlTerm Float Float Float Unit)
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Control := Float)
      (Args := Unit)
  let solverFinite :=
    Milstein.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := FiniteDiffProdJacobianDiffusion (ControlTerm Float Float Float Unit))
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Control := Float)
      (Args := Unit)
  let solverExact :=
    Milstein.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := JacobianProdDiffusion (ControlTerm Float Float Float Unit) Float Float Unit)
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Control := Float)
      (Args := Unit)

  let y0 := 0.7
  let solveBase :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (ControlTerm Float Float Float Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      termsBase solverBase 0.0 1.0 (some 1.0) y0 () (saveat := { t1 := true })
  let solveFinite :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit)
        (FiniteDiffProdJacobianDiffusion (ControlTerm Float Float Float Unit)))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      termsFinite solverFinite 0.0 1.0 (some 1.0) y0 () (saveat := { t1 := true })
  let solveExact :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit)
        (JacobianProdDiffusion (ControlTerm Float Float Float Unit) Float Float Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      termsExact solverExact 0.0 1.0 (some 1.0) y0 () (saveat := { t1 := true })

  let yBase ← finalSaved "Milstein scalar fallback base" solveBase
  let yFinite ← finalSaved "Milstein scalar finite wrapper" solveFinite
  let yExact ← finalSaved "Milstein scalar exact wrapper" solveExact

  let dControl : Float := path.increment 0.0 1.0
  let qv := MilsteinControl.quadraticVariation dControl
  let g0 := (y0 * y0) * dControl
  let gg0 := 2.0 * y0 * y0 * y0
  let expected := y0 + g0 + (0.5 * (qv - 1.0)) * gg0

  LeanTest.assertTrue (approx yExact expected 1.0e-12)
    s!"Milstein scalar exact wrapper should match analytic one-step value: {expected} vs {yExact}"
  LeanTest.assertTrue (approx yBase yExact 2.0e-7)
    s!"Milstein scalar fallback parity mismatch: expected {yExact}, got {yBase}"
  LeanTest.assertTrue (approx yFinite yExact 2.0e-7)
    s!"Milstein scalar finite-diff wrapper parity mismatch: expected {yExact}, got {yFinite}"

@[test] def testMilsteinFiniteDiffFallbackVectorControlParity : IO Unit := do
  let drift : ODETerm Float Unit := { vectorField := fun _t _y _ => 0.0 }
  let c0 : Vector 2 Float := ⟨#[0.0, 0.0], by decide⟩
  let c1 : Vector 2 Float := ⟨#[2.0, 1.0], by decide⟩
  let path : AbstractPath (Vector 2 Float) :=
    AbstractPath.linearInterpolation 0.0 1.0 c0 c1
  let i0 : Fin 2 := ⟨0, by decide⟩
  let i1 : Fin 2 := ⟨1, by decide⟩
  let diffusionBase : ControlTerm Float Float (Vector 2 Float) Unit :=
    ControlTerm.ofPath
      (fun _t y _ => y * y)
      path
      (fun vf control => vf * ((control.get i0) + (control.get i1)))
  let diffusionFinite := withFiniteDiffJacobianProd diffusionBase
  let diffusionExact :
      JacobianProdDiffusion (ControlTerm Float Float (Vector 2 Float) Unit) Float (Vector 2 Float) Unit :=
    withJacobianProd diffusionBase (fun _t y _args control =>
      let s := (control.get i0) + (control.get i1)
      let qv := MilsteinControl.quadraticVariation control
      let qvSafe := safeQvDenom qv
      (2.0 * y * y * y) * ((s * s) / qvSafe))

  let termsBase : MultiTerm (ODETerm Float Unit) (ControlTerm Float Float (Vector 2 Float) Unit) := {
    term1 := drift
    term2 := diffusionBase
  }
  let termsFinite :
      MultiTerm (ODETerm Float Unit)
        (FiniteDiffProdJacobianDiffusion (ControlTerm Float Float (Vector 2 Float) Unit)) := {
    term1 := drift
    term2 := diffusionFinite
  }
  let termsExact :
      MultiTerm (ODETerm Float Unit)
        (JacobianProdDiffusion (ControlTerm Float Float (Vector 2 Float) Unit) Float (Vector 2 Float) Unit) := {
    term1 := drift
    term2 := diffusionExact
  }

  let solverBase :=
    Milstein.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := ControlTerm Float Float (Vector 2 Float) Unit)
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Control := Vector 2 Float)
      (Args := Unit)
  let solverFinite :=
    Milstein.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := FiniteDiffProdJacobianDiffusion (ControlTerm Float Float (Vector 2 Float) Unit))
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Control := Vector 2 Float)
      (Args := Unit)
  let solverExact :=
    Milstein.solver
      (Drift := ODETerm Float Unit)
      (Diffusion := JacobianProdDiffusion (ControlTerm Float Float (Vector 2 Float) Unit) Float (Vector 2 Float) Unit)
      (Y := Float)
      (VFd := Float)
      (VFg := Float)
      (Control := Vector 2 Float)
      (Args := Unit)

  let y0 := 0.6
  let solveBase :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (ControlTerm Float Float (Vector 2 Float) Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Vector 2 Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      termsBase solverBase 0.0 1.0 (some 1.0) y0 () (saveat := { t1 := true })
  let solveFinite :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit)
        (FiniteDiffProdJacobianDiffusion (ControlTerm Float Float (Vector 2 Float) Unit)))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Vector 2 Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      termsFinite solverFinite 0.0 1.0 (some 1.0) y0 () (saveat := { t1 := true })
  let solveExact :=
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit)
        (JacobianProdDiffusion (ControlTerm Float Float (Vector 2 Float) Unit) Float (Vector 2 Float) Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Vector 2 Float))
      (Args := Unit)
      (Controller := ConstantStepSize)
      termsExact solverExact 0.0 1.0 (some 1.0) y0 () (saveat := { t1 := true })

  let yBase ← finalSaved "Milstein vector fallback base" solveBase
  let yFinite ← finalSaved "Milstein vector finite wrapper" solveFinite
  let yExact ← finalSaved "Milstein vector exact wrapper" solveExact

  let dControl := path.increment 0.0 1.0
  let s := (dControl.get i0) + (dControl.get i1)
  let qv := MilsteinControl.quadraticVariation dControl
  let qvSafe := safeQvDenom qv
  let g0 := (y0 * y0) * s
  let gg0 := (2.0 * y0 * y0 * y0) * ((s * s) / qvSafe)
  let expected := y0 + g0 + (0.5 * (qv - 1.0)) * gg0

  LeanTest.assertTrue (approx yExact expected 1.0e-12)
    s!"Milstein vector exact wrapper should match analytic one-step value: {expected} vs {yExact}"
  LeanTest.assertTrue (approx yBase yExact 3.0e-7)
    s!"Milstein vector fallback parity mismatch: expected {yExact}, got {yBase}"
  LeanTest.assertTrue (approx yFinite yExact 3.0e-7)
    s!"Milstein vector finite-diff wrapper parity mismatch: expected {yExact}, got {yFinite}"

def run : IO Unit := do
  testMilsteinFiniteDiffFallbackScalarParity
  testMilsteinFiniteDiffFallbackVectorControlParity

end Tests.DiffEqMilsteinAutodiffParity
