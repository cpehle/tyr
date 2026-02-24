import LeanTest
import Tyr.DiffEq
import Tyr.DiffEq.Adjoint.Torch

/-!
# `Tests.TestDiffEqAdjoint`

Adjoint tests that compare gradient computations against finite-difference references.

## Overview
- Regression and behavior checks run by the LeanTest-based test suite.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace Tests.DiffEqAdjoint

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def getY1 {S C : Type} (label : String) (sol : Solution (T #[]) S C) (fallback : T #[]) :
    IO (T #[]) := do
  match sol.ys with
  | some ys =>
      if ys.size == 0 then
        LeanTest.fail s!"Expected y1 for {label}"
        return fallback
      else
        return ys[ys.size - 1]!
  | none =>
      LeanTest.fail s!"Expected ys for {label}"
      return fallback

@[test] def testBacksolveAdjointLinearODE : IO Unit := do
  let term : ODETerm (T #[]) (T #[]) := { vectorField := fun _t y a => mul a y }
  let solver :=
    RK4.solver
      (Term := ODETerm (T #[]) (T #[]))
      (Y := T #[])
      (VF := T #[])
      (Args := T #[])
  let adjSolver :=
    RK4.solver
      (Term := ODETerm (AdjointState (T #[]) (T #[])) (T #[]))
      (Y := AdjointState (T #[]) (T #[]))
      (VF := AdjointState (T #[]) (T #[]))
      (Args := T #[])
  let y0 := full #[] 2.0
  let a := full #[] 0.3
  let t0 := 0.0
  let t1 := 1.0
  let dt0 := some 0.01
  let sol :=
    diffeqsolve
      (Term := ODETerm (T #[]) (T #[]))
      (Y := T #[])
      (VF := T #[])
      (Control := Time)
      (Args := T #[])
      (Controller := ConstantStepSize)
      term solver t0 t1 dt0 y0 a (saveat := { steps := true })
  let adjY1 := ones #[]
  match backsolveAdjoint term adjSolver sol a adjY1 with
  | none => LeanTest.fail "Expected adjoint result"
  | some adj => do
      let eps := 1.0e-3
      let y0p := add_scalar y0 eps
      let y0m := add_scalar y0 (-eps)
      let solp :=
        diffeqsolve
          (Term := ODETerm (T #[]) (T #[]))
          (Y := T #[])
          (VF := T #[])
          (Control := Time)
          (Args := T #[])
          (Controller := ConstantStepSize)
          term solver t0 t1 dt0 y0p a (saveat := { t1 := true })
      let solm :=
        diffeqsolve
          (Term := ODETerm (T #[]) (T #[]))
          (Y := T #[])
          (VF := T #[])
          (Control := Time)
          (Args := T #[])
          (Controller := ConstantStepSize)
          term solver t0 t1 dt0 y0m a (saveat := { t1 := true })
      let y1p ← getY1 "y0+eps" solp y0
      let y1m ← getY1 "y0-eps" solm y0
      let lp := nn.item y1p
      let lm := nn.item y1m
      let fdY0 := (lp - lm) / (2.0 * eps)

      let ap := add_scalar a eps
      let am := add_scalar a (-eps)
      let solap :=
        diffeqsolve
          (Term := ODETerm (T #[]) (T #[]))
          (Y := T #[])
          (VF := T #[])
          (Control := Time)
          (Args := T #[])
          (Controller := ConstantStepSize)
          term solver t0 t1 dt0 y0 ap (saveat := { t1 := true })
      let solam :=
        diffeqsolve
          (Term := ODETerm (T #[]) (T #[]))
          (Y := T #[])
          (VF := T #[])
          (Control := Time)
          (Args := T #[])
          (Controller := ConstantStepSize)
          term solver t0 t1 dt0 y0 am (saveat := { t1 := true })
      let y1ap ← getY1 "a+eps" solap y0
      let y1am ← getY1 "a-eps" solam y0
      let lap := nn.item y1ap
      let lam := nn.item y1am
      let fdA := (lap - lam) / (2.0 * eps)

      let adjY0 := nn.item adj.adjY0
      let adjA := nn.item adj.adjArgs
      LeanTest.assertTrue (approx adjY0 fdY0 1.0e-2)
        s!"adjoint dy0 expected {fdY0}, got {adjY0}"
      LeanTest.assertTrue (approx adjA fdA 1.0e-2)
        s!"adjoint da expected {fdA}, got {adjA}"

@[test] def testDirectAdjointLinearODE : IO Unit := do
  let term : ODETerm (T #[]) (T #[]) := { vectorField := fun _t y a => mul a y }
  let solver :=
    Euler.solver
      (Term := ODETerm (T #[]) (T #[]))
      (Y := T #[])
      (VF := T #[])
      (Args := T #[])
  let y0 := full #[] 2.0
  let a := full #[] 0.3
  let t0 := 0.0
  let t1 := 1.0
  let dt0 := some 0.01
  let adjY1 := ones #[]
  let (_, adjOpt) :=
    diffeqsolveDirectAdjoint term solver t0 t1 dt0 y0 a adjY1 (saveat := { t1 := true })
  match adjOpt with
  | none => LeanTest.fail "Expected direct adjoint result"
  | some adj => do
      let eps := 1.0e-3
      let y0p := add_scalar y0 eps
      let y0m := add_scalar y0 (-eps)
      let solp :=
        diffeqsolve
          (Term := ODETerm (T #[]) (T #[]))
          (Y := T #[])
          (VF := T #[])
          (Control := Time)
          (Args := T #[])
          (Controller := ConstantStepSize)
          term solver t0 t1 dt0 y0p a (saveat := { t1 := true })
      let solm :=
        diffeqsolve
          (Term := ODETerm (T #[]) (T #[]))
          (Y := T #[])
          (VF := T #[])
          (Control := Time)
          (Args := T #[])
          (Controller := ConstantStepSize)
          term solver t0 t1 dt0 y0m a (saveat := { t1 := true })
      let y1p ← getY1 "y0+eps" solp y0
      let y1m ← getY1 "y0-eps" solm y0
      let lp := nn.item y1p
      let lm := nn.item y1m
      let fdY0 := (lp - lm) / (2.0 * eps)

      let ap := add_scalar a eps
      let am := add_scalar a (-eps)
      let solap :=
        diffeqsolve
          (Term := ODETerm (T #[]) (T #[]))
          (Y := T #[])
          (VF := T #[])
          (Control := Time)
          (Args := T #[])
          (Controller := ConstantStepSize)
          term solver t0 t1 dt0 y0 ap (saveat := { t1 := true })
      let solam :=
        diffeqsolve
          (Term := ODETerm (T #[]) (T #[]))
          (Y := T #[])
          (VF := T #[])
          (Control := Time)
          (Args := T #[])
          (Controller := ConstantStepSize)
          term solver t0 t1 dt0 y0 am (saveat := { t1 := true })
      let y1ap ← getY1 "a+eps" solap y0
      let y1am ← getY1 "a-eps" solam y0
      let lap := nn.item y1ap
      let lam := nn.item y1am
      let fdA := (lap - lam) / (2.0 * eps)

      let adjY0 := nn.item adj.adjY0
      let adjA := nn.item adj.adjArgs
      LeanTest.assertTrue (approx adjY0 fdY0 1.0e-2)
        s!"direct adjoint dy0 expected {fdY0}, got {adjY0}"
      LeanTest.assertTrue (approx adjA fdA 1.0e-2)
        s!"direct adjoint da expected {fdA}, got {adjA}"

end Tests.DiffEqAdjoint
