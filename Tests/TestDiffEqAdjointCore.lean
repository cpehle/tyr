import LeanTest
import Tyr.DiffEq
import Tyr.DiffEq.Adjoint.Core

namespace Tests.DiffEqAdjointCore

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def getY1 {S C : Type} (label : String) (sol : Solution Float S C) : IO Float := do
  match sol.ys with
  | some ys =>
      if ys.size == 0 then
        LeanTest.fail s!"Expected y1 for {label}"
        return 0.0
      else
        return ys[ys.size - 1]!
  | none =>
      LeanTest.fail s!"Expected ys for {label}"
      return 0.0

section
  -- Valid for the linear vf used in this test: f(t,y,a) = a*y.
  local instance : AdjointBackend Float Float where
    vjp vf t y a adjY :=
      let f := vf t y a
      let gradY := a * adjY
      let gradA := y * adjY
      (f, gradY, gradA)

  @[test] def testBacksolveAdjointLinearODE : IO Unit := do
    let term : ODETerm Float Float := { vectorField := fun _t y a => a * y }
    let solver :=
      RK4.solver
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Args := Float)
    let adjSolver :=
      RK4.solver
        (Term := ODETerm (AdjointState Float Float) Float)
        (Y := AdjointState Float Float)
        (VF := AdjointState Float Float)
        (Args := Float)
    let y0 := 2.0
    let a := 0.3
    let t0 := 0.0
    let t1 := 1.0
    let dt0 := some 0.01
    let sol :=
      diffeqsolve
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Control := Time)
        (Args := Float)
        (Controller := ConstantStepSize)
        term solver t0 t1 dt0 y0 a (saveat := { steps := true })
    let adjY1 := 1.0
    match backsolveAdjoint term adjSolver sol a adjY1 with
    | none => LeanTest.fail "Expected adjoint result"
    | some adj => do
        let eps := 1.0e-4
        let solp :=
          diffeqsolve
            (Term := ODETerm Float Float)
            (Y := Float)
            (VF := Float)
            (Control := Time)
            (Args := Float)
            (Controller := ConstantStepSize)
            term solver t0 t1 dt0 (y0 + eps) a (saveat := { t1 := true })
        let solm :=
          diffeqsolve
            (Term := ODETerm Float Float)
            (Y := Float)
            (VF := Float)
            (Control := Time)
            (Args := Float)
            (Controller := ConstantStepSize)
            term solver t0 t1 dt0 (y0 - eps) a (saveat := { t1 := true })
        let y1p ← getY1 "y0+eps" solp
        let y1m ← getY1 "y0-eps" solm
        let fdY0 := (y1p - y1m) / (2.0 * eps)

        let solap :=
          diffeqsolve
            (Term := ODETerm Float Float)
            (Y := Float)
            (VF := Float)
            (Control := Time)
            (Args := Float)
            (Controller := ConstantStepSize)
            term solver t0 t1 dt0 y0 (a + eps) (saveat := { t1 := true })
        let solam :=
          diffeqsolve
            (Term := ODETerm Float Float)
            (Y := Float)
            (VF := Float)
            (Control := Time)
            (Args := Float)
            (Controller := ConstantStepSize)
            term solver t0 t1 dt0 y0 (a - eps) (saveat := { t1 := true })
        let y1ap ← getY1 "a+eps" solap
        let y1am ← getY1 "a-eps" solam
        let fdA := (y1ap - y1am) / (2.0 * eps)

        LeanTest.assertTrue (approx adj.adjY0 fdY0 1.0e-3)
          s!"adjoint dy0 expected {fdY0}, got {adj.adjY0}"
        LeanTest.assertTrue (approx adj.adjArgs fdA 1.0e-3)
          s!"adjoint da expected {fdA}, got {adj.adjArgs}"

  @[test] def testBacksolveAdjointWrapper : IO Unit := do
    let term : ODETerm Float Float := { vectorField := fun _t y a => a * y }
    let solver :=
      RK4.solver
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Args := Float)
    let adjSolver :=
      RK4.solver
        (Term := ODETerm (AdjointState Float Float) Float)
        (Y := AdjointState Float Float)
        (VF := AdjointState Float Float)
        (Args := Float)
    let adjoint : BacksolveAdjoint Float Float := { adjSolver := adjSolver }
    let y0 := 2.0
    let a := 0.3
    let t0 := 0.0
    let t1 := 1.0
    let dt0 := some 0.01
    let adjY1 := 1.0
    let (solAdj, adj) :=
      diffeqsolveBacksolveAdjoint (Controller := ConstantStepSize)
        term solver adjoint t0 t1 dt0 y0 a adjY1 (saveat := { t1 := true })
    let sol :=
      diffeqsolve
        (Term := ODETerm Float Float)
        (Y := Float)
        (VF := Float)
        (Control := Time)
        (Args := Float)
        (Controller := ConstantStepSize)
        term solver t0 t1 dt0 y0 a (saveat := { t1 := true })
    let yAdj ← getY1 "wrapper" solAdj
    let yDir ← getY1 "direct" sol
    LeanTest.assertTrue (approx yAdj yDir 1.0e-6)
      s!"Wrapper solution expected {yDir}, got {yAdj}"
    match adj with
    | some _ => pure ()
    | none => LeanTest.fail "Expected adjoint result from wrapper"
end

end Tests.DiffEqAdjointCore
