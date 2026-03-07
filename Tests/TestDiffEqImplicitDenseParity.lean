import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqImplicitDenseParity

open LeanTest
open torch
open torch.DiffEq

private def denseValue {S C : Type}
    (label : String)
    (sol : Solution Float S C)
    (t : Time) : IO Float := do
  match sol.interpolation with
  | some _ =>
      pure (sol.evaluate t)
  | none =>
      LeanTest.fail s!"{label}: expected dense interpolation"
      pure 0.0

private def firstSavedValue {S C : Type}
    (label : String)
    (sol : Solution Float S C) : IO Float := do
  match sol.ys with
  | some ys =>
      if ys.size > 0 then
        pure ys[0]!
      else
        LeanTest.fail s!"{label}: empty ys"
        pure 0.0
  | none =>
      LeanTest.fail s!"{label}: expected ys"
      pure 0.0

@[test] def testKvaerno3SplitDenseImprovesOverHermiteFallback : IO Unit := do
  let tProbe : Time := 0.37
  let term : ODETerm Float Unit := {
    vectorField := fun t y _ => -2.0 * y + y * y + Float.sin t
  }
  let solverSplit :=
    Kvaerno3.solver
      (cfg := { denseKind := .splitAtStage 1 "kvaerno3-stage2-split-hermite" })
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Args := Unit)
  let solverHermite :=
    Kvaerno3.solver
      (cfg := { denseKind := .hermite })
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Args := Unit)
  let solveDense := fun (solver : AbstractSolver (ODETerm Float Unit) Float Float Time Unit) =>
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.25) (1.0 : Float) ()
      (saveat := { dense := true, t1 := false })
  let reference :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solverHermite 0.0 1.0 (some 0.005) (1.0 : Float) ()
      (saveat := { ts := some #[tProbe], t1 := false })
  LeanTest.assertTrue (reference.result == Result.successful)
    "Kvaerno3 reference solve should succeed"
  let yRef ← firstSavedValue "Kvaerno3 reference" reference
  let splitSol := solveDense solverSplit
  let hermiteSol := solveDense solverHermite
  LeanTest.assertTrue (splitSol.result == Result.successful)
    "Kvaerno3 split dense solve should succeed"
  LeanTest.assertTrue (hermiteSol.result == Result.successful)
    "Kvaerno3 Hermite fallback solve should succeed"
  let ySplit ← denseValue "Kvaerno3 split" splitSol tProbe
  let yHermite ← denseValue "Kvaerno3 hermite" hermiteSol tProbe
  let errSplit := Float.abs (ySplit - yRef)
  let errHermite := Float.abs (yHermite - yRef)
  LeanTest.assertTrue (errSplit < errHermite)
    s!"Kvaerno3 split dense should improve over Hermite fallback: {errSplit} vs {errHermite}"
  LeanTest.assertTrue (errSplit < 0.8 * errHermite)
    s!"Kvaerno3 split dense improvement should be meaningful: {errSplit} vs {errHermite}"

@[test] def testKencarp3DefaultDenseImprovesOverHermiteFallback : IO Unit := do
  let tProbe : Time := 0.54
  let explicit : ODETerm Float Unit := { vectorField := fun _t y _ => -1.0 * y }
  let implicit : ODETerm Float Unit := { vectorField := fun _t y _ => -0.9 * y }
  let terms : MultiTerm (ODETerm Float Unit) (ODETerm Float Unit) := {
    term1 := explicit
    term2 := implicit
  }
  let solverDefault :=
    Kencarp3.solver
      (cfg := {})
      (ExplicitTerm := ODETerm Float Unit)
      (ImplicitTerm := ODETerm Float Unit)
      (Y := Float)
      (VFe := Float)
      (VFi := Float)
      (Args := Unit)
  let solverHermite :=
    Kencarp3.solver
      (cfg := { denseKind := .hermite })
      (ExplicitTerm := ODETerm Float Unit)
      (ImplicitTerm := ODETerm Float Unit)
      (Y := Float)
      (VFe := Float)
      (VFi := Float)
      (Args := Unit)
  let solveDense := fun (solver :
      AbstractSolver
        (MultiTerm (ODETerm Float Unit) (ODETerm Float Unit))
        Float
        (Float × Float)
        (Time × Time)
        Unit) =>
    diffeqsolve
      (Term := MultiTerm (ODETerm Float Unit) (ODETerm Float Unit))
      (Y := Float)
      (VF := (Float × Float))
      (Control := (Time × Time))
      (Args := Unit)
      (Controller := ConstantStepSize)
      terms solver 0.0 1.0 (some 0.5) (1.0 : Float) ()
      (saveat := { dense := true, t1 := false })
  let solDefault := solveDense solverDefault
  let solHermite := solveDense solverHermite
  LeanTest.assertTrue (solDefault.result == Result.successful)
    "KenCarp3 default dense solve should succeed"
  LeanTest.assertTrue (solHermite.result == Result.successful)
    "KenCarp3 Hermite fallback solve should succeed"
  let yDefault ← denseValue "KenCarp3 default" solDefault tProbe
  let yHermite ← denseValue "KenCarp3 hermite" solHermite tProbe
  let exact := Float.exp (-1.9 * tProbe)
  let errDefault := Float.abs (yDefault - exact)
  let errHermite := Float.abs (yHermite - exact)
  LeanTest.assertTrue (errDefault < errHermite)
    s!"KenCarp3 default split dense should improve over Hermite fallback: {errDefault} vs {errHermite}"
  LeanTest.assertTrue (errDefault < 0.8 * errHermite)
    s!"KenCarp3 default split improvement should be meaningful: {errDefault} vs {errHermite}"

end Tests.DiffEqImplicitDenseParity

unsafe def main : IO Unit := do
  Tests.DiffEqImplicitDenseParity.testKvaerno3SplitDenseImprovesOverHermiteFallback
  Tests.DiffEqImplicitDenseParity.testKencarp3DefaultDenseImprovesOverHermiteFallback
  IO.println "TestDiffEqImplicitDenseParity: ok"
