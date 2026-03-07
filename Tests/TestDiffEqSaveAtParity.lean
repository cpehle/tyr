import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqSaveAtParity

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def assertSavedTsYsEq {S C : Type}
    (label : String)
    (sol : Solution Float S C)
    (expected : Array Time) : IO Unit := do
  match sol.ts, sol.ys with
  | some ts, some ys =>
      LeanTest.assertTrue (ts.size == expected.size)
        s!"{label}: expected {expected.size} saved times, got {ts.size}"
      LeanTest.assertTrue (ys.size == expected.size)
        s!"{label}: expected {expected.size} saved values, got {ys.size}"
      for i in [:expected.size] do
        let tExpected := expected[i]!
        LeanTest.assertTrue (approx ts[i]! tExpected 1e-12)
          s!"{label}: ts[{i}] expected {tExpected}, got {ts[i]!}"
        LeanTest.assertTrue (approx ys[i]! tExpected 1e-12)
          s!"{label}: ys[{i}] expected {tExpected}, got {ys[i]!}"
  | _, _ =>
      LeanTest.fail s!"{label}: expected ts/ys output"

@[test] def testSaveAtSubsIgnoresSyntheticRootPayload : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let leaf : SubSaveAt := { ts := some #[0.25] }
  let saveat : SaveAt := {
    -- `SaveAt.t1` defaults to true, but with `subs` this should act like a tree root
    -- container and not emit an extra payload entry.
    subs := #[leaf]
  }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (0.0 : Float) () (saveat := saveat)
  LeanTest.assertTrue (sol.result == Result.successful)
    "SaveAt(subs=...) solve should succeed"
  assertSavedTsYsEq "SaveAt(subs) root payload suppression" sol #[0.25]

@[test] def testNestedContainerSubSaveAtPayloadIgnored : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let leaf : SubSaveAt := { ts := some #[0.6] }
  let container : SubSaveAt := {
    -- Diffrax-style tree semantics: payload is leaf-defined; non-leaf nodes are
    -- structural containers.
    t1 := true
    subs := #[leaf]
  }
  let saveat : SaveAt := {
    t1 := false
    subs := #[container]
  }
  let sol :=
    diffeqsolve
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (0.0 : Float) () (saveat := saveat)
  LeanTest.assertTrue (sol.result == Result.successful)
    "Nested SubSaveAt container solve should succeed"
  assertSavedTsYsEq "Nested SubSaveAt container payload suppression" sol #[0.6]

def run : IO Unit := do
  testSaveAtSubsIgnoresSyntheticRootPayload
  testNestedContainerSubSaveAtPayloadIgnored

end Tests.DiffEqSaveAtParity
