import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqEventTreeParity

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def leafHits (tree : EventMaskTree) : Array Bool :=
  match tree with
  | .leaf hit => #[hit]
  | .branch children =>
      children.foldl (fun acc child => acc ++ leafHits child) #[]

@[test] def testEventTreeEarliestRootMaskParity : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let evEarly : EventSpec Float Unit := {
    condition := .real (fun _t y _ => y - 0.2)
    terminate := true
  }
  let evLate : EventSpec Float Unit := {
    condition := .real (fun _t y _ => y - 0.8)
    terminate := false
  }
  let evBool : EventSpec Float Unit := {
    condition := .boolean (fun t _y _ => t >= 0.5)
    terminate := false
  }
  let eventTree : EventTree Float Unit :=
    .branch #[
      .leaf evEarly,
      .branch #[.leaf evLate, .leaf evBool]
    ]
  let treeSol :=
    diffeqsolveEventTree
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 1.0) (0.0 : Float) () eventTree
      (saveat := { t1 := true })
  let sol := treeSol.base
  LeanTest.assertTrue (sol.result == Result.eventOccurred)
    "Event-tree solve should terminate at earliest terminating root"
  LeanTest.assertTrue (approx sol.t1 0.2 1.0e-6)
    s!"Expected earliest root at t=0.2, got {sol.t1}"
  match treeSol.eventMaskTree, treeSol.eventMaskLastTree with
  | some maskTree, some lastTree =>
      let hits := leafHits maskTree
      let lastHits := leafHits lastTree
      LeanTest.assertTrue (hits.size == 3 && lastHits.size == 3)
        s!"Expected three leaf mask entries, got {hits.size}/{lastHits.size}"
      LeanTest.assertTrue (hits[0]! && !hits[1]! && !hits[2]!)
        s!"Expected only early root to be committed in eventMask, got {hits}"
      LeanTest.assertTrue (lastHits[0]! && !lastHits[1]! && !lastHits[2]!)
        s!"Expected only early root at chosen time in eventMaskLast, got {lastHits}"
  | _, _ =>
      LeanTest.fail "Expected tree-shaped event masks"

@[test] def testEventTreeTieTimeMaskParity : IO Unit := do
  let term : ODETerm Float Unit := { vectorField := fun _t _y _ => 1.0 }
  let solver :=
    Euler.solver (Term := ODETerm Float Unit) (Y := Float) (VF := Float) (Args := Unit)
  let evKeep : EventSpec Float Unit := {
    condition := .boolean (fun t _y _ => t >= 0.5)
    terminate := false
  }
  let evStop : EventSpec Float Unit := {
    condition := .boolean (fun t _y _ => t >= 0.5)
    terminate := true
  }
  let eventTree : EventTree Float Unit :=
    .branch #[
      .leaf evKeep,
      .branch #[.leaf evStop]
    ]
  let treeSol :=
    diffeqsolveEventTree
      (Term := ODETerm Float Unit)
      (Y := Float)
      (VF := Float)
      (Control := Time)
      (Args := Unit)
      (Controller := ConstantStepSize)
      term solver 0.0 1.0 (some 0.1) (0.0 : Float) () eventTree
      (saveat := { t1 := true })
  let sol := treeSol.base
  LeanTest.assertTrue (sol.result == Result.eventOccurred)
    "Terminating tied event in tree should stop solve"
  LeanTest.assertTrue (approx sol.t1 0.5 1.0e-12)
    s!"Expected tie-time termination at t=0.5, got {sol.t1}"
  match treeSol.eventMaskTree, treeSol.eventMaskLastTree with
  | some maskTree, some lastTree =>
      let hits := leafHits maskTree
      let lastHits := leafHits lastTree
      LeanTest.assertTrue (hits.size == 2 && lastHits.size == 2)
        s!"Expected two leaf mask entries, got {hits.size}/{lastHits.size}"
      LeanTest.assertTrue (hits[0]! && hits[1]!)
        s!"Both tie-time events should be committed in eventMask, got {hits}"
      LeanTest.assertTrue (lastHits[0]! && lastHits[1]!)
        s!"Both tie-time events should appear in eventMaskLast, got {lastHits}"
  | _, _ =>
      LeanTest.fail "Expected tree-shaped tie-time masks"

def run : IO Unit := do
  testEventTreeEarliestRootMaskParity
  testEventTreeTieTimeMaskParity

end Tests.DiffEqEventTreeParity

unsafe def main : IO Unit := do
  Tests.DiffEqEventTreeParity.run
  IO.println "TestDiffEqEventTreeParity: ok"
