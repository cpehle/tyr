import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqBrownianFinParity

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

private def finIndices4 : Array (Fin 4) :=
  #[⟨0, by decide⟩, ⟨1, by decide⟩, ⟨2, by decide⟩, ⟨3, by decide⟩]

@[test] def testFinBrownianAdditivityOverIntervalSplit : IO Unit := do
  let tree : VirtualBrownianTree (Fin 4 → Float) := {
    t0 := 0.0
    t1 := 1.0
    tol := 1.0e-3
    seed := 602001
    shape := fun _ => (0.0 : Float)
  }

  let tMid := 0.37
  let inc01 := VirtualBrownianTree.increment tree 0.0 1.0
  let inc0m := VirtualBrownianTree.increment tree 0.0 tMid
  let incm1 := VirtualBrownianTree.increment tree tMid 1.0

  LeanTest.assertTrue (approx inc01.dt (inc0m.dt + incm1.dt) 1e-12)
    s!"Fin container dt additivity mismatch: {inc01.dt} vs {inc0m.dt + incm1.dt}"

  for i in finIndices4 do
    let lhs := inc01.W i
    let rhs := inc0m.W i + incm1.W i
    LeanTest.assertTrue (approx lhs rhs 1e-6)
      s!"Fin Brownian component {i.1} not additive: {lhs} vs {rhs}"

@[test] def testFinBrownianDeterministicRepeatability : IO Unit := do
  let tree : VirtualBrownianTree (Fin 4 → Float) := {
    t0 := 0.0
    t1 := 1.0
    tol := 1.0e-3
    seed := 602002
    shape := fun _ => (0.0 : Float)
  }

  let tStart := 0.15
  let tStop := 0.92

  let incA := VirtualBrownianTree.increment tree tStart tStop
  let incB := VirtualBrownianTree.increment tree tStart tStop

  let stA := VirtualBrownianTree.incrementSpaceTime tree tStart tStop
  let stB := VirtualBrownianTree.incrementSpaceTime tree tStart tStop

  let sttA := VirtualBrownianTree.incrementSpaceTimeTime tree tStart tStop
  let sttB := VirtualBrownianTree.incrementSpaceTimeTime tree tStart tStop

  LeanTest.assertTrue (approx incA.dt incB.dt 1e-12)
    s!"Fin Brownian dt repeatability mismatch: {incA.dt} vs {incB.dt}"
  LeanTest.assertTrue (approx stA.dt stB.dt 1e-12)
    s!"Fin space-time dt repeatability mismatch: {stA.dt} vs {stB.dt}"
  LeanTest.assertTrue (approx sttA.dt sttB.dt 1e-12)
    s!"Fin space-time-time dt repeatability mismatch: {sttA.dt} vs {sttB.dt}"

  for i in finIndices4 do
    LeanTest.assertTrue (approx (incA.W i) (incB.W i) 1e-12)
      s!"Fin Brownian W component {i.1} repeatability mismatch"

    LeanTest.assertTrue (approx (stA.W i) (stB.W i) 1e-12)
      s!"Fin space-time W component {i.1} repeatability mismatch"
    LeanTest.assertTrue (approx (stA.H i) (stB.H i) 1e-12)
      s!"Fin space-time H component {i.1} repeatability mismatch"

    LeanTest.assertTrue (approx (sttA.W i) (sttB.W i) 1e-12)
      s!"Fin space-time-time W component {i.1} repeatability mismatch"
    LeanTest.assertTrue (approx (sttA.H i) (sttB.H i) 1e-12)
      s!"Fin space-time-time H component {i.1} repeatability mismatch"
    LeanTest.assertTrue (approx (sttA.K i) (sttB.K i) 1e-12)
      s!"Fin space-time-time K component {i.1} repeatability mismatch"

def run : IO Unit := do
  testFinBrownianAdditivityOverIntervalSplit
  testFinBrownianDeterministicRepeatability

end Tests.DiffEqBrownianFinParity
