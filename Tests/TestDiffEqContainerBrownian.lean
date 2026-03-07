import LeanTest
import Tyr.DiffEq

namespace Tests.DiffEqContainerBrownian

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

@[test] def testBrownianArrayIncrementsAdditive : IO Unit := do
  let tree : VirtualBrownianTree (Array Float) := {
    t0 := 0.0
    t1 := 1.0
    tol := 1.0e-3
    seed := 123456
    shape := #[0.0, 0.0, 0.0]
  }
  let inc01 := VirtualBrownianTree.increment tree 0.0 1.0
  let inc05 := VirtualBrownianTree.increment tree 0.0 0.5
  let inc51 := VirtualBrownianTree.increment tree 0.5 1.0
  LeanTest.assertTrue (inc01.W.size == inc05.W.size && inc01.W.size == inc51.W.size)
    "Array Brownian sizes should match"
  for i in [:inc01.W.size] do
    let lhs := inc01.W[i]!
    let rhs := inc05.W[i]! + inc51.W[i]!
    LeanTest.assertTrue (approx lhs rhs 1e-6)
      s!"Array Brownian component {i} not additive: {lhs} vs {rhs}"

@[test] def testBrownianListIncrementsAdditive : IO Unit := do
  let tree : VirtualBrownianTree (List Float) := {
    t0 := 0.0
    t1 := 1.0
    tol := 1.0e-3
    seed := 654321
    shape := [0.0, 0.0]
  }
  let inc01 := VirtualBrownianTree.increment tree 0.0 1.0
  let inc05 := VirtualBrownianTree.increment tree 0.0 0.5
  let inc51 := VirtualBrownianTree.increment tree 0.5 1.0
  let a01 := inc01.W.toArray
  let a05 := inc05.W.toArray
  let a51 := inc51.W.toArray
  LeanTest.assertTrue (a01.size == a05.size && a01.size == a51.size)
    "List Brownian sizes should match"
  for i in [:a01.size] do
    let lhs := a01[i]!
    let rhs := a05[i]! + a51[i]!
    LeanTest.assertTrue (approx lhs rhs 1e-6)
      s!"List Brownian component {i} not additive: {lhs} vs {rhs}"

def run : IO Unit := do
  testBrownianArrayIncrementsAdditive
  testBrownianListIncrementsAdditive

end Tests.DiffEqContainerBrownian
