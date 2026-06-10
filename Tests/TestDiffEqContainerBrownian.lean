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

/-- Tensor-valued VBT must agree element-for-element with the `Array Float`
    tree of the same seed (same element-index key derivation), modulo the
    Float32 storage of the packed tensor. -/
@[test] def testBrownianTensorMatchesArray : IO Unit := do
  let mkArr : VirtualBrownianTree (Array Float) := {
    t0 := 0.0, t1 := 1.0, tol := 1.0e-3, seed := 4242
    shape := #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  }
  let mkTen : VirtualBrownianTree (T #[2, 3]) := {
    t0 := 0.0, t1 := 1.0, tol := 1.0e-3, seed := 4242
    shape := torch.zeros #[2, 3]
  }
  for (a, b) in [(0.0, 1.0), (0.1, 0.7), (0.25, 0.5)] do
    let wArr := (VirtualBrownianTree.increment mkArr a b).W
    let wTen ← data.tensorToFloatArray' (VirtualBrownianTree.increment mkTen a b).W
    LeanTest.assertEqual wTen.size 6 "tensor increment should have 6 elements"
    for i in [:6] do
      LeanTest.assertTrue (approx wArr[i]! wTen[i]! 1e-6)
        s!"tensor[{i}] = {wTen[i]!} ≠ array[{i}] = {wArr[i]!} on [{a}, {b}]"

@[test] def testBrownianTensorIncrementsAdditive : IO Unit := do
  let tree : VirtualBrownianTree (T #[4]) := {
    t0 := 0.0, t1 := 1.0, tol := 1.0e-3, seed := 99
    shape := torch.zeros #[4]
  }
  let w01 ← data.tensorToFloatArray' (VirtualBrownianTree.increment tree 0.0 1.0).W
  let w05 ← data.tensorToFloatArray' (VirtualBrownianTree.increment tree 0.0 0.5).W
  let w51 ← data.tensorToFloatArray' (VirtualBrownianTree.increment tree 0.5 1.0).W
  for i in [:4] do
    LeanTest.assertTrue (approx w01[i]! (w05[i]! + w51[i]!) 1e-5)
      s!"tensor Brownian component {i} not additive"

@[test] def testBrownianTensorLevyAreasAndReplay : IO Unit := do
  let tree : VirtualBrownianTree (T #[3]) := {
    t0 := 0.0, t1 := 2.0, tol := 1.0e-3, seed := 7
    shape := torch.zeros #[3]
  }
  -- replay determinism: identical queries give identical samples
  let i1 := VirtualBrownianTree.incrementSpaceTimeTime tree 0.3 1.4
  let i2 := VirtualBrownianTree.incrementSpaceTimeTime tree 0.3 1.4
  let w1 ← data.tensorToFloatArray' i1.W
  let w2 ← data.tensorToFloatArray' i2.W
  let h1 ← data.tensorToFloatArray' i1.H
  let h2 ← data.tensorToFloatArray' i2.H
  let k1 ← data.tensorToFloatArray' i1.K
  let k2 ← data.tensorToFloatArray' i2.K
  for i in [:3] do
    LeanTest.assertTrue (w1[i]! == w2[i]! && h1[i]! == h2[i]! && k1[i]! == k2[i]!)
      s!"tensor VBT replay should be deterministic (component {i})"
  -- Lévy fields match the scalar trees with the same per-element keys
  let arrTree : VirtualBrownianTree (Array Float) := {
    t0 := 0.0, t1 := 2.0, tol := 1.0e-3, seed := 7
    shape := #[0.0, 0.0, 0.0]
  }
  let arrInc := VirtualBrownianTree.incrementSpaceTimeTime arrTree 0.3 1.4
  for i in [:3] do
    LeanTest.assertTrue (approx arrInc.W[i]! w1[i]! 1e-6)
      s!"tensor W[{i}] should match array path"
    LeanTest.assertTrue (approx arrInc.H[i]! h1[i]! 1e-6)
      s!"tensor H[{i}] should match array path"
    LeanTest.assertTrue (approx arrInc.K[i]! k1[i]! 1e-6)
      s!"tensor K[{i}] should match array path"

def run : IO Unit := do
  testBrownianArrayIncrementsAdditive
  testBrownianListIncrementsAdditive
  testBrownianTensorMatchesArray
  testBrownianTensorIncrementsAdditive
  testBrownianTensorLevyAreasAndReplay

end Tests.DiffEqContainerBrownian
