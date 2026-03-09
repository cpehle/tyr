import Tyr
import LeanTest

namespace Tests.ADTensorStructFlatten

open torch
open LeanTest

structure Inner where
  bias : T #[4]
  tag : Static String
  cache : Frozen #[4]
  deriving TensorStruct, ToTensorStructSchema, TensorStructFlatten

structure Outer where
  weight : T #[2, 4]
  inner : Inner
  gates : torch.Vector 2 (Frozen #[1])
  optScale : Option (T #[3])
  flag : Bool
  deriving TensorStruct, ToTensorStructSchema, TensorStructFlatten

private def templateOuter : Outer :=
  { weight := zeros #[2, 4]
    inner := {
      bias := zeros #[4]
      tag := "template-tag"
      cache := (full #[4] 7.0 : Frozen #[4])
    }
    gates := ⟨#[
      (full #[1] 10.0 : Frozen #[1]),
      (full #[1] 20.0 : Frozen #[1])
    ], rfl⟩
    optScale := some (zeros #[3])
    flag := true }

private def replacementOuter : Outer :=
  { weight := full #[2, 4] 3.0
    inner := {
      bias := full #[4] 4.0
      tag := "replacement-tag"
      cache := (full #[4] 8.0 : Frozen #[4])
    }
    gates := ⟨#[
      (full #[1] 30.0 : Frozen #[1]),
      (full #[1] 40.0 : Frozen #[1])
    ], rfl⟩
    optScale := some (full #[3] 5.0)
    flag := false }

private def expectedDiffOnlyRebuild : Outer :=
  { weight := replacementOuter.weight
    inner := {
      bias := replacementOuter.inner.bias
      tag := templateOuter.inner.tag
      cache := templateOuter.inner.cache
    }
    gates := templateOuter.gates
    optScale := replacementOuter.optScale
    flag := templateOuter.flag }

private def expectedDiffAndFrozenRebuild : Outer :=
  { weight := replacementOuter.weight
    inner := {
      bias := replacementOuter.inner.bias
      tag := templateOuter.inner.tag
      cache := replacementOuter.inner.cache
    }
    gates := replacementOuter.gates
    optScale := replacementOuter.optScale
    flag := templateOuter.flag }

private def leafRoles (leaves : Array TensorLeafValue) : Array TensorLeafRole :=
  leaves.map (·.role)

private def leafShapes (leaves : Array TensorLeafValue) : Array (Array Nat) :=
  leaves.map (·.shape)

private def leavesAllclose (lhs rhs : Array TensorLeafValue) : Bool := Id.run do
  if lhs.size != rhs.size then
    return false
  for h : i in [:lhs.size] do
    let l := lhs[i]
    let some r := rhs[i]? | return false
    if l.role != r.role then
      return false
    match l.payload, r.payload with
    | ⟨ls, lt⟩, ⟨rs, rt⟩ =>
        if hshape : ls = rs then
          let rt' : T ls := hshape ▸ rt
          if !(allclose lt rt') then
            return false
        else
          return false
  return true

private def unwrapOrFail {α : Type} (result : Except String α) : IO α :=
  match result with
  | .ok value => pure value
  | .error err => LeanTest.fail err

@[test]
def testTensorStructFlattenDiffOnlyUsesDifferentiableLeafOrder : IO Unit := do
  let leaves := TensorStructFlatten.flatten templateOuter .diffOnly
  LeanTest.assertEqual leaves.size 3
    "diffOnly flattening should keep only differentiable tensor leaves."
  let expectedRoles : Array TensorLeafRole := #[.diff, .diff, .diff]
  LeanTest.assertTrue (leafRoles leaves == expectedRoles)
    s!"diffOnly roles should be purely differentiable, got {reprStr (leafRoles leaves)}"
  let expectedShapes : Array (Array Nat) := #[#[2, 4], #[4], #[3]]
  LeanTest.assertTrue (leafShapes leaves == expectedShapes)
    s!"diffOnly flattening should preserve declaration-order tensor shapes, got {reprStr (leafShapes leaves)}"

@[test]
def testTensorStructFlattenDiffAndFrozenIncludesForwardOnlyLeaves : IO Unit := do
  let leaves := TensorStructFlatten.flatten templateOuter .diffAndFrozen
  LeanTest.assertEqual leaves.size 6
    "diffAndFrozen flattening should include forward-only frozen tensor leaves."
  let expectedRoles : Array TensorLeafRole := #[
    .diff,
    .diff,
    .frozen,
    .frozen,
    .frozen,
    .diff
  ]
  LeanTest.assertTrue (leafRoles leaves == expectedRoles)
    s!"diffAndFrozen roles should retain differentiable/frozen ordering, got {reprStr (leafRoles leaves)}"
  let expectedShapes : Array (Array Nat) := #[
    #[2, 4],
    #[4],
    #[4],
    #[1],
    #[1],
    #[3]
  ]
  LeanTest.assertTrue (leafShapes leaves == expectedShapes)
    s!"diffAndFrozen flattening should preserve all selected tensor shapes, got {reprStr (leafShapes leaves)}"

@[test]
def testTensorStructRebuildFromDiffOnlyReplacesOnlyDiffLeaves : IO Unit := do
  let rebuilt ← unwrapOrFail <|
    TensorStructFlatten.rebuildFrom templateOuter
      (TensorStructFlatten.flatten replacementOuter .diffOnly)
      .diffOnly
  LeanTest.assertTrue (rebuilt.flag == templateOuter.flag)
    "Rebuilding from diffOnly leaves should preserve static fields from the template."
  LeanTest.assertTrue (rebuilt.inner.tag == templateOuter.inner.tag)
    "Rebuilding from diffOnly leaves should preserve nested static fields from the template."
  LeanTest.assertTrue
    (leavesAllclose
      (TensorStructFlatten.flatten rebuilt .diffAndFrozen)
      (TensorStructFlatten.flatten expectedDiffOnlyRebuild .diffAndFrozen))
    "Rebuilding from diffOnly leaves should update only differentiable tensors."

@[test]
def testTensorStructRebuildFromDiffAndFrozenReplacesAllSelectedLeaves : IO Unit := do
  let rebuilt ← unwrapOrFail <|
    TensorStructFlatten.rebuildFrom templateOuter
      (TensorStructFlatten.flatten replacementOuter .diffAndFrozen)
      .diffAndFrozen
  LeanTest.assertTrue (rebuilt.flag == templateOuter.flag)
    "Rebuilding from diffAndFrozen leaves should still preserve static fields from the template."
  LeanTest.assertTrue (rebuilt.inner.tag == templateOuter.inner.tag)
    "Rebuilding from diffAndFrozen leaves should still preserve nested static fields from the template."
  LeanTest.assertTrue
    (leavesAllclose
      (TensorStructFlatten.flatten rebuilt .diffAndFrozen)
      (TensorStructFlatten.flatten expectedDiffAndFrozenRebuild .diffAndFrozen))
    "Rebuilding from diffAndFrozen leaves should update differentiable and frozen tensors."

@[test]
def testTensorStructRebuildRejectsShapeMismatch : IO Unit := do
  let diffLeaves := TensorStructFlatten.flatten replacementOuter .diffOnly
  let badLeaves := diffLeaves.set! 0 (TensorLeafValue.ofTensor .diff (full #[8] 1.0))
  match TensorStructFlatten.rebuildFrom templateOuter badLeaves .diffOnly with
  | .ok _ =>
      LeanTest.fail "Rebuild should reject tensor leaves whose shapes do not match the template."
  | .error err =>
      LeanTest.assertTrue (err.containsSubstr "weight")
        s!"Shape mismatch should point at the failing path, got: {err}"

@[test]
def testTensorStructRebuildRejectsMissingAndExtraLeaves : IO Unit := do
  let diffLeaves := TensorStructFlatten.flatten replacementOuter .diffOnly
  let missingLeaves := diffLeaves.extract 0 2
  match TensorStructFlatten.rebuildFrom templateOuter missingLeaves .diffOnly with
  | .ok _ =>
      LeanTest.fail "Rebuild should reject missing tensor leaves."
  | .error err =>
      LeanTest.assertTrue (err.containsSubstr "optScale")
        s!"Missing-leaf diagnostics should point at the first missing path, got: {err}"

  let extraLeaves := diffLeaves.push (TensorLeafValue.ofTensor .diff (full #[1] 0.0))
  match TensorStructFlatten.rebuildFrom templateOuter extraLeaves .diffOnly with
  | .ok _ =>
      LeanTest.fail "Rebuild should reject extra tensor leaves after consuming the template shape."
  | .error err =>
      LeanTest.assertTrue (err.containsSubstr "Unexpected extra tensor leaves")
        s!"Extra-leaf diagnostics should explain the unused suffix, got: {err}"

end Tests.ADTensorStructFlatten
