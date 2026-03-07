import LeanTest
import Tyr.AD.Sparse

namespace Tests.ADSparse

open LeanTest
open Tyr.AD.Sparse

private def mkEntry (src dst : Nat) (weight : Float) : SparseEntry :=
  { src := src, dst := dst, weight := weight }

private def mkMap
    (repr : String)
    (inDim outDim : Nat)
    (entries : Array SparseEntry) :
    SparseLinearMap :=
  {
    repr := SparseMapTag.namedStr repr
    inDim? := some inDim
    outDim? := some outDim
    entries := entries
  }

@[test]
def testValidateRejectsDuplicateCoords : IO Unit := do
  let m := mkMap "dup" 2 2 #[
    mkEntry 0 1 1.0,
    mkEntry 0 1 2.0
  ]
  match validateMap m with
  | .ok () =>
    LeanTest.fail "validateMap should reject duplicate sparse coordinates"
  | .error msg =>
    LeanTest.assertTrue (msg.contains "duplicate")
      s!"Expected duplicate-coordinate diagnostic, got: {msg}"

@[test]
def testAddCoalescesEntries : IO Unit := do
  let lhs := mkMap "lhs" 2 2 #[
    mkEntry 0 0 1.0,
    mkEntry 0 1 2.0
  ]
  let rhs := mkMap "rhs" 2 2 #[
    mkEntry 0 1 3.0,
    mkEntry 1 1 4.0
  ]
  match add lhs rhs with
  | .error msg =>
    LeanTest.fail s!"add should succeed, got: {msg}"
  | .ok out =>
    LeanTest.assertEqual out.inDim? (some 2) "Input dimension should be preserved"
    LeanTest.assertEqual out.outDim? (some 2) "Output dimension should be preserved"
    let expected : Array SparseEntry := #[
      mkEntry 0 0 1.0,
      mkEntry 0 1 5.0,
      mkEntry 1 1 4.0
    ]
    LeanTest.assertTrue (out.entries == expected)
      s!"add should coalesce duplicate coordinates by summing weights; expected={reprStr expected}, got={reprStr out.entries}"

@[test]
def testComposeMultipliesAndCoalesces : IO Unit := do
  let inMap := mkMap "in" 2 3 #[
    mkEntry 0 0 2.0,
    mkEntry 1 1 3.0,
    mkEntry 1 2 4.0
  ]
  let outMap := mkMap "out" 3 2 #[
    mkEntry 0 0 5.0,
    mkEntry 1 1 7.0,
    mkEntry 2 1 11.0
  ]
  match compose inMap outMap with
  | .error msg =>
    LeanTest.fail s!"compose should succeed on compatible maps, got: {msg}"
  | .ok composed =>
    LeanTest.assertEqual composed.inDim? (some 2) "Composed input dimension should be inherited from inMap"
    LeanTest.assertEqual composed.outDim? (some 2) "Composed output dimension should be inherited from outMap"
    let expected : Array SparseEntry := #[
      mkEntry 0 0 10.0,
      mkEntry 1 1 65.0
    ]
    LeanTest.assertTrue (composed.entries == expected)
      s!"compose should implement sparse multiplication and coalescing; expected={reprStr expected}, got={reprStr composed.entries}"

@[test]
def testComposeRejectsDimMismatch : IO Unit := do
  let inMap := mkMap "in" 2 3 #[mkEntry 0 0 1.0]
  let outMap := mkMap "out" 4 1 #[mkEntry 0 0 1.0]
  match compose inMap outMap with
  | .ok _ =>
    LeanTest.fail "compose should fail when middle dimensions do not match"
  | .error msg =>
    LeanTest.assertTrue (msg.contains "dimension mismatch")
      s!"Expected dimension-mismatch diagnostic, got: {msg}"

@[test]
def testComposeIdentityLikeKeepsKnownDimMetadata : IO Unit := do
  let idLike : SparseLinearMap := identityLike
  let outMap := mkMap "out" 3 2 #[mkEntry 0 0 1.0]
  match compose idLike outMap with
  | .error msg =>
    LeanTest.fail s!"compose should accept identity-like placeholder, got: {msg}"
  | .ok composed =>
    LeanTest.assertEqual composed.inDim? (some 3)
      "Identity-like placeholder should preserve known input dimensions from the composed map"

end Tests.ADSparse
