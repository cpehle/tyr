import Tyr
import Tyr.Manifolds.Orthogonal
import Tyr.Manifolds.Grassmann
import Tyr.Model.BranchingFlows
import LeanTest

/-!
# `Tests.TestBranchingFlows`

Branching flow tests for tree sampling behavior and manifold-related helpers.

## Overview
- Regression and behavior checks run by the LeanTest-based test suite.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

open torch
open torch.branching
open Tyr.AD

private def clamp01 (x : Float) : Float :=
  max 0.0 (min x 1.0)

private def uniformTimeDist : TimeDist :=
  { cdf := fun t => clamp01 t,
    pdf := fun t => if t < 0.0 || t > 1.0 then 0.0 else 1.0,
    quantile := fun p => clamp01 p }

private def mkState (vals : Array Nat) (groups : Array Int) : BranchingState Nat :=
  let n := vals.size
  { state := vals,
    groupings := groups,
    del := Array.replicate n false,
    ids := (Array.range n).map (fun i => Int.ofNat (i + 1)),
    branchmask := Array.replicate n true,
    flowmask := Array.replicate n true,
    padmask := Array.replicate n true }

@[test]
def testBranchingSampleForestSimple : IO Unit := do
  let elements : Array Nat := #[1, 2]
  let groupings : Array Int := #[0, 0]
  let branchable : Array Bool := #[true, true]
  let flowable : Array Bool := #[true, true]
  let deleted : Array Bool := #[false, false]
  let ids : Array Int := #[1, 2]
  let merger := fun (a b : Nat) (_w1 _w2 : Nat) => a + b
  let (roots, times, _rng) :=
    sampleForest elements groupings branchable flowable deleted ids uniformTimeDist
      (sequentialUniformPolicy Nat) 1.0 merger none { state := 1 }
  LeanTest.assertEqual roots.size 1 "Forest collapses to one root"
  LeanTest.assertEqual times.size 1 "Two leaves yield one split time"
  let t := times[0]!
  LeanTest.assertTrue (t >= 0.0 && t <= 1.0) "Split time in [0,1]"
  let root := roots[0]!
  LeanTest.assertEqual root.weight 2 "Merged weight is 2"
  LeanTest.assertEqual root.group 0 "Group preserved"
  LeanTest.assertEqual root.children.size 2 "Root has two children"

@[test]
def testBranchingFixedcountInsertionsLength : IO Unit := do
  let x := mkState #[10, 20, 30] #[0, 0, 1]
  let (x', _rng) := fixedcountDelInsertions x 2 { state := 42 }
  LeanTest.assertEqual x'.state.size (x.state.size + 2) "Adds exactly numEvents elements"
  LeanTest.assertEqual x'.del.size x'.state.size "Deletion flags match length"
  LeanTest.assertEqual x'.groupings.size x'.state.size "Groupings match length"
  LeanTest.assertEqual x'.ids.size x'.state.size "Ids match length"
  LeanTest.assertEqual x'.branchmask.size x'.state.size "Branchmask match length"
  LeanTest.assertEqual x'.flowmask.size x'.state.size "Flowmask match length"
  LeanTest.assertEqual x'.padmask.size x'.state.size "Padmask match length"

@[test]
def testGeodesicInterpolateFloat : IO Unit := do
  let x := 0.0
  let y := 10.0
  let z := canonicalAnchorMerge x y 1 3
  LeanTest.assertTrue (Float.abs (z - 7.5) < 1.0e-6) "Geodesic interpolate matches weighted average"

@[test]
def testOrthogonalLogExp : IO Unit := do
  let Y ← Orthogonal.random 2
  let prod := torch.nn.mm (torch.nn.transpose2d Y.matrix) Y.matrix
  LeanTest.assertTrue (torch.allclose prod (torch.eye 2) 1.0e-4 1.0e-4) "exp preserves orthogonality"

@[test]
def testGrassmannDistanceSelf : IO Unit := do
  let X ← Grassmann.random 3 1
  let d := Grassmann.distance X X
  LeanTest.assertTrue (Float.abs d < 1.0e-4) "Grassmann self-distance is near zero"
