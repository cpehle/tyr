import LeanTest
import Tyr.DiffEq.Path

namespace Tests.DiffEqPathComposeParity

open LeanTest
open torch
open torch.DiffEq

private def approx (a b tol : Float) : Bool :=
  Float.abs (a - b) < tol

@[test] def testComposePointEvaluationParity : IO Unit := do
  let leftPath : AbstractPath Float :=
    AbstractPath.ofPosition 0.0 1.0 (fun t => t)
  let rightPath : AbstractPath Float :=
    AbstractPath.ofPosition 1.0 2.0 (fun t => t + 10.0)
  let path := AbstractPath.compose leftPath rightPath

  let beforeSplit := path.evaluate 0.25 none true
  LeanTest.assertTrue (approx beforeSplit 0.25 1e-12)
    s!"Expected point value 0.25 before split, got {beforeSplit}"

  let afterSplit := path.evaluate 1.25 none true
  LeanTest.assertTrue (approx afterSplit 11.25 1e-12)
    s!"Expected point value 11.25 after split, got {afterSplit}"

  let splitLeft := path.evaluate 1.0 none true
  LeanTest.assertTrue (approx splitLeft 1.0 1e-12)
    s!"Expected left-continuous split value 1.0, got {splitLeft}"

  let splitRight := path.evaluate 1.0 none false
  LeanTest.assertTrue (approx splitRight 11.0 1e-12)
    s!"Expected right-continuous split value 11.0, got {splitRight}"

@[test] def testComposeCrossSplitIncrements : IO Unit := do
  let leftPath := AbstractPath.linearInterpolation 0.0 1.0 (0.0 : Float) (2.0 : Float)
  let rightPath := AbstractPath.linearInterpolation 1.0 2.0 (2.0 : Float) (3.0 : Float)
  let path := AbstractPath.compose leftPath rightPath

  let forward := path.evaluate 0.25 (some 1.5) true
  LeanTest.assertTrue (approx forward 2.0 1e-12)
    s!"Expected forward crossing increment 2.0, got {forward}"

  let backward := path.evaluate 1.5 (some 0.25) true
  LeanTest.assertTrue (approx backward (-2.0) 1e-12)
    s!"Expected backward crossing increment -2.0, got {backward}"

end Tests.DiffEqPathComposeParity
