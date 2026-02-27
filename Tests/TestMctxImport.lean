import LeanTest
import Tyr.Mctx

open torch.mctx

@[test]
def testMctxImportSymbols : IO Unit := do
  let root : RootFnOutput Unit := {
    priorLogits := #[0.0, 1.0]
    value := 0.0
    embedding := ()
  }

  let recurrent : RecurrentFn Unit Unit := fun _ _ _ _ =>
    ({ reward := 0.0, discount := 1.0, priorLogits := #[0.0, 0.0], value := 0.0 }, ())

  let _p := muzeroPolicy (params := ()) (rngKey := 0) (root := root) (recurrentFn := recurrent) (numSimulations := 1)
  let _a := alphazeroPolicy (params := ()) (rngKey := 0) (root := root) (recurrentFn := recurrent) (numSimulations := 1)
  let _g := gumbelMuZeroPolicy (params := ()) (rngKey := 0) (root := root) (recurrentFn := recurrent) (numSimulations := 1)
  let _sub := getSubtree (_a.searchTree) 0
  let _rst := resetSearchTree _sub
  let _q1 := qtransformByMinMax (_p.searchTree) ROOT_INDEX (-1.0) 1.0
  let _q2 := qtransformByParentAndSiblings (_p.searchTree) ROOT_INDEX
  let _q3 := qtransformCompletedByMixValue (_g.searchTree) ROOT_INDEX

  LeanTest.assertTrue true "Mctx symbols should resolve and run"
