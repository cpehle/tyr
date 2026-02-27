import LeanTest
import Tyr.Mctx

open torch.mctx

private def checkVisits
    (expected : Array Nat)
    (maxNumConsideredActions numSimulations : Nat)
    : IO Unit := do
  LeanTest.assertEqual expected.size numSimulations "Expected sequence length should match num_simulations"
  let got := getSequenceOfConsideredVisits maxNumConsideredActions numSimulations
  LeanTest.assertEqual got expected s!"Unexpected considered-visit sequence for m={maxNumConsideredActions}, sims={numSimulations}"

@[test]
def testSeqHalvingConsideredMinSims : IO Unit := do
  let expected : Array Nat := #[
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 4, 4, 5, 5, 6, 6
  ]
  checkVisits expected 8 24

@[test]
def testSeqHalvingConsideredExtraSims : IO Unit := do
  let expected : Array Nat := #[
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    4, 4, 5, 5, 6, 6, 7, 7,
    8, 8, 9, 9, 10, 10, 11, 11,
    12, 12, 13, 13, 14, 14, 15, 15,
    16, 16, 17
  ]
  checkVisits expected 8 47

@[test]
def testSeqHalvingConsideredLessSims : IO Unit := do
  let expected : Array Nat := #[0, 0]
  checkVisits expected 8 2

@[test]
def testSeqHalvingConsideredLessSims2 : IO Unit := do
  let expected : Array Nat := #[
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1,
    2
  ]
  checkVisits expected 8 13

@[test]
def testSeqHalvingNotPowerOfTwo : IO Unit := do
  let expected : Array Nat := #[
    0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 2, 2, 2,
    3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8
  ]
  checkVisits expected 7 24

@[test]
def testSeqHalvingAction0 : IO Unit := do
  let expected := (List.range 16).toArray
  checkVisits expected 0 16

@[test]
def testSeqHalvingAction1 : IO Unit := do
  let expected := (List.range 16).toArray
  checkVisits expected 1 16
