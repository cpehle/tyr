/-
  TestDataLoader.lean

  Tests for the DataLoader module including:
  - Sequential data loading (Shakespeare style)
  - BOS token finding
  - Epoch iteration
-/
import Tyr
import Tyr.DataLoader
import Examples.GPT.GPTDataLoader
import LeanTest

open torch
open torch.DataLoader

@[test]
def testSequentialLoader : IO Unit := do
  let trainPath := "data/shakespeare_char/train.bin"

  -- Check if data exists
  let fileExists ← data.fileExists trainPath
  if !fileExists then
    LeanTest.assertFalse false
    return

  -- Load the data
  let ⟨n, _loader⟩ ← SequentialLoader.fromFile trainPath

  -- Verify we got reasonable data
  if n < 1000 then
    LeanTest.fail "Expected at least 1000 tokens"
    return

  LeanTest.assertTrue true

@[test]
def testRandomBatchSampling : IO Unit := do
  let trainPath := "data/shakespeare_char/train.bin"

  let fileExists ← data.fileExists trainPath
  if !fileExists then
    return

  let ⟨_, loader⟩ ← SequentialLoader.fromFile trainPath

  -- Sample a random batch with explicit types
  let batchSize : UInt64 := 4
  let blockSize : UInt64 := 32

  let result ← loader.sampleRandomBatch batchSize blockSize
  let input : T #[4, 32] := result.1
  let target : T #[4, 32] := result.2

  -- Compute sum to verify we got valid tensors (mean requires float, tokens are int64)
  let inputSum := nn.itemInt (nn.sumAll input)
  let targetSum := nn.itemInt (nn.sumAll target)
  
  LeanTest.assertTrue (inputSum > 0) "Input sum positive"
  LeanTest.assertTrue (targetSum > 0) "Target sum positive"

@[test]
def testBosFinderInit : IO Unit := do
  let trainPath := "data/shakespeare_char/train.bin"
  let fileExists ← data.fileExists trainPath
  if !fileExists then
    return

  let ⟨_, loader⟩ ← SequentialLoader.fromFile trainPath

  -- Use newline (token 0 in Shakespeare vocab) as BOS equivalent
  let bosToken : UInt64 := 0
  let finder ← BOSFinder.init loader.tokens bosToken

  LeanTest.assertTrue (finder.bosPositions.size > 0) "Found BOS positions"

@[test]
def testSequentialBatchIterator : IO Unit := do
  let trainPath := "data/shakespeare_char/train.bin"

  let fileExists ← data.fileExists trainPath
  if !fileExists then
    return

  let ⟨_, loader⟩ ← SequentialLoader.fromFile trainPath

  let batchSize : UInt64 := 4
  let seqLen : UInt64 := 32
  let iter := SequentialBatchIterator.new loader batchSize seqLen

  -- Get a few batches
  let mut currentIter := iter
  let mut batchCount : Nat := 0
  for _ in [:5] do
    let (maybeBatch, nextIter) := currentIter.next
    match maybeBatch with
    | none =>
      pure ()
    | some _ =>
      batchCount := batchCount + 1
    currentIter := nextIter

  LeanTest.assertEqual batchCount 5

@[test]
def testEpochReset : IO Unit := do
  let trainPath := "data/shakespeare_char/train.bin"

  let fileExists ← data.fileExists trainPath
  if !fileExists then
    return

  let ⟨_, loader⟩ ← SequentialLoader.fromFile trainPath

  -- Use large batch to quickly exhaust data
  let batchSize : UInt64 := 100
  let seqLen : UInt64 := 64
  let iter := SequentialBatchIterator.new loader batchSize seqLen

  -- Iterate until we hit epoch boundary
  let mut currentIter := iter
  let mut batchCount : Nat := 0
  let mut sawEpochEnd := false

  for _ in [:10000] do  -- Safety limit
    if sawEpochEnd then break
    let (maybeBatch, nextIter) := currentIter.next
    match maybeBatch with
    | none =>
      sawEpochEnd := true
    | some _ =>
      batchCount := batchCount + 1
    currentIter := nextIter

  if !sawEpochEnd then
    LeanTest.fail "Did not see epoch end (data might be very large)"
  else
    if currentIter.epoch != 1 then
      LeanTest.fail "Expected epoch 1 after reset"
    else
      LeanTest.assertTrue true

@[test]
def testDocumentAwareLoader : IO Unit := do
  let trainPath := "data/shakespeare_char/train.bin"

  let fileExists ← data.fileExists trainPath
  if !fileExists then
    return

  -- Load shard (single process, so shard 0 of 1)
  let ⟨_, shard⟩ ← DataShard.loadFromFile trainPath 0 1 0  -- bosToken = 0 (newline)

  -- Test batch extraction
  let batchSize : UInt64 := 4
  let seqLen : UInt64 := 32

  let (maybeBatch, _) ← shard.bosFinder.getBatch shard.tokens batchSize seqLen
  match maybeBatch with
  | none =>
    LeanTest.fail "Could not get batch"
  | some _ =>
    LeanTest.assertTrue true

@[test]
def testShuffleDeterminism : IO Unit := do
  let trainPath := "data/shakespeare_char/train.bin"

  let fileExists ← data.fileExists trainPath
  if !fileExists then
    return

  let ⟨_, loader⟩ ← SequentialLoader.fromFile trainPath
  let finder ← BOSFinder.init loader.tokens 0  -- Use newline as BOS

  -- Shuffle with same seed twice
  let seed : UInt64 := 42
  let shuffled1 := finder.shuffle seed
  let shuffled2 := finder.shuffle seed

  -- Check first few positions match
  let match1 := shuffled1.bosPositions.toList.take 10
  let match2 := shuffled2.bosPositions.toList.take 10

  if match1 == match2 then
    LeanTest.assertTrue true
  else
    LeanTest.fail "Shuffle not deterministic!"


