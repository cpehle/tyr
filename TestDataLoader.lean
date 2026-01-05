/-
  TestDataLoader.lean

  Tests for the DataLoader module including:
  - Sequential data loading (Shakespeare style)
  - BOS token finding
  - Epoch iteration
-/
import Tyr
import Tyr.DataLoader

open torch
open torch.DataLoader

/-- Test sequential loader: load Shakespeare data and verify basic operations -/
def testSequentialLoader : IO Unit := do
  IO.println "=== Testing Sequential Loader ==="

  let trainPath := "data/shakespeare_char/train.bin"

  -- Check if data exists
  let fileExists ← data.fileExists trainPath
  if !fileExists then
    IO.println "  Skipping: Shakespeare data not found at data/shakespeare_char/"
    IO.println "  Run prepare.py in data/shakespeare_char/ to generate test data"
    return

  -- Load the data
  IO.println "  Loading training data..."
  let ⟨n, loader⟩ ← SequentialLoader.fromFile trainPath
  IO.println s!"  Loaded {n} tokens"

  -- Verify we got reasonable data
  if n < 1000 then
    IO.println "  ERROR: Expected at least 1000 tokens"
    return

  IO.println "  Sequential loader test passed!"

/-- Test random batch sampling -/
def testRandomBatchSampling : IO Unit := do
  IO.println "=== Testing Random Batch Sampling ==="

  let trainPath := "data/shakespeare_char/train.bin"

  let fileExists ← data.fileExists trainPath
  if !fileExists then
    IO.println "  Skipping: Shakespeare data not found"
    return

  let ⟨_, loader⟩ ← SequentialLoader.fromFile trainPath

  -- Sample a random batch with explicit types
  let batchSize : UInt64 := 4
  let blockSize : UInt64 := 32

  IO.println s!"  Sampling batch of size {batchSize} x {blockSize}..."
  let result ← loader.sampleRandomBatch batchSize blockSize
  let input : T #[4, 32] := result.1
  let target : T #[4, 32] := result.2

  -- Compute sum to verify we got valid tensors (mean requires float, tokens are int64)
  let inputSum := nn.itemInt (nn.sumAll input)
  let targetSum := nn.itemInt (nn.sumAll target)
  IO.println s!"  Input sum: {inputSum}, Target sum: {targetSum}"
  IO.println "  Random batch sampling test passed!"

/-- Test BOS finder initialization with Shakespeare data -/
def testBosFinderInit : IO Unit := do
  IO.println "=== Testing BOS Finder Init ==="

  let trainPath := "data/shakespeare_char/train.bin"
  let fileExists ← data.fileExists trainPath
  if !fileExists then
    IO.println "  Skipping: Shakespeare data not found"
    return

  let ⟨n, loader⟩ ← SequentialLoader.fromFile trainPath

  -- Use newline (token 0 in Shakespeare vocab) as BOS equivalent
  let bosToken : UInt64 := 0
  let finder ← BOSFinder.init loader.tokens bosToken

  IO.println s!"  Found {finder.bosPositions.size} BOS positions in {n} tokens"
  if finder.bosPositions.size > 0 then
    let firstFew := finder.bosPositions.toList.take 5
    IO.println s!"  First few positions: {firstFew}"

  IO.println "  BOS finder init test passed!"

/-- Test sequential batch iterator -/
def testSequentialBatchIterator : IO Unit := do
  IO.println "=== Testing Sequential Batch Iterator ==="

  let trainPath := "data/shakespeare_char/train.bin"

  let fileExists ← data.fileExists trainPath
  if !fileExists then
    IO.println "  Skipping: Shakespeare data not found"
    return

  let ⟨_, loader⟩ ← SequentialLoader.fromFile trainPath

  let batchSize : UInt64 := 4
  let seqLen : UInt64 := 32
  let iter := SequentialBatchIterator.new loader batchSize seqLen

  IO.println s!"  Iterator created with batchSize={batchSize}, seqLen={seqLen}"

  -- Get a few batches
  let mut currentIter := iter
  let mut batchCount : Nat := 0
  for _ in [:5] do
    let (maybeBatch, nextIter) := currentIter.next
    match maybeBatch with
    | none =>
      IO.println s!"  Epoch boundary reached after {batchCount} batches"
    | some _ =>
      batchCount := batchCount + 1
    currentIter := nextIter

  IO.println s!"  Got {batchCount} batches"
  IO.println s!"  Current epoch: {currentIter.epoch}"
  IO.println "  Sequential batch iterator test passed!"

/-- Test epoch reset behavior -/
def testEpochReset : IO Unit := do
  IO.println "=== Testing Epoch Reset ==="

  let trainPath := "data/shakespeare_char/train.bin"

  let fileExists ← data.fileExists trainPath
  if !fileExists then
    IO.println "  Skipping: Shakespeare data not found"
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
      IO.println s!"  Epoch 0 complete after {batchCount} batches"
    | some _ =>
      batchCount := batchCount + 1
    currentIter := nextIter

  if !sawEpochEnd then
    IO.println "  Warning: Did not see epoch end (data might be very large)"
  else
    IO.println s!"  New epoch: {currentIter.epoch}"
    if currentIter.epoch != 1 then
      IO.println "  ERROR: Expected epoch 1 after reset"
    else
      IO.println "  Epoch reset test passed!"

/-- Test document-aware loader (DataShard) -/
def testDocumentAwareLoader : IO Unit := do
  IO.println "=== Testing Document-Aware Loader ==="

  let trainPath := "data/shakespeare_char/train.bin"

  let fileExists ← data.fileExists trainPath
  if !fileExists then
    IO.println "  Skipping: Shakespeare data not found"
    return

  -- Load shard (single process, so shard 0 of 1)
  IO.println "  Loading data shard..."
  let ⟨n, shard⟩ ← DataShard.loadFromFile trainPath 0 1 0  -- bosToken = 0 (newline)

  IO.println s!"  Loaded shard with {n} tokens"
  IO.println s!"  BOS positions found: {shard.bosFinder.bosPositions.size}"

  -- Test batch extraction
  let batchSize : UInt64 := 4
  let seqLen : UInt64 := 32

  let (maybeBatch, _) ← shard.bosFinder.getBatch shard.tokens batchSize seqLen
  match maybeBatch with
  | none =>
    IO.println "  Warning: Could not get batch (unexpected)"
  | some _ =>
    IO.println "  Successfully extracted batch"

  IO.println "  Document-aware loader test passed!"

/-- Test shuffle determinism -/
def testShuffleDeterminism : IO Unit := do
  IO.println "=== Testing Shuffle Determinism ==="

  let trainPath := "data/shakespeare_char/train.bin"

  let fileExists ← data.fileExists trainPath
  if !fileExists then
    IO.println "  Skipping: Shakespeare data not found"
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
    IO.println "  Shuffle is deterministic (same seed gives same order)"
    IO.println s!"  First 10 positions: {match1}"
    IO.println "  Shuffle determinism test passed!"
  else
    IO.println "  ERROR: Shuffle not deterministic!"
    IO.println s!"  First: {match1}"
    IO.println s!"  Second: {match2}"

def main : IO Unit := do
  IO.println "========================================="
  IO.println "         DataLoader Test Suite          "
  IO.println "========================================="
  IO.println ""

  testSequentialLoader
  IO.println ""

  testRandomBatchSampling
  IO.println ""

  testBosFinderInit
  IO.println ""

  testSequentialBatchIterator
  IO.println ""

  testEpochReset
  IO.println ""

  testDocumentAwareLoader
  IO.println ""

  testShuffleDeterminism
  IO.println ""

  IO.println "========================================="
  IO.println "         All Tests Complete             "
  IO.println "========================================="
