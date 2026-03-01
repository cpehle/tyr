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
import Lean.Util.Path

open torch
open torch.DataLoader

private def shakespeareTrainPath : String := "data/shakespeare_char/train.bin"
private def nanochatFixtureDir : String := "data/nanochat"

private def skipMissingFixture (testName path : String) : IO Unit := do
  IO.println s!"[skip] {testName}: missing fixture {path}"

private def requireFixture (testName path : String) : IO Bool := do
  let fixtureExists ← data.fileExists path
  if fixtureExists then
    pure true
  else
    skipMissingFixture testName path
    pure false

private def requireNanochatFixtures (testName : String) : IO Bool := do
  let dirPath : System.FilePath := ⟨nanochatFixtureDir⟩
  let dirExists ← dirPath.pathExists
  if !dirExists then
    skipMissingFixture testName nanochatFixtureDir
    return false
  let trainPath := s!"{nanochatFixtureDir}/fineweb_train_1m.bin"
  let valPath := s!"{nanochatFixtureDir}/fineweb_val_1m.bin"
  let trainExists ← data.fileExists trainPath
  let valExists ← data.fileExists valPath
  if !trainExists then
    skipMissingFixture testName trainPath
  if !valExists then
    skipMissingFixture testName valPath
  return trainExists && valExists

@[test]
def testSequentialLoader : IO Unit := do
  if !(← requireFixture "testSequentialLoader" shakespeareTrainPath) then
    return

  -- Load the data
  let ⟨n, _loader⟩ ← SequentialLoader.fromFile shakespeareTrainPath
  LeanTest.assertTrue (n >= 1000) s!"Expected at least 1000 tokens, got {n}"

@[test]
def testRandomBatchSampling : IO Unit := do
  if !(← requireFixture "testRandomBatchSampling" shakespeareTrainPath) then
    return

  let ⟨_, loader⟩ ← SequentialLoader.fromFile shakespeareTrainPath

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
  if !(← requireFixture "testBosFinderInit" shakespeareTrainPath) then
    return

  let ⟨_, loader⟩ ← SequentialLoader.fromFile shakespeareTrainPath

  -- Use newline (token 0 in Shakespeare vocab) as BOS equivalent
  let bosToken : UInt64 := 0
  let finder ← BOSFinder.init loader.tokens bosToken

  LeanTest.assertTrue (finder.bosPositions.size > 0) "Found BOS positions"

@[test]
def testSequentialBatchIterator : IO Unit := do
  if !(← requireFixture "testSequentialBatchIterator" shakespeareTrainPath) then
    return

  let ⟨_, loader⟩ ← SequentialLoader.fromFile shakespeareTrainPath

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
  if !(← requireFixture "testEpochReset" shakespeareTrainPath) then
    return

  let ⟨_, loader⟩ ← SequentialLoader.fromFile shakespeareTrainPath

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

  LeanTest.assertTrue sawEpochEnd "Did not see epoch end (data might be very large)"
  LeanTest.assertEqual currentIter.epoch 1 "Expected epoch 1 after reset"

@[test]
def testDocumentAwareLoader : IO Unit := do
  if !(← requireFixture "testDocumentAwareLoader" shakespeareTrainPath) then
    return

  -- Load shard (single process, so shard 0 of 1)
  let ⟨_, shard⟩ ← DataShard.loadFromFile shakespeareTrainPath 0 1 0  -- bosToken = 0 (newline)

  -- Test batch extraction
  let batchSize : UInt64 := 4
  let seqLen : UInt64 := 32

  let (maybeBatch, _) ← shard.bosFinder.getBatch shard.tokens batchSize seqLen
  LeanTest.assertTrue maybeBatch.isSome "Could not get batch"

@[test]
def testShuffleDeterminism : IO Unit := do
  if !(← requireFixture "testShuffleDeterminism" shakespeareTrainPath) then
    return

  let ⟨_, loader⟩ ← SequentialLoader.fromFile shakespeareTrainPath
  let finder ← BOSFinder.init loader.tokens 0  -- Use newline as BOS

  -- Shuffle with same seed twice
  let seed : UInt64 := 42
  let shuffled1 := finder.shuffle seed
  let shuffled2 := finder.shuffle seed

  -- Check first few positions match
  let match1 := shuffled1.bosPositions.toList.take 10
  let match2 := shuffled2.bosPositions.toList.take 10

  LeanTest.assertEqual match1 match2 "Shuffle not deterministic!"

@[test]
def testResolveShardPathsDirectoryAndPrefix : IO Unit := do
  if !(← requireNanochatFixtures "testResolveShardPathsDirectoryAndPrefix") then
    return

  let trainFromDir ← resolveShardPaths nanochatFixtureDir .train
  let valFromDir ← resolveShardPaths nanochatFixtureDir .val

  LeanTest.assertTrue (trainFromDir.size >= 2)
    s!"Expected at least two train shards, got {trainFromDir.size}"
  LeanTest.assertTrue (!valFromDir.isEmpty)
    s!"Expected at least one validation shard in {nanochatFixtureDir}"

  let trainFromPrefix ← resolveShardPaths s!"{nanochatFixtureDir}/fineweb_train" .train
  let valFromPrefix ← resolveShardPaths s!"{nanochatFixtureDir}/fineweb_val" .val

  LeanTest.assertEqual trainFromPrefix trainFromDir
  LeanTest.assertEqual valFromPrefix valFromDir

@[test]
def testDistributedGeneratorRotatesAcrossTrainShards : IO Unit := do
  if !(← requireNanochatFixtures "testDistributedGeneratorRotatesAcrossTrainShards") then
    return

  let cfg : Config := {
    dataPath := nanochatFixtureDir
    valPath := none
    seqLen := 128
    bosToken := 50256
    numWorkers := 1
    bufferSize := 1
    seed := 42
  }

  let gen ← DistributedDataGenerator.init cfg 8 128
  if gen.trainPaths.size < 2 then
    LeanTest.fail s!"Expected at least two train shards, got {gen.trainPaths.size}"
    return

  let oldIdx := gen.trainPathIdx
  let exhaustedFinder := {
    gen.iterator.shard.bosFinder with
      currentPos := gen.iterator.shard.bosFinder.dataLen
  }
  let exhaustedShard := { gen.iterator.shard with bosFinder := exhaustedFinder }
  let exhaustedIter := { gen.iterator with shard := exhaustedShard }
  let exhaustedGen := { gen with iterator := exhaustedIter }

  let (maybeBatch, rotatedGen) ← exhaustedGen.nextBatch
  LeanTest.assertTrue maybeBatch.isSome "Expected a batch after rotating to next shard"

  let expectedIdx := (oldIdx + 1) % gen.trainPaths.size
  LeanTest.assertEqual rotatedGen.trainPathIdx expectedIdx
  LeanTest.assertTrue (gen.trainPaths[oldIdx]! != rotatedGen.trainPaths[rotatedGen.trainPathIdx]!)
    "Expected rotation to switch the active shard path"

/-- Parse command line arguments into a LeanTest RunConfig. -/
private def parseArgs (args : List String) : IO LeanTest.RunConfig := do
  let mut config : LeanTest.RunConfig := {}
  let mut remaining := args
  while _h : !remaining.isEmpty do
    match remaining with
    | "--filter" :: pattern :: rest =>
      config := { config with filter := some pattern }
      remaining := rest
    | "--ignored" :: rest =>
      config := { config with includeIgnored := true }
      remaining := rest
    | "--fail-fast" :: rest =>
      config := { config with failFast := true }
      remaining := rest
    | "--help" :: _ =>
      IO.println "Usage: TestDataLoader [OPTIONS]"
      IO.println ""
      IO.println "Options:"
      IO.println "  --filter PATTERN  Only run tests matching PATTERN"
      IO.println "  --ignored         Include tests marked as ignored"
      IO.println "  --fail-fast       Stop on first failure"
      IO.println "  --help            Show this help"
      IO.Process.exit 0
    | _ :: rest =>
      remaining := rest
    | [] => remaining := []
  return config

/-- Standalone executable entrypoint for the DataLoader test module. -/
unsafe def main (args : List String) : IO UInt32 := do
  let config ← parseArgs args
  Lean.initSearchPath (← Lean.findSysroot)
  Lean.enableInitializersExecution
  let env ← Lean.importModules
    #[{ module := `LeanTest }, { module := `Tests.TestDataLoader }]
    {}
  LeanTest.runTestsAndExit env {} config
