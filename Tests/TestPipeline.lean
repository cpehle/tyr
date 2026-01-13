/-
  Tests/TestPipeline.lean

  Tests for Pipeline orchestration (Tyr/Pipeline.lean) and
  Data/Pipeline scheduling (Tyr/Data/Pipeline.lean).
-/
import Tyr.Pipeline
import Tyr.Data.Pipeline
import LeanTest

namespace Tests.Pipeline

open torch.Pipeline
open torch.Data.Pipeline

/-! ## Float Assertion Helpers -/

/-- Assert two floats are approximately equal within epsilon -/
def assertFloatEq (actual expected : Float) (epsilon : Float := 0.001)
    (msg : String := "") : IO Unit := do
  let diff := Float.abs (actual - expected)
  if diff > epsilon then
    LeanTest.fail s!"{msg}: expected {expected}, got {actual} (diff {diff} > epsilon {epsilon})"

/-- Assert a float is less than a threshold -/
def assertFloatLt (actual threshold : Float) (msg : String := "") : IO Unit := do
  if actual >= threshold then
    LeanTest.fail s!"{msg}: expected {actual} < {threshold}"

/-- Assert a float is greater than a threshold -/
def assertFloatGt (actual threshold : Float) (msg : String := "") : IO Unit := do
  if actual <= threshold then
    LeanTest.fail s!"{msg}: expected {actual} > {threshold}"

/-- Assert a float is in range [lo, hi] -/
def assertFloatInRange (actual lo hi : Float) (msg : String := "") : IO Unit := do
  if actual < lo || actual > hi then
    LeanTest.fail s!"{msg}: expected {actual} in [{lo}, {hi}]"

/-! ## RetryPolicy Tests -/

@[test]
def testRetryDelayExponentialBackoff : IO Unit := do
  let policy : RetryPolicy := { initialDelayMs := 100, backoffMultiplier := 2.0 }
  -- 100 * 2^0 = 100, 100 * 2^1 = 200, 100 * 2^2 = 400
  LeanTest.assertEqual (policy.delayForAttempt 0) 100
  LeanTest.assertEqual (policy.delayForAttempt 1) 200
  LeanTest.assertEqual (policy.delayForAttempt 2) 400

@[test]
def testRetryDelayMaxCap : IO Unit := do
  let policy : RetryPolicy := { initialDelayMs := 1000, maxDelayMs := 5000, backoffMultiplier := 2.0 }
  -- At attempt 10: 1000 * 2^10 = 1024000, but should be capped at 5000
  let delay := policy.delayForAttempt 10
  LeanTest.assertTrue (delay <= 5000) s!"delay {delay} should be capped at maxDelayMs 5000"

/-! ## StageStatus Serialization Tests -/

@[test]
def testStageStatusJsonRoundTrip : IO Unit := do
  -- Test all variants
  for status in [StageStatus.pending, .running, .completed, .failed "test error"] do
    let json := toJson status
    match (fromJson? json : Except String StageStatus) with
    | .ok status' => LeanTest.assertTrue (status == status') "status should round-trip"
    | .error e => LeanTest.fail s!"Failed to parse: {e}"

/-! ## StageInfo Tests -/

@[test]
def testStageInfoJsonRoundTrip : IO Unit := do
  let info : StageInfo := {
    name := "test-stage"
    status := .completed
    startTime := some 1000
    endTime := some 2000
    retryCount := 1
  }
  let json := toJson info
  match (fromJson? json : Except String StageInfo) with
  | .ok info' =>
    LeanTest.assertEqual info.name info'.name
    LeanTest.assertEqual info.retryCount info'.retryCount
  | .error e => LeanTest.fail s!"Failed to parse StageInfo: {e}"

@[test]
def testStageInfoDurationMs : IO Unit := do
  let info : StageInfo := {
    name := "test"
    status := .completed
    startTime := some 1000
    endTime := some 5000
  }
  LeanTest.assertEqual info.durationMs (some 4000)

/-! ## PipelineCheckpoint Serialization Tests -/

@[test]
def testPipelineCheckpointJsonRoundTrip : IO Unit := do
  let checkpoint : PipelineCheckpoint := {
    stages := #[
      { name := "stage1", status := .completed, startTime := some 100, endTime := some 200 },
      { name := "stage2", status := .failed "error", startTime := some 300, endTime := some 400 }
    ]
    timestamp := 12345
    runId := "test-run-123"
  }
  let json := toJson checkpoint
  match (fromJson? json : Except String PipelineCheckpoint) with
  | .ok checkpoint' =>
    LeanTest.assertEqual checkpoint.stages.size checkpoint'.stages.size
    LeanTest.assertEqual checkpoint.runId checkpoint'.runId
  | .error e => LeanTest.fail s!"Failed to parse PipelineCheckpoint: {e}"

/-! ## Duration Formatting Tests -/

@[test]
def testFormatDuration : IO Unit := do
  LeanTest.assertEqual (formatDuration 5000) "5s"
  LeanTest.assertEqual (formatDuration 65000) "1m5s"
  LeanTest.assertEqual (formatDuration 3665000) "1h1m5s"

/-! ## Report Section Markdown Tests -/

@[test]
def testReportSectionMarkdown : IO Unit := do
  let header := ReportSection.header "content"
  LeanTest.assertTrue (header.toMarkdown.containsSubstr "# Training Report") "header format"

  let stage := ReportSection.stage "pretrain" "Duration: 10m"
  LeanTest.assertTrue (stage.toMarkdown.containsSubstr "## pretrain") "stage format"

  let metrics := ReportSection.metrics "loss" [("train", "0.5")]
  LeanTest.assertTrue (metrics.toMarkdown.containsSubstr "train: 0.5") "metrics format"

/-! ## Data/Pipeline LR Scheduling Tests -/

@[test]
def testBatchSizeScale : IO Unit := do
  -- Same batch size = scale of 1.0
  assertFloatEq (batchSizeScale 524288 524288) 1.0 (msg := "same batch size")
  -- Smaller batch = smaller scale; sqrt(0.5) ≈ 0.707
  assertFloatEq (batchSizeScale 262144 524288) 0.7071 (epsilon := 0.01) "sqrt scaling"
  -- Larger batch = larger scale; sqrt(2) ≈ 1.414
  assertFloatEq (batchSizeScale 1048576 524288) 1.414 (epsilon := 0.01) "sqrt scaling"

@[test]
def testWeightDecayScale : IO Unit := do
  -- Reference depth (12) should give scale of 1.0
  assertFloatEq (weightDecayScale 12) 1.0 (msg := "reference depth")
  -- Deeper model = smaller weight decay
  assertFloatLt (weightDecayScale 24) (weightDecayScale 12) "deeper model smaller WD"

@[test]
def testStageLRMultiplierPlateau : IO Unit := do
  -- In plateau phase (no warmup, before warmdown), multiplier should be 1.0
  let cfg : StageLRConfig := {
    warmupRatio := 0.0
    warmdownRatio := 0.4
    initLrFrac := 1.0
    minLrFrac := 0.0
  }
  -- At step 100 of 1000 total (10%), in plateau phase
  assertFloatEq (getStageLRMultiplier cfg 100 1000) 1.0 (msg := "plateau phase")

@[test]
def testStageLRMultiplierWarmdown : IO Unit := do
  let cfg : StageLRConfig := {
    warmupRatio := 0.0
    warmdownRatio := 0.4
    initLrFrac := 1.0
    minLrFrac := 0.1
  }
  -- At step 900 of 1000 (90%), should be decreasing
  let mult := getStageLRMultiplier cfg 900 1000
  assertFloatLt mult 1.0 "warmdown decreasing"
  -- At final step, should be near minLrFrac
  assertFloatInRange (getStageLRMultiplier cfg 999 1000) 0.1 0.2 "near minLrFrac at end"

@[test]
def testPretrainingLRConfig : IO Unit := do
  let cfg := pretrainingLRConfig
  assertFloatEq cfg.embeddingLr 0.3 (msg := "embedding LR")
  assertFloatEq cfg.matrixLr 0.02 (msg := "matrix LR")

@[test]
def testSFTLRConfig : IO Unit := do
  let cfg := sftLRConfig
  assertFloatEq cfg.initLrFrac 0.02 (msg := "start at 2%")
  assertFloatEq cfg.warmdownRatio 1.0 (msg := "linear decay throughout")

end Tests.Pipeline
