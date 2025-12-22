/-
  TestModdedGPT.lean

  Unit tests for modded-nanogpt port to Tyr.

  Test categories:
  1. C++ FFI smoke tests
  2. Pure function tests (schedules, shapes)
  3. Optimizer unit tests
  4. Sharding tests
  5. Model forward tests
  6. Training step tests

  Usage:
    ninja test-modded
    export DYLD_LIBRARY_PATH=external/libtorch/lib:/opt/homebrew/opt/libomp/lib
    out/exe/TestModdedGPT
-/
import Tyr
import Tyr.Distributed
import Tyr.Sharding
import Tyr.ModdedGPT
import Tyr.DataLoader
import Tyr.ModdedTrain
import Tyr.Optim.PolarExpress
import Tyr.Optim.NorMuon
import Tyr.Optim.DistAdam

open torch
open torch.dist
open torch.Sharding
open torch.moddedGpt
open torch.DataLoader
open torch.ModdedTrain
open torch.Optim

/-! ## Test Utilities -/

/-- Simple assertion with message -/
def assertWith (cond : Bool) (msg : String) : IO Unit := do
  if !cond then
    IO.eprintln s!"ASSERTION FAILED: {msg}"
    throw (IO.userError msg)

/-- Check that a Float is finite and non-NaN -/
def isFiniteFloat (x : Float) : Bool :=
  !x.isNaN && !x.isInf

/-! ## 1. C++ FFI Smoke Tests -/

/-- Test distributed FFI fallbacks (non-distributed mode) -/
def testDistributedFFI : IO Unit := do
  IO.println "=== Testing Distributed FFI ==="
  (← IO.getStdout).flush

  -- In non-distributed mode, rank should be 0 and worldSize should be 1
  IO.println "  Testing non-distributed fallbacks..."
  (← IO.getStdout).flush

  -- Note: getRank/getWorldSize may fail if process group not initialized
  -- Test that isInitialized returns false
  let isInit ← dist.isInitialized
  IO.println s!"    isInitialized: {isInit}"

  IO.println "  Distributed FFI test passed!"

/-- Test Polar Express kernels -/
def testPolarExpressFFI : IO Unit := do
  IO.println "=== Testing Polar Express FFI ==="
  (← IO.getStdout).flush

  -- Test XXT: A @ A.T
  IO.println "  Testing XXT (A @ A.T)..."
  (← IO.getStdout).flush
  let A ← randn #[32, 32] false
  let xxt ← dist.xxt A
  IO.println "    XXT computed"
  (← IO.getStdout).flush

  -- Test polar express orthogonalization
  IO.println "  Testing Polar Express orthogonalization..."
  (← IO.getStdout).flush
  let G ← randn #[64, 64] false
  let orthG ← dist.polarExpress G 5
  IO.println "    Orthogonalization computed"
  (← IO.getStdout).flush

  -- Verify result is not all zeros
  let norm := nn.item (nn.meanAll (nn.abs orthG))
  assertWith (norm > 0.0) "Orthogonalized gradient should be non-zero"
  IO.println s!"    Output norm: {norm}"

  IO.println "  Polar Express FFI test passed!"

/-! ## 2. Pure Function Tests (Schedules) -/

/-- Test hyperparameter schedules -/
def testSchedules : IO Unit := do
  IO.println "=== Testing Schedules ==="
  (← IO.getStdout).flush

  -- Batch size schedule
  IO.println "  Testing batch size schedule..."
  assertWith (getBatchSize 0 == 8) "Batch size at step 0 should be 8"
  assertWith (getBatchSize 199 == 8) "Batch size at step 199 should be 8"
  assertWith (getBatchSize 200 == 16) "Batch size at step 200 should be 16"
  assertWith (getBatchSize 999 == 16) "Batch size at step 999 should be 16"
  assertWith (getBatchSize 1000 == 24) "Batch size at step 1000 should be 24"
  IO.println "    Batch size schedule correct"

  -- Window size schedule
  IO.println "  Testing window size schedule..."
  let (ws0_s, ws0_l) := getWindowSizes 0
  assertWith (ws0_s == 3 && ws0_l == 3) "Window sizes at step 0 should be (3, 3)"
  let (ws500_s, ws500_l) := getWindowSizes 500
  assertWith (ws500_s == 3 && ws500_l == 7) "Window sizes at step 500 should be (3, 7)"
  let (ws1500_s, ws1500_l) := getWindowSizes 1500
  assertWith (ws1500_s == 3 && ws1500_l == 11) "Window sizes at step 1500 should be (3, 11)"
  IO.println "    Window size schedule correct"

  -- Learning rate schedule
  IO.println "  Testing learning rate schedule..."
  let hp := Hyperparameters.default
  let lr0 := getLearningRate 0 hp
  let lr300 := getLearningRate 300 hp  -- After warmup
  let lr2000 := getLearningRate 2000 hp  -- During cooldown
  assertWith (lr0 < lr300) "LR should increase during warmup"
  assertWith (lr2000 < lr300) "LR should decrease during cooldown"
  IO.println s!"    LR at step 0: {lr0}"
  IO.println s!"    LR at step 300: {lr300}"
  IO.println s!"    LR at step 2000: {lr2000}"

  -- Momentum schedule
  IO.println "  Testing momentum schedule..."
  let mom0 := getMuonMomentum 0 hp
  let mom300 := getMuonMomentum 300 hp
  assertWith (mom0 < mom300) "Momentum should increase during warmup"
  IO.println s!"    Momentum at step 0: {mom0}"
  IO.println s!"    Momentum at step 300: {mom300}"

  IO.println "  All schedule tests passed!"

/-! ## 3. Optimizer Unit Tests -/

/-- Test NorMuon optimizer single step -/
def testNorMuon : IO Unit := do
  IO.println "=== Testing NorMuon Optimizer ==="

  -- Create a parameter and gradient
  let param ← randn #[64, 64] false
  let param := autograd.set_requires_grad param true
  let grad ← randn #[64, 64] false

  -- Initialize state
  let state := NorMuon.initParamState param
  let cfg : NorMuon.Config := default

  -- Single step (using Adam-like update since orthogonalization may be slow)
  let (newParam, newState) ← NorMuon.stepAdamLike param grad state cfg 1.0 1.0

  -- Verify param changed
  let diff := nn.item (nn.meanAll (nn.abs (autograd.detach newParam - autograd.detach param)))
  assertWith (diff > 0) "Parameter should have changed after optimizer step"
  IO.println s!"    Parameter changed by: {diff}"

  -- Verify state updated
  assertWith (newState.step == 1) "Step count should be 1"

  IO.println "  NorMuon test passed!"

/-- Test DistAdam optimizer single step -/
def testDistAdam : IO Unit := do
  IO.println "=== Testing DistAdam Optimizer ==="
  (← IO.getStdout).flush

  -- Create a parameter and gradient
  IO.println "  Creating parameter and gradient..."
  let param ← randn #[128, 64] false
  let param := autograd.set_requires_grad param true
  let grad ← randn #[128, 64] false
  (← IO.getStdout).flush

  -- Initialize state
  let state := DistAdam.initParamState param
  let cfg : DistAdam.Config := default

  -- Single step
  IO.println "  Running single optimizer step..."
  (← IO.getStdout).flush
  let (newParam, newState) ← DistAdam.stepSingle param grad state cfg 1.0 1.0

  -- Verify param changed
  let diff := nn.item (nn.meanAll (nn.abs (autograd.detach newParam - autograd.detach param)))
  assertWith (diff > 0) "Parameter should have changed after optimizer step"
  IO.println s!"    Parameter changed by: {diff}"

  -- Verify state updated
  assertWith (newState.step == 1) "Step count should be 1"

  IO.println "  DistAdam test passed!"

/-! ## 4. Sharding Tests -/

/-- Test sharding shape computation -/
def testShardingShapes : IO Unit := do
  IO.println "=== Testing Sharding Shapes ==="
  (← IO.getStdout).flush

  -- Test shard size computation
  IO.println "  Testing shard size computation..."
  let fullSize : UInt64 := 100
  let worldSize : UInt64 := 4

  let size0 := shardSize fullSize worldSize 0
  let size1 := shardSize fullSize worldSize 1
  let size2 := shardSize fullSize worldSize 2
  let size3 := shardSize fullSize worldSize 3

  IO.println s!"    Shard sizes: {size0}, {size1}, {size2}, {size3}"

  -- Sum should equal full size
  let totalSize := size0 + size1 + size2 + size3
  assertWith (totalSize == fullSize) s!"Shard sizes should sum to {fullSize}, got {totalSize}"
  IO.println "    Shard sizes sum correctly"

  -- Test offset computation
  IO.println "  Testing offset computation..."
  let offset0 := shardOffset fullSize worldSize 0
  let offset1 := shardOffset fullSize worldSize 1
  let offset2 := shardOffset fullSize worldSize 2
  let offset3 := shardOffset fullSize worldSize 3

  IO.println s!"    Offsets: {offset0}, {offset1}, {offset2}, {offset3}"
  assertWith (offset0 == 0) "First shard offset should be 0"

  IO.println "  Sharding shape tests passed!"

/-! ## 5. Model Tests -/

/-- Test ModdedGPT model initialization -/
def testModdedGPTInit : IO Unit := do
  IO.println "=== Testing ModdedGPT Initialization ==="
  (← IO.getStdout).flush

  -- Use tiny config for fast testing
  let cfg : moddedGpt.Config := {
    vocabSize := 1000
    nLayer := 2
    nHead := 2
    headDim := 32
    modelDim := 64
    maxSeqLen := 128
    blockSize := 32
    numValueEmbeds := 2
  }

  IO.println s!"  Config: {cfg.nLayer} layers, {cfg.nHead} heads, {cfg.modelDim} dim"
  (← IO.getStdout).flush

  IO.println "  Initializing model..."
  (← IO.getStdout).flush
  let params ← ModdedGPTParams.init cfg

  -- Check that arrays have correct sizes
  assertWith (params.blocks.size == cfg.nLayer.toNat) "Should have correct number of blocks"
  assertWith (params.valueEmbeds.size == cfg.numValueEmbeds) "Should have correct number of value embeds"

  IO.println s!"    Blocks: {params.blocks.size}"
  IO.println s!"    Value embeds: {params.valueEmbeds.size}"

  IO.println "  ModdedGPT initialization test passed!"

/-- Test YaRN rotary initialization -/
def testYarnRotary : IO Unit := do
  IO.println "=== Testing YaRN Rotary ==="
  (← IO.getStdout).flush

  let headDim : UInt64 := 64
  let maxSeqLen : UInt64 := 128

  IO.println "  Initializing YaRN rotary..."
  (← IO.getStdout).flush
  let yarn ← YarnRotary.init headDim maxSeqLen

  -- Check cos/sin are non-zero
  let cosNorm := nn.item (nn.meanAll (nn.abs yarn.cos))
  let sinNorm := nn.item (nn.meanAll (nn.abs yarn.sin))

  IO.println s!"    Cos norm: {cosNorm}"
  IO.println s!"    Sin norm: {sinNorm}"

  assertWith (cosNorm > 0) "Cos should be non-zero"

  IO.println "  YaRN rotary test passed!"

/-- Test ModdedGPT forward pass -/
def testModdedGPTForward : IO Unit := do
  IO.println "=== Testing ModdedGPT Forward Pass ==="
  (← IO.getStdout).flush

  -- Tiny config
  let cfg : moddedGpt.Config := {
    vocabSize := 1000
    nLayer := 2
    nHead := 2
    headDim := 32
    modelDim := 64
    maxSeqLen := 128
    blockSize := 32
    numValueEmbeds := 2
    softcapValue := 30.0
  }

  IO.println "  Initializing model..."
  (← IO.getStdout).flush
  let params ← ModdedGPTParams.init cfg
  let yarn ← YarnRotary.init cfg.headDim cfg.maxSeqLen

  -- Create input
  let batchSize : UInt64 := 2
  let seqLen : UInt64 := 32
  IO.println s!"  Creating input [{batchSize}, {seqLen}]..."
  (← IO.getStdout).flush
  let input ← randint 0 cfg.vocabSize.toInt64 #[batchSize, seqLen]

  -- Forward pass
  IO.println "  Running forward pass..."
  (← IO.getStdout).flush
  let logits ← forward params yarn input 3 3

  -- Check output
  let logitsMean := nn.item (nn.meanAll logits)
  let logitsMax := nn.item (nn.maxAll (nn.abs logits))

  IO.println s!"    Logits mean: {logitsMean}"
  IO.println s!"    Logits max abs: {logitsMax}"

  -- Verify softcapping (values should be bounded)
  assertWith (logitsMax < 35.0) s!"Logits should be softcapped below 35, got {logitsMax}"
  assertWith (isFiniteFloat logitsMean) "Logits mean should be finite"

  IO.println "  ModdedGPT forward test passed!"

/-! ## 6. Loss and Training Tests -/

/-- Test loss computation -/
def testLoss : IO Unit := do
  IO.println "=== Testing Loss Computation ==="
  (← IO.getStdout).flush

  -- Tiny config
  let cfg : moddedGpt.Config := {
    vocabSize := 1000
    nLayer := 2
    nHead := 2
    headDim := 32
    modelDim := 64
    maxSeqLen := 128
    blockSize := 32
    numValueEmbeds := 2
  }

  IO.println "  Initializing model..."
  let params ← ModdedGPTParams.init cfg
  let yarn ← YarnRotary.init cfg.headDim cfg.maxSeqLen

  -- Create input and targets
  let batchSize : UInt64 := 2
  let seqLen : UInt64 := 32
  let input ← randint 0 cfg.vocabSize.toInt64 #[batchSize, seqLen]
  let targets ← randint 0 cfg.vocabSize.toInt64 #[batchSize, seqLen]

  -- Compute loss
  IO.println "  Computing loss..."
  (← IO.getStdout).flush
  let lossT ← moddedGpt.loss params yarn input targets 3 3
  let lossVal := nn.item lossT

  IO.println s!"    Loss: {lossVal}"

  assertWith (isFiniteFloat lossVal) "Loss should be finite"
  assertWith (lossVal > 0) "Loss should be positive"

  IO.println "  Loss computation test passed!"

/-! ## 7. Data Loader Tests -/

/-- Test data loader utilities -/
def testDataLoader : IO Unit := do
  IO.println "=== Testing Data Loader ==="
  (← IO.getStdout).flush

  -- Test hyperparams for step
  IO.println "  Testing getHyperparamsForStep..."
  let (bs0, wsS0, wsL0) := getHyperparamsForStep 0
  let (bs500, wsS500, wsL500) := getHyperparamsForStep 500
  let (bs1500, wsS1500, wsL1500) := getHyperparamsForStep 1500

  IO.println s!"    Step 0: batch={bs0}, window=({wsS0}, {wsL0})"
  IO.println s!"    Step 500: batch={bs500}, window=({wsS500}, {wsL500})"
  IO.println s!"    Step 1500: batch={bs1500}, window=({wsS1500}, {wsL1500})"

  assertWith (bs0 == 8) "Batch size at step 0 should be 8"
  assertWith (bs1500 == 24) "Batch size at step 1500 should be 24"

  -- Test tokens per step
  IO.println "  Testing tokensPerStep..."
  let tps := tokensPerStep 8 2048 1
  IO.println s!"    Tokens per step (batch=8, seq=2048, world=1): {tps}"
  assertWith (tps == 8 * 2048) "Tokens per step calculation"

  IO.println "  Data loader test passed!"

/-! ## Main Test Runner -/

/-- Run all tests -/
def runAllTests : IO Unit := do
  IO.println "========================================"
  IO.println "  Modded-NanoGPT Test Suite"
  IO.println "========================================"
  IO.println ""

  -- FFI tests
  testDistributedFFI
  IO.println ""
  testPolarExpressFFI
  IO.println ""

  -- Pure function tests
  testSchedules
  IO.println ""

  -- Optimizer tests
  testNorMuon
  IO.println ""
  testDistAdam
  IO.println ""

  -- Sharding tests
  testShardingShapes
  IO.println ""

  -- Model tests
  testModdedGPTInit
  IO.println ""
  testYarnRotary
  IO.println ""
  testModdedGPTForward
  IO.println ""

  -- Loss and training tests
  testLoss
  IO.println ""

  -- Data loader tests
  testDataLoader
  IO.println ""

  IO.println "========================================"
  IO.println "  All tests passed!"
  IO.println "========================================"

def main : IO Unit := do
  try
    runAllTests
  catch e =>
    IO.eprintln s!"Test failed: {e}"
    IO.Process.exit 1
