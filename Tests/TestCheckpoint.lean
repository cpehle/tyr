/-
  Tests/TestCheckpoint.lean

  Tests for checkpointing functionality and TensorStruct integration.
-/
import Tyr
import Examples.GPT.GPT
import Examples.GPT.ModdedTrain
import Tyr.Optim.DistAdam
import Tyr.Optim.NorMuon
import LeanTest

open torch
open torch.ModdedTrain
open torch.Optim

@[test]
def testCheckpointCreation : IO Unit := do
  -- Create tiny config
  let cfg : moddedGpt.Config := {
    vocabSize := 128
    nLayer := 2
    nHead := 2
    headDim := 16
    modelDim := 32
    maxSeqLen := 64
    blockSize := 32
    numValueEmbeds := 1
  }

  -- Init params
  let params ← moddedGpt.ModdedGPTParams.init cfg
  
  -- Init optimizer state
  let optState := OptimizerState.init false 1

  -- Create checkpoint
  let ckpt : Checkpoint cfg := {
    params := params
    optState := optState
    step := 10
    bestValLoss := 0.5
  }

  -- Test saveCheckpoint (functionality check)
  -- This currently just logs to stdout, but we verify it runs without error
  -- and computes statistics using TensorStruct
  saveCheckpoint ckpt "dummy_path.ckpt"

  -- Verify TensorStruct on params directly
  let numParams := TensorStruct.fold (fun {s} _t acc => acc + s.foldl (· * ·) 1) 0 params
  LeanTest.assertTrue (numParams > 0) "Parameter count should be positive"
  
  -- Rough calculation: 
  -- embed: 128*32 = 4096
  -- valueEmbeds: 1 * 128*32 = 4096
  -- blocks (2):
  --   attn qkvo: 4*32 * 2*16 = 128 * 32 = 4096
  --   attn gate: 12 * 2 = 24
  --   mlp fc: 4*32 * 32 = 4096
  --   mlp proj: 4*32 * 32 = 4096
  -- lmHead: 32 * 128 = 4096
  -- scalars: ...
  -- Total should be > 20000
  LeanTest.assertTrue (numParams > 20000) s!"Parameter count {numParams} too low"

@[test]
def testDistAdamCheckpointing : IO Unit := do
  -- Test that DistAdam.ParamState implements TensorStruct correctly
  let shape : Shape := #[10, 10]
  let param := zeros shape
  let state := DistAdam.initParamState param
  
  -- Verify structure traversal
  let numTensors := TensorStruct.fold (fun {_s} _t acc => acc + 1) 0 state
  LeanTest.assertEqual numTensors 2 "DistAdam state should have 2 tensors (expAvg, expAvgSq)"
  
  let totalElements := TensorStruct.fold (fun {s} _t acc => acc + s.foldl (· * ·) 1) 0 state
  LeanTest.assertEqual totalElements 200 "DistAdam state should have 200 elements"

  -- Verify map works
  let stateOnes := TensorStruct.map (fun t => add_scalar t 1.0) state
  let sumOnes := TensorStruct.fold (fun {_s} t acc => acc + nn.item (nn.sumAll t)) 0.0 stateOnes
  LeanTest.assertEqual sumOnes 200.0 "Mapped state should sum to 200"

@[test]
def testNorMuonCheckpointing : IO Unit := do
  -- Test that NorMuon.ParamState implements TensorStruct correctly
  let shape : Shape := #[10, 10]
  let param := zeros shape
  let state := NorMuon.initParamState param
  
  -- Initial state has no tensors (Options are none)
  let numTensors := TensorStruct.fold (fun {_s} _t acc => acc + 1) 0 state
  LeanTest.assertEqual numTensors 0 "Initial NorMuon state should have 0 tensors"
  
  -- Simulate state update
  let momentum := ones shape
  let secondMoment := ones #[]
  let stateWithData : NorMuon.ParamState shape := {
    momentumBuffer := some momentum
    secondMoment := some secondMoment
    step := 1
  }
  
  -- Verify structure traversal with data
  let numTensorsFilled := TensorStruct.fold (fun {_s} _t acc => acc + 1) 0 stateWithData
  LeanTest.assertEqual numTensorsFilled 2 "Filled NorMuon state should have 2 tensors"
  
  let totalElements := TensorStruct.fold (fun {s} _t acc => acc + s.foldl (· * ·) 1) 0 stateWithData
  LeanTest.assertEqual totalElements 101 "Filled NorMuon state should have 101 elements (100 + 1)"

@[test]
def testLoadCheckpointMock : IO Unit := do
  -- Test loadCheckpoint stub
  let cfg : moddedGpt.Config := moddedGpt.Config.default
  let ckpt ← loadCheckpoint cfg "nonexistent.ckpt"
  match ckpt with
  | none => LeanTest.assertTrue true
  | some _ => LeanTest.fail "Should return none for stub implementation"
