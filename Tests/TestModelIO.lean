/-
  Tests/TestModelIO.lean

  Tests for Model Parameter serialization and structure traversal using TensorStruct.
-/
import Tyr
import Examples.NanoChat.ModdedGPT
import LeanTest

open torch
open torch.moddedGpt

/-- Helper to flatten model to a list of tensors -/
def flattenModel [TensorStruct α] (model : α) : List (Σ s, T s) :=
  TensorStruct.fold (fun {s} t acc => (⟨s, t⟩ : Σ s, T s) :: acc) [] model |>.reverse

/-- Helper to count elements in flattened model -/
def countElements (flat : List (Σ s, T s)) : Nat :=
  flat.foldl (fun acc ⟨s, _⟩ => acc + s.foldl (fun p d => p * d.toNat) 1) 0

@[test]
def testModdedGPTStructure : IO Unit := do
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

  -- Init model
  let params ← ModdedGPTParams.init cfg

  -- Flatten
  let flat := flattenModel params
  
  -- Verify count
  -- embed: 1
  -- smearGate: 1 (weight)
  -- valueEmbeds: 1
  -- blocks (2):
  --   attn qkvo: 1
  --   attn gate: 1
  --   mlp fc: 1
  --   mlp proj: 1
  --   (4 per block * 2 = 8)
  -- lmHead: 1
  -- scalars: 1
  -- Total tensors: 1 + 1 + 1 + 8 + 1 + 1 = 13
  let numTensors := flat.length
  -- Note: CastedLinear has weight. MLP has cFc, cProj. CausalSelfAttention has qkvoWeight, attnGate.
  -- Let's count accurately.
  -- 1. embed (T)
  -- 2. smearGate (CastedLinear -> weight)
  -- 3. valueEmbeds (Array T -> 1 T)
  -- 4. blocks[0].attn.qkvoWeight
  -- 5. blocks[0].attn.attnGate.weight
  -- 6. blocks[0].mlp.cFc
  -- 7. blocks[0].mlp.cProj
  -- 8-11. blocks[1]...
  -- 12. lmHead.weight
  -- 13. scalars.values
  -- Total = 13 tensors.
  
  LeanTest.assertEqual numTensors 13 "Should have 13 tensors in the model structure"

  -- Verify total elements
  let totalElems := countElements flat
  LeanTest.assertTrue (totalElems > 0) "Total elements should be positive"

@[test]
def testParameterUpdate : IO Unit := do
  -- Verify we can map over parameters to update them (simulating optimizer step or loading)
  let cfg : moddedGpt.Config := {
    vocabSize := 10
    nLayer := 1
    nHead := 1
    headDim := 4
    modelDim := 4
    maxSeqLen := 10
    blockSize := 10
    numValueEmbeds := 1
  }
  
  let params ← ModdedGPTParams.init cfg
  
  -- Apply update: p = p + 1
  let updatedParams := TensorStruct.map (fun t => t + (1.0 : Float)) params
  
  -- Check one tensor to verify update
  let embedSumOriginal := nn.item (nn.sumAll params.embed)
  let embedSumUpdated := nn.item (nn.sumAll updatedParams.embed)
  
  let diff := embedSumUpdated - embedSumOriginal
  let expectedDiff := (cfg.vocabSize * cfg.modelDim).toFloat -- adding 1.0 to each element
  
  LeanTest.assertTrue (Float.abs (diff - expectedDiff) < 1e-4) "Parameters should be updated"

@[test]
unsafe def testStateDictSimulation : IO Unit := do
  -- Simulate saving to a state dict (list of tensors) and loading back
  let cfg : moddedGpt.Config := {
    vocabSize := 10
    nLayer := 1
    nHead := 1
    headDim := 4
    modelDim := 4
    maxSeqLen := 10
    blockSize := 10
    numValueEmbeds := 1
  }
  
  let params ← ModdedGPTParams.init cfg
  
  -- "Save" to list
  let stateList := flattenModel params
  
  -- "Load" from list (Mock implementation of loading logic)
  -- We use mapM with a stateful iterator to consume the list
  let loadFromList (list : List (Σ s, T s)) (model : ModdedGPTParams cfg) 
      : IO (ModdedGPTParams cfg) := do
    let ref ← IO.mkRef list
    TensorStruct.mapM (fun {s} _oldT => do
      let remaining ← ref.get
      match remaining with
      | [] => throw $ IO.userError "Not enough tensors in state list"
      | ⟨s', t⟩ :: rest => 
        -- In real loading, we'd check shape match here (s == s')
        -- But T s and T s' are different types if s != s' statically
        -- We cast or rely on runtime check
        ref.set rest
        -- Unsafe cast for this test since we know shapes match
        -- In real code, we'd load by name or validate shape
        pure (unsafeCast t : T s)
    ) model

  let loadedParams ← loadFromList stateList params
  
  -- Verify loaded matches original
  let diff := TensorStruct.zipWith (fun a b => a - b) params loadedParams
  let totalDiff := TensorStruct.fold (fun {s} t acc => acc + nn.item (nn.sumAll (nn.abs t))) 0.0 diff
  
  LeanTest.assertTrue (totalDiff < 1e-6) "Loaded parameters should match saved ones"
