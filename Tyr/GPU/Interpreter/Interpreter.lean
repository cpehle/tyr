/-
  Tyr/GPU/Interpreter/Interpreter.lean

  Main module for GPU interpreter.
  Re-exports all components and provides high-level API.
-/
import Tyr.GPU.Interpreter.Config
import Tyr.GPU.Interpreter.Instruction.Base
import Tyr.GPU.Interpreter.Instruction.Tile
import Tyr.GPU.Interpreter.Instruction.Transformer
import Tyr.GPU.Interpreter.DAG.Node
import Tyr.GPU.Interpreter.DAG.Builder
import Tyr.GPU.Interpreter.Scheduler.Sequential
import Tyr.GPU.Interpreter.Scheduler.DAGBased

namespace Tyr.GPU.Interpreter

/-- High-level interpreter specification -/
structure InterpreterSpec where
  /-- Name of the interpreter -/
  name : String
  /-- Configuration (warp roles, memory model, scheduling) -/
  config : InterpreterConfig
  /-- Supported operations (for dispatch generation) -/
  supportedOpcodes : List Nat
  deriving Repr, Inhabited

/-- ThunderKittens-style interpreter specification -/
def thunderKittensSpec : InterpreterSpec := {
  name := "thunderkittens"
  config := thunderKittensConfig
  supportedOpcodes := [0, 1, 2, 3, 4, 5, 6, 7, 8]  -- TileOp opcodes
}

/-- Megakernels-style interpreter specification -/
def megakernelsSpec : InterpreterSpec := {
  name := "megakernels"
  config := megakernelsConfig
  supportedOpcodes := [0, 101, 102, 103, 104, 105, 106, 107]  -- TransformerOp opcodes
}

/-- Build and schedule a DAG using the given config -/
def buildAndSchedule (dag : DAG) (config : InterpreterConfig) (smCount : Nat) : Schedule :=
  schedule dag smCount config.scheduling

/-- Example: Build a simple FlashAttention-style DAG (ThunderKittens level) -/
def exampleFlashAttentionDAG (seqLen headDim : Nat := 64) : DAG := buildDAG do
  -- Load Q once
  let qLoad ← DAGBuilder.emitTile
    (.load (.global 0) (.shared 0 0) ⟨64, headDim⟩)

  -- For each K block, compute QK^T, softmax, SV
  let mut prevSV : Option NodeId := none
  for kBlock in [:seqLen / 64] do
    let kOffset := kBlock * 64 * headDim * 2
    let vOffset := kOffset + 64 * headDim * 2

    -- Load K and V
    let kLoad ← DAGBuilder.emitTileAfter [qLoad]
      (.load (.global kOffset) (.shared 1 0) ⟨64, headDim⟩)
    let vLoad ← DAGBuilder.emitTileAfter [qLoad]
      (.load (.global vOffset) (.shared 2 0) ⟨64, headDim⟩)

    -- QK^T
    let qkGemm ← DAGBuilder.emitTileAfter [qLoad, kLoad]
      (.gemm (.shared 0 0) (.shared 1 0) (.register 0) 64 64 headDim)

    -- Softmax
    let softmax ← DAGBuilder.emitTileAfter [qkGemm]
      (.elementwise [.register 0] (.register 1) ⟨64, 64⟩ .softmax)

    -- SV (accumulate into output)
    let deps := match prevSV with
      | some prev => [softmax, vLoad, prev]
      | none => [softmax, vLoad]
    let svGemm ← DAGBuilder.emitTileAfter deps
      (.gemm (.register 1) (.shared 2 0) (.register 2) 64 headDim 64
        (if prevSV.isSome then .add else .overwrite))

    prevSV := some svGemm

  -- Store output
  match prevSV with
  | some lastSV =>
    let _ ← DAGBuilder.emitTileAfter [lastSV]
      (.store (.register 2) (.global 0) ⟨64, headDim⟩)
  | none => pure ()

/-- Example: Build a transformer layer DAG (Megakernels level) -/
def exampleTransformerLayerDAG (layerIdx : Nat) (numKVHeads : Nat := 8) : DAG := buildDAG do
  -- QKV projection (2 blocks)
  let qkv1 ← DAGBuilder.emitTransformer
    (.layerNormQKVRope layerIdx ⟨0, 4⟩)
  let qkv2 ← DAGBuilder.emitTransformer
    (.layerNormQKVRope layerIdx ⟨4, 8⟩)

  -- Partial attention for each KV head
  let mut attnNodes : List NodeId := []
  for kvHead in [:numKVHeads] do
    let partial0 ← DAGBuilder.emitTransformerAfter [qkv1, qkv2]
      (.partialAttention layerIdx kvHead 4 0)
    let partial1 ← DAGBuilder.emitTransformerAfter [qkv1, qkv2]
      (.partialAttention layerIdx kvHead 4 1)
    let partial2 ← DAGBuilder.emitTransformerAfter [qkv1, qkv2]
      (.partialAttention layerIdx kvHead 4 2)
    let partial3 ← DAGBuilder.emitTransformerAfter [qkv1, qkv2]
      (.partialAttention layerIdx kvHead 4 3)

    -- Reduce partials
    let reduced ← DAGBuilder.emitTransformerAfter [partial0, partial1, partial2, partial3]
      (.attentionReduction layerIdx (kvHead * 4) 4 true [0, 1, 2, 3])
    attnNodes := reduced :: attnNodes

  -- O projection
  let oProj1 ← DAGBuilder.emitTransformerAfter attnNodes
    (.oProjResidual layerIdx ⟨0, 4⟩ 0)
  let oProj2 ← DAGBuilder.emitTransformerAfter attnNodes
    (.oProjResidual layerIdx ⟨4, 8⟩ 1)

  -- MLP: Gate+Up projection with SiLU
  let gate1 ← DAGBuilder.emitTransformerAfter [oProj1, oProj2]
    (.layerNormGateSiLU layerIdx [0, 1, 2, 3])
  let gate2 ← DAGBuilder.emitTransformerAfter [oProj1, oProj2]
    (.layerNormGateSiLU layerIdx [4, 5, 6, 7])

  -- Down projection
  let _ ← DAGBuilder.emitTransformerAfter [gate1, gate2]
    (.downProjResidual layerIdx ⟨0, 4⟩ 0)
  let _ ← DAGBuilder.emitTransformerAfter [gate1, gate2]
    (.downProjResidual layerIdx ⟨4, 8⟩ 1)

/-- Example: Build a hybrid DAG mixing tile and fused ops -/
def exampleHybridDAG (layerIdx : Nat) : DAG := buildDAG do
  -- Start with fused QKV (Megakernels style)
  let qkv ← DAGBuilder.emitTransformer
    (.layerNormQKVRope layerIdx ⟨0, 8⟩)

  -- Drop to tile level for custom attention
  let qkGemm ← DAGBuilder.emitTileAfter [qkv]
    (.gemm (.shared 0 0) (.shared 1 0) (.register 0) 64 64 128)

  let softmax ← DAGBuilder.emitTileAfter [qkGemm]
    (.elementwise [.register 0] (.register 1) ⟨64, 64⟩ .softmax)

  let svGemm ← DAGBuilder.emitTileAfter [softmax]
    (.gemm (.register 1) (.shared 2 0) (.register 2) 64 128 64)

  -- Back to fused level for O projection
  let _ ← DAGBuilder.emitTransformerAfter [svGemm]
    (.oProjResidual layerIdx ⟨0, 8⟩ 0)

end Tyr.GPU.Interpreter
