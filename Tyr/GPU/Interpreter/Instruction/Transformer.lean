/-
  Tyr/GPU/Interpreter/Instruction/Transformer.lean

  Transformer-level fused operations (Megakernels style).
  High-level instructions for LLM inference.

  Based on Megakernels/megakernels/demos/latency/instructions.py
-/
import Tyr.GPU.Interpreter.Instruction.Base

namespace Tyr.GPU.Interpreter

open Tyr.GPU

/-- Block range for parallelization -/
structure BlockRange where
  start : Nat
  stop : Nat
  deriving Repr, BEq, Hashable, Inhabited

/-- Transformer fused operations (Megakernels level) -/
inductive TransformerOp where
  /-- Fused: LayerNorm → QKV projection → RoPE → KV cache append -/
  | layerNormQKVRope
      (layerIdx : Nat)
      (blockRange : BlockRange)

  /-- Partial attention over a segment of the KV cache -/
  | partialAttention
      (layerIdx : Nat)
      (kvHeadIdx : Nat)
      (numPartials : Nat)
      (partialIdx : Nat)

  /-- Reduce partial attention results (tree reduction) -/
  | attentionReduction
      (layerIdx : Nat)
      (headStartIdx : Nat)
      (numPartials : Nat)
      (isTerminal : Bool)
      (reductionList : List Nat)
      (outputPartialIdx : Option Nat := none)

  /-- Output projection with residual add -/
  | oProjResidual
      (layerIdx : Nat)
      (blockRange : BlockRange)
      (reductionBlockIdx : Nat)

  /-- Fused: LayerNorm → Gate/Up projection → SiLU -/
  | layerNormGateSiLU
      (layerIdx : Nat)
      (blockIdxs : List Nat)

  /-- Down projection with residual add -/
  | downProjResidual
      (layerIdx : Nat)
      (blockRange : BlockRange)
      (reductionBlockIdx : Nat)

  /-- Final: RMSNorm → LM Head projection -/
  | rmsLmHead
      (blockRange : BlockRange)

  deriving Repr, BEq, Hashable, Inhabited

namespace TransformerOp

/-- Opcode for each transformer operation -/
def opcode : TransformerOp → Nat
  | .layerNormQKVRope .. => 101
  | .partialAttention .. => 102
  | .attentionReduction .. => 103
  | .oProjResidual .. => 104
  | .layerNormGateSiLU .. => 105
  | .downProjResidual .. => 106
  | .rmsLmHead .. => 107

/-- Previous opcode (for implicit dependency ordering) -/
def prevOpcode : TransformerOp → Option Nat
  | .layerNormQKVRope .. => some 106  -- After downProjResidual (or start)
  | .partialAttention .. => some 101  -- After layerNormQKVRope
  | .attentionReduction .. => some 102  -- After partialAttention
  | .oProjResidual .. => some 103  -- After attentionReduction
  | .layerNormGateSiLU .. => some 104  -- After oProjResidual
  | .downProjResidual .. => some 105  -- After layerNormGateSiLU
  | .rmsLmHead .. => some 106  -- After last downProjResidual

/-- Serialize block range -/
private def serializeBlockRange (br : BlockRange) : Array UInt32 :=
  #[natToUInt32 br.start, natToUInt32 br.stop]

/-- Serialize a transformer operation -/
def serialize (op : TransformerOp) : Array UInt32 :=
  let fields := match op with
    | .layerNormQKVRope layerIdx blockRange =>
      #[natToUInt32 layerIdx] ++ serializeBlockRange blockRange

    | .partialAttention layerIdx kvHeadIdx numPartials partialIdx =>
      #[natToUInt32 layerIdx, natToUInt32 kvHeadIdx,
        natToUInt32 numPartials, natToUInt32 partialIdx]

    | .attentionReduction layerIdx headStartIdx numPartials isTerminal reductionList outputPartialIdx =>
      #[natToUInt32 layerIdx, natToUInt32 headStartIdx,
        natToUInt32 numPartials, if isTerminal then 1 else 0] ++
      serializeNatList reductionList ++
      #[serializeOptionNat outputPartialIdx]

    | .oProjResidual layerIdx blockRange reductionBlockIdx =>
      #[natToUInt32 layerIdx] ++ serializeBlockRange blockRange ++
      #[natToUInt32 reductionBlockIdx]

    | .layerNormGateSiLU layerIdx blockIdxs =>
      #[natToUInt32 layerIdx] ++ serializeNatList blockIdxs

    | .downProjResidual layerIdx blockRange reductionBlockIdx =>
      #[natToUInt32 layerIdx] ++ serializeBlockRange blockRange ++
      #[natToUInt32 reductionBlockIdx]

    | .rmsLmHead blockRange =>
      serializeBlockRange blockRange

  serializeWithOpcode op.opcode fields

/-- Model configuration for cost estimation -/
structure ModelConfig where
  hiddenSize : Nat := 4096
  intermediateSize : Nat := 11008
  numAttentionHeads : Nat := 32
  numKVHeads : Nat := 8
  headDim : Nat := 128
  vocabSize : Nat := 32000
  qkvBlockSize : Nat := 256
  oProjBlockSize : Nat := 256
  upGateBlockSize : Nat := 512
  downProjBlockSize : Nat := 256
  lmHeadBlockSize : Nat := 512
  seqLen : Nat := 2048
  deriving Repr, Inhabited

/-- Estimated cost with model config -/
def costWithConfig (op : TransformerOp) (cfg : ModelConfig) : Float :=
  match op with
  | .layerNormQKVRope _ blockRange =>
    let numBlocks := blockRange.stop - blockRange.start
    (numBlocks * cfg.qkvBlockSize * cfg.hiddenSize).toFloat

  | .partialAttention _ _ numPartials _ =>
    let loadedSeqLen := cfg.seqLen / numPartials
    (loadedSeqLen * cfg.headDim * 2).toFloat  -- KV cache loads

  | .attentionReduction _ _ _ _ reductionList _ =>
    (reductionList.length * cfg.headDim).toFloat

  | .oProjResidual _ blockRange _ =>
    let numBlocks := blockRange.stop - blockRange.start
    (numBlocks * cfg.oProjBlockSize * cfg.hiddenSize).toFloat

  | .layerNormGateSiLU _ blockIdxs =>
    (blockIdxs.length * cfg.upGateBlockSize * cfg.hiddenSize * 2).toFloat

  | .downProjResidual _ blockRange _ =>
    let numBlocks := blockRange.stop - blockRange.start
    (numBlocks * cfg.downProjBlockSize * cfg.hiddenSize).toFloat

  | .rmsLmHead blockRange =>
    let numBlocks := blockRange.stop - blockRange.start
    (numBlocks * cfg.lmHeadBlockSize * cfg.hiddenSize).toFloat

/-- Default cost (without config) -/
def cost (op : TransformerOp) : Float :=
  costWithConfig op {}

/-- Resource pool for scheduling -/
def pool : TransformerOp → ResourcePool
  | .layerNormQKVRope .. => .compute
  | .partialAttention .. => .memory  -- Memory bound (KV cache)
  | .attentionReduction .. => .compute
  | .oProjResidual .. => .compute
  | .layerNormGateSiLU .. => .compute
  | .downProjResidual .. => .compute
  | .rmsLmHead .. => .compute

end TransformerOp

instance : GpuInstruction TransformerOp where
  level := .fused
  opcode := 100  -- Base; actual opcode varies
  serialize := TransformerOp.serialize
  cost := TransformerOp.cost
  tags := { pool := .mixed }

/-- Get the actual opcode for a specific TransformerOp value -/
def TransformerOp.getOpcode (op : TransformerOp) : Nat := op.opcode

end Tyr.GPU.Interpreter
