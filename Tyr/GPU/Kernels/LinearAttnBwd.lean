import Tyr.GPU.Kernels.Prelude
import Tyr.GPU.Kernels.Support

/-!
  Tyr/GPU/Kernels/LinearAttnBwd.lean

  Reverse-scan backward for the decayed recurrent linear-attention forward in
  `LinearAttn.lean`.

  ThunderKittens only ships a forward kernel for this family, so this module is
  not a direct transliteration. It is instead built to match the exact forward
  decomposition now used in Tyr:

  - local masked-decayed `QK^T V`,
  - recurrent `Q_decay @ KV_prev`,
  - state update `KV_t = block_decay * KV_prev + K_decay^T V`.

  The backward relies on the per-chunk `KV_prev` checkpoints emitted by the
  forward kernel so it can run the correct reverse recurrence without
  approximating the state history.
-/

namespace Tyr.GPU.Kernels.LinearAttn

open Tyr.GPU
open Tyr.GPU.Codegen

set_option maxRecDepth 4096

private abbrev chunkSize : Nat := 64
private abbrev featDim : Nat := 128
private abbrev halfFeat : Nat := 64
private abbrev numChunks : Nat := 16

/-- Convert an FP32 row-major tile into BF16 row-major form. -/
private def toBf16Row {rows cols : Nat}
    (src : RT GpuFloat.Float32 rows cols .Row) : KernelM (RT GpuFloat.BFloat16 rows cols .Row) := do
  let dst : RT GpuFloat.BFloat16 rows cols ← allocRT .BFloat16 rows cols
  convert dst src
  pure dst

/-- Convert an FP32 row-major tile into BF16 column-major form. -/
private def toBf16Col {rows cols : Nat}
    (src : RT GpuFloat.Float32 rows cols .Row) : KernelM (RT GpuFloat.BFloat16 rows cols .Col) := do
  let rowTile ← toBf16Row src
  let colTile : RT GpuFloat.BFloat16 rows cols .Col ← allocRT .BFloat16 rows cols .Col
  swapLayout colTile rowTile
  pure colTile

/-- Convert a BF16 row-major tile into column-major form. -/
private def bf16ToCol {rows cols : Nat}
    (src : RT GpuFloat.BFloat16 rows cols .Row) : KernelM (RT GpuFloat.BFloat16 rows cols .Col) := do
  let colTile : RT GpuFloat.BFloat16 rows cols .Col ← allocRT .BFloat16 rows cols .Col
  swapLayout colTile src
  pure colTile

/-- Build the masked local score tile used by both the forward and backward
branches. -/
private def recomputeMaskedScores
    (q : RT GpuFloat.BFloat16 chunkSize featDim)
    (k : RT GpuFloat.BFloat16 chunkSize featDim)
    (qDecay : RV GpuFloat.Float32 chunkSize)
    (kDecay : RV GpuFloat.Float32 chunkSize)
    : KernelM (RT GpuFloat.Float32 chunkSize chunkSize) := do
  let rawScores : RT GpuFloat.Float32 chunkSize chunkSize ← zeroRT .Float32 chunkSize chunkSize
  let maskedScores : RT GpuFloat.Float32 chunkSize chunkSize ← allocRT .Float32 chunkSize chunkSize
  mmaT rawScores q k rawScores
  Support.applyLinearAttnDecayMask maskedScores rawScores qDecay kDecay
  pure maskedScores

/-- Canonical reverse-scan backward for the decayed recurrent linear-attention
kernel. -/
@[gpu_kernel .SM90]
def linearAttnBwd
    (Q_ptr : GPtr GpuFloat.BFloat16)
    (K_ptr : GPtr GpuFloat.BFloat16)
    (V_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (kv_history_top_ptr : GPtr GpuFloat.Float32)
    (kv_history_bottom_ptr : GPtr GpuFloat.Float32)
    (slope : KVal Float32)
    (dQ_ptr : GPtr GpuFloat.Float32)
    (dK_ptr : GPtr GpuFloat.Float32)
    (dV_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit := do
  comment "=== Decayed Linear Attention Backward ==="
  comment "Reverse scan over the same local-plus-recurrent state decomposition used in forward"

  let coord ← blockCoord2D
  let (qDecay, kDecay, stateDecay) ← Support.buildLinearAttnDecayVectors slope
  let lastChunk ← constIntVal (numChunks - 1) "last_chunk"

  let dStateTop : RT GpuFloat.Float32 halfFeat featDim ← zeroRT .Float32 halfFeat featDim
  let dStateBottom : RT GpuFloat.Float32 halfFeat featDim ← zeroRT .Float32 halfFeat featDim

  let qShared : ST GpuFloat.BFloat16 chunkSize featDim ← allocST .BFloat16 chunkSize featDim
  let kShared : ST GpuFloat.BFloat16 chunkSize featDim ← allocST .BFloat16 chunkSize featDim
  let vShared : ST GpuFloat.BFloat16 chunkSize featDim .Col ← allocST .BFloat16 chunkSize featDim .Col
  let dOShared : ST GpuFloat.BFloat16 chunkSize featDim ← allocST .BFloat16 chunkSize featDim
  let stateTopShared : ST GpuFloat.Float32 halfFeat featDim ← allocST .Float32 halfFeat featDim
  let stateBottomShared : ST GpuFloat.Float32 halfFeat featDim ← allocST .Float32 halfFeat featDim
  let dQShared : ST GpuFloat.Float32 chunkSize featDim ← allocST .Float32 chunkSize featDim
  let dKShared : ST GpuFloat.Float32 chunkSize featDim ← allocST .Float32 chunkSize featDim
  let dVShared : ST GpuFloat.Float32 chunkSize featDim ← allocST .Float32 chunkSize featDim

  for loopIdx in krange 0 numChunks do
    let iterVal : KVal UInt32 := ⟨loopIdx.id, "iter"⟩
    let revIdx ← scalarSub lastChunk iterVal "rev_idx"
    let chunkCoord := coord.withRow revIdx.id

    let q : RT GpuFloat.BFloat16 chunkSize featDim ← allocRT .BFloat16 chunkSize featDim
    let k : RT GpuFloat.BFloat16 chunkSize featDim ← allocRT .BFloat16 chunkSize featDim
    let v : RT GpuFloat.BFloat16 chunkSize featDim .Col ← allocRT .BFloat16 chunkSize featDim .Col
    let dO : RT GpuFloat.BFloat16 chunkSize featDim ← allocRT .BFloat16 chunkSize featDim
    let qF : RT GpuFloat.Float32 chunkSize featDim ← allocRT .Float32 chunkSize featDim
    let kF : RT GpuFloat.Float32 chunkSize featDim ← allocRT .Float32 chunkSize featDim
    let stateTopPrev : RT GpuFloat.Float32 halfFeat featDim ← allocRT .Float32 halfFeat featDim
    let stateBottomPrev : RT GpuFloat.Float32 halfFeat featDim ← allocRT .Float32 halfFeat featDim

    comment "Load the current chunk and the forward KV checkpoints"
    loadGlobal qShared Q_ptr chunkCoord
    loadGlobal kShared K_ptr chunkCoord
    loadGlobal vShared V_ptr chunkCoord
    loadGlobal dOShared dO_ptr chunkCoord
    loadGlobal stateTopShared kv_history_top_ptr chunkCoord
    loadGlobal stateBottomShared kv_history_bottom_ptr chunkCoord
    sync
    load q qShared
    load k kShared
    load v vShared
    load dO dOShared
    load stateTopPrev stateTopShared
    load stateBottomPrev stateBottomShared
    convert qF q
    convert kF k

    let (qLeft, qRight) ← Support.splitCols qF
    let (kLeft, kRight) ← Support.splitCols kF
    let maskedScores ← recomputeMaskedScores q k qDecay kDecay

    let maskedScoresBf ← toBf16Row maskedScores
    let maskedScoresT : RT GpuFloat.BFloat16 chunkSize chunkSize ← allocRT .BFloat16 chunkSize chunkSize
    transpose maskedScoresT maskedScoresBf

    let vRow : RT GpuFloat.BFloat16 chunkSize featDim ← allocRT .BFloat16 chunkSize featDim
    swapLayout vRow v
    let qCol ← bf16ToCol q
    let kCol ← bf16ToCol k
    let dOCol ← bf16ToCol dO

    comment "Local masked-decayed path"
    let dScores : RT GpuFloat.Float32 chunkSize chunkSize ← zeroRT .Float32 chunkSize chunkSize
    let dScoresMasked : RT GpuFloat.Float32 chunkSize chunkSize ← allocRT .Float32 chunkSize chunkSize
    let dScoresMaskedBf : RT GpuFloat.BFloat16 chunkSize chunkSize ← allocRT .BFloat16 chunkSize chunkSize
    let dScoresMaskedT : RT GpuFloat.BFloat16 chunkSize chunkSize ← allocRT .BFloat16 chunkSize chunkSize
    mmaT dScores dO vRow dScores
    Support.applyLinearAttnDecayMask dScoresMasked dScores qDecay kDecay
    convert dScoresMaskedBf dScoresMasked
    transpose dScoresMaskedT dScoresMaskedBf

    let dQLocal : RT GpuFloat.Float32 chunkSize featDim ← zeroRT .Float32 chunkSize featDim
    let dKLocal : RT GpuFloat.Float32 chunkSize featDim ← zeroRT .Float32 chunkSize featDim
    let dVLocal : RT GpuFloat.Float32 chunkSize featDim ← zeroRT .Float32 chunkSize featDim
    mma dQLocal dScoresMaskedBf kCol dQLocal
    mma dKLocal dScoresMaskedT qCol dKLocal
    mma dVLocal maskedScoresT dOCol dVLocal

    comment "Recurrent output path"
    let qLeftDecayed : RT GpuFloat.Float32 chunkSize halfFeat ← allocRT .Float32 chunkSize halfFeat
    let qRightDecayed : RT GpuFloat.Float32 chunkSize halfFeat ← allocRT .Float32 chunkSize halfFeat
    mulCol qLeftDecayed qLeft qDecay
    mulCol qRightDecayed qRight qDecay

    let stateTopPrevBf ← toBf16Row stateTopPrev
    let stateBottomPrevBf ← toBf16Row stateBottomPrev
    let dQTopRecur : RT GpuFloat.Float32 chunkSize halfFeat ← zeroRT .Float32 chunkSize halfFeat
    let dQBottomRecur : RT GpuFloat.Float32 chunkSize halfFeat ← zeroRT .Float32 chunkSize halfFeat
    mmaT dQTopRecur dO stateTopPrevBf dQTopRecur
    mmaT dQBottomRecur dO stateBottomPrevBf dQBottomRecur
    mulCol dQTopRecur dQTopRecur qDecay
    mulCol dQBottomRecur dQBottomRecur qDecay

    let qLeftDecayedCol ← toBf16Col qLeftDecayed
    let qRightDecayedCol ← toBf16Col qRightDecayed
    let dStateTopFromOut : RT GpuFloat.Float32 halfFeat featDim ← zeroRT .Float32 halfFeat featDim
    let dStateBottomFromOut : RT GpuFloat.Float32 halfFeat featDim ← zeroRT .Float32 halfFeat featDim
    mmaAtB dStateTopFromOut qLeftDecayedCol dOCol dStateTopFromOut
    mmaAtB dStateBottomFromOut qRightDecayedCol dOCol dStateBottomFromOut

    comment "State-update path"
    let totalStateTop : RT GpuFloat.Float32 halfFeat featDim ← allocRT .Float32 halfFeat featDim
    let totalStateBottom : RT GpuFloat.Float32 halfFeat featDim ← allocRT .Float32 halfFeat featDim
    add totalStateTop dStateTop dStateTopFromOut
    add totalStateBottom dStateBottom dStateBottomFromOut

    let kLeftDecayed : RT GpuFloat.Float32 chunkSize halfFeat ← allocRT .Float32 chunkSize halfFeat
    let kRightDecayed : RT GpuFloat.Float32 chunkSize halfFeat ← allocRT .Float32 chunkSize halfFeat
    mulCol kLeftDecayed kLeft kDecay
    mulCol kRightDecayed kRight kDecay
    let kLeftDecayedBf ← toBf16Row kLeftDecayed
    let kRightDecayedBf ← toBf16Row kRightDecayed
    let totalStateTopBf ← toBf16Row totalStateTop
    let totalStateBottomBf ← toBf16Row totalStateBottom
    let totalStateTopBfCol ← toBf16Col totalStateTop
    let totalStateBottomBfCol ← toBf16Col totalStateBottom

    let dKTopState : RT GpuFloat.Float32 chunkSize halfFeat ← zeroRT .Float32 chunkSize halfFeat
    let dKBottomState : RT GpuFloat.Float32 chunkSize halfFeat ← zeroRT .Float32 chunkSize halfFeat
    mmaT dKTopState vRow totalStateTopBf dKTopState
    mmaT dKBottomState vRow totalStateBottomBf dKBottomState
    mulCol dKTopState dKTopState kDecay
    mulCol dKBottomState dKBottomState kDecay

    let dVStateTop : RT GpuFloat.Float32 chunkSize featDim ← zeroRT .Float32 chunkSize featDim
    let dVStateBottom : RT GpuFloat.Float32 chunkSize featDim ← zeroRT .Float32 chunkSize featDim
    mma dVStateTop kLeftDecayedBf totalStateTopBfCol dVStateTop
    mma dVStateBottom kRightDecayedBf totalStateBottomBfCol dVStateBottom

    let dQRecur : RT GpuFloat.Float32 chunkSize featDim ← allocRT .Float32 chunkSize featDim
    let dKState : RT GpuFloat.Float32 chunkSize featDim ← allocRT .Float32 chunkSize featDim
    concatCols dQRecur dQTopRecur dQBottomRecur
    concatCols dKState dKTopState dKBottomState

    let dVState : RT GpuFloat.Float32 chunkSize featDim ← allocRT .Float32 chunkSize featDim
    let dQChunk : RT GpuFloat.Float32 chunkSize featDim ← allocRT .Float32 chunkSize featDim
    let dKChunk : RT GpuFloat.Float32 chunkSize featDim ← allocRT .Float32 chunkSize featDim
    let dVChunk : RT GpuFloat.Float32 chunkSize featDim ← allocRT .Float32 chunkSize featDim
    add dVState dVStateTop dVStateBottom
    add dQChunk dQLocal dQRecur
    add dKChunk dKLocal dKState
    add dVChunk dVLocal dVState

    store dQShared dQChunk
    store dKShared dKChunk
    store dVShared dVChunk
    storeGlobal dQ_ptr dQShared chunkCoord
    storeGlobal dK_ptr dKShared chunkCoord
    storeGlobal dV_ptr dVShared chunkCoord

    comment "Propagate reverse recurrent state to the previous chunk"
    mulCol dStateTop totalStateTop stateDecay
    mulCol dStateBottom totalStateBottom stateDecay
    sync

/-- Canonical ThunderKittens-aligned backward name for the decayed recurrent
forward surface. -/
abbrev tkLinearAttnBwd := linearAttnBwd

end Tyr.GPU.Kernels.LinearAttn
