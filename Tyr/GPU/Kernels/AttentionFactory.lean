/-
  Tyr/GPU/Kernels/AttentionFactory.lean

  Parameterized FlashAttention kernel factory.

  This module consolidates the near-identical attention forward/backward kernels
  in `Kernels/FlashAttn.lean`, `Kernels/FlashAttnBwd.lean`, `Kernels/MhaH100.lean`,
  `Kernels/FlashAttnCausal64.lean`, and parts of `Kernels/FlashAttn3.lean` into a
  single template driven by a `FAVariantConfig`.

  Covered (non-warp-specialized) variants:
  - FlashAttention forward, with optional LSE output
  - MhaH100-style forward (LSE output scaled as `l = -8 * lse` for head_dim=64)
  - Causal forward (`isCausal := true`)
  - FlashAttention backward prep (`dVec = rowSum(dO * O)`)
  - FlashAttention backward (partials: final `dQ` plus per-tile `dK`/`dV` blocks)

  Explicitly NOT covered in this first pass:
  - FA3 warp-specialized producer/consumer pipelines (`useWarpSpec := true`
    panics with a TODO). Once the warp-spec primitives stabilise we can fold
    those variants in without breaking the API.
  - GQA (`enableGqa := true`) — the factory accepts the flag but currently
    does not rewrite tile layouts; callers must keep `enableGqa := false` until
    the flag is implemented. It still panics via `panic!` to fail loudly.
-/
import Tyr.GPU.Codegen.Macros

import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels.AttentionFactory

open Tyr.GPU
open Tyr.GPU.Codegen

/-- Configuration for a FlashAttention-family kernel variant produced by the
    factory.

    Setting `useWarpSpec := true` currently triggers a `panic!` in the factory;
    FA3-style warp-specialized kernels stay hand-written until the primitives
    they rely on (TMA, async MMA pipelines) gain stable typed wrappers.

    `enableGqa` is reserved for grouped-query attention and likewise panics
    today.  `isCausal := true` selects an upper-triangle mask applied to the
    score tile. -/
structure FAVariantConfig where
  /-- Query/output sequence length (e.g. 64, 128, 768). -/
  seq : Nat
  /-- Head dimension. Must be a multiple of 16. Today every variant uses 64. -/
  headDim : Nat := 64
  /-- Number of KV tile blocks iterated by the inner loop (= `seq / tileSize`). -/
  kvBlocks : Nat
  /-- Q/K/V/O tile row size (e.g. 64). -/
  tileSize : Nat := 64
  /-- Input dtype (Q/K/V/O storage). -/
  dtype : GpuFloat := .BFloat16
  /-- Accumulator dtype used for the softmax/MMA accumulator and LSE vector. -/
  accDtype : GpuFloat := .Float32
  /-- Softmax scale (e.g. `1 / sqrt(headDim) = 0.125` for `headDim = 64`). -/
  scale : Float
  /-- Whether to write a log-sum-exp output vector for the backward pass. -/
  emitLse : Bool := false
  /-- Optional scale applied to the LSE vector before storing. `MhaH100`
      variants use `lseScale = -8.0` (matching ThunderKittens' `l = -8 * lse`
      convention for `head_dim = 64`); plain LSE variants keep `1.0`.
      Only honored when `emitLse := true`. -/
  lseScale : Float := 1.0
  /-- Target architecture. -/
  arch : GpuArch := .SM90
  /-- Use FA3-style warp specialization (producer + consumers). Currently
      unimplemented — the factory panics when this is `true`. -/
  useWarpSpec : Bool := false
  /-- Apply causal masking to the score tile. -/
  isCausal : Bool := false
  /-- Enable grouped-query attention. Currently unimplemented. -/
  enableGqa : Bool := false
  /-- Custom kernel name. If empty, callers should rely on the
      `@[gpu_kernel]` attribute to derive the name from the declaration. -/
  kernelName : String := ""
  deriving Repr, Inhabited

/-! ## Causal-aware softmax mask value

Shared by forward and backward factories. -/

/-- Standard `-inf`-like fill value used for causal masks before softmax. -/
def causalMaskVal : Float := -1.0e10

/-! ## Forward factory

The forward template mirrors the body of `tkFlashAttnFwd2Block`,
`tkFlashAttnFwd12Block`, `tkMhaH100Fwd2Block`, `tkMhaH100Fwd12Block`,
`tkFlashAttnCausalFwd1Block`, and their `...BlockLse` counterparts. When
`emitLse := true`, the factory appends the `computeLSE`/`scalarMulVec`/
`storeVec`/`storeVecGlobalRow` suffix used by the LSE-producing variants. -/

/-- Build a FlashAttention forward kernel body.

    The `dtype` and `accDtype` type parameters must match the runtime `cfg.dtype`
    and `cfg.accDtype` values; the factory only consumes the config fields that
    are not encoded in the pointer types. We keep the dtype at the type level so
    the caller supplies concrete `GPtr GpuFloat.BFloat16` / `GPtr GpuFloat.Float32`
    parameter types that match the hand-written kernels exactly.

    When `cfg.emitLse := false`, `lse_ptr` must be `none`; when
    `cfg.emitLse := true`, it must be `some lse`. Mismatch `panic!`s, which is
    the intended failure mode. -/
def mkFlashAttnFwdBody {dtype accDtype : GpuFloat} (cfg : FAVariantConfig)
    (q_ptr k_ptr v_ptr o_ptr : GPtr dtype)
    (lse_ptr : Option (GPtr accDtype))
    : KernelM Unit := do
  if cfg.useWarpSpec then
    KernelM.fail "AttentionFactory.mkFlashAttnFwdBody: useWarpSpec := true is not yet \
            supported. Use the hand-written FA3 kernels in Kernels/FlashAttn3.lean."
  if cfg.enableGqa then
    KernelM.fail "AttentionFactory.mkFlashAttnFwdBody: enableGqa := true is not yet \
            supported."

  let tileSize : Nat := cfg.tileSize
  let numKvBlocks : Nat := cfg.kvBlocks
  let scale : Float := cfg.scale
  let lseScale : Float := cfg.lseScale
  let isCausal : Bool := cfg.isCausal

  let coord ← blockCoord2D

  let q ← allocRT dtype tileSize tileSize
  let k ← allocRT dtype tileSize tileSize
  let v ← allocRT dtype tileSize tileSize .Col
  let o ← zeroRT accDtype tileSize tileSize

  let softmaxState ← allocSoftmaxState accDtype tileSize

  let qShared ← allocST dtype tileSize tileSize
  let kShared ← allocST dtype tileSize tileSize
  let vShared ← allocST dtype tileSize tileSize .Col
  let oShared ← allocST dtype tileSize tileSize

  -- LSE shared vector, only allocated when we'll store it.
  let lseShared? : Option (SV accDtype tileSize) ←
    if cfg.emitLse then do
      let s ← allocSV accDtype tileSize
      pure (some s)
    else
      pure none

  loadGlobal qShared q_ptr coord
  sync
  load q qShared

  for kvIdx in krange 0 numKvBlocks do
    let s ← zeroRT accDtype tileSize tileSize
    let p ← allocRT dtype tileSize tileSize

    loadGlobal kShared k_ptr (coord.withRow kvIdx.id)
    loadGlobal vShared v_ptr (coord.withRow kvIdx.id)
    sync
    load k kShared
    load v vShared

    mmaT s q k s
    scalarMul s s scale
    if isCausal then
      makeCausal s s (some causalMaskVal)
    onlineSoftmax s o softmaxState
    convert p s
    mma o p v o
    sync

  finalizeSoftmax o softmaxState

  let oBf16 ← allocRT dtype tileSize tileSize
  convert oBf16 o
  store oShared oBf16
  storeGlobal o_ptr oShared coord

  if cfg.emitLse then
    let lse ← computeLSE softmaxState
    if lseScale != 1.0 then
      scalarMulVec lse lse lseScale
    match lse_ptr, lseShared? with
    | some lptr, some lshared =>
        storeVec lshared lse
        storeVecGlobalRow lptr lshared coord
    | _, _ =>
        KernelM.fail "AttentionFactory.mkFlashAttnFwdBody: emitLse := true but no \
                lse_ptr was supplied"

/-- Convenience entry without an LSE pointer. -/
@[inline] def mkFlashAttnFwd {dtype : GpuFloat} (cfg : FAVariantConfig)
    (q_ptr k_ptr v_ptr o_ptr : GPtr dtype) : KernelM Unit :=
  -- `accDtype` is irrelevant when `emitLse := false`; default to Float32 so
  -- elaboration still has a concrete type to pin.
  mkFlashAttnFwdBody (dtype := dtype) (accDtype := .Float32)
    cfg q_ptr k_ptr v_ptr o_ptr none

/-- Convenience entry that always writes an LSE vector. -/
@[inline] def mkFlashAttnFwdLse {dtype accDtype : GpuFloat} (cfg : FAVariantConfig)
    (q_ptr k_ptr v_ptr o_ptr : GPtr dtype)
    (lse_ptr : GPtr accDtype) : KernelM Unit :=
  mkFlashAttnFwdBody (dtype := dtype) (accDtype := accDtype)
    { cfg with emitLse := true } q_ptr k_ptr v_ptr o_ptr (some lse_ptr)

/-! ## Backward prep factory

Mirrors `tkMhaH100BwdPrep2Block` and `flashAttnBwdPrep`: computes
`d[row] = sum_j(dO[row,j] * O[row,j])` per query tile. -/

/-- Build a FlashAttention backward-prep kernel body. -/
def mkFlashAttnBwdPrep {dtype accDtype : GpuFloat} (cfg : FAVariantConfig)
    (dO_ptr o_ptr : GPtr dtype)
    (d_ptr : GPtr accDtype)
    : KernelM Unit := do
  if cfg.useWarpSpec then
    KernelM.fail "AttentionFactory.mkFlashAttnBwdPrep: useWarpSpec := true is not yet supported."
  let tileSize : Nat := cfg.tileSize
  let coord ← blockCoord2D

  let dO ← allocRT dtype tileSize tileSize
  let o ← allocRT dtype tileSize tileSize
  let dOf ← allocRT accDtype tileSize tileSize
  let of ← allocRT accDtype tileSize tileSize
  let prod ← allocRT accDtype tileSize tileSize
  let dVec ← allocRV accDtype tileSize

  let dOShared ← allocST dtype tileSize tileSize
  let oShared ← allocST dtype tileSize tileSize
  let dShared ← allocSV accDtype tileSize

  loadGlobal dOShared dO_ptr coord
  loadGlobal oShared o_ptr coord
  sync
  load dO dOShared
  load o oShared

  convert dOf dO
  convert of o
  mul prod dOf of
  rowSum dVec prod

  storeVec dShared dVec
  storeVecGlobalRow d_ptr dShared coord

/-! ## Backward (partials) factory

Mirrors `tkMhaH100Bwd2BlockPartials` and `tkMhaH100Bwd12BlockPartials`.
Produces final `dQ` plus per-(q_tile, kv_tile) partial `dK`/`dV` blocks that
are reduced on the host. The `lScale`/`invLScale` convention matches the
`-8 * lse` MhaH100 storage format when `cfg.lseScale = -8.0`; for plain
`lse` forwards, set `cfg.lseScale = 1.0` (and `invLScale` below will be
`cfg.scale`). -/

private def invLScaleFor (cfg : FAVariantConfig) : Float :=
  -- For MhaH100's `l = -8 * lse` storage with head_dim=64, this is -0.125.
  -- For plain `lse` (lseScale = 1.0), reading-back is identity (scale=1.0).
  if cfg.lseScale == -8.0 then -0.125
  else if cfg.lseScale == 1.0 then 1.0
  else 1.0 / cfg.lseScale

/-- Build a FlashAttention backward kernel body that writes final `dQ` plus
    partial `dK`/`dV` tiles. -/
def mkFlashAttnBwdPartials {dtype accDtype : GpuFloat} (cfg : FAVariantConfig)
    (q_ptr k_ptr v_ptr dO_ptr : GPtr dtype)
    (l_ptr d_ptr dQ_ptr dK_part_ptr dV_part_ptr : GPtr accDtype)
    : KernelM Unit := do
  if cfg.useWarpSpec then
    KernelM.fail "AttentionFactory.mkFlashAttnBwdPartials: useWarpSpec := true is not yet supported."
  if cfg.enableGqa then
    KernelM.fail "AttentionFactory.mkFlashAttnBwdPartials: enableGqa := true is not yet supported."

  let tileSize : Nat := cfg.tileSize
  let numKvBlocks : Nat := cfg.kvBlocks
  let scale : Float := cfg.scale
  let invLScale : Float := invLScaleFor cfg
  let isCausal : Bool := cfg.isCausal

  let coord ← blockCoord2D

  let q ← allocRT dtype tileSize tileSize
  let dO ← allocRT dtype tileSize tileSize
  let dQ ← zeroRT accDtype tileSize tileSize
  let lTk ← allocRV accDtype tileSize
  let lse ← allocRV accDtype tileSize
  let dVec ← allocRV accDtype tileSize

  let rowShared ← allocST dtype tileSize tileSize
  let colShared ← allocST dtype tileSize tileSize .Col
  let dKShared ← allocST accDtype tileSize tileSize
  let dVShared ← allocST accDtype tileSize tileSize

  loadGlobal rowShared q_ptr coord
  sync
  load q rowShared
  loadGlobal rowShared dO_ptr coord
  sync
  load dO rowShared

  loadVecGlobalRowRV lTk l_ptr coord
  loadVecGlobalRowRV dVec d_ptr coord
  scalarMulVec lse lTk invLScale

  for kvIdx in krange 0 numKvBlocks do
    let k ← allocRT dtype tileSize tileSize
    let v ← allocRT dtype tileSize tileSize .Col
    let sT ← zeroRT accDtype tileSize tileSize
    let pT ← allocRT accDtype tileSize tileSize
    let dPT ← zeroRT accDtype tileSize tileSize
    let dST ← allocRT accDtype tileSize tileSize
    let dKPart ← zeroRT accDtype tileSize tileSize
    let dVPart ← zeroRT accDtype tileSize tileSize

    loadGlobal rowShared k_ptr (coord.withRow kvIdx.id)
    loadGlobal colShared v_ptr (coord.withRow kvIdx.id)
    sync
    load k rowShared
    load v colShared

    -- Reference-style orientation: keep score/probability blocks in KxQ order.
    mmaT sT k q sT
    scalarMul sT sT scale
    if isCausal then
      makeCausal sT sT (some causalMaskVal)
    subRow sT sT lse
    exp pT sT

    let vRow ← allocRT dtype tileSize tileSize
    swapLayout vRow v
    mmaT dPT vRow dO dPT
    subRow dPT dPT dVec

    mul dST pT dPT
    let dSTScaled ← allocRT accDtype tileSize tileSize
    scalarMul dSTScaled dST scale

    let pTBf16 ← allocRT dtype tileSize tileSize
    convert pTBf16 pT
    let dSTBf16 ← allocRT dtype tileSize tileSize
    convert dSTBf16 dSTScaled

    let dOCol ← allocRT dtype tileSize tileSize .Col
    swapLayout dOCol dO
    mma dVPart pTBf16 dOCol dVPart

    let dSRow ← allocRT dtype tileSize tileSize
    transpose dSRow dSTBf16
    let dSCol ← allocRT dtype tileSize tileSize .Col
    swapLayout dSCol dSRow
    let qCol ← allocRT dtype tileSize tileSize .Col
    swapLayout qCol q
    mmaAtB dKPart dSCol qCol dKPart
    let kCol ← allocRT dtype tileSize tileSize .Col
    swapLayout kCol k
    mma dQ dSRow kCol dQ

    let dkvCoord := (coord.withRow kvIdx.id).withCol coord.r
    store dKShared dKPart
    sync
    storeGlobal dK_part_ptr dKShared dkvCoord
    store dVShared dVPart
    sync
    storeGlobal dV_part_ptr dVShared dkvCoord
    sync

  store dKShared dQ
  sync
  storeGlobal dQ_ptr dKShared coord

/-! ## Ready-made configs for the existing hand-written variants

These mirror the literal configs baked into `Kernels/FlashAttn.lean`,
`Kernels/FlashAttnBwd.lean`, `Kernels/MhaH100.lean`, and
`Kernels/FlashAttnCausal64.lean`. They are useful both as direct factory
callers and as canonical examples when adding new shapes. -/

namespace Configs

/-- Non-causal FA forward, seq=128, head_dim=64, two KV blocks, no LSE. -/
def flashAttnFwd2Block : FAVariantConfig :=
  { seq := 128, kvBlocks := 2, scale := 0.125 }

/-- Non-causal FA forward, seq=128, head_dim=64, two KV blocks, with LSE. -/
def flashAttnFwd2BlockLse : FAVariantConfig :=
  { seq := 128, kvBlocks := 2, scale := 0.125, emitLse := true }

/-- MhaH100 forward, seq=128, two KV blocks, head_dim=64, LSE stored as `-8*lse`. -/
def mhaH100Fwd2Block : FAVariantConfig :=
  { seq := 128, kvBlocks := 2, scale := 0.125, emitLse := true, lseScale := -8.0 }

/-- Non-causal FA forward, seq=768, 12 KV blocks. -/
def flashAttnFwd12Block : FAVariantConfig :=
  { seq := 768, kvBlocks := 12, scale := 0.125 }

/-- Non-causal FA forward (LSE), seq=768, 12 KV blocks. -/
def flashAttnFwd12BlockLse : FAVariantConfig :=
  { seq := 768, kvBlocks := 12, scale := 0.125, emitLse := true }

/-- MhaH100 forward, seq=768, 12 KV blocks. -/
def mhaH100Fwd12Block : FAVariantConfig :=
  { seq := 768, kvBlocks := 12, scale := 0.125, emitLse := true, lseScale := -8.0 }

/-- Causal FA forward, seq=64, single KV block, head_dim=64, no LSE. -/
def flashAttnCausalFwd1Block : FAVariantConfig :=
  { seq := 64, kvBlocks := 1, scale := 0.125, isCausal := true }

/-- Causal FA forward with LSE, seq=64, single KV block. -/
def flashAttnCausalFwd1BlockLse : FAVariantConfig :=
  { seq := 64, kvBlocks := 1, scale := 0.125, emitLse := true, isCausal := true }

/-- MhaH100 bwd partials, seq=128, 2 KV blocks. -/
def mhaH100Bwd2Block : FAVariantConfig :=
  { seq := 128, kvBlocks := 2, scale := 0.125, emitLse := true, lseScale := -8.0 }

/-- MhaH100 bwd partials, seq=768, 12 KV blocks. -/
def mhaH100Bwd12Block : FAVariantConfig :=
  { seq := 768, kvBlocks := 12, scale := 0.125, emitLse := true, lseScale := -8.0 }

end Configs

/-! ## Factory-generated kernels

These re-implementations of the existing hand-written kernels serve as
both regression targets and usage examples. They share signatures with
the originals so launcher callers can swap them in directly. The factory
bodies should emit byte-identical CUDA (modulo kernel name) for each
covered variant. -/

/-- Factory-backed FA forward, seq=128, 2 KV blocks (no LSE). -/
@[gpu_kernel .SM90]
def tkFlashAttnFwd2BlockFactory
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit :=
  mkFlashAttnFwd Configs.flashAttnFwd2Block q_ptr k_ptr v_ptr o_ptr

/-- Factory-backed FA forward with LSE, seq=128, 2 KV blocks. -/
@[gpu_kernel .SM90]
def tkFlashAttnFwd2BlockLseFactory
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (lse_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit :=
  mkFlashAttnFwdLse Configs.flashAttnFwd2BlockLse q_ptr k_ptr v_ptr o_ptr lse_ptr

/-- Factory-backed MhaH100 forward, seq=128, 2 KV blocks. -/
@[gpu_kernel .SM90]
def tkMhaH100Fwd2BlockFactory
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit :=
  mkFlashAttnFwdLse Configs.mhaH100Fwd2Block q_ptr k_ptr v_ptr o_ptr l_ptr

/-- Factory-backed FA forward, seq=768, 12 KV blocks (no LSE). -/
@[gpu_kernel .SM90]
def tkFlashAttnFwd12BlockFactory
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit :=
  mkFlashAttnFwd Configs.flashAttnFwd12Block q_ptr k_ptr v_ptr o_ptr

/-- Factory-backed FA forward with LSE, seq=768, 12 KV blocks. -/
@[gpu_kernel .SM90]
def tkFlashAttnFwd12BlockLseFactory
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (lse_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit :=
  mkFlashAttnFwdLse Configs.flashAttnFwd12BlockLse q_ptr k_ptr v_ptr o_ptr lse_ptr

/-- Factory-backed MhaH100 forward, seq=768, 12 KV blocks. -/
@[gpu_kernel .SM90]
def tkMhaH100Fwd12BlockFactory
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit :=
  mkFlashAttnFwdLse Configs.mhaH100Fwd12Block q_ptr k_ptr v_ptr o_ptr l_ptr

/-- Factory-backed causal FA forward, seq=64, 1 KV block (no LSE). -/
@[gpu_kernel .SM90]
def tkFlashAttnCausalFwd1BlockFactory
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit :=
  mkFlashAttnFwd Configs.flashAttnCausalFwd1Block q_ptr k_ptr v_ptr o_ptr

/-- Factory-backed causal FA forward with LSE, seq=64, 1 KV block. -/
@[gpu_kernel .SM90]
def tkFlashAttnCausalFwd1BlockLseFactory
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (lse_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit :=
  mkFlashAttnFwdLse Configs.flashAttnCausalFwd1BlockLse q_ptr k_ptr v_ptr o_ptr lse_ptr

/-- Factory-backed MhaH100 backward-prep kernel. -/
@[gpu_kernel .SM90]
def tkMhaH100BwdPrep2BlockFactory
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (d_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit :=
  mkFlashAttnBwdPrep Configs.mhaH100Bwd2Block dO_ptr o_ptr d_ptr

/-- Factory-backed MhaH100 backward (partials), seq=128, 2 KV blocks. -/
@[gpu_kernel .SM90]
def tkMhaH100Bwd2BlockPartialsFactory
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32)
    (dK_part_ptr : GPtr GpuFloat.Float32)
    (dV_part_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit :=
  mkFlashAttnBwdPartials Configs.mhaH100Bwd2Block
    q_ptr k_ptr v_ptr dO_ptr l_ptr d_ptr dQ_ptr dK_part_ptr dV_part_ptr

/-- Factory-backed MhaH100 backward (partials), seq=768, 12 KV blocks. -/
@[gpu_kernel .SM90]
def tkMhaH100Bwd12BlockPartialsFactory
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (dO_ptr : GPtr GpuFloat.BFloat16)
    (l_ptr : GPtr GpuFloat.Float32)
    (d_ptr : GPtr GpuFloat.Float32)
    (dQ_ptr : GPtr GpuFloat.Float32)
    (dK_part_ptr : GPtr GpuFloat.Float32)
    (dV_part_ptr : GPtr GpuFloat.Float32)
    (_seq_len : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit :=
  mkFlashAttnBwdPartials Configs.mhaH100Bwd12Block
    q_ptr k_ptr v_ptr dO_ptr l_ptr d_ptr dQ_ptr dK_part_ptr dV_part_ptr

end Tyr.GPU.Kernels.AttentionFactory
