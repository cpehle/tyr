/- ThunderKittens-style decode-attention forward kernels for BF16 Q/K/V on H100.

Three specializations covering the head dims that matter in practice:

- `tkMhaH100DecodeFwd`    : `head_dim = 128` (Llama-3, Qwen2-7B, Mistral-7B)
- `tkMhaH100DecodeFwd64`  : `head_dim = 64`  (Qwen3-4B, Llama-2-7B/13B variants)
- `tkMhaH100DecodeFwd256` : `head_dim = 256` (Qwen 3.5/3.6 family, Gemma-2 27B)

All three share the same schedule (factored into `decodeFwdBodyImpl`) — only
the WGMMA tile head dim and the score scale change.

Semantics:
- Q/O layout: [batch, q_heads, 1, head_dim]
- K/V layout: [batch, kv_heads, kv_seq, head_dim]
- GQA: `kv_head = q_head / (q_heads / kv_heads)`
- mask/dropout/causal/custom scale are not part of this specialization
- accumulation and online softmax are Float32, output is BF16

Schedule (V1, modeled on TK's `mha_decode` reference but simplified):
- Grid: one CTA per `(batch, q_head)`, `gridX = batch * q_heads`.
  No GQA-group Q packing in V1; KV reads across a GQA group rely on L2 reuse.
- Block: 128 threads (1 warpgroup).
- Tiles: 64×D BF16 Q (only row 0 carries valid data; rows 1..63 are zero from
  TMA OOB-fill), 64×D BF16 K, 64×D BF16 V (column-major), 64×64 F32 S,
  64×64 BF16 P, 64×D F32 O accumulator.  D ∈ {64, 128}.
- Loop: runtime over `ceil(kv_seq / 64)` KV blocks. Async TMA load K+V → WGMMA
  QK^T → tail mask (last block, runtime cutoff) → online log2 softmax → convert
  to BF16 → WGMMA PV.
- Tail mask: `right_fill(scores, scores, kv_seq - kvIdx*64, -inf)` runs every
  iteration; for non-tail blocks the runtime cutoff is `>= 64` (no-op), for the
  tail block it masks OOB score columns to -inf so they don't contribute to
  softmax. Mirrors TK's `mha_h100_lcf.cu:71` exactly.
- Single-buffer KV (no producer/consumer split, no ring buffer); the producer/
  consumer + ring-buffered version is a planned V2 follow-up. See
  `dev/decode_kernel_v2_plan.md`.

Constraints checked at runtime selection:
- head_dim == 128 (selects `tkMhaH100DecodeFwd`)  OR
- head_dim == 64  (selects `tkMhaH100DecodeFwd64`)
-/
import Tyr.GPU.Codegen.Macros
import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels

open Tyr.GPU
open Tyr.GPU.Codegen

private def asyncLoadDecodeTile {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : GPtr dtype)
    (coord : RTileCoord)
    (sem : Semaphore) : KernelM Unit := do
  initSemaphore sem 0 1
  blockSync
  expectBytes sem (rows * cols * dtype.bytes)
  loadGlobalAsync dst src coord sem.id
  waitSemaphore sem

private def asyncLoadDecodePair {dtype : GpuFloat} {rows cols : Nat}
    {layoutA layoutB : TileLayout}
    (dstA : ST dtype rows cols layoutA)
    (srcA : GPtr dtype)
    (coordA : RTileCoord)
    (dstB : ST dtype rows cols layoutB)
    (srcB : GPtr dtype)
    (coordB : RTileCoord)
    (sem : Semaphore) : KernelM Unit := do
  initSemaphore sem 0 1
  blockSync
  expectBytes sem (2 * rows * cols * dtype.bytes)
  loadGlobalAsync dstA srcA coordA sem.id
  loadGlobalAsync dstB srcB coordB sem.id
  waitSemaphore sem

/-- Shared kernel body for all head-dim specializations.

    `hdim` is a compile-time `Nat` (64, 128, or 256) that determines the
    WGMMA tile head dimension and the corresponding score scale
    `scoreScaleLog2e`. The body is otherwise identical for all
    specializations.

    Tile-row strategy: shared tiles are 64 rows (the full warpgroup-level
    WGMMA m=64), but per-warp register tiles are 16 rows — each warp owns
    1/4 of the warpgroup output. This matches the TK reference structure
    (`mha_h100_lcf.cu` uses `rt_fl<16, qo_tile::cols>` for the
    accumulator) and was the missing piece for d=256: per-warp
    `rt<float, 64, 256>` was 512 fp32/thread, exactly the
    65536/128-thread budget under `__launch_bounds__(128, 1)`, so q/p/state
    pushed it over the limit and ptxas reported `wgmma.mma_async
    instructions are serialized due to insufficient register resources`.

    Q stays in shared throughout (`warpgroupMmaSharedT64x16` consumes Q
    directly from `qShared`); avoiding the `load q qShared` step keeps
    register usage flat for the Q tile, since with qSeq=1 only one Q row
    is real anyway. -/
private def decodeFwdBodyImpl (hdim : Nat) (scoreScaleLog2e : Float)
    (hHdim : hdim % 16 = 0)
    (q_ptr k_ptr v_ptr o_ptr : GPtr GpuFloat.BFloat16)
    (q_heads kv_heads kv_seq : KVal UInt64) : KernelM Unit := do
  let tileM : Nat := 64       -- warpgroup-level M (shared tile rows)
  let tileMWarp : Nat := 16   -- per-warp register tile rows (warpgroup distributes 64 rows across 4 warps × 16)
  let tileN : Nat := 64       -- KV block (kv positions per WGMMA inner loop)
  setLaunchBounds 128 1

  let blockId32 ← getBlockIdx 0 "decode_block"
  let blockId ← castUInt64 blockId32 "decode_block_u64"
  let qHead ← scalarMod blockId q_heads "q_head"
  let batchId ← scalarDivVal blockId q_heads "batch_id"
  let kvHead ← runtimeGqaKvHead qHead q_heads kv_heads

  let zero ← constIntVal 0 "zero"
  let qCoord := makeRTileCoord batchId.id qHead.id zero.id zero.id
  let oCoord := qCoord

  -- Per-warp register tiles (16 rows = warp's slice of the 64-row warpgroup output)
  let o ← zeroRT .Float32 tileMWarp hdim
  let softmaxState ← allocSoftmaxState .Float32 tileMWarp

  -- Shared tiles (64 rows for warpgroup-level WGMMA)
  let qShared ← allocST .BFloat16 tileM hdim
  let kShared ← allocST .BFloat16 tileN hdim
  let vShared ← allocST .BFloat16 tileN hdim .Col
  let oShared ← allocST .BFloat16 tileM hdim
  let qSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadDecodeTile qShared q_ptr qCoord qSem
  -- Q stays in shared; consumed via `warpgroupMmSharedT64x16` below.

  let blockN64 ← constUInt64Val (tileN : Int) "block_n"
  let blockNMinusOne ← constUInt64Val ((tileN - 1 : Nat) : Int) "block_n_minus_one"
  let kvSeqPlusPad ← scalarAddVal kv_seq blockNMinusOne "kv_seq_plus_pad"
  let numKvBlocks ← scalarDivVal kvSeqPlusPad blockN64 "num_kv_blocks"
  let numKvBlocks32 : KVal UInt32 ← castScalar .UInt32 numKvBlocks "num_kv_blocks_u32"
  let kvSeq32 : KVal UInt32 ← castScalar .UInt32 kv_seq "kv_seq_u32"
  let blockN32 : KVal UInt32 ← constIntVal (tileN : Int) "block_n_u32"

  -- Loop-peel the last KV block out of the inner loop. Even an `if`-guarded
  -- `right_fill` keeps the unrolled mask body in the inner-loop function and
  -- disrupts WGMMA register scheduling, breaking d=64 multi-iter parity
  -- (verified empirically). Inner iterations never need the mask
  -- (cutoff >= tileN); the single peeled-out tail iteration carries it.
  let oneU32 ← constIntVal (1 : Int) "one_u32"
  let numKvBlocksMinusOne ← scalarSub numKvBlocks32 oneU32 "n_kv_blocks_minus_one"

  for kvIdx in kvrange 0 numKvBlocksMinusOne do
    let s ← zeroRT .Float32 tileMWarp tileN
    let p ← allocRT .BFloat16 tileMWarp tileN
    let kvCoord := makeRTileCoord batchId.id kvHead.id kvIdx.id zero.id

    asyncLoadDecodePair kShared k_ptr kvCoord vShared v_ptr kvCoord kvSem
    -- Shared/shared mma_ABt: dst per-warp 16×64 ← qShared (64×hdim) @ kShared^T
    warpgroupMmSharedT64x16 s qShared kShared (hK := hHdim)
    mmaAsyncWait
    onlineSoftmaxLog2 s o softmaxState scoreScaleLog2e
    convert p s
    -- Register/shared mma_AB: dst per-warp 16×hdim ← p (16×64) @ vShared (64×hdim)
    warpgroupMma o p vShared (hN := hHdim)
    mmaAsyncWait

  -- Peeled tail iteration: same body, plus the runtime-guarded right_fill.
  let s ← zeroRT .Float32 tileMWarp tileN
  let p ← allocRT .BFloat16 tileMWarp tileN
  let kvCoord := makeRTileCoord batchId.id kvHead.id numKvBlocksMinusOne.id zero.id

  asyncLoadDecodePair kShared k_ptr kvCoord vShared v_ptr kvCoord kvSem
  warpgroupMmSharedT64x16 s qShared kShared (hK := hHdim)
  mmaAsyncWait
  let kvBase ← scalarMulVal numKvBlocksMinusOne blockN32 "kv_base_tail"
  let cutoff ← scalarSub kvSeq32 kvBase "tail_cutoff"
  let needsMask ← scalarLt cutoff blockN32 "needs_tail_mask"
  emitIf needsMask.id do
    rightFillVal s s cutoff (some (-3.4028234663852886e38))
  onlineSoftmaxLog2 s o softmaxState scoreScaleLog2e
  convert p s
  warpgroupMma o p vShared (hN := hHdim)
  mmaAsyncWait

  finalizeSoftmax o softmaxState

  let oBf16 ← allocRT .BFloat16 tileMWarp hdim
  convert oBf16 o
  -- Warpgroup-distributed store: each warp's 16 rows → its strip of the
  -- 64-row shared tile, then a single TMA store writes all 64 rows.
  warpgroupStore oShared oBf16
  -- Use a TMA store (vs. non-TMA `warp::store`) so OOB writes get dropped.
  -- The 64×D shared tile has only 1 valid query row (row 0 of warp 0);
  -- rows 1..63 came from TMA OOB-zero-fill on the Q load and produce
  -- garbage softmax outputs that must NOT be written to global. TMA stores
  -- drop OOB writes automatically because the global gl is configured with
  -- q_seq=1.
  storeGlobalAsync o_ptr oShared oCoord
  tmaStoreCommitGroup
  tmaStoreAsyncWait

/-- Decode attention forward, head_dim = 128.

    Launch configuration: `gridX = batch * q_heads`, `blockX = 128`. One CTA
    per `(batch, q_head)`. Dynamic shared memory must accommodate the full
    set of shared tiles (qShared + kShared + vShared + oShared, 64×128 BF16
    each ≈ 64 KiB). -/
@[gpu_kernel .SM90]
def tkMhaH100DecodeFwd
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (_batch : KVal UInt64)
    (q_heads : KVal UInt64)
    (kv_heads : KVal UInt64)
    (kv_seq : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit :=
  -- 1/sqrt(128) * log2(e) ≈ 0.0883883476 * 1.44269504 ≈ 0.12750032696
  decodeFwdBodyImpl 128 0.12750032696 (by decide)
    q_ptr k_ptr v_ptr o_ptr q_heads kv_heads kv_seq

/-- Decode attention forward, head_dim = 64.

    Launch configuration: `gridX = batch * q_heads`, `blockX = 128`. One CTA
    per `(batch, q_head)`. Shared-memory budget is half the 128-dim variant
    (~32 KiB total tile area). -/
@[gpu_kernel .SM90]
def tkMhaH100DecodeFwd64
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (_batch : KVal UInt64)
    (q_heads : KVal UInt64)
    (kv_heads : KVal UInt64)
    (kv_seq : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit :=
  -- 1/sqrt(64) * log2(e) = 0.125 * 1.44269504 = 0.18033688011125
  decodeFwdBodyImpl 64 0.18033688011125 (by decide)
    q_ptr k_ptr v_ptr o_ptr q_heads kv_heads kv_seq

/-- Decode attention forward, head_dim = 256.

    Targets Qwen 3.5/3.6 (`head_dim=256`, `q_heads/kv_heads = 8` GQA ratio)
    and Gemma-2 27B. Launch configuration: `gridX = batch * q_heads`,
    `blockX = 128`. One CTA per `(batch, q_head)`.

    SMEM: 4 × 64 × 256 × 2 = 128 KiB tile area + ~4 KiB headroom — fits in
    the 227 KiB H100 dynamic SMEM cap.

    Register pressure: per-warp `rt<float, 64, 256>` is 512 fp32/thread
    plus q/p tiles + softmax state. With `__launch_bounds__(128, 1)` the
    thread reg budget is 65536/128 = 512, so this almost certainly spills
    to local memory. Correctness holds; perf is capped until we refactor
    to 16-row warp tiles (TK reference structure). Tracked in
    `dev/issues.md` D01b. -/
@[gpu_kernel .SM90]
def tkMhaH100DecodeFwd256
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (_batch : KVal UInt64)
    (q_heads : KVal UInt64)
    (kv_heads : KVal UInt64)
    (kv_seq : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit :=
  -- 1/sqrt(256) * log2(e) = 0.0625 * 1.44269504 = 0.090168440034
  decodeFwdBodyImpl 256 0.090168440034 (by decide)
    q_ptr k_ptr v_ptr o_ptr q_heads kv_heads kv_seq

/-! ## GQA-packed decode (V2 — D03)

Decode-V2 step 1 from `dev/decode_kernel_v2_plan.md`. Pack `R = q_heads /
kv_heads` query heads of a GQA group into the first R rows of the 64-row
Q tile; grid drops from `batch * q_heads` to `batch * kv_heads`. Each CTA
loads K/V once for the whole group instead of R times.

Eligibility (enforced by the C++ dispatcher in `cc/src/tyr_ops.cpp`):
- `q_heads % kv_heads == 0`
- `R = q_heads / kv_heads ∈ {2, 4, 8, 16}` (so R | 64; rows beyond R are
  TMA OOB-zero-fill on Q load and TMA OOB-drop on O store)
- otherwise V1 (`tkMhaH100DecodeFwd*`) is selected and computes the
  same answer with worse bandwidth.

Body is a near-clone of `decodeFwdBodyImpl` differing only in:
- grid axis: `kvHead = blockId % kv_heads; batchId = blockId / kv_heads`
- Q coord: `(batch, kvHead * R, 0, 0)` — picks R contiguous heads
- O coord: same as Q coord — TMA store drops rows ≥ R

The WGMMA / softmax / convert / store machinery is unchanged: per-warp
16-row tiles still operate the same way, rows beyond R compute on
OOB-zero Q rows producing OOB outputs that get dropped on TMA store.
-/
private def decodeFwdBodyImplGqa (hdim : Nat) (scoreScaleLog2e : Float)
    (hHdim : hdim % 16 = 0)
    (q_ptr k_ptr v_ptr o_ptr : GPtr GpuFloat.BFloat16)
    (q_heads kv_heads kv_seq : KVal UInt64) : KernelM Unit := do
  let tileM : Nat := 64
  let tileMWarp : Nat := 16
  let tileN : Nat := 64
  setLaunchBounds 128 1

  let blockId32 ← getBlockIdx 0 "decode_block"
  let blockId ← castUInt64 blockId32 "decode_block_u64"
  -- D03 grid: batch * kv_heads (one CTA per GQA group).
  let kvHead ← scalarMod blockId kv_heads "kv_head"
  let batchId ← scalarDivVal blockId kv_heads "batch_id"
  -- R = q_heads / kv_heads. firstQHead = kvHead * R is the leading head
  -- index in this group; the next R-1 heads in q_heads layout follow it.
  let gqaRatio ← scalarDivVal q_heads kv_heads "gqa_ratio"
  let firstQHead ← scalarMulVal kvHead gqaRatio "first_q_head"

  let zero ← constIntVal 0 "zero"
  let qCoord := makeRTileCoord batchId.id firstQHead.id zero.id zero.id
  let oCoord := qCoord

  let o ← zeroRT .Float32 tileMWarp hdim
  let softmaxState ← allocSoftmaxState .Float32 tileMWarp

  let qShared ← allocST .BFloat16 tileM hdim
  let kShared ← allocST .BFloat16 tileN hdim
  let vShared ← allocST .BFloat16 tileN hdim .Col
  let oShared ← allocST .BFloat16 tileM hdim
  let qSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadDecodeTile qShared q_ptr qCoord qSem

  let blockN64 ← constUInt64Val (tileN : Int) "block_n"
  let blockNMinusOne ← constUInt64Val ((tileN - 1 : Nat) : Int) "block_n_minus_one"
  let kvSeqPlusPad ← scalarAddVal kv_seq blockNMinusOne "kv_seq_plus_pad"
  let numKvBlocks ← scalarDivVal kvSeqPlusPad blockN64 "num_kv_blocks"
  let numKvBlocks32 : KVal UInt32 ← castScalar .UInt32 numKvBlocks "num_kv_blocks_u32"
  let kvSeq32 : KVal UInt32 ← castScalar .UInt32 kv_seq "kv_seq_u32"
  let blockN32 : KVal UInt32 ← constIntVal (tileN : Int) "block_n_u32"

  let oneU32 ← constIntVal (1 : Int) "one_u32"
  let numKvBlocksMinusOne ← scalarSub numKvBlocks32 oneU32 "n_kv_blocks_minus_one"

  for kvIdx in kvrange 0 numKvBlocksMinusOne do
    let s ← zeroRT .Float32 tileMWarp tileN
    let p ← allocRT .BFloat16 tileMWarp tileN
    let kvCoord := makeRTileCoord batchId.id kvHead.id kvIdx.id zero.id

    asyncLoadDecodePair kShared k_ptr kvCoord vShared v_ptr kvCoord kvSem
    warpgroupMmSharedT64x16 s qShared kShared (hK := hHdim)
    mmaAsyncWait
    onlineSoftmaxLog2 s o softmaxState scoreScaleLog2e
    convert p s
    warpgroupMma o p vShared (hN := hHdim)
    mmaAsyncWait

  let s ← zeroRT .Float32 tileMWarp tileN
  let p ← allocRT .BFloat16 tileMWarp tileN
  let kvCoord := makeRTileCoord batchId.id kvHead.id numKvBlocksMinusOne.id zero.id

  asyncLoadDecodePair kShared k_ptr kvCoord vShared v_ptr kvCoord kvSem
  warpgroupMmSharedT64x16 s qShared kShared (hK := hHdim)
  mmaAsyncWait
  let kvBase ← scalarMulVal numKvBlocksMinusOne blockN32 "kv_base_tail"
  let cutoff ← scalarSub kvSeq32 kvBase "tail_cutoff"
  let needsMask ← scalarLt cutoff blockN32 "needs_tail_mask"
  emitIf needsMask.id do
    rightFillVal s s cutoff (some (-3.4028234663852886e38))
  onlineSoftmaxLog2 s o softmaxState scoreScaleLog2e
  convert p s
  warpgroupMma o p vShared (hN := hHdim)
  mmaAsyncWait

  finalizeSoftmax o softmaxState

  let oBf16 ← allocRT .BFloat16 tileMWarp hdim
  convert oBf16 o
  warpgroupStore oShared oBf16
  -- TMA store at coord (batch, firstQHead, 0, 0). Rows 0..R-1 land in the
  -- valid `[batch, firstQHead..firstQHead+R-1, 0, :]` global slots; rows
  -- R..63 are TMA OOB-dropped because the gl is configured with q_seq=1.
  storeGlobalAsync o_ptr oShared oCoord
  tmaStoreCommitGroup
  tmaStoreAsyncWait

/-- Decode attention forward, head_dim = 128, GQA-packed (V2 D03).
    Launch: `gridX = batch * kv_heads`, `blockX = 128`. -/
@[gpu_kernel .SM90]
def tkMhaH100DecodeFwdGqa
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (_batch : KVal UInt64)
    (q_heads : KVal UInt64)
    (kv_heads : KVal UInt64)
    (kv_seq : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit :=
  decodeFwdBodyImplGqa 128 0.12750032696 (by decide)
    q_ptr k_ptr v_ptr o_ptr q_heads kv_heads kv_seq

/-- Decode attention forward, head_dim = 64, GQA-packed (V2 D03). -/
@[gpu_kernel .SM90]
def tkMhaH100DecodeFwdGqa64
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (_batch : KVal UInt64)
    (q_heads : KVal UInt64)
    (kv_heads : KVal UInt64)
    (kv_seq : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit :=
  decodeFwdBodyImplGqa 64 0.18033688011125 (by decide)
    q_ptr k_ptr v_ptr o_ptr q_heads kv_heads kv_seq

/-- Decode attention forward, head_dim = 256, GQA-packed (V2 D03). -/
@[gpu_kernel .SM90]
def tkMhaH100DecodeFwdGqa256
    (q_ptr : GPtr GpuFloat.BFloat16)
    (k_ptr : GPtr GpuFloat.BFloat16)
    (v_ptr : GPtr GpuFloat.BFloat16)
    (o_ptr : GPtr GpuFloat.BFloat16)
    (_batch : KVal UInt64)
    (q_heads : KVal UInt64)
    (kv_heads : KVal UInt64)
    (kv_seq : KVal UInt64)
    (_head_dim : KVal UInt64) : KernelM Unit :=
  decodeFwdBodyImplGqa 256 0.090168440034 (by decide)
    q_ptr k_ptr v_ptr o_ptr q_heads kv_heads kv_seq

end Tyr.GPU.Kernels
