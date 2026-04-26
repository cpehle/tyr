/- ThunderKittens-style decode-attention forward kernels for BF16 Q/K/V on H100.

Two specializations covering the head dims that matter in practice:

- `tkMhaH100DecodeFwd`   : `head_dim = 128` (Llama-3, Qwen2-7B, Mistral-7B)
- `tkMhaH100DecodeFwd64` : `head_dim = 64`  (Qwen3-4B, Llama-2-7B/13B variants)

Both share the same schedule (factored into `decodeFwdBodyImpl`) — only the
WGMMA tile head dim and the score scale change.

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

/-- Shared kernel body for both head-dim specializations.

    `hdim` is a compile-time `Nat` (64 or 128) that determines the WGMMA tile
    head dimension and the corresponding score scale `scoreScaleLog2e`. The
    body is otherwise identical for both specializations. -/
private def decodeFwdBodyImpl (hdim : Nat) (scoreScaleLog2e : Float)
    (hHdim : hdim % 16 = 0)
    (q_ptr k_ptr v_ptr o_ptr : GPtr GpuFloat.BFloat16)
    (q_heads kv_heads kv_seq : KVal UInt64) : KernelM Unit := do
  let tileM : Nat := 64
  let tileN : Nat := 64
  setLaunchBounds 128 1

  let blockId32 ← getBlockIdx 0 "decode_block"
  let blockId ← castUInt64 blockId32 "decode_block_u64"
  let qHead ← scalarMod blockId q_heads "q_head"
  let batchId ← scalarDivVal blockId q_heads "batch_id"
  let kvHead ← runtimeGqaKvHead qHead q_heads kv_heads

  let zero ← constIntVal 0 "zero"
  let qCoord := makeRTileCoord batchId.id qHead.id zero.id zero.id
  let oCoord := qCoord

  let q ← allocRT .BFloat16 tileM hdim
  let o ← zeroRT .Float32 tileM hdim
  let softmaxState ← allocSoftmaxState .Float32 tileM

  let qShared ← allocST .BFloat16 tileM hdim
  let kShared ← allocST .BFloat16 tileN hdim
  let vShared ← allocST .BFloat16 tileN hdim .Col
  let oShared ← allocST .BFloat16 tileM hdim
  let qSem ← allocSemaphore
  let kvSem ← allocSemaphore

  asyncLoadDecodeTile qShared q_ptr qCoord qSem
  load q qShared

  let blockN64 ← constUInt64Val (tileN : Int) "block_n"
  let blockNMinusOne ← constUInt64Val ((tileN - 1 : Nat) : Int) "block_n_minus_one"
  let kvSeqPlusPad ← scalarAddVal kv_seq blockNMinusOne "kv_seq_plus_pad"
  let numKvBlocks ← scalarDivVal kvSeqPlusPad blockN64 "num_kv_blocks"
  let numKvBlocks32 : KVal UInt32 ← castScalar .UInt32 numKvBlocks "num_kv_blocks_u32"
  let kvSeq32 : KVal UInt32 ← castScalar .UInt32 kv_seq "kv_seq_u32"
  let blockN32 : KVal UInt32 ← constIntVal (tileN : Int) "block_n_u32"

  -- Loop-peel the last KV block out of the inner loop. Even an `if`-guarded
  -- `right_fill` keeps the unrolled mask body in the inner-loop function and
  -- disrupts WGMMA register scheduling for the 64-row warpgroup score tile,
  -- breaking d=64 multi-iter parity (verified empirically). Inner iterations
  -- never need the mask (cutoff >= tileN), so emit them with no mask code at
  -- all. The single peeled-out tail iteration carries the conditional mask.
  let oneU32 ← constIntVal (1 : Int) "one_u32"
  let numKvBlocksMinusOne ← scalarSub numKvBlocks32 oneU32 "n_kv_blocks_minus_one"

  for kvIdx in kvrange 0 numKvBlocksMinusOne do
    let s ← zeroRT .Float32 tileM tileN
    let p ← allocRT .BFloat16 tileM tileN
    let kvCoord := makeRTileCoord batchId.id kvHead.id kvIdx.id zero.id

    asyncLoadDecodePair kShared k_ptr kvCoord vShared v_ptr kvCoord kvSem
    warpgroupMmT s q kShared (hK := hHdim)
    mmaAsyncWait
    onlineSoftmaxLog2 s o softmaxState scoreScaleLog2e
    convert p s
    warpgroupMma o p vShared (hN := hHdim)
    mmaAsyncWait

  -- Peeled tail iteration: same body, plus the runtime-guarded right_fill.
  let s ← zeroRT .Float32 tileM tileN
  let p ← allocRT .BFloat16 tileM tileN
  let kvCoord := makeRTileCoord batchId.id kvHead.id numKvBlocksMinusOne.id zero.id

  asyncLoadDecodePair kShared k_ptr kvCoord vShared v_ptr kvCoord kvSem
  warpgroupMmT s q kShared (hK := hHdim)
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

  let oBf16 ← allocRT .BFloat16 tileM hdim
  convert oBf16 o
  store oShared oBf16
  -- Use a TMA store (vs. non-TMA `warp::store`) so OOB writes get dropped.
  -- The 64xD tile only has 1 valid query row (row 0); rows 1..63 came from
  -- TMA OOB-zero-fill on the Q load and produce garbage softmax outputs that
  -- must NOT be written to global. With a non-TMA `warp::store` they would
  -- spill into the next q_head's row 0 (since [B,Hq,1,D] has stride S*D=D
  -- between heads), corrupting all output. TMA stores drop OOB writes
  -- automatically because the global gl is configured with q_seq=1.
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

end Tyr.GPU.Kernels
