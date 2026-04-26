# TK-style decode kernel — V2 roadmap

V1 (landed): `tkMhaH100DecodeFwd` and `tkMhaH100DecodeFwd64` in
`Tyr/GPU/Kernels/MhaH100Decode.lean`. Single warpgroup, single-buffer KV,
WGMMA QKᵀ + PV, online log2 softmax, runtime tail mask via TK `right_fill`.
Eligible: BF16, qSeq=1, head_dim ∈ {64, 128}, GQA-valid, any kv_seq ≥ 1.

V2 candidates ranked by ROI for production decode workloads.

## 1. GQA-group Q-packing — **highest impact, medium effort**

**Problem in V1.** Grid is `batch * q_heads`. For Llama-3-8B (32 q heads, 8 kv
heads, GQA ratio 4) at batch=1 this yields 32 CTAs over 132 SMs (~25% util),
and each of 4 sibling CTAs in a GQA group redundantly loads the same K/V
tiles from HBM (mitigated only partially by L2 reuse).

**V2 design.** Pack the R = gqaRatio query heads of a GQA group into the
first R rows of a single 64-row Q tile (rest zero from TMA OOB-fill). Grid
becomes `batch * kv_heads`. Each CTA owns the GQA group, loads K/V once for
the group, computes attention for R query heads in parallel.

- Llama-3-8B, batch=1: grid drops to 8 CTAs but **KV bandwidth drops 4×**.
  Compute per CTA grows 4× (still using the same 64-row Q tile, just with
  R=4 valid rows instead of 1). Net win on memory-bound decode: ~3×.
- Output write: scatter the first R rows of the 64×D O tile back to
  positions `[batch, kvHead*R + r, 0, :]` for r in [0, R). Needs either a
  custom store helper or per-row TMA stores.
- Q load: TMA from `[B, Hq, 1, D]` viewed as a 4D layout where the row
  dimension is Hq*qSeq=Hq. A 64-row tile load at coord
  `(batch, 0, kvHead*R/64_tiles, 0)` would pick up R contiguous q_heads if
  R | 64 (true for ratio ∈ {2, 4, 8, 16}). Rows beyond R are TMA OOB-fill
  (= zero), giving the desired "valid Q rows × correct K, V" semantics.
- WGMMA tile shapes unchanged; only the launch grid + Q layout interpretation
  shifts.

**Effort:** ~2 days of kernel work + parity tests + bench. Expected to land
us close to SDPA on Llama-3-8B at small batch (where we currently regress).

## 2. Producer/consumer warpgroup split + 2-stage KV ring buffer — **medium impact, larger effort**

**Problem in V1.** Single warpgroup serially: load K,V → WGMMA QKᵀ → softmax
→ WGMMA PV → load K,V (next iter) → … . The async TMA + WGMMA pipeline can
overlap the load with prior compute, but only one stage deep. With long kv_seq
the kernel spends time waiting on KV transfers.

**V2 design.** 1 producer warpgroup (128 threads) + 1 consumer warpgroup
(128 threads), 256 threads/CTA total. K/V tiles allocated as
`STArray BFloat16 64 hdim 2` (2 stages). Per-stage semaphores in a
`SemaphoreArray 2`.

- Producer: pre-issues TMA loads for stage 0, waits for stage 0 consumed,
  loads stage 1; ping-pongs.
- Consumer: waits stage 0 produced, computes WGMMA on stage 0, signals
  consumed, waits stage 1, …
- Use `warpgroupDecreaseRegisters` (producer, ~24 regs/thread) and
  `warpgroupIncreaseRegisters` (consumer, ~232 regs/thread) per TK convention
  to maximize register file allocation for the consumer's WGMMA accumulator.

**Existing primitives that cover this:** `allocSTArray`,
`allocSemaphoreArray`, `loadGlobalAsyncArraySemArray`,
`warpgroupMmaRhsArray`, `warpgroupMmSharedArrayT64x16`, `asProducer`,
`asConsumer`, `waitSemaphoreArrayPhaseVal`, `arriveSemaphoreArrayWarp`,
`expectBytesArrayWarp`. All exist in `Tyr/GPU/Codegen/Primitives.lean`.

**Risk:** semaphore phase tracking inside a runtime KV loop. The existing
training kernel uses single-buffer + serial pattern, so this is the first
real producer/consumer kernel in Tyr's codegen. Expect to spend time on the
phase-bit ergonomics.

**Effort:** ~3 days. Expected speedup: 1.5–2× for kv_seq ≥ 1k where load
latency dominates.

## 3. Split-KV reduction — **critical for low-batch long-context**

**Problem.** At batch=1, kv_seq=16k, even with GQA packing the grid is just
`1 * kv_heads = 8` CTAs. Each CTA processes 16k/64 = 256 KV blocks
sequentially. H100 has 132 SMs idle.

**V2 design.** Split the kv_seq dimension across multiple CTAs (typical: 8
or 16 splits). Each CTA computes a partial `(max, sum, acc)` tuple over its
kv slice. A second reduction kernel combines partials.

- First kernel: same as V1 but processes `kv_seq / num_splits` instead of
  `kv_seq`, writes partials to `[batch, q_heads, num_splits, hdim+2]`
  scratch buffer (the `+2` is for max and sum).
- Second kernel: per `(batch, q_head)`, combine `num_splits` partials via
  the streaming softmax-rescale recurrence. Tiny per-CTA, fits in registers.

**Effort:** ~3 days (two kernels + scratch allocation + reduction launcher).
Expected speedup: ~10× at batch=1, kv_seq=16k.

## 4. Paged KV cache — **deferred until production serving needs it**

vLLM-style: KV stored as `[num_blocks, H_kv, block_size, D]` (block_size
typically 16 or 32) plus per-sequence int32 block table. Each K/V tile
load goes through a block-table indirection.

- Add a `PagedAttentionProblem` variant or extend `AttentionProblem` with
  a paged-cache field.
- Kernel TMA descriptor changes: the K/V GL is per-block, indexed by the
  block-table value at the current (kvBlockIdx / block_size) position.
- Runtime contract change: callers must allocate the paged buffer and
  block table.

**Effort:** ~4 days kernel + 2 days runtime + serving integration. Defer
until there's a concrete user.

## 5. head_dim = 256 — **small effort, low priority**

Gemma-2 27B uses head_dim=256. Trivial extension: another specialization
`tkMhaH100DecodeFwd256` calling `decodeFwdBodyImpl 256 …` with
`scoreScaleLog2e = 1/sqrt(256) * log2(e) ≈ 0.0901684400557`. SMEM cost
per CTA ~128 KiB (tile area: 4 × 64 × 256 × 2 = 128 KiB) — fits in the
228 KiB H100 dynamic SMEM budget. ~1 day.

## 6. FP8 K/V cache — **defer until a model demands it**

Hopper has FP8 WGMMA (E4M3 / E5M2). Storing K/V in FP8 halves cache memory
bandwidth at the cost of one scale-factor multiply per block. Requires:
- FP8 tile types in `Tyr/GPU/Codegen/TileTypes.lean` (already exist as
  `GpuFloat.FP8E4M3` etc; check tile primitives).
- Per-tensor or per-token scale factors in the eligibility / launcher.
- Reference parity tolerance of ~5e-2 (FP8 quantization error).

**Effort:** ~5 days plus a model that wants it. Defer.

## Recommended order

1. **GQA Q-packing** — highest impact, unblocks Llama-3 perf parity.
2. **Producer/consumer + ring buffer** — biggest pure-kernel speedup, lays
   ground for FA3-style structure across the codebase.
3. **Split-KV reduction** — required for any long-context-low-batch
   demo / serving.
4. Then optional: head_dim=256 (when Gemma integration lands), paged KV
   (when serving demands it), FP8 (when a target model demands it).

## Cross-cutting investments

- **Numerical correctness harness** (`Examples/GPU/RunMhaH100Decode.lean`,
  see this branch): the same harness scales to V2 — same parity assertion
  against torch SDPA, just at more shapes. Add `RunMhaH100DecodeBench.lean`
  for the perf side.
- **Bench tracking**: extend `benchmarks/results/flash_attn_cpp_native_*.jsonl`
  with one new row per (batch, q_heads, kv_heads, kv_seq, head_dim). The
  existing `cc/tools/bench_flash_attn.cpp` already wraps the dispatch ops;
  add a `--shape` arg that takes a quintuple.
- **A second kernel template** (`tkMhaH100DecodePackedQ`, `…RingBuffer`)
  alongside the V1 kernel. Don't replace V1 in place — keep it as a
  fallback / debug aid. The C++ select_route picks the most capable variant
  the kernel registry has built.
