# Tyr Code Review & Technical Debt

This document tracks architectural, performance, and correctness issues identified during deep semantic review.

## 🚀 Performance Bottlenecks

### P01: Recursive KV-Cache Prefill
- **Issue**: `prefillCachesFromEmbeds` in `Qwen35/Model.lean` was performing $N$ separate forward passes for a prompt of length $N$.
- **Status**: Partially addressed via Batch Prefill refactor. Needs verification for other models.
- **Impact**: Drastic first-token latency reduction.

### P02: Sequential SafeTensors Loading
- **Issue**: Sharded model loading is fully sequential. Each tensor load involves re-parsing metadata and re-opening file handles.
- **Status**: Standard handle caching implemented.
- **Recommendation**: Implement `mmap`-backed loading in `cc/src/tyr.cpp` to eliminate copies and allow zero-copy weight access. Parallelize the Lean `Array.mapM` calls during layer loading.

### P03: FFI Boundary Overhead
- **Issue**: High-level Lean code calls Torch for every atomic op (e.g. `add`, `mul`, `relu`). In a 32-layer GPT, this incurs thousands of FFI context switches.
- **Recommendation**: Implement **Kernel Fusion** for hot blocks (e.g., Fused Attention, Fused MLP) where a single C++ call handles the entire block.

### P04: AD IR-Level Overhead
- **Issue**: `Tyr/AutoGrad.lean` uses heavy Lean IR rewriting. Tuple packing/unpacking for multiple gradients creates excessive allocations in the generated C code.
- **Recommendation**: Optimize `unpackTupleValues` and `mkTupleReturn` to avoid intermediate list/array allocations where possible.

## 🏛️ Architectural Refactoring

### A01: Monolithic `Tyr/Torch.lean`
- **Issue**: `Torch.lean` is 1450+ lines, mixing low-level FFI, high-level wrappers, and domain-specific submodules (`rotary`, `nn`, `linalg`).
- **Recommendation**: Split into:
  - `Tyr/FFI.lean`: Raw `@extern` declarations.
  - `Tyr/Ops.lean`: Safe Lean wrappers with shape logic.
  - `Tyr/NN/*.lean`: Move functional primitives to dedicated files.

### A02: TensorStruct Higher-Kinded Data (HKD)
- **Issue**: `TensorStruct` is hardcoded to `T s` leaves. This makes generic `tree_transpose` (e.g., `Model (Array α) -> Array (Model α)`) nearly impossible to express cleanly.
- **Recommendation**: Refactor model structures to be parameterized by a leaf provider: `Linear (f : Shape -> Type)`.

### A03: Redundant Linear/Affine Wrappers
- **Issue**: `lean_torch_linear`, `lean_torch_linear3d`, `lean_torch_affine`, and `lean_torch_affine3d` all map to the same `torch::linear` call in C++.
- **Recommendation**: Consolidate into a single polymorphic `torch.linear` in Lean that handles rank-2 and rank-3 via dependent types.

## ⚖️ Correctness & Type Safety

### C01: Shape Inference Runtime Checks
- **Issue**: Complex shapes (e.g. `matmulShape`) are computed at runtime and then "cast" via `reshape`. This deferment hides shape errors until execution.
- **Recommendation**: Move more shape arithmetic to the type-level using Lean 4's macro system and typeclasses.

### C02: Device Consistency
- **Issue**: Some FFI calls (like `rand`, `full`) take an optional `device`, but most ops assume all inputs are on the same device. Mismatches lead to hard Torch crashes.
- **Recommendation**: Add a `Device` index to the `T` type: `T (s : Shape) (d : Device)`. This would provide compile-time guarantees for device-locality.

## ⚡ Decode kernel — open issues

V1 of the TK-style H100 decode kernel landed (`tkMhaH100DecodeFwd[64]` in
`Tyr/GPU/Kernels/MhaH100Decode.lean`). Tracker for the remaining work.
Longer-form rationale per item in `dev/decode_kernel_v2_plan.md`.

### D01: head_dim=256 (unblock Qwen 3.6 / Gemma-2 27B) — **done**
- Landed in `feat(gpu-kernels): head_dim=256 decode kernel for Qwen
  3.5/3.6 family`. `tkMhaH100DecodeFwd256` calls `decodeFwdBodyImpl`
  with `hdim=256` and `scoreScaleLog2e = 1/sqrt(256) * log2(e)`.
- Verified: `qwen36_35B` (B=1, qHeads=16, kvHeads=2, kvSeq=2048,
  head_dim=256) parity-passes vs SDPA at mae≈4.4e-5, max_abs≈4.9e-4
  — same precision as head_dim=128 cases.
- Eligibility test in `TestGPUKernels.lean` updated: head_dim=256
  selects `tkMhaH100Decode`; head_dim=192 (and other unsupported
  dims) still route to portable.

### D01b: refactor decode kernel to 16-row warp tiles — **pending**
- **Issue**: V1 uses 64-row warpgroup tiles. Per-warp
  `rt<float, 64, 256>` is 512 fp32/thread plus q/p/state, which
  exceeds the 65536/128 = 512 32-bit reg/thread budget under
  `launch_bounds(128, 1)` and spills to local memory. ptxas confirms:
  `wgmma.mma_async instructions are serialized due to insufficient
  register resources`. Correctness holds; perf is capped.
- **Plan**: mirror the TK reference (`mha_h100_lcf.cu`) structure —
  per-warp `rt_fl<16, hdim>` accumulator, warpgroup distributes 64
  rows across 4 warps × 16. Q is loaded once into shared and each
  warp pulls its 16-row strip. Same online softmax math; only the
  tile dimensions and indexing change.
- Sequence after D02 (bench V1 first to know what we're improving on).

### D02: Benchmark V1 vs SDPA
- **Issue**: No measured perf number yet for any decode shape.
- **Plan**: Mirror `Examples/GPU/RunFlashAttnBench.lean` shape onto the
  decode fixtures, emit JSONL to `benchmarks/results/decode_*.jsonl`.
  ~half day.

### D03: GQA-group Q-packing
- **Issue**: At Llama-3 batch=1, grid is `batch * q_heads = 32` CTAs
  (~25% util on 132-SM H100); each GQA group's R sibling CTAs
  redundantly load the same K/V tiles.
- **Plan**: Pack the R = `q_heads/kv_heads` query heads of each group
  into the first R rows of one 64-row Q tile. Grid drops to
  `batch * kv_heads`, KV bandwidth drops R×, output write becomes a
  per-row scatter. ~2 days. Detail in `decode_kernel_v2_plan.md` §1.
- **Expected**: ~3× on Llama-3-8B batch=1.

### D04: Producer/consumer + 2-stage KV ring buffer
- **Issue**: V1 is single-buffer; load and compute are only one stage
  deep, so long-`kv_seq` decode spends measurable time waiting on
  KV transfers.
- **Plan**: 1 producer + 1 consumer warpgroup, `STArray BFloat16 64
  hdim 2`, per-stage `SemaphoreArray 2`. All required primitives are
  already in `Tyr/GPU/Codegen/Primitives.lean`; main effort is
  semaphore phase-bit ergonomics. ~3 days. Detail in §2.
- **Expected**: 1.5–2× for `kv_seq ≥ 1k`.

### D05: Split-KV reduction
- **Issue**: At batch=1, kv_seq=16k, even with GQA packing each CTA
  processes 256 KV blocks sequentially while ~124 SMs sit idle.
- **Plan**: Split kv_seq across 8–16 CTAs writing partial
  `(max, sum, acc)` to scratch; tiny combine kernel applies the
  streaming softmax-rescale recurrence. ~3 days. Detail in §3.
- **Expected**: ~10× at batch=1, kv_seq=16k.

### D06 (deferred): Paged KV cache
- vLLM-style block-table indirection. Detail in §4. Defer until a
  concrete serving user lands.

### D07 (deferred): FP8 K/V cache
- Halves KV bandwidth on Hopper at the cost of one scale-factor
  multiply per block. Detail in §6. Defer until a target model demands
  it (Llama-3 / Qwen 3.6 still in BF16 today).

### D08 (cross-cutting): Decode backward
- Decode forward only. Decode is typically inference-time so backward
  isn't needed yet, but the kernel doc-comment should call it out
  explicitly.
