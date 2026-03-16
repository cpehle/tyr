# ThunderKittens Porting Tracker

This is the working tracker for the `Tyr.GPU.Kernels` ThunderKittens port.
It tracks the canonical public surface, the vendored-source coverage matrix, and
the remaining fidelity follow-ups after removing the older sketch-only modules.

## Goal

Port the vendored ThunderKittens kernels in
[`thirdparty/ThunderKittens/kernels`](/Users/pehle/dev/tyr/thirdparty/ThunderKittens/kernels)
into the Lean GPU catalog so that:

- every vendored source has a built Lean counterpart,
- the catalog is part of the normal build and doc graph,
- the public surface is grouped by logical kernel family,
- redundant sketch/alias layers are removed from the core catalog,
- Tyr-only derived kernels are clearly separated from vendored-source parity.

## Commit History

| Commit | Scope |
| --- | --- |
| `f8443f37` | Build/doc integration for kernel catalog, removed duplicated `RT/ST/RV/SV/GPtr/KVal` alias layer, cleaned duplicate forward/backward ownership. |
| `0f6dfebd` | Silenced kernel build warnings across the GPU tree. |
| `7a2fc13e` | Canonicalized the ThunderKittens fused residual + layernorm port in `FusedLayerNorm.lean`. |
| `8a01adfc` | Wired `Mamba2.lean` decay/state flow instead of leaving `A_ptr`/`state_ptr` unused. |
| `06dafb55` | Aligned `FFTConv.lean` and `Hedgehog.lean` to source-backed ThunderKittens chunk/state surfaces. |
| `9c2a9d3d` | Refined `Distributed.lean`, `RingAttn*.lean`, and `UlyssesAttn*.lean` around concrete collective/transport phases. |
| `23bcfbf7` | Reworked `Based.lean`, `LinearAttn.lean`, and `MOE.lean` into canonical source-backed forward surfaces. |
| `891627fb` | Tightened H100 FP8 and B200 NVFP4 GEMM surfaces. |
| `323f24b0` | Added the remaining dedicated ThunderKittens source counterparts: `mha_h100_lcf`, BF16 GEMM, Blackwell FP8/MXFP8 GEMM exports, and the distributed educational/B200/LCSC surfaces. |
| `a22de68d` | Extended the GPU scalar DSL and rewrote `LinearAttn*.lean` around the decayed recurrent/local ThunderKittens contract. |
| `c4a6a16a` | Encoded Blackwell GEMM parity surfaces and tightened `Mamba2` / `MhaH100LCF` source staging. |
| `in progress` | Added runtime-bounded typed kernel loops and removed the remaining raw attention-side holdouts in `RingAttn.lean` and `MhaH100LCF.lean`. |

## Catalog Organization

The public family entrypoints are:

- `Tyr.GPU.Kernels.Attention`
- `Tyr.GPU.Kernels.StateSpace`
- `Tyr.GPU.Kernels.Parallel`
- `Tyr.GPU.Kernels.Gemm`
- `Tyr.GPU.Kernels.Normalization`
- `Tyr.GPU.Kernels.Experimental`

The root `Tyr.GPU.Kernels` module remains the full-catalog umbrella built out of
those family modules instead of a flat import list of leaf files.

## Vendored Source Coverage

Every vendored ThunderKittens `.cu` source now has an implemented Lean
counterpart that is part of the normal build/doc graph.

Status meanings:

- `implemented`: dedicated Lean surface exists and is built/documented

| Vendored source | Lean counterpart | Status | Notes |
| --- | --- | --- | --- |
| `attention/mha_h100/mha_h100.cu` | `Tyr/GPU/Kernels/MhaH100.lean` | `implemented` | Closest attention-side transliteration in the tree. |
| `attention/mha_h100_lcf/mha_h100_lcf.cu` | `Tyr/GPU/Kernels/MhaH100LCF.lean` (`tkMhaH100LCFFwd64`, `tkMhaH100LCFFwd128`) | `implemented` | Typed stationary-Q / streamed-KV shell; the multi-worker CTA packing is intentionally flattened to one logical query tile per kernel instance. |
| `based/linear_attn.cu` | `Tyr/GPU/Kernels/Based.lean` (`tkBasedLinearAttnFwd`) | `implemented` | Source-backed forward owns the explicit `a0/a1/a2` state and local polynomial attention contract. |
| `fftconv/fftconv_non_pc.cu` | `Tyr/GPU/Kernels/FFTConv.lean` (`tkFFTConvNonPC64`) | `implemented` | Dedicated non-persistent counterpart exists. |
| `fftconv/fftconv_pc.cu` | `Tyr/GPU/Kernels/FFTConv.lean` (`tkFFTConvPC1024`) | `implemented` | Persistent producer/consumer counterpart exists. |
| `flux/flux_gate.cu` | `Tyr/GPU/Kernels/Flux.lean` (`tkFluxMatmulGateFwd`) | `implemented` | Dedicated gate+bias+residual counterpart exists. |
| `flux/flux_gelu.cu` | `Tyr/GPU/Kernels/Flux.lean` (`tkFluxMatmulGeluFwd`) | `implemented` | Dedicated GELU+bias counterpart exists. |
| `gemm/bf16_b200/bf16_b200_gemm.cu` | `Tyr/GPU/Kernels/Bf16Gemm.lean` (`tkB200Bf16GemmFwd`) | `implemented` | Blackwell BF16 surface now uses a typed 256x256 tiled mainloop and typed epilogue. |
| `gemm/bf16_h100/bf16_h100_gemm.cu` | `Tyr/GPU/Kernels/Bf16Gemm.lean` (`tkH100Bf16GemmFwd`) | `implemented` | Dedicated Hopper BF16 counterpart exists. |
| `gemm/fp8_b200/fp8_b200_gemm_1cta.cu` | `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkB200Fp8E4M3Gemm1CtaFwd`) | `implemented` | Dedicated Blackwell 1-CTA FP8 counterpart now uses a typed 128x256 Blackwell-sized mainloop. |
| `gemm/fp8_b200/fp8_b200_gemm_2cta.cu` | `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkB200Fp8E4M3Gemm2CtaFwd`) | `implemented` | Dedicated Blackwell 2-CTA FP8 counterpart now uses the same typed tile family with cluster-sized output geometry. |
| `gemm/fp8_h100/fp8_h100_gemm.cu` | `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkH100Fp8E4M3GemmFwd`) | `implemented` | Primary H100 FP8 surface. |
| `gemm/fp8_h100_scaled/fp8_h100_gemm_scaled.cu` | `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkH100Fp8ScaledGemmFwd`) | `implemented` | Explicit scale epilogue represented. |
| `gemm/mxfp8_b200/mxfp8_b200_gemm.cu` | `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkB200MxFp8GemmFwd`) | `implemented` | MXFP8 counterpart now uses typed e8m0 scale vectors converted into an FP32 epilogue. |
| `gemm/nvfp4_b200/nvfp4_b200_gemm.cu` | `Tyr/GPU/Kernels/NvFp4Gemm.lean` (`tkB200NvFp4GemmFwd`) | `implemented` | Packed-fp4 local/global scale contract now rides on typed accumulator, converted scale vectors, and typed scalar global IO. |
| `hedgehog/hedgehog.cu` | `Tyr/GPU/Kernels/Hedgehog.lean` (`tkHedgehogFwd`) | `implemented` | Canonical chunk/state surface exists. |
| `layernorm/layernorm.cu` | `Tyr/GPU/Kernels/FusedLayerNorm.lean` (`tkFusedLayerNormResidual1024`) | `implemented` | Canonical fused residual + layernorm port. |
| `linear_attention/linear_attention.cu` | `Tyr/GPU/Kernels/LinearAttn.lean` (`tkLinearAttnFwd`) | `implemented` | Dedicated decayed recurrent/local forward surface. |
| `mamba2/mamba2.cu` | `Tyr/GPU/Kernels/Mamba2.lean` (`mamba2Fwd`) | `implemented` | Typed chunk/state shell with runtime-bounded chunk loops, decay-prefix construction, local masked-decayed `QK^T`, and recurrent `K^T V` updates. |
| `parallel/ag_gemm/ag_gemm_b200.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`agGemmB200Fwd`) | `implemented` | Dedicated Blackwell AG+GEMM counterpart now uses the typed producer/consumer distributed surface. |
| `parallel/ag_gemm/ag_gemm_h100.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`agGemmFwd`) | `implemented` | Dedicated H100 AG+GEMM counterpart now uses the typed producer/consumer distributed surface. |
| `parallel/ag_gemm_fp8/ag_gemm_fp8_b200.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`agGemmFp8B200Fwd`) | `implemented` | Dedicated Blackwell FP8 AG+GEMM counterpart now uses the typed producer/consumer distributed surface. |
| `parallel/all_gather/all_gather.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`allGatherFwd`) | `implemented` | The transport path is now encoded through typed layout-dimension and scalar-control DSL primitives. |
| `parallel/all_reduce/all_reduce.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`allReduceFwd`) | `implemented` | The out-of-place collective now rides on the typed tile/multimem surface. |
| `parallel/all_reduce_educational/all_reduce_educational.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`allReduceEducationalFwd`) | `implemented` | Educational in-place all-reduce counterpart exists. |
| `parallel/all_to_all/all_to_all.cu` | `Tyr/GPU/Kernels/Distributed.lean`, `Tyr/GPU/Kernels/UlyssesAttn.lean` | `implemented` | The shared all-to-all transport/indexing surface is now encoded through typed layout-dimension and scalar-control DSL primitives. |
| `parallel/gemm_ar/gemm_ar_h100.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`gemmArFwd`) | `implemented` | Dedicated H100 GEMM+all-reduce counterpart now uses the typed distributed epilogue surface. |
| `parallel/gemm_ar/gemm_ar_h100_lcsc.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`gemmArH100LcscFwd`) | `implemented` | Dedicated LCSC GEMM+all-reduce counterpart now uses the typed distributed epilogue surface. |
| `parallel/gemm_rs/gemm_rs_b200.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`gemmRsB200Fwd`) | `implemented` | Dedicated Blackwell GEMM+reduce-scatter counterpart now uses the typed distributed `store_add` surface. |
| `parallel/gemm_rs/gemm_rs_h100.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`gemmRsFwd`) | `implemented` | Dedicated H100 GEMM+reduce-scatter counterpart now uses the typed distributed `store_add` surface. |
| `parallel/gemm_rs_fp8/gemm_rs_fp8_b200.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`gemmRsFp8B200Fwd`) | `implemented` | Dedicated Blackwell FP8 GEMM+reduce-scatter counterpart now uses the typed distributed `store_add` surface. |
| `parallel/moe_dispatch_gemm/moe_dispatch_gemm_h100.cu` | `Tyr/GPU/Kernels/MOE.lean` (`tkMoeDispatchGemm`) | `implemented` | Canonical fused dispatch/grouped-GEMM surface exists. |
| `parallel/reduce_scatter/reduce_scatter.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`reduceScatterFwd`) | `implemented` | The sharded transport path is now encoded through the typed tile/multimem surface. |
| `parallel/ring_attn/ring_attn_h100.cu` | `Tyr/GPU/Kernels/RingAttn.lean` (`ringAttnPartial`, `ringAttnComm`, `ringAttnReduce`) | `implemented` | All three forward phases now use typed DSL code; the partial phase runs as a typed single-query-tile shell over a runtime-bounded KV-shard loop. |
| `parallel/ulysses_attn/ulysses_attn.cu` | `Tyr/GPU/Kernels/UlyssesAttn.lean` (`allToAllFwd`, `ulyssesQkvAllToAll`, `ulyssesAttnFwd`) | `implemented` | Ulysses transport/orchestration now rides on the typed shared all-to-all surface instead of a raw backend block. |
| `rotary/rotary.cu` | `Tyr/GPU/Kernels/Rotary.lean` | `implemented` | Reasonably faithful tile split / rotate / concat structure. |

## Tyr-Only Derived Kernels

These are part of Tyr's GPU catalog, but they are not required for vendored
ThunderKittens source coverage:

- `Tyr/GPU/Kernels/LinearAttnBwd.lean`
- `Tyr/GPU/Kernels/RingAttnBwd.lean`
- `Tyr/GPU/Kernels/UlyssesAttnBwd.lean`

They remain useful, but they should be treated as Tyr-native extensions around
the source-backed forward kernels rather than parity blockers.

## Fidelity Follow-Ups

The remaining work is about first-class DSL fidelity and exact launch
arithmetic, not missing kernel families. The items below are ordered by
dependency: later items often require earlier ones.

### 1. Tensor Memory (TMEM) as a First-Class IR Tile Kind

**Status**: `hasTMEM` flag exists in `ArchConfig`; no IR support.

The B200 ThunderKittens kernels store MMA output accumulators and scale tiles in
TMEM (144 KB per CTA). The current IR only has `declRT`/`declST`/`declRV`/`declSV`.

**What to add to `KStmt`**:

| Statement | Semantics |
| --- | --- |
| `declTT v dtype rows cols` | Tensor-memory tile declaration (analogous to `declRT`) |
| `tmemAllocate v pool offset` | Allocate a TT from a provisioned TMEM pool at `offset` |
| `tmemProvision pool clusterSize` | Provision a TMEM address region across the cluster |
| `tmemDeprovision pool` | Release the provisioned region |

**Emit mapping** (`EmitNew.lean`):
- `declTT` → `kittens::tt<dtype, rows, cols> v;`
- `tmemAllocate` → `v = pool.allocate<tt_t>(offset);`
- `tmemProvision` → `pool.provision(tmem_addr);` (inside `if(elect_one)`)
- `tmemDeprovision` → `pool.deprovision();`

**Affected kernel surfaces**: `Bf16Gemm.lean` (B200), `PrecisionGemm.lean`
(FP8/MXFP8 B200), `NvFp4Gemm.lean` (NVFP4 B200). These currently compress TMEM
fragments into register-tile accumulators.

### 2. Cluster Coordinate Accessors and Cluster-Aware TMA

**Status**: `hasClusterBarrier` flag exists; `clusterSize` in `ArchKernelConfig`.
No IR statements for cluster coordination.

The B200 2-CTA GEMM kernels query cluster rank and use cluster-scoped TMA to
share B-tiles across CTAs via distributed shared memory (DSMEM).

**What to add to `KStmt`**:

| Statement | Semantics |
| --- | --- |
| `clusterIdx dst axis` | Read cluster coordinate (`cluster_ctarank()`) |
| `clusterTmaLoad dst src coords sem` | TMA load visible to the full cluster |
| `clusterTmaStore dst src coords` | TMA store from cluster scope |
| `clusterArrive sem` | Cluster-scope barrier arrive |
| `clusterWait sem` | Cluster-scope barrier wait |

**Emit mapping**: These map directly to `tma::cluster::load_async()`,
`tma::cluster::store_async()`, and cluster barrier primitives.

**Affected kernel surfaces**: `Bf16Gemm.lean` (B200 2-CTA), `PrecisionGemm.lean`
(FP8 B200 2-CTA).

### 3. tcgen05 MMA with TMEM Destination and Scale Operands

**Status**: `tcgen05Mma` exists in IR but has no DSL wrapper and cannot target
TMEM destinations or accept scale-tile operands.

The B200 MMA uses `mm2_ABt` / `mma2_ABt` which write directly to TMEM and
optionally accept E8M0 scale tiles for block-scaled formats.

**What to add to `KStmt`**:

| Statement | Semantics |
| --- | --- |
| `tcgen05Mm trans dst a b` | First MMA (zero-init) to TMEM |
| `tcgen05Mma trans dst a b c` | Accumulating MMA to TMEM (already exists, needs TMEM dst) |
| `tcgen05MmaScaled trans dst a b c scaleA scaleB` | Scaled MMA with E8M0 scale tiles |
| `tcgen05Commit sem` | Commit tcgen05 results with cluster-aware semaphore |

**Emit mapping**:
- `tcgen05Mm` → `mm2_ABt(dst, a, b);` (or `mm2_AB` etc. per transpose mode)
- `tcgen05MmaScaled` → `mma2_ABt(dst, a, b, c, scaleA, scaleB, sem);`
- `tcgen05Commit` → `detail::tcgen05::commit<CLUSTER_SIZE>(sem);`

**Affected kernel surfaces**: All B200 GEMM variants.

### 4. Scale-Tile TMEM Loading

**Status**: Not represented. Scale vectors are currently loaded as shared-memory
vectors and applied in an explicit FP32 epilogue.

The MXFP8 and NVFP4 kernels pipeline E8M0 scale tiles through TMEM alongside
the main A/B tiles. Scale subtiles are indexed per pipeline stage.

**What to add to `KStmt`**:

| Statement | Semantics |
| --- | --- |
| `loadScaleTmem dst src stage` | Load scale tile (E8M0) into TMEM subtile for pipeline stage |
| `tmemSubtile dst src offset` | Extract a subtile from a pipelined TMEM allocation |

This is lower priority than items 1–3; the current explicit epilogue is
functionally correct.

### 5. Pipelined Barrier Sequencing (Phasebits)

**Status**: Semaphore ops exist (`Init`, `Wait`, `Arrive`, etc.) but no
ring-buffer phasebit tracking.

The ThunderKittens mainloops track pipeline stages via a `phasebits` word:
`get_phasebit<N>(phasebits, ring_idx)` / `update_phasebit<N>(phasebits, ring_idx)`.
This enables overlapping loads of stage N+2 with compute on stage N.

This can likely be handled as a DSL-level abstraction over the existing semaphore
ops rather than new IR statements. A `PipelineRing` combinator in the DSL that
emits the right `semaphore` / `tmaExpect` / `arrive` sequence per stage would
suffice.

### 6. CTA Worker Packing — Per-Kernel Status

The five kernels below flatten the source's multi-worker CTA packing to one
logical tile per kernel instance. Tightening means adding warpgroup-level role
assignment and (where applicable) multi-tile-per-CTA launch arithmetic.

**Prerequisites**: `ifWarpGroup` already exists in `KStmt`. What's missing is
the launch-side grid calculation that divides work across `NUM_WORKERS` query
tiles per CTA.

| Kernel | Source Workers | Current Lean Model | What to Tighten |
| --- | --- | --- | --- |
| `MhaH100LCF.lean` | 1 producer + 3 consumer warpgroups (12 consumer warps), each consumer holds one 64×D query tile | Single query tile per kernel | Add `ifWarpGroup 0` producer path for async K/V loads; replicate consumer path across warpgroups 1–3; adjust grid to `⌈seqLen / (3 × 64)⌉` |
| `Based.lean` | 1 warpgroup (4 warps), warp 0 prefetches TMA; lane-level shuffles for `mul_slice_row`/`mul_slice_col` | All threads compute uniformly; quadratic features materialized via explicit slice + broadcast | Add warp-0 TMA prefetch path; replace slice+broadcast with warp-shuffle DSL op when available |
| `LinearAttn.lean` | 2 warpgroups: WG0 builds `q_decay`, WG1 builds `k_decay`; 3-ring K/V buffering | Single-threaded decay computation; double-buffer staging | Split decay construction across `ifWarpGroup 0` / `ifWarpGroup 1`; upgrade to 3-ring staging when pipeline abstraction lands |
| `Hedgehog.lean` | 2 warpgroups: WG0 = sliding-window attention, WG1 = linear attention; 3-ring K/V; per-head alpha/beta mixing | Both paths computed sequentially in one thread group; 2-buffer staging | Partition sliding/linear paths across warpgroups; upgrade to 3-ring; keep combined normalization after barrier |
| `Mamba2.lean` | 8 consumer warps (2 warpgroups), producer warp rotates per iteration; per-warpgroup Hillis-Steele cumsum; 2-stage input/output pipeline | Single sequential chunk loop; decay built in-kernel | Add rotating producer warp; split cumsum across warpgroups; use `PipelineRing` (item 5) for I/O overlap |

### Dependency Graph

```
(1) TMEM Tiles ──┬──→ (3) tcgen05 MMA w/ TMEM dst
                 │
                 └──→ (4) Scale-Tile TMEM Loading
                           │
(2) Cluster TMA ─────→ (3) tcgen05 MMA w/ TMEM dst
                           │
(5) Pipeline Ring ────→ (6) CTA Worker Packing
                           ↑
                    ifWarpGroup (already in IR)
```

Items 1–3 unblock faithful Blackwell GEMM surfaces.
Item 5 unblocks most of the CTA worker packing in item 6.
Item 6 can be partially started now using `ifWarpGroup` for the kernels that
only need role assignment (MhaH100LCF, Hedgehog, LinearAttn).

## Notes

- The redundant sketch modules `LayerNorm.lean`, `LayerNormBwd.lean`,
  `LayerNormResidual.lean`, `Mamba.lean`, and `MambaBwd.lean` were removed from
  the core catalog once the canonical source-backed surfaces were in place.
- Do not add new alias-only modules or compatibility shims for kernel families
  that already have a canonical surface.
