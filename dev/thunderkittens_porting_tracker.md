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
- `implemented+raw`: dedicated Lean surface exists, but relies on `emitRaw`
  backend blocks for TMEM/cluster/PGL-style constructs that are not yet
  first-class in the Lean DSL

| Vendored source | Lean counterpart | Status | Notes |
| --- | --- | --- | --- |
| `attention/mha_h100/mha_h100.cu` | `Tyr/GPU/Kernels/MhaH100.lean` | `implemented` | Closest attention-side transliteration in the tree. |
| `attention/mha_h100_lcf/mha_h100_lcf.cu` | `Tyr/GPU/Kernels/MhaH100LCF.lean` (`tkMhaH100LCFFwd64`, `tkMhaH100LCFFwd128`) | `implemented+raw` | Dedicated LCF load-compute-finish surfaces with raw backend staging for the CTA packing details. |
| `based/linear_attn.cu` | `Tyr/GPU/Kernels/Based.lean` (`tkBasedLinearAttnFwd`) | `implemented` | Source-backed forward owns the explicit `a0/a1/a2` state and local polynomial attention contract. |
| `fftconv/fftconv_non_pc.cu` | `Tyr/GPU/Kernels/FFTConv.lean` (`tkFFTConvNonPC64`) | `implemented` | Dedicated non-persistent counterpart exists. |
| `fftconv/fftconv_pc.cu` | `Tyr/GPU/Kernels/FFTConv.lean` (`tkFFTConvPC1024`) | `implemented` | Persistent producer/consumer counterpart exists. |
| `flux/flux_gate.cu` | `Tyr/GPU/Kernels/Flux.lean` (`tkFluxMatmulGateFwd`) | `implemented` | Dedicated gate+bias+residual counterpart exists. |
| `flux/flux_gelu.cu` | `Tyr/GPU/Kernels/Flux.lean` (`tkFluxMatmulGeluFwd`) | `implemented` | Dedicated GELU+bias counterpart exists. |
| `gemm/bf16_b200/bf16_b200_gemm.cu` | `Tyr/GPU/Kernels/Bf16Gemm.lean` (`tkB200Bf16GemmFwd`) | `implemented+raw` | Blackwell BF16 producer/consumer + TMEM structure encoded through raw backend blocks. |
| `gemm/bf16_h100/bf16_h100_gemm.cu` | `Tyr/GPU/Kernels/Bf16Gemm.lean` (`tkH100Bf16GemmFwd`) | `implemented` | Dedicated Hopper BF16 counterpart exists. |
| `gemm/fp8_b200/fp8_b200_gemm_1cta.cu` | `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkB200Fp8E4M3Gemm1CtaFwd`) | `implemented+raw` | Dedicated Blackwell 1-CTA FP8 counterpart exists. |
| `gemm/fp8_b200/fp8_b200_gemm_2cta.cu` | `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkB200Fp8E4M3Gemm2CtaFwd`) | `implemented+raw` | Dedicated Blackwell 2-CTA FP8 counterpart exists. |
| `gemm/fp8_h100/fp8_h100_gemm.cu` | `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkH100Fp8E4M3GemmFwd`) | `implemented` | Primary H100 FP8 surface. |
| `gemm/fp8_h100_scaled/fp8_h100_gemm_scaled.cu` | `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkH100Fp8ScaledGemmFwd`) | `implemented` | Explicit scale epilogue represented. |
| `gemm/mxfp8_b200/mxfp8_b200_gemm.cu` | `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkB200MxFp8GemmFwd`) | `implemented+raw` | MXFP8 counterpart uses raw backend blocks for the scale-tile/TMEM structure. |
| `gemm/nvfp4_b200/nvfp4_b200_gemm.cu` | `Tyr/GPU/Kernels/NvFp4Gemm.lean` (`tkB200NvFp4GemmFwd`) | `implemented+raw` | Packed-fp4/local-scale/global-scale contract encoded directly in raw backend code. |
| `hedgehog/hedgehog.cu` | `Tyr/GPU/Kernels/Hedgehog.lean` (`tkHedgehogFwd`) | `implemented` | Canonical chunk/state surface exists. |
| `layernorm/layernorm.cu` | `Tyr/GPU/Kernels/FusedLayerNorm.lean` (`tkFusedLayerNormResidual1024`) | `implemented` | Canonical fused residual + layernorm port. |
| `linear_attention/linear_attention.cu` | `Tyr/GPU/Kernels/LinearAttn.lean` (`tkLinearAttnFwd`) | `implemented` | Dedicated decayed recurrent/local forward surface. |
| `mamba2/mamba2.cu` | `Tyr/GPU/Kernels/Mamba2.lean` (`mamba2Fwd`) | `implemented+raw` | Dedicated lcsf-style producer/consumer surface exists. |
| `parallel/ag_gemm/ag_gemm_b200.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`agGemmB200Fwd`) | `implemented+raw` | Dedicated Blackwell AG+GEMM counterpart exists. |
| `parallel/ag_gemm/ag_gemm_h100.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`agGemmFwd`) | `implemented+raw` | Dedicated H100 AG+GEMM counterpart exists. |
| `parallel/ag_gemm_fp8/ag_gemm_fp8_b200.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`agGemmFp8B200Fwd`) | `implemented+raw` | Dedicated Blackwell FP8 AG+GEMM counterpart exists. |
| `parallel/all_gather/all_gather.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`allGatherFwd`) | `implemented` | The transport path is now encoded through typed layout-dimension and scalar-control DSL primitives. |
| `parallel/all_reduce/all_reduce.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`allReduceFwd`) | `implemented` | The out-of-place collective now rides on the typed tile/multimem surface. |
| `parallel/all_reduce_educational/all_reduce_educational.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`allReduceEducationalFwd`) | `implemented` | Educational in-place all-reduce counterpart exists. |
| `parallel/all_to_all/all_to_all.cu` | `Tyr/GPU/Kernels/Distributed.lean`, `Tyr/GPU/Kernels/UlyssesAttn.lean` | `implemented` | The shared all-to-all transport/indexing surface is now encoded through typed layout-dimension and scalar-control DSL primitives. |
| `parallel/gemm_ar/gemm_ar_h100.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`gemmArFwd`) | `implemented+raw` | Dedicated H100 GEMM+all-reduce counterpart exists. |
| `parallel/gemm_ar/gemm_ar_h100_lcsc.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`gemmArH100LcscFwd`) | `implemented+raw` | Dedicated LCSC GEMM+all-reduce counterpart exists. |
| `parallel/gemm_rs/gemm_rs_b200.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`gemmRsB200Fwd`) | `implemented+raw` | Dedicated Blackwell GEMM+reduce-scatter counterpart exists. |
| `parallel/gemm_rs/gemm_rs_h100.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`gemmRsFwd`) | `implemented+raw` | Dedicated H100 GEMM+reduce-scatter counterpart exists. |
| `parallel/gemm_rs_fp8/gemm_rs_fp8_b200.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`gemmRsFp8B200Fwd`) | `implemented+raw` | Dedicated Blackwell FP8 GEMM+reduce-scatter counterpart exists. |
| `parallel/moe_dispatch_gemm/moe_dispatch_gemm_h100.cu` | `Tyr/GPU/Kernels/MOE.lean` (`tkMoeDispatchGemm`) | `implemented+raw` | Canonical fused dispatch/grouped-GEMM surface exists. |
| `parallel/reduce_scatter/reduce_scatter.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`reduceScatterFwd`) | `implemented` | The sharded transport path is now encoded through the typed tile/multimem surface. |
| `parallel/ring_attn/ring_attn_h100.cu` | `Tyr/GPU/Kernels/RingAttn.lean` (`ringAttnPartial`, `ringAttnComm`, `ringAttnReduce`) | `implemented+raw` | Forward ring attention is split into the same coarse phases as the vendored kernel. |
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

The remaining work is mostly about first-class DSL coverage and exact launch
arithmetic, not missing kernel families:

1. Promote TMEM / cluster-specialized storage / scale-tile concepts so the
   Blackwell GEMM family no longer needs raw backend blocks.
2. Add first-class distributed PGL / peer arithmetic for the remaining
   communication+GEMM and ring-attention kernels. The shared all-to-all,
   all-gather, all-reduce, and reduce-scatter transport paths are now typed,
   but the broader multi-peer topology DSL is still thin.
3. Tighten exact CTA worker packing for `MhaH100LCF.lean`, `Based.lean`,
   `LinearAttn.lean`, `Hedgehog.lean`, and `Mamba2.lean`.

## Notes

- The redundant sketch modules `LayerNorm.lean`, `LayerNormBwd.lean`,
  `LayerNormResidual.lean`, `Mamba.lean`, and `MambaBwd.lean` were removed from
  the core catalog once the canonical source-backed surfaces were in place.
- Do not add new alias-only modules or compatibility shims for kernel families
  that already have a canonical surface.
