# ThunderKittens Porting Tracker

This is the working tracker for the `Tyr.GPU.Kernels` ThunderKittens port.
Unlike the docs-facing status note, this file tracks tranche order, commit
history, and which modules are still partial ports versus source-backed
canonical surfaces.

## Goal

Port the vendored ThunderKittens kernels in
[`thirdparty/ThunderKittens/kernels`](/Users/pehle/dev/tyr/thirdparty/ThunderKittens/kernels)
into the Lean GPU DSL so that:

- every imported kernel module is part of the normal build,
- docs generation includes the kernel catalog,
- the catalog is grouped into logical family entrypoints,
- canonical surfaces are separated from sketch/demo kernels,
- duplicated APIs and alias layers are removed,
- the remaining gaps are explicitly recorded by module.

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
| `891627fb` | Tightened H100 FP8 and B200 NVFP4 compatibility GEMM surfaces. |
| `323f24b0` | Added the remaining dedicated ThunderKittens source counterparts: `mha_h100_lcf`, BF16 GEMM, Blackwell FP8/MXFP8 GEMM exports, and the distributed educational/B200/LCSC surfaces. |
| `a22de68d` | Extended the GPU scalar DSL and rewrote `LinearAttn*.lean` around the decayed recurrent/local ThunderKittens contract. |

## Current Module Status

## Catalog Organization

The public import surface is now intended to be:

- `Tyr.GPU.Kernels.Attention`
- `Tyr.GPU.Kernels.StateSpace`
- `Tyr.GPU.Kernels.Parallel`
- `Tyr.GPU.Kernels.Gemm`
- `Tyr.GPU.Kernels.Normalization`
- `Tyr.GPU.Kernels.Experimental`

The root `Tyr.GPU.Kernels` module should remain the full-catalog umbrella built
out of those family modules rather than a flat import list of every leaf file.

## Exhaustive Source Parity Matrix

This table is keyed by the vendored `.cu` sources under
[`thirdparty/ThunderKittens/kernels`](/Users/pehle/dev/tyr/thirdparty/ThunderKittens/kernels).
Status meanings:

- `canonical`: source-backed primary Lean surface exists
- `partial`: Lean surface exists but still compresses or abstracts key source behavior
- `compatibility`: Lean surface exists but is explicitly limited by current DSL/runtime types

Every vendored ThunderKittens `.cu` source now has a dedicated Lean
counterpart. The remaining work is fidelity, not coverage.

| Vendored source | Lean counterpart | Status | Notes |
| --- | --- | --- | --- |
| `attention/mha_h100/mha_h100.cu` | `Tyr/GPU/Kernels/MhaH100.lean` | `canonical` | Closest attention-side port today. |
| `attention/mha_h100_lcf/mha_h100_lcf.cu` | `Tyr/GPU/Kernels/MhaH100LCF.lean` (`tkMhaH100LCFFwd64`, `tkMhaH100LCFFwd128`) | `partial` | Dedicated LCF forward surfaces now follow the vendored load-compute-finish structure directly, but CTA multi-worker packing is still compressed. |
| `based/linear_attn.cu` | `Tyr/GPU/Kernels/Based.lean` | `partial` | Canonical forward exists; exact shuffle helpers still abstracted. |
| `fftconv/fftconv_non_pc.cu` | `Tyr/GPU/Kernels/FFTConv.lean` (`tkFFTConvNonPC64`) | `partial` | Dedicated non-persistent counterpart exists; complex global factor layouts remain abstract shared inputs. |
| `fftconv/fftconv_pc.cu` | `Tyr/GPU/Kernels/FFTConv.lean` (`tkFFTConvPC1024`) | `partial` | Persistent producer/consumer path exists; complex globals still abstract. |
| `flux/flux_gate.cu` | `Tyr/GPU/Kernels/Flux.lean` (`tkFluxMatmulGateFwd`) | `canonical` | Dedicated gate+bias+residual source counterpart now exists. |
| `flux/flux_gelu.cu` | `Tyr/GPU/Kernels/Flux.lean` (`tkFluxMatmulGeluFwd`) | `canonical` | Dedicated GELU+bias source counterpart now exists. |
| `gemm/bf16_b200/bf16_b200_gemm.cu` | `Tyr/GPU/Kernels/Bf16Gemm.lean` (`tkB200Bf16GemmFwd`) | `partial` | Dedicated Blackwell BF16 counterpart now follows the vendored producer/consumer + TMEM structure via raw backend blocks; TMEM/cluster semantics are not yet modeled as first-class Lean ops. |
| `gemm/bf16_h100/bf16_h100_gemm.cu` | `Tyr/GPU/Kernels/Bf16Gemm.lean` (`tkH100Bf16GemmFwd`) | `canonical` | Dedicated Hopper BF16 source counterpart now exists. |
| `gemm/fp8_b200/fp8_b200_gemm_1cta.cu` | `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkB200Fp8E4M3Gemm1CtaFwd`) | `partial` | Dedicated 1-CTA Blackwell FP8 counterpart now follows the vendored TMEM-backed producer/consumer structure via raw backend blocks. |
| `gemm/fp8_b200/fp8_b200_gemm_2cta.cu` | `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkB200Fp8E4M3Gemm2CtaFwd`) | `partial` | Dedicated 2-CTA Blackwell FP8 counterpart now follows the vendored split-cluster structure via raw backend blocks. |
| `gemm/fp8_h100/fp8_h100_gemm.cu` | `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkH100Fp8E4M3GemmFwd`) | `canonical` | Primary H100 FP8 surface. |
| `gemm/fp8_h100_scaled/fp8_h100_gemm_scaled.cu` | `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkH100Fp8ScaledGemmFwd`) | `canonical` | Explicit scale epilogue represented. |
| `gemm/mxfp8_b200/mxfp8_b200_gemm.cu` | `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkB200MxFp8GemmFwd`) | `partial` | Dedicated MXFP8 counterpart now uses explicit `FP8E8M0` scale-tile pointers and a raw source-shaped cluster/TMEM structure. |
| `gemm/nvfp4_b200/nvfp4_b200_gemm.cu` | `Tyr/GPU/Kernels/NvFp4Gemm.lean` (`tkB200NvFp4GemmFwd`) | `partial` | The B200 NVFP4 surface now follows the packed-fp4/local-scale/global-scale source contract directly via raw backend blocks. |
| `hedgehog/hedgehog.cu` | `Tyr/GPU/Kernels/Hedgehog.lean` (`tkHedgehogFwd`) | `partial` | Canonical chunk/state surface; full 3-ring schedule compressed. |
| `layernorm/layernorm.cu` | `Tyr/GPU/Kernels/FusedLayerNorm.lean` (`tkFusedLayerNormResidual1024`) | `canonical` | Canonical fused residual + layernorm port. |
| `linear_attention/linear_attention.cu` | `Tyr/GPU/Kernels/LinearAttn.lean`, `Tyr/GPU/Kernels/LinearAttnBwd.lean` | `partial` | Forward now builds decay vectors from the runtime slope and records KV checkpoints; backward matches that exact decomposition, but ThunderKittens itself only ships a forward kernel. |
| `mamba2/mamba2.cu` | `Tyr/GPU/Kernels/Mamba2.lean` | `partial` | The surface now follows the vendored lcsf-style producer/consumer structure directly, but exact double-buffering and warpgroup packing are still compressed. |
| `parallel/ag_gemm/ag_gemm_b200.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`agGemmB200CompatFwd`) | `compatibility` | Dedicated Blackwell AG+GEMM counterpart exists; exact cluster-specialized scheduling remains compressed. |
| `parallel/ag_gemm/ag_gemm_h100.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`agGemmFwd`) | `partial` | Source family represented, but peer arithmetic is simplified. |
| `parallel/ag_gemm_fp8/ag_gemm_fp8_b200.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`agGemmFp8B200CompatFwd`) | `compatibility` | Dedicated Blackwell FP8 AG+GEMM counterpart exists; peer arithmetic and cluster exchange remain compatibility-level. |
| `parallel/all_gather/all_gather.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`allGatherFwd`) | `partial` | Collective phase exists; not yet full PGL-indexed implementation. |
| `parallel/all_reduce/all_reduce.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`allReduceFwd`) | `partial` | Collective phase exists; peer indexing and topology are compressed. |
| `parallel/all_reduce_educational/all_reduce_educational.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`allReduceEducationalCompatFwd`) | `compatibility` | Dedicated educational counterpart exists, flattened to the tile-based Lean DSL. |
| `parallel/all_to_all/all_to_all.cu` | `Tyr/GPU/Kernels/Distributed.lean` / `Tyr/GPU/Kernels/UlyssesAttn*.lean` | `partial` | Generic transport exists; exact axis/peer arithmetic is still abstracted. |
| `parallel/gemm_ar/gemm_ar_h100.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`gemmArFwd`) | `partial` | H100 family present; multimem flow modeled, but not exact source launch structure. |
| `parallel/gemm_ar/gemm_ar_h100_lcsc.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`gemmArH100LcscCompatFwd`) | `compatibility` | Dedicated LCSC counterpart exists; exact loader/consumer/communicator role split is compressed. |
| `parallel/gemm_rs/gemm_rs_b200.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`gemmRsB200CompatFwd`) | `compatibility` | Dedicated Blackwell reduce-scatter GEMM counterpart exists; exact PGL shard type remains abstracted. |
| `parallel/gemm_rs/gemm_rs_h100.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`gemmRsFwd`) | `partial` | H100 family present; full device arithmetic still simplified. |
| `parallel/gemm_rs_fp8/gemm_rs_fp8_b200.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`gemmRsFp8B200CompatFwd`) | `compatibility` | Dedicated Blackwell FP8 reduce-scatter GEMM counterpart exists; exact PGL and cluster semantics remain compatibility-level. |
| `parallel/moe_dispatch_gemm/moe_dispatch_gemm_h100.cu` | `Tyr/GPU/Kernels/MOE.lean` (`tkMoeDispatchGemm`) | `partial` | Canonical fused surface exists; sparse metadata + exact warpgroup protocol missing. |
| `parallel/reduce_scatter/reduce_scatter.cu` | `Tyr/GPU/Kernels/Distributed.lean` (`reduceScatterFwd`) | `partial` | Collective phase exists; cluster/PGL arithmetic still abstracted. |
| `parallel/ring_attn/ring_attn_h100.cu` | `Tyr/GPU/Kernels/RingAttn*.lean` | `partial` | Forward phase split exists; backward still speculative. |
| `parallel/ulysses_attn/ulysses_attn.cu` | `Tyr/GPU/Kernels/UlyssesAttn*.lean` | `partial` | Transport/orchestration split exists; local attention boundary remains external. |
| `rotary/rotary.cu` | `Tyr/GPU/Kernels/Rotary.lean` | `canonical` | Reasonably faithful tile split / rotate / concat structure. |

### Canonical Or Near-Canonical

| Module | Vendored source | Status |
| --- | --- | --- |
| `Tyr/GPU/Kernels/Copy.lean` | `include/ops/group/memory/*` | Small direct example; acceptable as-is. |
| `Tyr/GPU/Kernels/Rotary.lean` | `kernels/rotary/rotary.cu` | Reasonably faithful tile split / rotate / concat structure. |
| `Tyr/GPU/Kernels/FusedLayerNorm.lean` (`tkFusedLayerNormResidual1024`) | `kernels/layernorm/layernorm.cu` | Canonical fused residual + layernorm surface. |
| `Tyr/GPU/Kernels/FFTConv.lean` (`tkFFTConvPC1024`) | `kernels/fftconv/fftconv_pc.cu` | Canonical persistent-cache surface now models per-head filter reload and producer/consumer tile flow; complex factor globals are still abstract shared inputs. |
| `Tyr/GPU/Kernels/MhaH100LCF.lean` | `kernels/attention/mha_h100_lcf/mha_h100_lcf.cu` | Dedicated LCF forward surfaces now follow the vendored load-compute-finish staging; CTA worker packing is still compressed. |
| `Tyr/GPU/Kernels/Hedgehog.lean` (`tkHedgehogFwd`) | `kernels/hedgehog/hedgehog.cu` | Canonical chunk/state surface now models long-resident feature maps, previous/current sliding blocks, and final `k_state` / `kv_state` writeout; the full 3-ring/TMA schedule is still compressed. |
| `Tyr/GPU/Kernels/MhaH100.lean` | `kernels/attention/mha_h100/mha_h100.cu` | Closest attention-side port in the tree. |
| `Tyr/GPU/Kernels/Bf16Gemm.lean` | `kernels/gemm/bf16_h100/bf16_h100_gemm.cu`, `kernels/gemm/bf16_b200/bf16_b200_gemm.cu` | H100 BF16 source counterpart is dedicated; the B200 side now follows the vendored producer/consumer + TMEM structure through raw backend code while first-class TMEM ops are still missing. |
| `Tyr/GPU/Kernels/Flux.lean` (`tkFluxMatmulGeluFwd`, `tkFluxMatmulGateFwd`) | `kernels/flux/flux_*.cu` | Dedicated source-facing flux surfaces now exist for both vendored kernels. |
| `Tyr/GPU/Kernels/Based.lean` (`tkBasedLinearAttnFwd`) | `kernels/based/linear_attn.cu` | Canonical forward now owns explicit `a0/a1/a2` state and local polynomial attention. |
| `Tyr/GPU/Kernels/LinearAttn.lean` (`tkLinearAttnFwd`) | `kernels/linear_attention/linear_attention.cu` | Canonical decayed forward now owns the local-vs-recurrent split instead of generic feature-map attention. |

### Partially Ported, Still Needs Source-Backed Cleanup

| Module | Vendored source | Main remaining gap |
| --- | --- | --- |
| `Tyr/GPU/Kernels/Based.lean` | `kernels/based/linear_attn.cu` | Still uses portable slice/broadcast helpers instead of the exact warp-shuffle `mul_slice_row` / `mul_slice_col` implementation. |
| `Tyr/GPU/Kernels/LinearAttn.lean` | `kernels/linear_attention/linear_attention.cu` | The remaining gap is exact multi-worker CTA packing; the decays now come from the runtime slope in-kernel. |
| `Tyr/GPU/Kernels/LinearAttnBwd.lean` | `kernels/linear_attention/linear_attention.cu` | Backward now matches Tyr’s decayed forward decomposition, but there is no vendored ThunderKittens backward source to transliterate directly. |
| `Tyr/GPU/Kernels/Mamba2.lean` | `kernels/mamba2/mamba2.cu` | The surface now follows the vendored lcsf staging directly, but exact double-buffering and warpgroup packing are still incomplete. |
| `Tyr/GPU/Kernels/MhaH100LCF.lean` | `kernels/attention/mha_h100_lcf/mha_h100_lcf.cu` | Multi-worker LCF CTA packing is still compressed to one logical query worker per kernel instance. |
| `Tyr/GPU/Kernels/RingAttn.lean` | `kernels/parallel/ring_attn/ring_attn_h100.cu` | Forward is now split into partial/comm/reduction kernels, but the exact peer scheduling and ping-pong launch structure are still simplified. |
| `Tyr/GPU/Kernels/RingAttnBwd.lean` | `kernels/parallel/ring_attn/ring_attn_h100.cu` | Explicitly speculative phased scaffold; needs correct global causal indexing and exact ring accumulation semantics. |
| `Tyr/GPU/Kernels/UlyssesAttn.lean` | `kernels/parallel/ulysses_attn/ulysses_attn.cu` | Now transport/orchestration over all-to-all, but the local attention launch boundary is still external to this module. |
| `Tyr/GPU/Kernels/UlyssesAttnBwd.lean` | `kernels/parallel/ulysses_attn/ulysses_attn.cu` | Backward mirrors the transport split, but the local FlashAttention backward launch is still only modeled as a speculative shell. |
| `Tyr/GPU/Kernels/Distributed.lean` | `kernels/parallel/*` | Collectives plus AG/GEMM-AR/GEMM-RS now have multimem/barrier/producer-consumer structure; still missing full PGL/cluster arithmetic and peer indexing. |
| `Tyr/GPU/Kernels/Bf16Gemm.lean` | `kernels/gemm/bf16_*.cu` | H100 BF16 is source-backed; B200 BF16 now follows the source structure via raw backend code, but first-class TMEM/cluster ops are still missing from the DSL. |
| `Tyr/GPU/Kernels/PrecisionGemm.lean` | `kernels/gemm/fp8_h100/*`, `kernels/gemm/fp8_b200/*`, `kernels/gemm/mxfp8_b200/*` | Dedicated H100 and B200/MXFP8 surfaces now exist; the remaining gap is first-class Blackwell TMEM / scale-tile modeling rather than missing source families. |
| `Tyr/GPU/Kernels/Flux.lean` | `kernels/flux/flux_*.cu` | Canonical source-facing flux surfaces now exist, but fixed-size matmul tiling still compresses some prototype layout details. |
| `Tyr/GPU/Kernels/NvFp4Gemm.lean` | `kernels/gemm/nvfp4_b200/nvfp4_b200_gemm.cu` | The control-flow and storage contract are now source-shaped, but the TMEM/NVFP4 story is still carried by raw backend code rather than first-class DSL constructs. |
| `Tyr/GPU/Kernels/MOE.lean` | `kernels/parallel/moe_dispatch_gemm/moe_dispatch_gemm_h100.cu` | Canonical fused dispatch/grouped-GEMM surface now models multimem publication and explicit cross-device barriers; remaining gap is sparse pull-dispatch metadata plus the exact warpgroup producer/consumer protocol. |

### Still Explicitly Sketch / Educational

| Module | Why it stays sketch-level for now |
| --- | --- |
| `Tyr/GPU/Kernels/LayerNorm.lean` | DSL/reference kernels separate from the canonical fused residual port. |
| `Tyr/GPU/Kernels/LayerNormBwd.lean` | Backward math sketch; useful for IR experimentation. |
| `Tyr/GPU/Kernels/Mamba.lean` | Educational stripped-down version rather than a source-backed ThunderKittens port. |
| `Tyr/GPU/Kernels/MambaBwd.lean` | Conceptual backward kernel. |
| `Tyr/GPU/Kernels/FusedLayerNorm.lean` (`FusedLayerNorm.*`) | Compact DSL sketches retained alongside the canonical kernel. |

## Planned Tranches

1. Finish exact peer arithmetic and launch boundaries for `RingAttn*.lean`, `UlyssesAttn*.lean`, and `Distributed.lean`
2. `Mamba2.lean` follow-up
3. Tighten the Blackwell GEMM surfaces in `Bf16Gemm.lean`, `PrecisionGemm.lean`, `NvFp4Gemm.lean`, and `Distributed.lean` once TMEM / cluster-specialized storage becomes available as first-class DSL ops

## Notes

- When a module gets a canonical source-backed surface, keep any older DSL
  sketches only if they are clearly documented as sketches or compatibility
  shims.
- New duplication should be removed as it appears; do not preserve parallel
  alias layers for the same kernel family.
