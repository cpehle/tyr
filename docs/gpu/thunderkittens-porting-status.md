# ThunderKittens Porting Status

This note tracks the public ThunderKittens parity surface inside
`Tyr.GPU.Kernels`.

The GPU catalog is grouped into logical family entrypoints:

- `Tyr.GPU.Kernels.Attention`
- `Tyr.GPU.Kernels.StateSpace`
- `Tyr.GPU.Kernels.Parallel`
- `Tyr.GPU.Kernels.Gemm`
- `Tyr.GPU.Kernels.Normalization`
- `Tyr.GPU.Kernels.Experimental`

## Coverage

Every vendored ThunderKittens `.cu` source under
[thirdparty/ThunderKittens/kernels](/Users/pehle/dev/tyr/thirdparty/ThunderKittens/kernels)
now has a built Lean counterpart in the catalog.

The important distinction now is not coverage vs missing families. It is:

- source-backed kernels that are fully represented in the Lean DSL today, and
- source-backed kernels that still use raw backend blocks for TMEM, cluster,
  or distributed PGL-style constructs that the DSL does not yet model as
  first-class operations.

The working exhaustive source-to-Lean matrix lives in
[dev/thunderkittens_porting_tracker.md](/Users/pehle/dev/tyr/dev/thunderkittens_porting_tracker.md).

## Canonical Public Surface

| Tyr module | Vendored ThunderKittens source | Notes |
| --- | --- | --- |
| `Tyr/GPU/Kernels/FusedLayerNorm.lean` (`tkFusedLayerNormResidual1024`) | `thirdparty/ThunderKittens/kernels/layernorm/layernorm.cu` | Canonical fused residual + layernorm port. |
| `Tyr/GPU/Kernels/MhaH100.lean` | `thirdparty/ThunderKittens/kernels/attention/mha_h100/mha_h100.cu` | Canonical Hopper MHA surface. |
| `Tyr/GPU/Kernels/MhaH100LCF.lean` (`tkMhaH100LCFFwd64`, `tkMhaH100LCFFwd128`) | `thirdparty/ThunderKittens/kernels/attention/mha_h100_lcf/mha_h100_lcf.cu` | Dedicated LCF load-compute-finish counterparts. |
| `Tyr/GPU/Kernels/Based.lean` (`tkBasedLinearAttnFwd`) | `thirdparty/ThunderKittens/kernels/based/linear_attn.cu` | Source-backed forward owns the local polynomial/state contract. |
| `Tyr/GPU/Kernels/LinearAttn.lean` (`tkLinearAttnFwd`) | `thirdparty/ThunderKittens/kernels/linear_attention/linear_attention.cu` | Canonical decayed recurrent/local forward surface. |
| `Tyr/GPU/Kernels/FFTConv.lean` (`tkFFTConvPC1024`, `tkFFTConvNonPC64`) | `thirdparty/ThunderKittens/kernels/fftconv/*.cu` | Persistent and non-persistent FFTConv counterparts. |
| `Tyr/GPU/Kernels/Hedgehog.lean` (`tkHedgehogFwd`) | `thirdparty/ThunderKittens/kernels/hedgehog/hedgehog.cu` | Canonical chunk/state surface. |
| `Tyr/GPU/Kernels/Mamba2.lean` (`mamba2Fwd`) | `thirdparty/ThunderKittens/kernels/mamba2/mamba2.cu` | Dedicated lcsf-style producer/consumer counterpart. |
| `Tyr/GPU/Kernels/Flux.lean` (`tkFluxMatmulGeluFwd`, `tkFluxMatmulGateFwd`) | `thirdparty/ThunderKittens/kernels/flux/flux_*.cu` | Dedicated source-facing flux surfaces. |
| `Tyr/GPU/Kernels/Bf16Gemm.lean` (`tkH100Bf16GemmFwd`, `tkB200Bf16GemmFwd`) | `thirdparty/ThunderKittens/kernels/gemm/bf16_*/*.cu` | Hopper and Blackwell BF16 GEMM counterparts. |
| `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkH100Fp8E4M3GemmFwd`, `tkH100Fp8ScaledGemmFwd`, `tkB200Fp8E4M3Gemm1CtaFwd`, `tkB200Fp8E4M3Gemm2CtaFwd`, `tkB200MxFp8GemmFwd`) | `thirdparty/ThunderKittens/kernels/gemm/fp8_*`, `thirdparty/ThunderKittens/kernels/gemm/mxfp8_b200/*` | H100 and Blackwell FP8/MXFP8 GEMM surfaces. |
| `Tyr/GPU/Kernels/NvFp4Gemm.lean` (`tkB200NvFp4GemmFwd`) | `thirdparty/ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_b200_gemm.cu` | Dedicated NVFP4 Blackwell GEMM counterpart. |
| `Tyr/GPU/Kernels/Distributed.lean` (`allGatherFwd`, `allReduceFwd`, `allReduceEducationalFwd`, `reduceScatterFwd`, `agGemmFwd`, `agGemmB200Fwd`, `agGemmFp8B200Fwd`, `gemmArFwd`, `gemmArH100LcscFwd`, `gemmRsFwd`, `gemmRsB200Fwd`, `gemmRsFp8B200Fwd`) | `thirdparty/ThunderKittens/kernels/parallel/*` | Collective and communication+compute counterparts for the distributed family. |
| `Tyr/GPU/Kernels/RingAttn.lean` (`ringAttnPartial`, `ringAttnComm`, `ringAttnReduce`) | `thirdparty/ThunderKittens/kernels/parallel/ring_attn/ring_attn_h100.cu` | Forward ring-attention phases are represented directly. |
| `Tyr/GPU/Kernels/UlyssesAttn.lean` (`allToAllFwd`, `ulyssesQkvAllToAll`, `ulyssesAttnFwd`) | `thirdparty/ThunderKittens/kernels/parallel/ulysses_attn/ulysses_attn.cu` | Ulysses transport/orchestration family built on the shared all-to-all surface. |
| `Tyr/GPU/Kernels/MOE.lean` (`tkMoeDispatchGemm`) | `thirdparty/ThunderKittens/kernels/parallel/moe_dispatch_gemm/moe_dispatch_gemm_h100.cu` | Canonical fused dispatch/grouped-GEMM surface. |
| `Tyr/GPU/Kernels/Rotary.lean` | `thirdparty/ThunderKittens/kernels/rotary/rotary.cu` | Canonical rotary position kernel. |

## Derived Tyr Kernels

These modules stay in the catalog, but they are Tyr-native extensions rather
than vendored ThunderKittens parity surfaces:

- `Tyr/GPU/Kernels/LinearAttnBwd.lean`
- `Tyr/GPU/Kernels/RingAttnBwd.lean`
- `Tyr/GPU/Kernels/UlyssesAttnBwd.lean`

## Follow-Ups

The remaining work is now mostly DSL expressiveness work:

1. Add first-class TMEM, cluster, and packed-scale constructs so the Blackwell
   GEMM family can move out of raw backend blocks.
2. Add first-class distributed PGL/topology primitives so the distributed,
   Ring, and Ulysses families no longer need raw peer-arithmetic scaffolding.
3. Tighten exact CTA worker packing for some attention/state-space families
   where the source structure is represented, but the runtime packing is still
   compressed.

## Cleanup Result

The old sketch-only kernel modules that duplicated the canonical ports were
removed from the core catalog:

- `Tyr/GPU/Kernels/LayerNorm.lean`
- `Tyr/GPU/Kernels/LayerNormBwd.lean`
- `Tyr/GPU/Kernels/LayerNormResidual.lean`
- `Tyr/GPU/Kernels/Mamba.lean`
- `Tyr/GPU/Kernels/MambaBwd.lean`

That leaves the build/docs surface centered on the actual source-backed
ThunderKittens counterparts instead of parallel educational shims.
