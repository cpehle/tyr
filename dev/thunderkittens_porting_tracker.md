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
| `pending` | Reworked `Based.lean` and `LinearAttn.lean` into canonical source-backed forward surfaces. |

## Current Module Status

### Canonical Or Near-Canonical

| Module | Vendored source | Status |
| --- | --- | --- |
| `Tyr/GPU/Kernels/Copy.lean` | `include/ops/group/memory/*` | Small direct example; acceptable as-is. |
| `Tyr/GPU/Kernels/Rotary.lean` | `kernels/rotary/rotary.cu` | Reasonably faithful tile split / rotate / concat structure. |
| `Tyr/GPU/Kernels/FusedLayerNorm.lean` (`tkFusedLayerNormResidual1024`) | `kernels/layernorm/layernorm.cu` | Canonical fused residual + layernorm surface. |
| `Tyr/GPU/Kernels/FFTConv.lean` (`tkFFTConvPC1024`) | `kernels/fftconv/fftconv_pc.cu` | Canonical persistent-cache surface now models per-head filter reload and producer/consumer tile flow; complex factor globals are still abstract shared inputs. |
| `Tyr/GPU/Kernels/Hedgehog.lean` (`tkHedgehogFwd`) | `kernels/hedgehog/hedgehog.cu` | Canonical chunk/state surface now models long-resident feature maps, previous/current sliding blocks, and final `k_state` / `kv_state` writeout; the full 3-ring/TMA schedule is still compressed. |
| `Tyr/GPU/Kernels/MhaH100.lean` | `kernels/attention/mha_h100/mha_h100.cu` | Closest attention-side port in the tree. |
| `Tyr/GPU/Kernels/Based.lean` (`tkBasedLinearAttnFwd`) | `kernels/based/linear_attn.cu` | Canonical forward now owns explicit `a0/a1/a2` state and local polynomial attention. |
| `Tyr/GPU/Kernels/LinearAttn.lean` (`tkLinearAttnFwd`) | `kernels/linear_attention/linear_attention.cu` | Canonical decayed forward now owns the local-vs-recurrent split instead of generic feature-map attention. |

### Partially Ported, Still Needs Source-Backed Cleanup

| Module | Vendored source | Main remaining gap |
| --- | --- | --- |
| `Tyr/GPU/Kernels/Based.lean` | `kernels/based/linear_attn.cu` | Still uses portable slice/broadcast helpers instead of the exact warp-shuffle `mul_slice_row` / `mul_slice_col` implementation. |
| `Tyr/GPU/Kernels/LinearAttn.lean` | `kernels/linear_attention/linear_attention.cu` | Uses explicit decay-vector inputs rather than constructing decays from slope in-kernel. |
| `Tyr/GPU/Kernels/LinearAttnBwd.lean` | `kernels/linear_attention/linear_attention.cu` | Still conceptual; needs alignment with forward state decomposition. |
| `Tyr/GPU/Kernels/Mamba2.lean` | `kernels/mamba2/mamba2.cu` | Decay/state are wired, but lcsf producer/consumer and exact warpgroup structure are incomplete. |
| `Tyr/GPU/Kernels/RingAttn.lean` | `kernels/parallel/ring_attn/ring_attn_h100.cu` | Forward is now split into partial/comm/reduction kernels, but the exact peer scheduling and ping-pong launch structure are still simplified. |
| `Tyr/GPU/Kernels/RingAttnBwd.lean` | `kernels/parallel/ring_attn/ring_attn_h100.cu` | Explicitly speculative phased scaffold; needs correct global causal indexing and exact ring accumulation semantics. |
| `Tyr/GPU/Kernels/UlyssesAttn.lean` | `kernels/parallel/ulysses_attn/ulysses_attn.cu` | Now transport/orchestration over all-to-all, but the local attention launch boundary is still external to this module. |
| `Tyr/GPU/Kernels/UlyssesAttnBwd.lean` | `kernels/parallel/ulysses_attn/ulysses_attn.cu` | Backward mirrors the transport split, but the local FlashAttention backward launch is still only modeled as a speculative shell. |
| `Tyr/GPU/Kernels/Distributed.lean` | `kernels/parallel/*` | Collectives plus AG/GEMM-AR/GEMM-RS now have multimem/barrier/producer-consumer structure; still missing full PGL/cluster arithmetic and peer indexing. |
| `Tyr/GPU/Kernels/PrecisionGemm.lean` | `kernels/gemm/fp8_h100/*` | Canonical H100 FP8 surfaces now live here; remaining gap is a separate source-backed B200/MxFP8 module rather than more H100 sketch kernels. |
| `Tyr/GPU/Kernels/NvFp4Gemm.lean` | `kernels/gemm/nvfp4_b200/nvfp4_b200_gemm.cu` | Canonical B200 control-flow surface is explicit, but intentionally compatibility-only until the DSL gains native packed NVFP4/tensor-memory types. |
| `Tyr/GPU/Kernels/MOE.lean` | `kernels/parallel/moe_dispatch_gemm/moe_dispatch_gemm_h100.cu` | Dense routing, dispatch, and combine structure are now modeled; remaining gap is sparse pull-dispatch plus inter-device barrier semantics. |

### Still Explicitly Sketch / Educational

| Module | Why it stays sketch-level for now |
| --- | --- |
| `Tyr/GPU/Kernels/LayerNorm.lean` | DSL/reference kernels separate from the canonical fused residual port. |
| `Tyr/GPU/Kernels/LayerNormBwd.lean` | Backward math sketch; useful for IR experimentation. |
| `Tyr/GPU/Kernels/Mamba.lean` | Educational stripped-down version rather than a source-backed ThunderKittens port. |
| `Tyr/GPU/Kernels/MambaBwd.lean` | Conceptual backward kernel. |
| `Tyr/GPU/Kernels/FusedLayerNorm.lean` (`FusedLayerNorm.*`) | Compact DSL sketches retained alongside the canonical kernel. |

## Planned Tranches

1. `LinearAttnBwd.lean`
2. Finish exact peer arithmetic and launch boundaries for `RingAttn*.lean`, `UlyssesAttn*.lean`, and `Distributed.lean`
3. `Mamba2.lean` follow-up
4. `PrecisionGemm.lean`, `NvFp4Gemm.lean`, `MOE.lean`

## Notes

- When a module gets a canonical source-backed surface, keep any older DSL
  sketches only if they are clearly documented as sketches or compatibility
  shims.
- New duplication should be removed as it appears; do not preserve parallel
  alias layers for the same kernel family.
