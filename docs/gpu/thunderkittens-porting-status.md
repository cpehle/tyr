# ThunderKittens Porting Status

This note tracks which `Tyr.GPU.Kernels` modules are already close to their
vendored ThunderKittens sources and which ones are still sketch-level DSL
prototypes.

The catalog is also grouped into logical family entrypoints:

- `Tyr.GPU.Kernels.Attention`
- `Tyr.GPU.Kernels.StateSpace`
- `Tyr.GPU.Kernels.Parallel`
- `Tyr.GPU.Kernels.Gemm`
- `Tyr.GPU.Kernels.Normalization`
- `Tyr.GPU.Kernels.Experimental`

## Canonical Or Closer Ports

| Tyr module | Vendored ThunderKittens source | Notes |
| --- | --- | --- |
| `Tyr/GPU/Kernels/Copy.lean` | `thirdparty/ThunderKittens/include/ops/group/memory/*` | Small, direct memory-movement example. |
| `Tyr/GPU/Kernels/Rotary.lean` | `thirdparty/ThunderKittens/kernels/rotary/rotary.cu` | Reasonably faithful tile split / rotate / concat structure. |
| `Tyr/GPU/Kernels/FusedLayerNorm.lean` (`tkFusedLayerNormResidual1024`) | `thirdparty/ThunderKittens/kernels/layernorm/layernorm.cu` | Canonical fused residual + layernorm porting surface. |
| `Tyr/GPU/Kernels/FFTConv.lean` (`tkFFTConvPC1024`) | `thirdparty/ThunderKittens/kernels/fftconv/fftconv_pc.cu` | Canonical persistent-cache surface now models per-head filter reload and a producer/consumer tile handoff; complex factor globals are still abstract shared inputs. |
| `Tyr/GPU/Kernels/Hedgehog.lean` (`tkHedgehogFwd`) | `thirdparty/ThunderKittens/kernels/hedgehog/hedgehog.cu` | Canonical chunk/state surface now models long-resident feature maps, previous/current sliding blocks, and final `k_state` / `kv_state` writeout; the full 3-ring/TMA schedule is still compressed. |
| `Tyr/GPU/Kernels/MhaH100.lean` | `thirdparty/ThunderKittens/kernels/attention/mha_h100/mha_h100.cu` | Closest attention-side port in the Lean tree today. |
| `Tyr/GPU/Kernels/Based.lean` (`tkBasedLinearAttnFwd`) | `thirdparty/ThunderKittens/kernels/based/linear_attn.cu` | Now owns explicit `a0/a1/a2` state plus local causal polynomial attention; still uses portable slice/broadcast helpers instead of exact warp-shuffle `mul_slice_*`. |
| `Tyr/GPU/Kernels/LinearAttn.lean` (`tkLinearAttnFwd`) | `thirdparty/ThunderKittens/kernels/linear_attention/linear_attention.cu` | Now a canonical decayed recurrent/local forward surface rather than generic feature-map attention; still relies on externally supplied decay vectors instead of in-kernel slope construction. |
| `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkH100Fp8E4M3GemmFwd`, `tkH100Fp8ScaledGemmFwd`) | `thirdparty/ThunderKittens/kernels/gemm/fp8_h100/*` | Canonical H100 FP8 surfaces now share a concrete 64x128x256 mainloop and explicit row/column-scale epilogues; B200/MxFP8 stays split into separate modules. |
| `Tyr/GPU/Kernels/NvFp4Gemm.lean` (`tkB200NvFp4GemmCompatFwd`) | `thirdparty/ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_b200_gemm.cu` | Canonical B200 control-flow surface is explicit, but intentionally compatibility-only until the DSL gains packed NVFP4 and tensor-memory types. |
| `Tyr/GPU/Kernels/MOE.lean` (`tkMoeDispatchGemm`) | `thirdparty/ThunderKittens/kernels/parallel/moe_dispatch_gemm/moe_dispatch_gemm_h100.cu` | Canonical fused dispatch/grouped-GEMM surface now models multimem publication and explicit cross-device barriers; sparse pull-dispatch metadata and the exact warpgroup protocol remain abstract. |

## Outstanding Sketch-Level Modules

| Tyr module | Vendored ThunderKittens source | Main gap |
| --- | --- | --- |
| `Tyr/GPU/Kernels/LinearAttnBwd.lean` (`linearAttnBwdSketch`) | `thirdparty/ThunderKittens/kernels/linear_attention/linear_attention.cu` | Backward path is now explicitly named as a sketch; it remains a conceptual derivation rather than a source-backed port. |
| `Tyr/GPU/Kernels/LayerNorm.lean` | `thirdparty/ThunderKittens/kernels/layernorm/layernorm.cu` | Generic tiling sketches, separate from the canonical fused residual port. |
| `Tyr/GPU/Kernels/LayerNormBwd.lean` | `thirdparty/ThunderKittens/kernels/layernorm/layernorm.cu` | Conceptual backward kernel with broadcast/gradient assumptions still mocked. |
| `Tyr/GPU/Kernels/FusedLayerNorm.lean` (`FusedLayerNorm.*`) | `thirdparty/ThunderKittens/kernels/layernorm/layernorm.cu` | Kept as sketches for IR experimentation; not the canonical port. |
| `Tyr/GPU/Kernels/Mamba2.lean` | `thirdparty/ThunderKittens/kernels/mamba2/mamba2.cu` | The decay vector and recurrent KV state are now wired, but the full lcsf producer/consumer structure and exact warpgroup decay handling are still missing. |
| `Tyr/GPU/Kernels/Mamba.lean` | `thirdparty/ThunderKittens/kernels/mamba2/mamba2.cu` | Educational sketch, not a faithful source-backed port. |
| `Tyr/GPU/Kernels/MambaBwd.lean` | `thirdparty/ThunderKittens/kernels/mamba2/mamba2.cu` | Backward logic is conceptual and still uses placeholder mask/cumsum handling. |
| `Tyr/GPU/Kernels/Distributed.lean` | `thirdparty/ThunderKittens/kernels/parallel/*` | Collectives and AG/GEMM-AR/GEMM-RS surfaces now use explicit multimem/barrier/producer-consumer structure, but they still rely on caller-provided views instead of full PGL/cluster arithmetic. |
| `Tyr/GPU/Kernels/RingAttn.lean` | `thirdparty/ThunderKittens/kernels/parallel/ring_attn/ring_attn_h100.cu` | Forward is now split into explicit partial/comm/reduction kernels, but the exact peer scheduling and ping-pong launch structure remain simplified. |
| `Tyr/GPU/Kernels/RingAttnBwd.lean` | `thirdparty/ThunderKittens/kernels/parallel/ring_attn/ring_attn_h100.cu` | Backward is explicitly demoted to a speculative phased scaffold; the phase split is there, but global causal indexing and exact ring accumulation are still incomplete. |
| `Tyr/GPU/Kernels/UlyssesAttn.lean` | `thirdparty/ThunderKittens/kernels/parallel/ulysses_attn/ulysses_attn.cu` | Ulysses is now modeled as all-to-all transport/orchestration around a separate local attention launch rather than a bespoke fused attention sketch. |
| `Tyr/GPU/Kernels/UlyssesAttnBwd.lean` | `thirdparty/ThunderKittens/kernels/parallel/ulysses_attn/ulysses_attn.cu` | Backward now mirrors the transport/orchestration split, but the local FlashAttention backward launch boundary is still represented only as a speculative shell. |

## Recommended Next Tranches

1. `LinearAttnBwd.lean` once the decayed forward contract is settled and shared derivative helpers exist.
   Current sketch surface: `linearAttnBwdSketch`.
2. `Mamba2.lean` to finish the lcsf / warp-specialized structure from `thirdparty/ThunderKittens/kernels/mamba2/mamba2.cu`.
3. Finish exact peer arithmetic and launch boundaries for `RingAttn*.lean`, `UlyssesAttn*.lean`, and `Distributed.lean` now that the collective/transport substrate is concrete.
4. `FFTConv.lean`, `Hedgehog.lean`, and the quantized GEMM family once the common pipeline abstractions are tightened.
