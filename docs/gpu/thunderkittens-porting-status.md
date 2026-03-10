# ThunderKittens Porting Status

This note tracks which `Tyr.GPU.Kernels` modules are already close to their
vendored ThunderKittens sources and which ones are still sketch-level DSL
prototypes.

## Canonical Or Closer Ports

| Tyr module | Vendored ThunderKittens source | Notes |
| --- | --- | --- |
| `Tyr/GPU/Kernels/Copy.lean` | `thirdparty/ThunderKittens/include/ops/group/memory/*` | Small, direct memory-movement example. |
| `Tyr/GPU/Kernels/Rotary.lean` | `thirdparty/ThunderKittens/kernels/rotary/rotary.cu` | Reasonably faithful tile split / rotate / concat structure. |
| `Tyr/GPU/Kernels/FusedLayerNorm.lean` (`tkFusedLayerNormResidual1024`) | `thirdparty/ThunderKittens/kernels/layernorm/layernorm.cu` | Canonical fused residual + layernorm porting surface. |
| `Tyr/GPU/Kernels/MhaH100.lean` | `thirdparty/ThunderKittens/kernels/attention/mha_h100/mha_h100.cu` | Closest attention-side port in the Lean tree today. |

## Outstanding Sketch-Level Modules

| Tyr module | Vendored ThunderKittens source | Main gap |
| --- | --- | --- |
| `Tyr/GPU/Kernels/Based.lean` | `thirdparty/ThunderKittens/kernels/based/linear_attn.cu` | Missing the real `mul_slice_row` / `mul_slice_col` state update structure and faithful `a2` propagation. |
| `Tyr/GPU/Kernels/LinearAttn.lean` | `thirdparty/ThunderKittens/kernels/linear_attention/linear_attention.cu` | Still a generic linear-attention sketch rather than the real slope/decay/allocation pattern. |
| `Tyr/GPU/Kernels/LinearAttnBwd.lean` | `thirdparty/ThunderKittens/kernels/linear_attention/linear_attention.cu` | Backward path is a conceptual derivation, not a source-backed port. |
| `Tyr/GPU/Kernels/LayerNorm.lean` | `thirdparty/ThunderKittens/kernels/layernorm/layernorm.cu` | Generic tiling sketches, separate from the canonical fused residual port. |
| `Tyr/GPU/Kernels/LayerNormBwd.lean` | `thirdparty/ThunderKittens/kernels/layernorm/layernorm.cu` | Conceptual backward kernel with broadcast/gradient assumptions still mocked. |
| `Tyr/GPU/Kernels/FusedLayerNorm.lean` (`FusedLayerNorm.*`) | `thirdparty/ThunderKittens/kernels/layernorm/layernorm.cu` | Kept as sketches for IR experimentation; not the canonical port. |
| `Tyr/GPU/Kernels/FFTConv.lean` | `thirdparty/ThunderKittens/kernels/fftconv/fftconv_pc.cu` | High-level semantic structure exists, but not the real producer/consumer or scratch choreography. |
| `Tyr/GPU/Kernels/Hedgehog.lean` | `thirdparty/ThunderKittens/kernels/hedgehog/hedgehog.cu` | Hybrid attention structure is present, but state handling and scheduling are simplified. |
| `Tyr/GPU/Kernels/Mamba2.lean` | `thirdparty/ThunderKittens/kernels/mamba2/mamba2.cu` | The decay vector and recurrent KV state are now wired, but the full lcsf producer/consumer structure and exact warpgroup decay handling are still missing. |
| `Tyr/GPU/Kernels/Mamba.lean` | `thirdparty/ThunderKittens/kernels/mamba2/mamba2.cu` | Educational sketch, not a faithful source-backed port. |
| `Tyr/GPU/Kernels/MambaBwd.lean` | `thirdparty/ThunderKittens/kernels/mamba2/mamba2.cu` | Backward logic is conceptual and still uses placeholder mask/cumsum handling. |
| `Tyr/GPU/Kernels/MOE.lean` | `thirdparty/ThunderKittens/kernels/parallel/moe_dispatch_gemm/moe_dispatch_gemm_h100.cu` | Top-k routing, dispatch, and combine are still mocked. |
| `Tyr/GPU/Kernels/Distributed.lean` | `thirdparty/ThunderKittens/kernels/parallel/*` | Collective semantics are illustrative only. |
| `Tyr/GPU/Kernels/RingAttn.lean` | `thirdparty/ThunderKittens/kernels/parallel/ring_attn/ring_attn_h100.cu` | Ring communication and stable reduction are still partially simulated. |
| `Tyr/GPU/Kernels/RingAttnBwd.lean` | `thirdparty/ThunderKittens/kernels/parallel/ring_attn/ring_attn_h100.cu` | Ring-step masking and communication are explicitly incomplete. |
| `Tyr/GPU/Kernels/UlyssesAttn.lean` | `thirdparty/ThunderKittens/kernels/parallel/ulysses_attn/ulysses_attn.cu` | NCCL/all-to-all behavior is still represented as copies. |
| `Tyr/GPU/Kernels/UlyssesAttnBwd.lean` | `thirdparty/ThunderKittens/kernels/parallel/ulysses_attn/ulysses_attn.cu` | Backward reuses local FlashAttention logic but still mocks the all-to-all path. |
| `Tyr/GPU/Kernels/PrecisionGemm.lean` | `thirdparty/ThunderKittens/kernels/gemm/*` | Missing the real TMA, scaling, and datatype-specific scheduling. |
| `Tyr/GPU/Kernels/NvFp4Gemm.lean` | `thirdparty/ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_b200_gemm.cu` | Still uses FP8 proxies instead of real FP4 packing/types. |

## Recommended Next Tranches

1. `Based.lean` against `thirdparty/ThunderKittens/kernels/based/linear_attn.cu`.
2. `LinearAttn.lean` against `thirdparty/ThunderKittens/kernels/linear_attention/linear_attention.cu`.
3. `Mamba2.lean` to finish the lcsf / warp-specialized structure from `thirdparty/ThunderKittens/kernels/mamba2/mamba2.cu`.
4. `RingAttn*.lean` and `UlyssesAttn*.lean` once the communication/runtime surface can express the real collectives.
