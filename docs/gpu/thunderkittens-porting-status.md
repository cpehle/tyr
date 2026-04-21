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
- source-backed kernels whose source structure is still compressed into typed
  tiled shells because the DSL does not yet model every TMEM / cluster /
  packed-scale construct as a first-class operation.

The working exhaustive source-to-Lean matrix lives in
[dev/thunderkittens_porting_tracker.md](/Users/pehle/dev/tyr/dev/thunderkittens_porting_tracker.md).

## Canonical Public Surface

| Tyr module | Vendored ThunderKittens source | Notes |
| --- | --- | --- |
| `Tyr/GPU/Kernels/FusedLayerNorm.lean` (`tkFusedLayerNormResidual1024`) | `thirdparty/ThunderKittens/kernels/layernorm/layernorm.cu` | Canonical fused residual + layernorm port. |
| `Tyr/GPU/Kernels/MhaH100.lean` | `thirdparty/ThunderKittens/kernels/attention/mha_h100/mha_h100.cu` | Canonical Hopper MHA surface. |
| `Tyr/GPU/Kernels/MhaH100LCF.lean` (`tkMhaH100LCFFwd64`, `tkMhaH100LCFFwd128`) | `thirdparty/ThunderKittens/kernels/attention/mha_h100_lcf/mha_h100_lcf.cu` | Dedicated LCF load-compute-finish counterparts, now fully typed as stationary-Q / streamed-KV shells. |
| `Tyr/GPU/Kernels/Based.lean` (`tkBasedLinearAttnFwd`) | `thirdparty/ThunderKittens/kernels/based/linear_attn.cu` | Source-backed forward owns the local polynomial/state contract. |
| `Tyr/GPU/Kernels/LinearAttn.lean` (`tkLinearAttnFwd`) | `thirdparty/ThunderKittens/kernels/linear_attention/linear_attention.cu` | Canonical decayed recurrent/local forward surface. |
| `Tyr/GPU/Kernels/FFTConv.lean` (`tkFFTConvPC1024`, `tkFFTConvNonPC64`) | `thirdparty/ThunderKittens/kernels/fftconv/*.cu` | Persistent and non-persistent FFTConv counterparts. |
| `Tyr/GPU/Kernels/Hedgehog.lean` (`tkHedgehogFwd`) | `thirdparty/ThunderKittens/kernels/hedgehog/hedgehog.cu` | Canonical chunk/state surface. |
| `Tyr/GPU/Kernels/Mamba2.lean` (`mamba2Fwd`) | `thirdparty/ThunderKittens/kernels/mamba2/mamba2.cu` | Dedicated typed chunk/state counterpart with runtime-bounded chunk loops and recurrent state updates. |
| `Tyr/GPU/Kernels/Flux.lean` (`tkFluxMatmulGeluFwd`, `tkFluxMatmulGateFwd`) | `thirdparty/ThunderKittens/kernels/flux/flux_*.cu` | Dedicated source-facing flux surfaces. |
| `Tyr/GPU/Kernels/Bf16Gemm.lean` (`tkH100Bf16GemmFwd`, `tkB200Bf16GemmFwd`) | `thirdparty/ThunderKittens/kernels/gemm/bf16_*/*.cu` | Hopper and Blackwell BF16 GEMM counterparts. |
| `Tyr/GPU/Kernels/PrecisionGemm.lean` (`tkH100Fp8E4M3GemmFwd`, `tkH100Fp8ScaledGemmFwd`, `tkB200Fp8E4M3Gemm1CtaFwd`, `tkB200Fp8E4M3Gemm2CtaFwd`, `tkB200MxFp8GemmFwd`) | `thirdparty/ThunderKittens/kernels/gemm/fp8_*`, `thirdparty/ThunderKittens/kernels/gemm/mxfp8_b200/*` | H100 and Blackwell FP8/MXFP8 GEMM surfaces. |
| `Tyr/GPU/Kernels/NvFp4Gemm.lean` (`tkB200NvFp4GemmFwd`) | `thirdparty/ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_b200_gemm.cu` | Dedicated NVFP4 Blackwell GEMM counterpart. |
| `Tyr/GPU/Kernels/Distributed.lean` (`allGatherFwd`, `allReduceFwd`, `allReduceEducationalFwd`, `reduceScatterFwd`, `agGemmFwd`, `agGemmB200Fwd`, `agGemmFp8B200Fwd`, `gemmArFwd`, `gemmArH100LcscFwd`, `gemmRsFwd`, `gemmRsB200Fwd`, `gemmRsFp8B200Fwd`) | `thirdparty/ThunderKittens/kernels/parallel/*` | Collective and communication+compute counterparts for the distributed family. |
| `Tyr/GPU/Kernels/RingAttn.lean` (`ringAttnPartial`, `ringAttnComm`, `ringAttnReduce`) | `thirdparty/ThunderKittens/kernels/parallel/ring_attn/ring_attn_h100.cu` | Forward ring-attention phases are represented directly, with the partial phase now using a typed runtime-bounded KV-shard loop. |
| `Tyr/GPU/Kernels/UlyssesAttn.lean` (`allToAllFwd`, `ulyssesQkvAllToAll`, `ulyssesAttnFwd`) | `thirdparty/ThunderKittens/kernels/parallel/ulysses_attn/ulysses_attn.cu` | Ulysses transport/orchestration family built on the typed shared all-to-all surface. |
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
   GEMM family can model the source structure directly rather than through typed
   compressed shells.
2. Tighten exact CTA worker packing for some attention/state-space families
   where the source structure is represented, but the runtime packing is still
   compressed.

## Training Bring-Up Roadmap (H100-First)

Kernel source coverage is complete, but end-to-end training integration is still
in progress. The execution order is:

1. Single-GPU bring-up on one H100.
2. FlashAttention performance/correctness benchmark milestone.
3. Model integration (NanoChat attention path).
4. Multi-GPU scaling to 4xH100.

### Stage 1: One-H100 baseline and kernel validation

- Load the expected toolchain before GPU runs:
  - `source ./load_modules.sh`
  - Set `LEAN_CC=$PWD/scripts/lean_cc_wrapper.sh` on this cluster before
    running `lake build`; this is currently required for Lean native shared
    library links.
  - Override module selections when needed with `TYR_CUDA_MODULE=...`,
    `TYR_NCCL_MODULE=...`, and `TYR_ARROW_MODULE=...`.
  - The default one-H100 path now uses `CUDA/12.9.1` and skips NCCL unless
    `TYR_NCCL_MODULE` is explicitly set.
  - `TYR_NCCL_MODULE=` cleanly skips NCCL loading, which is useful for the
    one-H100 direct path and keeps the default module stack smaller.
  - For non-login automation shells, `load_modules.sh` now bootstraps the
    `module` function from `/etc/profile.d/modules.sh` or the Lmod init script
    before loading the EasyBuild stack.
- Hopper note:
  - This host exposes newer CUDA modules including `CUDA/12.9.1` and
    `CUDA/13.1.0`, plus matching NCCL builds such as
    `NCCL/2.27.7-GCCcore-14.3.0-CUDA-12.9.1`.
  - A direct `nvcc` compile of the generated
    [Tyr_GPU_Kernels_MhaH100.cu](/grid/zador/home/pehle/dev/tyr/cc/src/generated/Tyr_GPU_Kernels_MhaH100.cu)
    object succeeded under `CUDA/12.9.1` without the old
    `cudaLaunchAttributePreferredClusterDimension` compatibility define, so the
    shim has now been removed from [cc/Makefile](/grid/zador/home/pehle/dev/tyr/cc/Makefile).
- Preferred direct build/run path:
  - `lake -R build GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 RunFlashAttn`
  - `lake -R env ./.lake/build/bin/GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 --out-dir cc/src/generated`
  - `make -C cc -j"$(nproc)"`
  - `lake -R env ./.lake/build/bin/RunFlashAttn`
  - The `GenerateGpuKernels` binary still needs `lake -R env`; invoking
    `./.lake/build/bin/GenerateGpuKernels` directly fails package lookup with
    `unknown module prefix 'Tyr'`.
  - Keep the older `scripts/gpu/*.sh` helpers as convenience wrappers only; the
    direct `lake` + `make` sequence is the path to stabilize.
  - For iteration speed, avoid the broad
    `lake -R build GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 RunFlashAttn RunMhaH100Train`
    command during proof/debug cycles. The faster loop is:
    - `lake -R build <module> +<module>:dynlib` for the failing Lean module
    - then rebuild only the runner you actually need
    - then rerun `GenerateGpuKernels` + `make -C cc`
  - If a replayed broad build dies late in `Tyr.GPU.Codegen.Primitives`, rebuild
    `Tyr.GPU.Codegen.Primitives` and `+Tyr.GPU.Codegen.Primitives:dynlib`
    directly. The concrete failure mode seen on this host was generic
    tensor-core complex wrappers not propagating the `M % 16 = 0`,
    `K % 16 = 0`, and `N % 16 = 0` shape proofs to `mma` / `mmaT`.
- Lean native-plugin note:
  - After changing import wiring in `Tyr.GPU.Codegen` / `Tyr.GPU.Kernels`,
    `lake build <module>` may refresh `.olean` / generated C without refreshing
    the corresponding native plugin. When debugging symbol mismatches in
    `.lake/build/lib/lean/*.so`, rebuild the affected module's `:dynlib`
    facet explicitly, for example:
    `lake build +Tyr.GPU.Codegen.Attribute:dynlib +Tyr.GPU.Kernels.Prelude:dynlib`.
- Run generated-kernel fixture checks on one visible GPU:
  - `lake -R build GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 RunFlashAttn`
  - `lake -R env ./.lake/build/bin/GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 --out-dir cc/src/generated`
  - `make -C cc -j"$(nproc)"`
  - `CUDA_VISIBLE_DEVICES=0 lake -R env ./.lake/build/bin/RunFlashAttn --regen`
  - `scripts/gpu/test_flashattn_e2e.sh` remains a convenience wrapper around
    the same flow.
- Run one-GPU kernel-vs-portable training benchmark:
  - `lake -R build GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 RunMhaH100Train`
  - `lake -R env ./.lake/build/bin/GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 --out-dir cc/src/generated`
  - `make -C cc -j"$(nproc)"`
  - `CUDA_VISIBLE_DEVICES=0 lake -R env ./.lake/build/bin/RunMhaH100Train --benchmark --warmup 20 --bench-iters 500 --lr 200.0 --noise 0.5`
  - `scripts/gpu/bench_mha_h100_train.sh` remains a convenience wrapper around
    the same direct commands.
- Acceptance:
  - fixture parity pass (forward and backward paths),
  - stable benchmark output from `RunMhaH100Train`,
  - no NaN/Inf regressions.

### Stage 2: Intermediate performance goal (SoTA FlashAttention vs Tyr TK)

- Add a reproducible benchmark harness that compares:
  - Tyr ThunderKittens-backed FlashAttention route,
  - PyTorch SDPA baseline,
  - SoTA FlashAttention backend available on the host stack.
- Record for each tested shape:
  - `ms/step`, `steps/s`, speedup ratio,
  - max/mean forward error and gradient error vs reference.
- Start with one H100 only before any multi-GPU rollout.

#### Benchmark Families

- `Family A: forward parity + throughput`
  - purpose: compare pure FlashAttention forward performance and numerical
    parity on identical inputs.
  - immediate implementation target: a dedicated one-command runner that
    launches the current direct TK kernels, PyTorch SDPA, and the strongest
    host FlashAttention backend available for the same shape.
- `Family B: train-step benchmark`
  - purpose: measure the practical end-to-end value of the H100 path using the
    existing forward + backward + SGD step loop.
  - immediate implementation target: extend the existing
    [RunMhaH100Train.lean](/grid/zador/home/pehle/dev/tyr/Examples/GPU/RunMhaH100Train.lean)
    path as the training benchmark reference until a generalized op-wrapper
    route exists.
- `Family C: kernel-generation generality sweep`
  - purpose: measure how broad the native kernel-generation surface actually is,
    not just how fast one supported benchmark shape runs.
  - immediate implementation target: a shape sweep that records whether a shape
    runs natively, falls back portably, or fails generation/build, together
    with the failure reason.

#### Benchmark Phases

- `Phase 0: lock the current exact-supported baseline`
  - benchmark only shapes already known to run through the direct path on one
    H100.
  - this is the release gate for the current bring-up work.
- `Phase 1: SoTA comparison on exact-supported shapes`
  - use the same input tensors and same dtype/layout for:
    - Tyr TK direct kernel path,
    - PyTorch SDPA default dispatch,
    - strongest host backend available for the same shape.
  - if the SoTA backend does not support the exact shape, report
    `unsupported`, not a silent fallback.
- `Phase 2: broaden along the near-term kernel family`
  - add benchmark rows only after a native kernel surface exists for them, for
    example through `MhaH100LCF` or `AttentionFactory`.
  - this is where `headDim = 128` and additional KV-tile counts should enter.
- `Phase 3: generality sweep`
  - run a broader matrix whose goal is coverage reporting:
    - native kernel hit,
    - portable fallback hit,
    - generation/build failure.
  - this is how to measure the generality problem systematically instead of
    discussing it abstractly.

#### Initial Matrix

- `Tier 0: must-run now`
  - `batch=1`, `heads=1`, `dtype=bf16`, `causal=false`
  - `(seq=128, headDim=64)` using the current 2-block path
  - `(seq=768, headDim=64)` once the 12-block direct runner is wired into the
    same benchmark harness
- `Tier 1: near-term native expansion`
  - `batch in {1, 4}`
  - `heads in {1, 8}`
  - `seq in {128, 768}`
  - `headDim in {64, 128}`
  - add rows only when the native path is actually present; otherwise report
    them in the generality sweep, not in the runtime leaderboard.
- `Tier 2: generality sweep only`
  - `seq in {64, 128, 256, 512, 768, 1024, 2048}`
  - `headDim in {64, 128}`
  - `causal in {false, true}`
  - `batch in {1, 4}`
  - `heads in {1, 8}`
  - for each row, log:
    - selected backend,
    - native/fallback/fail status,
    - failure reason if not native.

#### Metrics

- `Correctness`
  - forward: `out_mae`, `out_max`, `lse_mae`, `lse_max`
  - train-step: `dQ`, `dK`, `dV` max/mean error once exposed by the harness
  - convergence sanity: initial loss, final loss, relative improvement over a
    fixed small-step run
- `Performance`
  - median `ms/iter`
  - `iters/s`
  - speedup vs PyTorch SDPA default
  - speedup vs strongest SoTA backend available
  - estimated `tokens/s`
  - estimated effective `TFLOP/s` once the harness includes a FLOP model
- `Stability`
  - min/median/max over repeated runs
  - pass/fail/oom/nan status
  - backend-selected vs backend-requested, so silent fallbacks are visible
- `Generality`
  - generation success/failure
  - `lake` build time
  - `GenerateGpuKernels` time
  - `make -C cc` time for the generated kernel set

### Runtime Bridge Status (2026-04-21)

The high-level runtime bridge for FlashAttention now exists.

- `Tyr/Torch.lean`
  - `torch.nn.tyrFlashAttn4d` is no longer a temporary Lean fallback.
  - it now binds the real C++ entrypoint
    `lean_torch_tyr_flash_attn_4d`.
- `cc/src/tyr_ops.cpp`
  - now registers `tyr::flash_attn` through `TORCH_LIBRARY`.
  - current dispatch policy is:
    - native ThunderKittens-backed H100 route for the validated exact shapes
      (`bf16`, `batch=1`, `q_heads=kv_heads=1`, `headDim=64`,
      non-causal dense self-attention, `seq in {128, 768}`),
    - portable fallback to PyTorch SDPA for everything else.
  - the portable fallback now also covers:
    - causal attention,
    - padding-mask expansion,
    - grouped-query attention by repeating KV heads.
- `cc/Makefile`
  - now compiles `cc/src/tyr_ops.cpp` as part of `libTyrC`.

### Runtime Validation Snapshot

There is now a dedicated smoke test at
`Examples/GPU/RunFlashAttnOp.lean`.

It checks three cases:

1. native exact-shape H100 route,
2. portable causal fallback,
3. portable GQA fallback.

Current result on one H100:

- native route selection is correct: `route=tkKernel`
- native forward matches SDPA closely:
  - `out_mae=0.000057`
  - `out_max=0.001953`
- portable fallback paths are exact for the current checked cases:
  - causal fallback: all outputs and gradients matched exactly
  - GQA fallback: all outputs and gradients matched exactly
- native backward is not yet training-ready through the new high-level op:
  - `dQ` is close
  - `dK` and `dV` still miss the current tolerance window versus SDPA
  - one observed run produced:
    - `dk_mae=0.003548`, `dk_max=0.363281`
    - `dv_mae=0.004895`, `dv_max=0.550781`

Interpretation:

- the new runtime surface is now real and usable for forward benchmarking,
- portable fallback behavior is correct for non-native shapes and GQA,
- the remaining blocker for end-to-end training through `tyr::flash_attn` is
  the native backward mismatch on the exact H100 path.

### Build-System Note

While bringing up the runtime smoke test, a broader `lake -R build RunFlashAttnOp`
path exposed an unrelated package-native-plugin issue:

- `.lake/build/lib/lean/tyr_Tyr_GPU_Codegen_FFI.so` was observed as `file too short`
- later, `tyr_Tyr_GPU_Codegen_TileIR_Frontend.so` failed to materialize

This is not specific to the attention bridge itself, but it does affect the
fully packaged executable path. The narrower interpreter-based path with
explicit dynlib loading was sufficient to validate the attention bridge while
that broader package-native issue remains open.

#### Model-Driven Specialization Priorities

The current direct H100 path proves that Tyr can run generated TK-style
attention kernels on one H100, but it is still a narrow benchmark surface:

- `AttentionFactory` is already parameterized by `seq`, `headDim`, `kvBlocks`,
  `tileSize`, `dtype`, `arch`, `isCausal`, and `enableGqa`.
- The published wrapper/op layer is not yet parameterized in the same way:
  [MhaH100.lean](/grid/zador/home/pehle/dev/tyr/Tyr/GPU/Ops/MhaH100.lean)
  still hard-codes `[1, 1, seq, 64]` with manual dispatch for `seq=128` and
  `seq=768`, and
  [FlashAttn.lean](/grid/zador/home/pehle/dev/tyr/Tyr/GPU/Ops/FlashAttn.lean)
  still advertises TK coverage only for the same `headDim=64` / non-causal
  shapes.

The next backend milestone is therefore not "add one more benchmark shape". It
is to expose a stable runtime attention API over a registry of compiled
specializations, following the same pattern ThunderKittens uses from PyTorch:

- stable public call signature,
- runtime problem descriptor carrying tensor/layout/semantics metadata,
- dispatch to a compiled specialization keyed by concrete shape/mode,
- portable fallback when no specialization exists,
- benchmarking on top of that same surface so performance and generality are
  measured together.

This also matches how larger systems behave in practice:

- PyTorch exposes stable operator entrypoints and selects specialized kernels at
  runtime.
- JAX lowers a stable high-level operation and caches specialized executables
  per concrete signature.

#### Native Target Matrix Beyond the Current Demo Shapes

The next specialization work should be driven by the model code already in the
tree, not by the current `headDim=64` benchmark alone.

- `Qwen`
  - [Tyr/Model/Qwen/Attention.lean](/grid/zador/home/pehle/dev/tyr/Tyr/Model/Qwen/Attention.lean)
    uses grouped-query attention directly.
  - [Tyr/Model/Qwen35/Config.lean](/grid/zador/home/pehle/dev/tyr/Tyr/Model/Qwen35/Config.lean)
    defaults to `head_dim=256`, with common ratios such as
    `num_attention_heads=16`, `num_key_value_heads=4` and some variants with
    `num_key_value_heads=2`.
  - Qwen also needs decode-time KV-cache paths (`q_len=1`, growing `kv_len`)
    for incremental generation.
- `Gemma4`
  - [Tyr/Model/Gemma4/Config.lean](/grid/zador/home/pehle/dev/tyr/Tyr/Model/Gemma4/Config.lean)
    uses `head_dim=256`, sometimes `global_head_dim=512`, plus sliding-window
    attention with periodic full-attention layers.
  - Gemma variants also use nontrivial KV-head ratios and global/sliding mixes,
    so backend support cannot stop at plain non-causal dense MHA.

That implies the near-term native target matrix should be:

1. Existing H100 baseline:
   - `headDim=64`, `seq in {128, 768}`, non-causal
2. First model-relevant training specialization:
   - `headDim=256`, dense full-sequence attention on one H100
3. First grouped-query specialization:
   - `num_heads != num_kv_heads`
4. First decode specialization:
   - `q_len=1`, KV cache append/read path
5. First Gemma-style locality specialization:
   - sliding-window/full-attention split

#### Concrete Backend Refactor Sequence

1. Define a runtime `AttentionProblem` descriptor that includes:
   - `batch`
   - `num_q_heads`
   - `num_kv_heads`
   - `q_seq`
   - `kv_seq`
   - `head_dim`
   - `dtype`
   - `arch`
   - `is_causal`
   - attention mode flags such as dense/sliding/decode/GQA
2. Replace manual wrapper predicates with a specialization registry keyed by
   concrete problem classes, for example:
   - `(arch, dtype, q_seq, kv_seq, head_dim, causal, mode, gqa_class)`
3. Seed the registry with the already validated one-H100 variants:
   - `(128, 64, non-causal)`
   - `(768, 64, non-causal)`
   - the existing causal-64 family once wired through the same path
4. Move the benchmark harness onto that runtime surface so benchmark rows report
   both timing and dispatch outcome (`native`, `fallback`, `unsupported`).
5. Expand specializations in model order, not arbitrary shape order:
   - `headDim=256` full-sequence first
   - then GQA
   - then decode/KV-cache
   - then sliding-window/full-attention mixtures

Implementation status on 2026-04-20:

- The first operator-facing `AttentionProblem` layer is now in tree at
  [Tyr/GPU/Ops/AttentionProblem.lean](/grid/zador/home/pehle/dev/tyr/Tyr/GPU/Ops/AttentionProblem.lean).
- It records runtime attention metadata and centralizes the current
  specialization decision instead of leaving fixed-shape predicates duplicated
  across wrappers.
- The current selector is intentionally conservative and only exposes the
  already validated one-H100 native paths:
  - `tkMhaH1002Block`
  - `tkMhaH10012Block`
  - otherwise `portable`
- [MhaH100.lean](/grid/zador/home/pehle/dev/tyr/Tyr/GPU/Ops/MhaH100.lean)
  now dispatches through `AttentionProblem.currentSpecialization` and stores the
  selected specialization in its forward context.
- [FlashAttn.lean](/grid/zador/home/pehle/dev/tyr/Tyr/GPU/Ops/FlashAttn.lean)
  now constructs the same `AttentionProblem` descriptor instead of maintaining a
  separate hand-written shape predicate.
- Because the dedicated C++ op registration still does not exist, the current
  `torch.nn.tyrFlashAttn4d` symbol is temporarily implemented as a Lean-side
  fallback wrapper in [Tyr/Torch.lean](/grid/zador/home/pehle/dev/tyr/Tyr/Torch.lean).
  That keeps the wrapper layer buildable and makes the remaining missing piece
  explicit: the next step is still the real runtime bridge in `cc/src`.

#### Why This Feels Harder Than The ThunderKittens Demos

ThunderKittens itself does have a fairly clean integration story, but it is a
different story from "fully generic operator compiler".

- The ThunderKittens demos and PyTorch entrypoints expose a stable user-facing
  API and then launch a compiled specialization behind that API.
- The hard part is not the raw kernel call. The hard part is choosing the right
  specialization, handling fallback, and matching model semantics such as GQA,
  decode-time KV cache, and sliding-window attention.
- Tyr is currently paying extra complexity because the public wrapper layer
  still exposes fixed-shape typed variants directly instead of hiding them
  behind one runtime attention operator with specialization dispatch.

So the practical conclusion is:

- ThunderKittens is not the main blocker.
- The blocker is that Tyr still needs the same thin runtime operator layer that
  ThunderKittens-style integrations normally place above the kernels.
- Once that layer exists, adding or benchmarking new compiled variants should
  become much more mechanical.

#### How To Choose The "Right" Fixed Specializations

The right question is not "which exact fixed shapes should become the public
API?" The right question is "which problem axes materially change kernel
structure and therefore deserve specialization?"

For attention, those axes are usually:

- architecture: `SM90` vs future targets,
- dtype / accumulator dtype,
- attention mode:
  - dense prefill,
  - decode (`q_len = 1`),
  - sliding-window,
  - distributed/ring variants later,
- head-dimension class:
  - e.g. `64`, `128`, `256`, `512`,
- causal vs non-causal,
- grouped-query structure:
  - whether `num_q_heads == num_kv_heads`,
  - if not, the ratio class such as `2:1`, `4:1`, `8:1`,
- tile/block geometry:
  - tile sizes, warp layout, staging depth, shared-memory strategy.

By contrast, these should generally stay runtime values whenever possible:

- total `q_seq`,
- total `kv_seq`,
- batch size,
- exact number of heads.

That is especially important for large context lengths. If Tyr specializes on
exact full sequence length, the specialization matrix will explode and the
runtime surface will become brittle. For long-context support, the kernel should
usually specialize on:

- tile geometry,
- head-dimension class,
- mode,
- causal/GQA class,

while iterating over sequence tiles at runtime with tail masking for the final
partial tile.

In other words:

- good specialization: `SM90 + bf16 + dense_prefill + headDim=256 + causal +
  gqa_ratio=4 + tile=64`
- bad specialization: `SM90 + bf16 + qSeq=32768 + kvSeq=32768 + heads=16`

Exact-sequence specializations still have a place, but only when the algorithm
really depends on them:

- current bring-up/demo kernels with fixed block counts,
- tiny fixed-shape kernels used for validation,
- cases where full unrolling or staging decisions genuinely depend on exact
  small problem size.

They should not be the default strategy for scaling to large context lengths.

#### Practical Selection Rule For Tyr

Until there are production traces, Tyr should choose its first native
specializations from the model code already in tree and keep the matrix small.

1. Choose specialization families from model semantics:
   - `Qwen`: dense causal prefill, GQA, decode/KV-cache
   - `Gemma4`: sliding-window + periodic full attention
2. Choose head-dimension classes from model configs:
   - `256` first
   - `512` only when global-attention kernels are in scope
3. Choose GQA classes from model ratios:
   - equality case
   - `q_heads / kv_heads = 4`
   - `q_heads / kv_heads = 8` if needed by a concrete target model
4. Keep `q_seq` / `kv_seq` runtime-driven unless a kernel truly requires fixed
   loop counts.
5. Admit a new specialization only if it is both:
   - required by a target model path, and
   - measurably better than the portable or less-specialized path.

That gives a cleaner first internal matrix:

- `dense_prefill`, `headDim=256`, causal, equality/GQA
- `decode`, `headDim=256`, causal, KV-cache
- `sliding_window`, `headDim=256`, causal, window `{512,1024}` if Gemma paths
  need both

The current `headDim=64`, `seq in {128,768}` kernels remain valuable as:

- bring-up references,
- benchmarking scaffolding,
- proof that the runtime operator path works end-to-end,

but they should not define the long-term specialization policy.

#### Measurement Protocol

- Use one visible GPU only:
  - `CUDA_VISIBLE_DEVICES=0`
- Fix the module/toolchain stack in the result metadata:
  - `load_modules.sh` defaults
  - `nvcc --version`
  - `nvidia-smi` GPU/driver snapshot
- Fix input randomness:
  - one seed per benchmark row
  - reuse the same generated tensors across all compared backends
- For every reported runtime number:
  - 1 untimed cold start
  - 20 warmup iterations for stable runs
  - 100 to 500 timed iterations depending on runtime
  - 3 repeated benchmark runs
  - report the median of medians as the headline number
- Synchronize explicitly before reading timers.
- Keep correctness and performance modes separate:
  - correctness mode may use stricter references and fewer iterations
  - performance mode should avoid extra host-side checks inside the timed loop

#### Harness Design

- Add a dedicated runner rather than overloading the generalized op wrapper too
  early.
- Immediate preferred shape:
  - `Examples/GPU/RunFlashAttnBench.lean`
- Required backend modes:
  - `tyr_tk_direct`
  - `torch_sdpa`
  - `host_best_flash`
- Required output formats:
  - human-readable summary table on stdout
  - machine-readable `jsonl` or `csv` row per `(backend, shape, repeat)`
- Required metadata per row:
  - git commit
  - hostname
  - GPU model
  - driver version
  - CUDA toolkit version
  - module stack
  - shape, dtype, causal flag
  - backend requested
  - backend actually executed

#### Backend-Generalization Plan

- The next architectural priority is not just "more benchmark rows"; it is
  making the TK/codegen backend accept varying `seq` and `headDim` through one
  stable calling convention.
- Current state in-tree:
  - [AttentionFactory.lean](/grid/zador/home/pehle/dev/tyr/Tyr/GPU/Kernels/AttentionFactory.lean)
    already carries `seq`, `headDim`, `kvBlocks`, `tileSize`, `dtype`,
    `accDtype`, `scale`, and `isCausal` in `FAVariantConfig`,
  - but [MhaH100.lean](/grid/zador/home/pehle/dev/tyr/Tyr/GPU/Ops/MhaH100.lean)
    still publishes only a small fixed set of runtime variants (`seq=128` and
    `seq=768`, `headDim=64`),
  - and [FlashAttn.lean](/grid/zador/home/pehle/dev/tyr/Tyr/GPU/Ops/FlashAttn.lean)
    still classifies coverage with a hard-coded predicate before calling a
    torch-registered runtime op.
- Target direction, taking inspiration from ThunderKittens' PyTorch-side call
  pattern:
  - expose one stable runtime entrypoint that accepts runtime tensor metadata
    plus semantic flags (`causal`, `scale`, optional mask, maybe GQA mode),
  - lower that to a `globals`-style runtime descriptor,
  - dispatch to the best compiled specialization from a registry of
    `(seq, headDim, causal, dtype, arch)` variants,
  - make fallback selection explicit when no native specialization exists.
- This also matches how the larger frameworks tend to work:
  - PyTorch exposes stable high-level ops and then selects among specialized
    kernels/backends at runtime; `torch.compile` adds graph specialization and
    recompilation/guarding when needed,
  - JAX traces and lowers once per program shape/signature family, then
    compiles specialized executables per concrete shape/sharding; custom kernel
    routes such as Pallas still fit that specialization model rather than a
    single universally-generic kernel.
- Concretely, that means the benchmark harness should be built on top of the
  same future runtime surface we want NanoChat and the generalized ops to use,
  not on a permanently separate one-off interface.
- Practical implementation order:
  - first keep the direct one-H100 path as the correctness/perf baseline,
  - then build a runtime specialization registry over the existing factory +
    hand-written kernels,
  - then route the benchmark harness through that registry,
  - only after that widen the benchmark matrix to claim broader shape support.

#### Acceptance Criteria

- `Phase 0 gate`
  - all Tier-0 rows pass correctness
  - no NaN/Inf
  - training step reduces loss on the fixed 10-step sanity run
- `Phase 1 gate`
  - one command produces comparable rows for Tyr TK, PyTorch SDPA, and the
    strongest supported host backend on the same exact shape
  - unsupported backends are reported explicitly, not skipped silently
- `Phase 2 gate`
  - each newly-added native shape lands with both parity and runtime rows
- `Phase 3 gate`
  - the harness can report native coverage vs fallback coverage vs failure
    coverage for the broader shape matrix

## Current Bring-Up Notes

- The temporary `Tyr.GPU.Codegen.Ops` compatibility shim was removed again once
  the source tree no longer imported it.
- The concrete Lean-native blocker on 2026-04-20 was not a generic stale-cache
  issue; it was a package-prefix mismatch in generated native initializer
  symbols (`initialize_Tyr_GPU_Codegen_Monad` vs
  `initialize_tyr_Tyr_GPU_Codegen_Monad`) exposed by redundant direct
  `Monad` imports in:
  - `Tyr.GPU.Codegen.Attribute`
  - `Tyr.GPU.Kernels.Prelude`
- The current workaround is to route those modules through
  `Tyr.GPU.Codegen.Primitives` and rebuild the affected `:dynlib` facets.
- The remaining build-system blockers on this host turned out to be runtime
  linkage, not GPU codegen:
  - the same Lake `moreLinkArgs` are used for both
    `.lake/build/lib/lean/*.so` and `cc/build/libTyrC.so`,
  - a single relative LibTorch RPATH cannot be correct for both output
    locations,
  - the Linux link args now use an absolute repo-local LibTorch RPATH instead.
- The current cluster-specific compiler/linker constraint is:
  - Lean's bundled `clang` cannot relink native shared libraries on this host
    because of a glibc mismatch,
  - plain GCC cannot satisfy Lean's static `libc++` / `libc++abi` / `libgmp` /
    `libuv` link expectations by itself,
  - `scripts/lean_cc_wrapper.sh` remains necessary as a compatibility bridge
    for `LEAN_CC`, even though the build flow itself is otherwise the standard
    direct `lake` path.
- The current build-time bottleneck is not Lean elaboration anymore; it is the
  final link step for large executables such as `GenerateGpuKernels`,
  `RunFlashAttn`, and `RunMhaH100Train` once the broad graph has already been
  replayed.
- The default one-H100 direct path now works end-to-end under:
  - `CUDA/12.9.1`
  - no NCCL module by default
  - `lake -R env ./.lake/build/bin/GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 --out-dir cc/src/generated`
  - `make -C cc -B build/generated/Tyr_GPU_Kernels_MhaH100.o build/libTyrC.a build/libTyrC.so -j"$(nproc)"`
- The `cc` build needed one additional separation between one-GPU and
  distributed paths:
  - `TORCH_CUDA_LINK_FLAGS` stay enabled whenever LibTorch ships `torch_cuda`
    and `c10_cuda`,
  - `USE_C10D_NCCL` is now enabled only when an NCCL module is actually loaded
    (`EBROOTNCCL` present), which prevents one-GPU builds from failing on
    missing `nccl.h`.
- Validated one-H100 results after the CUDA/NCCL cleanup:
  - `RunFlashAttn --regen`:
    `fwd_ok=true`, `fwd_lse_ok=true`, `lse_ok=true`,
    `out_mae=0.000151`, `out_max=0.003906`
  - `RunMhaH100Train --steps 10 --log-every 5 --lr 200.0 --noise 0.5`:
    `init_loss=0.010400`, `final_loss=0.006443`,
    `rel_improvement=0.380499`, `ok=true`
  - `RunMhaH100Train --benchmark --warmup 5 --bench-iters 20 --lr 200.0 --noise 0.5`:
    `kernel_ms_per_step=0.157652`,
    `torch_ms_per_step=0.220110`,
    `kernel_speedup_vs_torch=1.396180`

### Stage 3: NanoChat integration

- Route compatible attention calls in `Examples/NanoChat/ModdedGPT.lean`
  through the GPU op wrapper path with explicit fallback.
- Keep unsupported shapes/features on the portable SDPA route.
- Add one-run fallback reason reporting and parity tests at block level.

### Stage 4: Scale out to 4xH100

- After one-H100 correctness/perf gates pass, run distributed training:
  - `NPROC_PER_NODE=4 ./scripts/nanochat/run_train_torchrun.sh ...`
- Compare 1/2/4 GPU throughput and convergence behavior.
- Keep the one-H100 benchmark as the release gate for kernel regressions.

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
