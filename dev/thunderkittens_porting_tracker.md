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

## Training Integration Log (H100-First)

### 2026-04-20

Objective shift for the next execution block:

- keep kernel-source parity work intact,
- prioritize end-to-end training readiness on H100,
- start on **one H100** before 4xH100 scaling,
- use a concrete intermediate benchmark goal: **SoTA FlashAttention vs Tyr
  ThunderKittens FlashAttention**.

Current status:

| Workstream | Status | Notes |
| --- | --- | --- |
| One-H100 fixture and train-step benchmark baseline | `in progress` | Existing entrypoints: `scripts/gpu/test_flashattn_e2e.sh`, `scripts/gpu/test_mha_h100_e2e.sh`, `scripts/gpu/bench_mha_h100_train.sh`. |
| Module loader stabilization | `in progress` | `load_modules.sh` now stays non-interactive for scripted runs and allows `TYR_CUDA_MODULE` / `TYR_NCCL_MODULE` overrides. |
| Hopper toolkit compatibility | `in progress` | Default path now uses `CUDA/12.9.1`; the old cluster-dimension compatibility define in `cc/Makefile` was removed after a direct `nvcc` validation on the generated H100 kernel. |
| SoTA-vs-Tyr FlashAttention benchmark harness | `pending` | Need a single command runner that reports parity + throughput for identical shapes/dtypes. Current blocker: `Tyr.GPU.Ops.FlashAttn` calls `torch.nn.tyrFlashAttn4d`, but the matching runtime registration does not appear to exist yet. |
| NanoChat attention integration to GPU op wrappers | `pending` | `Examples/NanoChat/ModdedGPT.lean` still calls SDPA directly (`nn.scaled_dot_product_attention` / `scaledDotProductAttentionGQAWindow`). |
| 4xH100 distributed training rollout | `pending` | Blocked on one-H100 correctness + benchmark gates. |

Immediate next gates:

1. Green one-H100 e2e fixture runs for FlashAttention/MHA.
2. Reproducible one-H100 benchmark output with kernel-vs-reference speed/accuracy.
3. Land FlashAttention benchmark harness comparing Tyr TK against SoTA backend.
4. Wire NanoChat attention through wrapper dispatch + fallback diagnostics.
5. Only then enable 4xH100 throughput/convergence runs.

Systematic benchmark plan:

1. Keep three benchmark families separate:
   - forward parity + throughput for FlashAttention itself,
   - train-step benchmark for practical end-to-end value,
   - kernel-generation generality sweep for native/fallback/fail coverage.
2. Start with exact-supported native shapes only:
   - `batch=1`, `heads=1`, `dtype=bf16`, `causal=false`
   - `(seq=128, headDim=64)` first
   - `(seq=768, headDim=64)` next once the direct runner is folded into the
     same harness.
3. Compare three backend classes on identical tensors:
   - `tyr_tk_direct`
   - `torch_sdpa`
   - `host_best_flash`
   If the strongest host backend cannot run the exact shape, record
   `unsupported`; do not silently substitute a different backend.
4. Require the harness to emit both human-readable and machine-readable output:
   - stdout summary table
   - `jsonl` or `csv` row per `(backend, shape, repeat)`
   - metadata must include git revision, GPU, driver, CUDA version, module
     stack, requested backend, executed backend, shape, dtype, and causal flag.
5. Fix the measurement protocol:
   - `CUDA_VISIBLE_DEVICES=0`
   - fixed RNG seed per row
   - same tensors reused across backends
   - 1 cold start
   - 20 warmup iterations
   - 100 to 500 timed iterations
   - 3 repeated runs
   - median of medians as the headline number
6. Treat generality as a measured workstream, not a side comment:
   - broader sweep over `seq in {64,128,256,512,768,1024,2048}`,
     `headDim in {64,128}`, `causal in {false,true}`,
     `batch in {1,4}`, `heads in {1,8}`
   - each row must log `native`, `fallback`, or `fail`, plus the failure reason.
7. Use explicit phase gates:
   - `Phase 0`: Tier-0 correctness + one-H100 benchmark sanity
   - `Phase 1`: SoTA comparison for exact-supported shapes
   - `Phase 2`: broaden to new native shapes only after they exist
   - `Phase 3`: full generality sweep and coverage reporting
8. Treat backend generality as the main architecture task, not a later cleanup:
   - `AttentionFactory` already exposes a parameterized configuration surface
     (`seq`, `headDim`, `kvBlocks`, `tileSize`, `dtype`, `isCausal`, etc.),
   - the published op/wrapper layer still hard-codes a few variants
     (`seq=128/768`, `headDim=64`) behind manual predicates,
   - the next backend milestone is a ThunderKittens-style runtime entrypoint:
     one stable call signature, runtime tensor metadata packed into a
     `globals`-like descriptor, and specialization dispatch through a registry
     of compiled variants,
   - this is also the pattern used by larger systems:
     PyTorch keeps a stable operator surface and selects specialized kernels at
     runtime, while JAX traces/lowers and then caches specialized executables
     per concrete signature,
   - benchmarking should move onto that runtime surface as soon as it exists so
     the harness measures both performance and actual backend generality.

Model-driven backend priorities after inspecting the current tree:

- Current architectural gap:
  - the kernel factory layer is already parameterized,
  - the public wrapper layer is still effectively a fixed-shape demo:
    `MhaH100.lean` hard-codes `[1,1,seq,64]` and manual `seq=128/768`
    dispatch,
  - `FlashAttn.lean` likewise only advertises native TK coverage for
    `batch=1`, `heads=1`, `kv_heads=1`, `headDim=64`, non-causal, and
    `seq in {128, 768}`.
- ThunderKittens reference pattern:
  - the vendored PyTorch entrypoints construct a runtime descriptor and launch
    compiled specializations behind a stable API (`mha_forward`, `mha_backward`)
    instead of publishing one wrapper per fixed kernel shape.
- Model pressure from code already in-tree:
  - `Qwen` attention uses grouped-query attention and decode-time KV cache,
  - `Qwen3.5` configs commonly need `head_dim=256` with
    `num_attention_heads != num_key_value_heads`,
  - `Gemma4` configs also center on `head_dim=256`, sometimes
    `global_head_dim=512`, and mix sliding-window with periodic full attention.
- Resulting milestone order:
  1. introduce a stable runtime `AttentionProblem` descriptor and specialization
     registry,
  2. seed it with the already validated `headDim=64` H100 variants,
  3. add a first `headDim=256` dense specialization,
  4. add grouped-query support,
  5. add decode/KV-cache support,
  6. add sliding-window/full-attention routing.
- Benchmark implication:
  - the benchmark harness should migrate onto that registry-backed runtime
    surface as soon as possible,
  - otherwise it will only measure one narrow fixed-shape demo path and will
    not tell us whether backend generality is improving.

Integration complexity note:

- ThunderKittens already shows the intended integration pattern:
  stable public entrypoints, runtime metadata packing, and dispatch into a
  compiled specialization.
- The main difficulty in Tyr is not "calling a TK kernel". It is that the
  current op layer still publishes a few fixed-shape typed variants directly,
  so specialization and dispatch concerns leak into the public API.
- The correct simplification is to move those concerns behind a single runtime
  attention operator and let the operator own specialization selection and
  portable fallback.

Specialization-selection rule for the next backend phase:

- Do not treat exact full sequence length as the main specialization axis for
  long-context support.
- Specialize on the axes that change kernel structure:
  - mode (`dense_prefill`, `decode`, `sliding_window`)
  - head-dimension class
  - causal flag
  - GQA class / `q_heads : kv_heads` ratio class
  - tile/block geometry
  - dtype / arch
- Keep these runtime when possible:
  - `q_seq`
  - `kv_seq`
  - batch
  - exact head count
- Exact-sequence fixed variants are acceptable only for:
  - current bring-up kernels,
  - tiny validation kernels,
  - genuinely fully-unrolled small kernels where exact size changes code
    structure.
- For Tyr's model-driven roadmap, the first real specialization family should
  be `headDim=256` rather than multiplying more `headDim=64` exact-sequence
  wrappers.

Implementation update:

- Landed the first runtime attention descriptor at
  `Tyr/GPU/Ops/AttentionProblem.lean`.
- It now owns:
  - runtime metadata (`batch`, `q_seq`, `kv_seq`, heads, `headDim`, dtype,
    device, mode, mask kind, causal flag, scale, GQA flag),
  - coarse planning helpers (`headDimClass`, `gqaClass`, `inferMode`),
  - the current conservative specialization selector.
- Current selector output is:
  - `tkMhaH1002Block`
  - `tkMhaH10012Block`
  - `portable`
- `Tyr.GPU.Ops.MhaH100` now dispatches through that selector instead of manual
  `seq == 128 || seq == 768` branching, and the chosen specialization is stored
  in `FwdCtxDispatch`.
- `Tyr.GPU.Ops.FlashAttn` now constructs the same `AttentionProblem` descriptor
  for its eligibility logic instead of maintaining a separate duplicated shape
  predicate.
- Because the dedicated `tyr::flash_attn` C++ registration is still absent,
  `torch.nn.tyrFlashAttn4d` is temporarily implemented as a Lean-side fallback
  wrapper in `Tyr/Torch.lean`.
- This clarifies the remaining split:
  - operator surface and selection logic now exist,
  - the missing next step is the real C++ runtime bridge that executes native
    kernels and falls back portably behind the same API.

Execution log additions:

- `load_modules.sh` was hardened for non-login automation shells:
  - if `module` is missing, it now attempts to source
    `/etc/profile.d/modules.sh` or `/usr/share/lmod/lmod/init/bash`
    before falling back to a no-op.
  - existing override knobs remain:
    `TYR_ARROW_MODULE`, `TYR_CUDA_MODULE`, `TYR_NCCL_MODULE`.
- Hopper toolkit compatibility fix landed in codegen:
  - Hopper/H100 generated C++ now guards `cuda_fp8.h` behind
    `KITTENS_HOPPER || KITTENS_BLACKWELL`,
  - `cuda_fp4.h` is only included for `KITTENS_BLACKWELL`.
- The temporary `Tyr.GPU.Codegen.Ops` compatibility shim was removed again
  after confirming the source tree no longer imports it.
- One concrete Lean-native failure mode is now identified:
  - after source edits, `lake build <module>` may refresh `.olean` / generated C
    while leaving the corresponding native plugin (`:dynlib`) stale,
  - symptom: `.so` still exports or requires
    `initialize_Tyr_GPU_Codegen_Monad` even after the regenerated `.c` uses the
    correct `initialize_tyr_Tyr_GPU_Codegen_*` symbols.
- Current mitigation in tree:
  - avoid redundant direct `import Tyr.GPU.Codegen.Monad` in the user-facing
    integration modules that trigger this mismatch,
  - route through `Tyr.GPU.Codegen.Primitives` in
    `Tyr.GPU.Codegen.Attribute`, `Tyr.GPU.Kernels.Prelude`,
    `Tyr.GPU.Kernels.FlashAttnCausal64`, and `Tyr.GPU`,
  - rebuild native plugins explicitly with `:dynlib` targets when needed.
- Current bring-up state:
  - `Tyr.GPU.Codegen.Attribute` and `Tyr.GPU.Kernels.Prelude` native plugins now
    rebuild cleanly with the correct package-prefixed initializer names,
  - `Examples.GPU.RunFlashAttn` has advanced past the earlier immediate native
    loader crash and is now in the broad native shared-object build phase.
  - The next blocker after that was `GenerateGpuKernels:exe` linking against a
    stale `Attribute.c.o.export` that still referenced helper-generated
    `buildKernelM` / `setArch` symbols.
  - The unused convenience layer at the bottom of
    `Tyr.GPU.Codegen.Attribute` (`buildGpuKernel`, `gpu_kernel` command macro,
    `GpuKernelFn`, `kernel`) has been removed, and `Attribute:o` was rebuilt so
    those stale unresolved references are no longer present in the object file.

### 2026-04-21

Runtime bridge update:

- landed `cc/src/tyr_ops.cpp` as the real C++ runtime bridge for
  `tyr::flash_attn`,
- added `tyr_ops.cpp` to `cc/Makefile`,
- removed the temporary Lean implementation of `torch.nn.tyrFlashAttn4d`,
  replacing it with the real extern
  `lean_torch_tyr_flash_attn_4d` in `Tyr/Torch.lean`,
- added `Examples/GPU/RunFlashAttnOp.lean` as a focused runtime smoke test,
- added `RunFlashAttnOp` to `lakefile.lean`.

Current dispatch behavior in the new C++ op:

- native exact route:
  - `bf16`
  - CUDA
  - dense self-attention
  - `batch=1`
  - `q_heads=kv_heads=1`
  - `headDim=64`
  - non-causal
  - no mask
  - `seq in {128, 768}`
- portable fallback:
  - everything else,
  - including causal mode and GQA.

Validation result from `Examples/GPU/RunFlashAttnOp.lean`:

- `flash_attn_native_dense`
  - route selection is correct: `route=tkKernel`
  - forward matches well:
    - `out_mae=0.000057`
    - `out_max=0.001953`
  - backward is not yet within the current tolerance for all components:
    - `dq_ok=true`
    - `dk_ok=false`
    - `dv_ok=false`
    - one observed run:
      - `dk_mae=0.003548`, `dk_max=0.363281`
      - `dv_mae=0.004895`, `dv_max=0.550781`
- `flash_attn_portable_causal`
  - route selection correct: `route=portable`
  - forward and all gradients matched exactly
- `flash_attn_portable_gqa`
  - route selection correct: `route=portable`
  - forward and all gradients matched exactly

Conclusion from the current bridge state:

- the operator surface is now real,
- forward benchmarking can proceed on the native route,
- portable fallback is good enough to support broader model shapes without
  crashing,
- native backward on the exact H100 path still needs work before the new
  high-level op should be used as the default training path.

Additional build-system observation:

- a broad `lake -R build RunFlashAttnOp` surfaced an unrelated package-native
  failure:
  - `.lake/build/lib/lean/tyr_Tyr_GPU_Codegen_FFI.so` appeared as `file too short`
  - later, `tyr_Tyr_GPU_Codegen_TileIR_Frontend.so` failed to materialize
- the narrower interpreter path with explicit dynlib loading was sufficient to
  validate the attention bridge while that package-wide native-plugin issue
  remains open.
  - Direct `lake build` on this host now has a more precise shape:
    - environment setup should stay minimal:
      `source ./load_modules.sh && LEAN_CC=$PWD/scripts/lean_cc_wrapper.sh ...`
    - the preferred non-wrapper build path is:
      `lake -R build GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 RunFlashAttn`
      followed by direct `GenerateGpuKernels`, `make -C cc`, and `RunFlashAttn`.
  - The runtime-linking failure mode was narrowed down further:
    - Lake reuses the same package `moreLinkArgs` for both
      `.lake/build/lib/lean/*.so` and `cc/build/libTyrC.so`,
    - the previous relative LibTorch RPATH could not be correct for both output
      trees at once,
    - switching Linux link args to an absolute repo-local LibTorch RPATH fixed
      the direct `libtorch.so` lookup failures in both the Lean dynlib path and
      `cc/build/libTyrC.so`.
  - The current cluster toolchain constraint is explicit:
    - Lean's bundled `clang` cannot relink native shared libraries here because
      its LLVM/libclang stack expects a newer glibc than the node provides,
    - plain GCC is not sufficient because Lake's Lean link flags still expect
      Lean-shipped `libc++`, `libc++abi`, `libgmp`, and `libuv`,
    - the existing `scripts/lean_cc_wrapper.sh` remains the necessary bridge
      behind `LEAN_CC`; this is compatibility glue, not an additional custom
      build flow.
  - Validation milestone reached:
    - `source ./load_modules.sh && LEAN_CC=$PWD/scripts/lean_cc_wrapper.sh lake -R build Tyr.GPU.Codegen.EmitNew`
      now completes successfully end-to-end,
    - this confirms the current blocker has moved from toolchain/runtime
      resolution into the larger `RunFlashAttn` dependency closure and runtime
      execution itself.
  - A later replayed build exposed a generic backend issue in
    `Tyr.GPU.Codegen.Primitives` itself:
    - `complexMma` / `complexMmaT` were calling `mma` / `mmaT` without
      forwarding the tensor-core divisibility proofs,
    - on Lean 4.29 this no longer elaborates when the dimensions are still free
      variables, so `Tyr.GPU.Codegen.Primitives` failed late in the
      `RunFlashAttn` dependency graph,
    - forwarding `hM`, `hK`, and `hN` explicitly fixed the module, and
      `lake -R build Tyr.GPU.Codegen.Primitives +Tyr.GPU.Codegen.Primitives:dynlib`
      now succeeds again.
  - The one-H100 canonical workflow is now documented as direct executables:
    - prefer `lake -R env ./.lake/build/bin/RunFlashAttn --regen` for fixture
      validation after `GenerateGpuKernels` + `make -C cc`,
    - prefer `lake -R env ./.lake/build/bin/RunMhaH100Train --benchmark ...`
      for the training-vs-portable benchmark,
    - keep `scripts/gpu/test_flashattn_e2e.sh` and
      `scripts/gpu/bench_mha_h100_train.sh` only as convenience wrappers.
  - The next concrete iteration-speed lesson on this host is:
    - avoid the broad
      `lake -R build GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 RunFlashAttn RunMhaH100Train`
      loop while fixing proof/elaboration failures,
    - it replays a large dependency graph and then spends minutes in final
      binary links,
    - the faster loop is targeted `lake -R build <module> +<module>:dynlib`
      until the failing helper compiles, followed by only the runner actually
      needed for the next test.
  - Module availability was broader than the initial notes assumed:
    - `module spider` on 2026-04-20 shows `CUDA/12.9.1`, `CUDA/13.1.0`, and
      `NCCL/2.27.7-GCCcore-14.3.0-CUDA-12.9.1` on this host,
    - a direct `nvcc` compile of `cc/src/generated/Tyr_GPU_Kernels_MhaH100.cu`
      to `/tmp/tyr_mha_h100_cuda1291.o` succeeded under `CUDA/12.9.1` without
      the old
      `cudaLaunchAttributePreferredClusterDimension ->
      cudaLaunchAttributeClusterDimension` define,
    - that compatibility define has therefore been deleted from `cc/Makefile`,
    - `load_modules.sh` now defaults to `CUDA/12.9.1` for the one-H100 path.
  - `load_modules.sh` now treats `TYR_NCCL_MODULE` as optional:
    - leaving it unset now defaults to no NCCL module for the one-GPU path,
    - setting `TYR_NCCL_MODULE=` cleanly skips NCCL for the one-GPU direct
      path, which reduces module churn while validating newer CUDA versions.
  - The first standard direct one-H100 path now works under the new defaults:
    - `source ./load_modules.sh`
    - `lake -R env ./.lake/build/bin/GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 --out-dir cc/src/generated`
    - `make -C cc -B build/generated/Tyr_GPU_Kernels_MhaH100.o build/libTyrC.a build/libTyrC.so -j"$(nproc)"`
    - note: invoking `./.lake/build/bin/GenerateGpuKernels` without
      `lake -R env` still fails with `unknown module prefix 'Tyr'`, so the Lake
      environment wrapper remains part of the canonical direct flow.
  - The `cc` build needed one more generality fix after NCCL became opt-in:
    - `TORCH_CUDA_LINK_FLAGS` stay enabled when LibTorch ships the CUDA libs,
    - `USE_C10D_NCCL` is now gated on `EBROOTNCCL`,
    - this separates the one-GPU CUDA path from the multi-GPU distributed path
      and avoids `nccl.h` failures when NCCL is intentionally absent.
  - One-H100 validation results after the CUDA/NCCL cleanup:
    - `RunFlashAttn --regen` passed with
      `out_mae=0.000151`, `out_max=0.003906`, `lse_mae=0.0`
    - `RunMhaH100Train --steps 10 --log-every 5 --lr 200.0 --noise 0.5`
      reduced loss from `0.010400` to `0.006443`
    - `RunMhaH100Train --benchmark --warmup 5 --bench-iters 20 --lr 200.0 --noise 0.5`
      reported `kernel_ms_per_step=0.157652`,
      `torch_ms_per_step=0.220110`,
      `kernel_speedup_vs_torch=1.396180`

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
