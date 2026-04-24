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

### 2026-04-23

Performance-first checkpoint:

- [x] Rebuilt the vendored ThunderKittens H100 MHA extension for Python 3.12
  using:
  - `module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0`
  - `module load CUDA/12.9.1`
- [x] Confirmed CUDA 12.6 is not sufficient for this vendored TK source on this
  host:
  - the compile fails on `cudaLaunchAttributePreferredClusterDimension`,
  - no compatibility shim was added.
- [x] Added the vendored-TK baseline harness:
  - [benchmarks/bench_tk_mha_h100.py](/grid/zador/home/pehle/dev/tyr/benchmarks/bench_tk_mha_h100.py)
- [x] Switched [cc/tools/bench_flash_attn.cpp](/grid/zador/home/pehle/dev/tyr/cc/tools/bench_flash_attn.cpp)
  to CUDA-event timing for kernel-style comparison.
- [x] Captured benchmark artifacts:
  - [benchmarks/results/flash_attn_cpp_native_h100_cuda_event.jsonl](/grid/zador/home/pehle/dev/tyr/benchmarks/results/flash_attn_cpp_native_h100_cuda_event.jsonl)
  - [benchmarks/results/thunderkittens_mha_h100_cuda_event.jsonl](/grid/zador/home/pehle/dev/tyr/benchmarks/results/thunderkittens_mha_h100_cuda_event.jsonl)
- [x] Direct comparable row is now generated-vs-TK parity:
  - row: `native_dense_768x64`, BF16, non-causal, fwd+bwd, `B=1,H=1`
  - generated Tyr: `0.135669 ms` / `0.152279 ms` across opposite backend
    orders,
  - vendored ThunderKittens: `0.137159 ms` / `0.153591 ms`,
  - PyTorch SDPA: `0.170481 ms` / `0.187031 ms`.
- [x] Landed the codegen prerequisite for TK-style producer/store warp split:
  - the generated IR/emitter now supports current-warp async TMA issue points
    instead of always hardwiring `load_async` / `store_add_async` to CTA warp
    `0`,
  - `warpid` / `laneid` are now first-class DSL values,
  - this unblocks generated `MhaH100` from assigning different producer warps
    to prefetch vs `qg` store-add, which is required to match the vendored TK
    backward synchronization pattern.
- [x] Non-comparable row:
  - `native_dense_128x64` is kept for Tyr correctness smoke,
  - vendored TK does not provide a valid row there.

Gap list:

- [x] Generated Tyr bridge is correctness-green on the fixed native rows.
- [x] Vendored TK is correctness-green on the direct comparable `768x64` row.
- [x] Native C++ benchmark can run SDPA, generated Tyr, and vendored TK in one
  process.
- [x] Generated H100 forward now mirrors TK's shared staging, typed TMA coords,
  `warpgroup::laneid()` compute-done arrive, and final LSE sequence.
- [x] Dynamic shared-memory allocation and launch-time shared-memory attributes
  are present for generated Hopper kernels.
- [x] D=64 backward parity target is met for `dQ` / `dK` errors on the
  comparable row.
- [~] Remaining structural delta:
  - generated backward uses a generated prep plus K/V sweep path, while TK uses
    the original monolithic templated kernel shape.
- [ ] Extend specializations to `seq >= 768`, `seq % 256 == 0`,
  `headDim in {64,128}` before Qwen/Gemma GQA and causal variants.
- [ ] Produce a persisted joined benchmark report that lists generated Tyr,
  vendored TK, and SDPA rows side by side from the JSONL artifacts.

### 2026-04-22

Current working checklist:

- [x] Fix the runtime reduction contract for stacked `dK` / `dV` partials in
  `Tyr.GPU.Ops.MhaH100`.
- [x] Add raw-partial dump support to the direct H100 runners so we can inspect
  `dK` / `dV` before any reduction.
- [x] Add a raw-partial tile comparator:
  - `scripts/gpu/compare_mha_partial_tile.py`
- [x] Fix the nested codegen execution path in `lakefile.lean` so the normal
  build can run `GenerateGpuKernels` through `lake -R env`.
- [x] Narrow the GPU-codegen invalidation surface in `lakefile.lean` by
  fingerprinting:
  - `TYR_GPU_CODEGEN_MODULE`
  - `TYR_SKIP_GPU_CODEGEN`
  - `TYR_BUILD_TYRC_DYLIB`
- [x] Add a reproducible one-H100 benchmark scaffold:
  - `Examples/GPU/RunFlashAttnBench.lean`
  - `scripts/gpu/bench_flash_attn_matrix.sh`
- [x] Wire an initial `flash_attention` baseline into that scaffold using the
  repo-local FA3 kernel for the exact `1x1x256x64` forward-only row.
- [x] Fix the native backward shared-memory overrun by loading `l` / `d`
  directly into RV registers instead of staging them through an extra shared
  vector.
- [x] Rebuild and rerun a fresh native `RunMhaH100` binary through the manual
  trace-based relink path.
- [x] Confirm the raw-partial dump path works from the rebuilt native binary.
- [x] Extend the PyTorch raw-partial comparator to unwrap TorchScript-wrapped
  single-tensor fixture payloads.
- [x] Stabilize fixed-row H100 runtime gradients through the direct `dQ` plus
  KV-sweep backward split.
- [~] Reduce the broad Lake rebuild fanout so the standard native validation
  loop is practical.
- [x] Benchmark the current Tyr runtime against:
  - PyTorch SDPA
  on one H100 with machine-readable output for the native 128/768 x 64 rows.
- [ ] Add apples-to-apples repo-local FA3 rows for the same native training
  shapes once FA3 backward coverage is available.
- [ ] Broaden the native runtime beyond the current fixed `headDim=64`
  families.
- [ ] Route real model attention paths through the runtime operator:
  - Qwen
  - Gemma

Notes from today:

- Native backward shared-memory compile blocker is cleared in source:
  - `Tyr/GPU/Codegen/GlobalLayout.lean` now provides a direct RV global row
    load helper,
  - `Tyr/GPU/Kernels/MhaH100.lean` and
    `Tyr/GPU/Kernels/AttentionFactory.lean` now load `l` / `d` directly into
    RV registers instead of allocating an extra shared `SV<float, 64>`,
  - this removes the `0x100` shared-byte overrun that had pushed the generated
    `tkMhaH100Bwd*Partials` kernels above the H100 `0xc000` limit.
- Fresh native validation now exists through a manual rebuild path:
  - refreshed the relevant `.olean` files directly,
  - regenerated `cc/src/generated/Tyr_GPU_Kernels_MhaH100.cu`,
  - rebuilt the generated object and `cc/build/libTyrC.a`,
  - relinked `/tmp/RunMhaH100.manual` from the traced link line and ran it
    successfully with `--dump-partials`.
- Raw partial dumps are now confirmed working from the rebuilt native path:
  - `diag_dK_tiles.pt`
  - `diag_dV_tiles.pt`
- The PyTorch comparator now handles the fixture payload format we actually
  dump:
  - `scripts/gpu/compare_mha_partial_tile.py` unwraps TorchScript
    `RecursiveScriptModule` objects that contain a single tensor in the state
    dict.
- Current diagnosis from the rebuilt binary plus comparator:
  - forward, `l`, `dQ`, and `dV` are on the expected route,
  - `dK` still mismatches,
  - the mismatch is already present in the raw per-tile `dK` dumps before any
    q-block reduction, so reduction is not the leading bug anymore.
- Build-system status:
  - the broad Lake rebuild fanout remains unresolved,
  - `lake -R build RunMhaH100` still replays a much larger graph than this
    kernel loop should require on this machine,
  - the current reliable validation path is the manual trace-based relink, not
    the normal narrow `lake build` loop.

- The key build failure moved from “generator binary not executable” to the
  stricter Lake issue:
  - nested `lake env` was re-reading an invalid compiled config
  - switching the generator execution path to `lake -R env` fixes that class
    of failure
- A second host-specific executable issue then surfaced:
  - some `lean_exe` outputs under `.lake/build/bin` are being produced as
    sparse zero-filled files instead of valid ELF binaries
  - concrete examples seen today:
    - `GenerateGpuKernels`
    - `RunFlashAttnBench`
  - this is not a linker-command bug:
    - the exact link command recorded in `GenerateGpuKernels.trace` produces a
      valid ELF when the output path is moved to `/tmp`
  - current mitigation in `lakefile.lean`:
    - if a built executable under `.lake/build/bin` is invalid, relink it from
      its `.trace` into `/tmp/tyr_relinked/<ExeName>` and run the repaired copy
      from there
- The benchmark wrapper now uses the same `-R` path for:
  - native codegen
  - benchmark executable launch
- The benchmark surface is intentionally honest:
  - `flash_attention` is not advertised as generic yet
  - it is only wired for the exact repo-local FA3-supported row
  - all other rows should still report `unsupported`

### 2026-04-21

Attention backward bring-up update:

- The native exact-route `dK` / `dV` mismatch is no longer treated as a
  numerical mystery in the softmax/MMA math.
- The concrete mismatch source was the reduction contract:
  - Tyr `MhaH100` backward had been writing `[seq, seq]` scratch partials for
    `dK` / `dV` and reducing them later on the host/runtime side,
  - upstream ThunderKittens `mha_h100` writes final global gradients directly.
- The Lean/runtime path was changed to match the source kernel contract:
  - `Tyr/GPU/Kernels/MhaH100.lean`
  - `Tyr/GPU/Ops/MhaH100.lean`
  - `Examples/GPU/RunMhaH100.lean`
  - `Examples/GPU/RunMhaH100Seq768.lean`
  - `Examples/GPU/RunMhaH100Train.lean`
  - `cc/src/tyr_ops.cpp`
  now allocate final `dK` / `dV` tensors and use direct `storeGlobalAdd`
  accumulation.
- That exposed the next real backend issue:
  - ThunderKittens `warp::tma::store_add_async` requires the destination
    `gl<...>` descriptor to carry the matching shared-tile type,
  - Tyr codegen was still rendering those params as bare
    `gl<float, 1, 1, -1, -1>`.
- Codegen work landed to infer and render required TMA descriptor types from
  kernel IR:
  - `Tyr/GPU/Codegen/EmitNew.lean`
  - `Tyr/GPU/Codegen/Attribute.lean`
  - `Tyr/GPU/Codegen/FFI.lean`
- The clean generator path is still not fully proven:
  - `GenerateGpuKernels` is still emitting bare `gl<...>` in the current
    `MhaH100` generated backward translation unit,
  - a narrow local patch to
    `cc/src/generated/Tyr_GPU_Kernels_MhaH100.cu`
    adding `st<float, 64, 64>` descriptors to the two backward accumulation
    outputs unblocked `nvcc`,
  - `make -C cc -B build/generated/Tyr_GPU_Kernels_MhaH100.o build/libTyrC.so`
    completes successfully with that local patch.
- Remaining runtime-bridge blocker:
  - `Examples/GPU/RunFlashAttnOp.lean` is still blocked by the build state of
    `Tyr.GPU.Ops.FlashAttn` / its missing `.olean` in the current Lake graph,
  - so the post-fix parity smoke has not yet been rerun end-to-end.

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
    - after a fresh generated-kernel rebuild and relink of `cc/build/libTyrC.so`:
      - `dk_mae=0.002040`, `dk_max=0.363281`
      - `dv_mae=0.002166`, `dv_max=0.550781`
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
  high-level op should be used as the default training path,
- the remaining mismatch is not explained by stale bridge linkage:
  `nm -D cc/build/libTyrC.so` initially showed the `tkMhaH100*` launcher
  symbols as undefined until `cc/src/generated/Tyr_GPU_Kernels_MhaH100.cu`
  was regenerated and the shared library relinked, but the `dK` / `dV`
  parity failure remained after that real rebuild.

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

## 2026-04-21 Runtime Bridge Checkpoint

- Verified current one-H100 `tyr::flash_attn` runtime-op status with
  `Examples/GPU/RunFlashAttnOp.lean` after rebuilding `cc/build/libTyrC.so`:
  - `route=tkKernel`
  - `out_ok=true`
  - `dq_ok=true`
  - `dk_ok=true`
  - `dv_ok=true`
  - `out_mae=0.000057`
  - `dq_mae=0.000032`
  - `dk_mae=0.000046`
  - `dv_mae=0.000172`
  - `dv_max=0.003906`

- Source changes made to remove the hand-patched generated-CUDA dependency for
  the non-crashing backward accumulation path:
  - `Tyr/GPU/Kernels/MhaH100.lean`
    - changed `tkMhaH100Bwd2BlockPartials` / `tkMhaH100Bwd12BlockPartials` to
      write q-block-major stacked `dK`/`dV` partials with
      `stack_row = qBlock * kvBlocks + kvBlock`
      and plain `storeGlobal`
    - updated comments to describe stacked partial outputs instead of final
      in-place accumulation
  - `Tyr/GPU/Ops/MhaH100.lean`
    - allocate `dKStack` / `dVStack : [1, 1, kvBlocks * seq, 64]`
    - reduce with `reshape -> nn.sumDim -> nn.unsqueeze`
  - `Examples/GPU/RunMhaH100.lean`
  - `Examples/GPU/RunMhaH100Train.lean`
  - `Examples/GPU/RunMhaH100Seq768.lean`
    - same stacked-partials reduction change for raw launcher examples

- Bridge-side correctness fix:
  - `cc/src/tyr_ops.cpp`
    - keep kernel `dQ` and stacked `dK`
    - compute exact `dV` in FP32 in `native_backward` via:
      - `scores = Q K^T / sqrt(d)`
      - `probs = softmax(scores)`
      - `dV = probs^T dO`
  - This is a deliberate temporary bridge fix so the high-level runtime op is
    training-correct while the raw kernel-side `dK` mismatch stays isolated.

- Validation notes:
  - forced rebuild of `build/generated/Tyr_GPU_Kernels_MhaH100.o` and
    `build/libTyrC.so` succeeded
  - confirmed generated `cc/src/generated/Tyr_GPU_Kernels_MhaH100.cu` now emits
    stacked row coordinates:
    - 2-block: `(((qBlock * 2) + kvIdx) * 64)`
    - 12-block: `(((qBlock * 12) + kvIdx) * 64)`
  - raw launcher examples cannot be validated with `lean --run`; they must be
    exercised as compiled `lean_exe` targets because the interpreter cannot
    resolve generated launcher externs

- Remaining open item:
  - raw kernel-side `dK` tile math/layout still appears off in the standalone
    `tkMhaH100Bwd*Partials` path
  - raw tile dumps plus the PyTorch comparator now show this happens before the
    q-block reduction step
  - the runtime op remains training-correct on the bridged path because `dV`
    is recomputed exactly in the C++ bridge and `dQ` / `dV` parity is already
    in line on the native route

## 2026-04-22 Native Rebuild And Tile-Math Checkpoint

- Source-level compile fix:
  - removed the extra shared `SV<float, 64>` staging for `l` / `d` in the
    backward partial kernels,
  - replaced it with direct RV global loads so the generated CUDA fits inside
    Hopper shared-memory limits without a generated-file compatibility shim.
- Fresh native run:
  - the current authoritative binary is `/tmp/RunMhaH100.manual`,
  - it was produced from the traced link line after refreshing the relevant
    Lean artifacts and rebuilding the native archive,
  - it runs successfully with `--dump-partials`.
- Comparator status:
  - TorchScript-wrapped fixture tensors now load directly,
  - `dV` tiles compare correctly,
  - `dK` tiles mismatch in every inspected `(qBlock, kvBlock)` tile, which
    makes this a raw tile-math or layout issue rather than a reduction bug.
- TODO state:
  - [x] Fix the shared-memory overrun in the backward partial kernels.
  - [x] Rebuild and run a fresh native manual `RunMhaH100` binary.
  - [x] Verify that native raw partial dumps are emitted.
  - [x] Make the PyTorch comparator accept TorchScript fixture payloads.
  - [x] Identify the true root cause of the apparent `dK` failure on the 128x64 example path.
  - [~] Reduce the broad Lake rebuild fanout in the standard native build path.
  - [ ] Collapse the manual trace-based rebuild into a simpler normal build flow.
  - [ ] Resume one-H100 benchmark rows now that native 128x64 parity is closed.

## 2026-04-22 Output Buffer Aliasing Root Cause

- The earlier 128x64 `dK` failure diagnosis was wrong.
- Actual root cause:
  - Lean treated identical `torch.zeros ...` constructors for mutable output
    buffers as pure and merged them during code generation,
  - the compiled `RunMhaH100` call site passed the same tensor for
    `dK_part_ptr` and `dV_part_ptr`,
  - because the kernel writes `dK` first and `dV` second, the aliased stack
    ended the launch containing `dV`, which made `dv_ref_ok=true` and
    `dk_ref_ok=false` look like a kernel-side `dK` bug.
- Concrete evidence:
  - in `.lake/build/ir/Examples/GPU/RunMhaH100.c`, the generated backward
    launch used `x_549` for both `dK_part_ptr` and `dV_part_ptr`,
  - a direct trace-based relink of the compiled example binary made temporary
    debug routes take effect immediately, which ruled out stale generated CUDA
    as the explanation,
  - after restoring the intended kernel path and making the caller allocate
    distinct fresh `dK` / `dV` stacks, the relinked compiled binary reported:
    - `overall_ok=true`
    - `dq_ref_ok=true`
    - `dk_ref_ok=true`
    - `dv_ref_ok=true`
    - `dk_mae=0.000166`
    - `dv_mae=0.000153`
- Source fix:
  - `Examples/GPU/RunMhaH100.lean`
  - `Examples/GPU/RunMhaH100Train.lean`
  - `Examples/GPU/RunMhaH100Seq768.lean`
  - `Tyr/GPU/Ops/MhaH100.lean`
  - each now seeds the partial stacks once and builds `dK` / `dV` outputs from
    distinct `torch.mul_scalar` expressions so Lean cannot CSE them into one
    mutable tensor.
- Kernel cleanup:
  - `Tyr/GPU/Kernels/MhaH100.lean` was restored to the intended backward path
    after the temporary `V` / `dO` debug stores and the speculative `dK`
    contraction edits.
- Runtime-op status:
  - the relinked compiled `RunFlashAttnOp` bridge test is now green on the
    native dense route too:
    - `route=tkKernel`
    - `out_ok=true`
    - `dq_ok=true`
    - `dk_ok=true`
    - `dv_ok=true`
    - representative native-dense errors:
      - `out_mae=0.000057`
      - `dq_mae=0.000032`
      - `dk_mae=0.000046`
      - `dv_mae=0.000172`
- Consequence:
  - the current 128x64 one-H100 blocker is no longer backward correctness,
  - the next priority is benchmarking plus build-flow simplification, not more
    `dK` tile-math forensics on this path.

## 2026-04-22 Writeback Staging Investigation

- Cross-checked Tyr `MhaH100` backward against ThunderKittens `mha_h100`.
- Main result:
  - TK `dK` and `dV` use separate shared staging plus async TMA-store ordering.
  - Tyr `storeGlobal` lowers to blocking `warp::store`, so the useful local change is shared-buffer separation, not adding a new post-store wait.
- Kernel and source changes made:
  - `Tyr/GPU/Kernels/MhaH100.lean`
    - replaced the single FP32 writeback tile with separate `dKShared` and `dVShared` tiles in both 2-block and 12-block backward kernels
    - kept the backward tile math unchanged
  - `Tyr/GPU/Kernels/AttentionFactory.lean`
    - applied the same K/V writeback staging split to the generator template so regeneration preserves the fix direction
- Build and validation note:
  - a clean narrow rebuild was started via `lake -R build GenerateGpuKernels Tyr.GPU.Kernels.MhaH100 RunMhaH100 RunMhaH100Seq768`
  - earlier concurrent example-only builds had produced non-runnable `.lake/build/bin/RunMhaH100*` artifacts identified by `file` as plain `data`, so the current rebuild is intended to replace those with a single authoritative pass
  - the first long rebuild attempt also exposed a Lake race on `GenerateGpuKernels`:
    overlapping work can fail with a missing
    `.lake/build/bin/GenerateGpuKernels` and leave
    `cc/src/generated/Tyr_GPU_Kernels_MhaH100.cu` stale
  - `lakefile.lean` already carries the narrower raw-example helpers:
    - `TYR_BUILD_TYRC_DYLIB=0`
    - `buildMhaH100Examples`
    - `runMhaH100Exe`
    - `runMhaH100Seq768Exe`
    - `validateMhaH100Examples`
  - `buildNamedExecutables` now tries plain `lake build` first and retries
    with `lake -R build` only when the failure text looks like a reconfigure
    problem
  - `extern_lib libtyr` now narrows the GPU IR invalidation set specifically
    for the `Tyr.GPU.Kernels.MhaH100` loop instead of tracking every
    `.c.o.export` under `.lake/build/ir/Tyr/GPU`
  - `extern_lib libtyr` now also fingerprints
    `TYR_GPU_CODEGEN_MODULE`, `TYR_SKIP_GPU_CODEGEN`, and
    `TYR_BUILD_TYRC_DYLIB` into a small build-dir config file so those
    environment-driven build modes participate in invalidation
  - these do not eliminate the heavy Lake replay, but they define the correct non-interpreter validation loop and avoid some unnecessary native churn
- Remaining known limitation:
  - the stacked-partials reduction still only scales naturally for the current fixed-shape cases because `qBlocks == kvBlocks` for `128` and `768`; this must be generalized before claiming arbitrary-sequence support
- Additional TK gap checklist captured during this pass:
  - [x] Split `dK` and `dV` writeback staging in the current Tyr `MhaH100` path
  - [x] Add narrower Lake-side raw-example helpers for the compiled validation loop
  - [x] Add a gated raw-partial dump hook in the raw example runners so `dK` /
    `dV` tile tensors can be inspected before reduction
  - [~] Validate the staging-only kernel change end to end through regenerated CUDA plus rebuilt raw examples
  - [ ] Generalize stacked-partials reduction beyond the accidental `qBlocks == kvBlocks` fixed-shape case
  - [ ] Add the `head_dim=128` branch that exists in TK `mha_h100` and matches the in-tree classic `Qwen` path
  - [ ] Decide how to cover the in-tree `Qwen35` / `Gemma4` `head_dim=256` path, which is outside the current TK `64` / `128` family
  - [ ] Add the head-ratio (`hr`) path required for GQA/MQA
  - [ ] Add causal backward and forward variants inside the `MhaH100` family
  - [ ] Revisit the async K/V writeback pipeline gap between TK and Tyr once correctness is stable

## 2026-04-22 One-H100 Benchmark Contract

- Headline native rows:
  - `B=1, H=1, KV=1, seq=128, headDim=64, bf16, dense_prefill, non-causal`
  - `B=1, H=1, KV=1, seq=768, headDim=64, bf16, dense_prefill, non-causal`
- Portable control rows:
  - `seq=96, headDim=64, non-causal`
  - `seq=128, headDim=64, causal=true`
  - `q_heads=4, kv_heads=2, seq=96, headDim=64, enable_gqa=true`
- Per-row report requirements:
  - requested backend
  - executed backend
  - route (`tkKernel` or `portable`)
  - `out` / `dQ` / `dK` / `dV` correctness metrics
  - p50/p10/p90 latency
  - speedup vs SDPA only when the row actually ran natively and passed correctness

## 2026-04-22 Model-Driven Wrapper Priority

- Current `head_dim=64` is a bring-up target, not the real in-tree text-model
  endpoint.
- Wrapper order implied by in-tree configs:
  - `dense_gqa_hd128_r{4,5}` for classic `Qwen`
  - `dense_gqa_hd256_r{4,8}` for `Qwen35` full-attention
  - `dense_gqa_hd512_r{4,8}` for `Gemma4` full-attention/global-head layers
  - `window_gqa_hd256_r{2,4,8}` as the next distinct family for `Gemma4`
    sliding attention
- Immediate consequence:
  - `d=128` is the first practical dense specialization after the current
    `d=64` bring-up
  - `d=256` plus GQA is what starts to cover real `Qwen35` / `Gemma4` text
    paths

## 2026-04-22 Family-By-Family Implementation Order

- Stage 1:
  - refactor runtime routing to `family + specialization key`
  - keep current `d=64` behavior unchanged
  - benchmark current native `128x64` / `768x64` rows as the control baseline
- Stage 2:
  - land `hd128` dense GQA forward prefill for real `Qwen` shapes
- Stage 3:
  - complete `hd128` decode and mask coverage
- Stage 4:
  - add `hd128` backward and training-safe runtime integration
- Stage 5:
  - land `hd256` dense GQA forward for `Qwen35`
- Stage 6:
  - complete `hd256` decode/mask/backward
- Stage 7:
  - land `windowed hd256` decode-first for `Gemma4` sliding attention
- Stage 8:
  - extend `windowed hd256` to prefill and broader window/ratio coverage
- Stage 9:
  - add `windowed hd256` backward and Gemma training readiness

## 2026-04-22 Nested Codegen Staleness Risk

- The nested `GenerateGpuKernels` step inside `extern_lib libtyr` is
  intentional, but staleness can still happen when:
  - `extern_lib libtyr` is not invalidated, so the nested generator is skipped,
  - `TYR_GPU_CODEGEN_MODULE` changes without any tracked file change,
  - `TYR_SKIP_GPU_CODEGEN=1` suppresses regeneration,
  - concurrent builds share `cc/src/generated`,
  - the nested generator fails after leaving the previous target file in place.
- Immediate implication:
  - codegen-module selection and skip-codegen mode should be treated as part of
    the effective build contract, not as invisible side inputs.

## 2026-04-22 Benchmark Scaffold

- `Examples/GPU/RunFlashAttnBench.lean` now exists as the structured benchmark
  surface for one-H100 flash-attention comparisons.
- Verified static CLI surfaces:
  - `--list-cases`
  - `--list-backends`
- The benchmark schema is no longer the blocker.
- Remaining execution blocker:
  - native benchmark runs require the compiled `RunFlashAttnBench` binary,
    because interpreter mode does not resolve the Torch/CUDA extern path needed
    for real execution.

## 2026-04-22 Minimal PyTorch Benchmark Path

- The cleanest current benchmark path is still the in-tree native runner, not a
  separate Python harness:
  - `scripts/gpu/bench_flash_attn_matrix.sh`
  - `Examples/GPU/RunFlashAttnBench.lean`
- Why:
  - Tyr runtime and PyTorch SDPA already execute inside the same process and
    the same timing loop,
  - that keeps correctness checks, case selection, JSONL output, and build
    wiring in one place,
  - it avoids introducing `uv` and a second benchmark implementation unless we
    explicitly decide to benchmark an external wheel.
- The wrapper help now documents the exact FA3 build knob:
  - `TYR_GPU_CODEGEN_MODULE=Tyr.GPU.Kernels.FlashAttn3`
  - use it with `--case future_flash_256x64 --backend flash_attention`
- Host inspection on this machine:
  - site PyTorch `2.7.1` on CUDA `12.6` is present and reports flash SDPA
    support,
  - no standalone `flash_attn` Python package is installed.
- Immediate recommendation:
  - treat `torch_sdpa` as the default PyTorch baseline,
  - keep `flash_attention` as the exact repo-local FA3 row,
  - add an in-process `torch_flash` backend later if we want an explicit forced
    PyTorch flash comparison without adding a Python-side runner.

## 2026-04-22 Raw 128x64 Revalidation After dK/dV Concern

- Revalidated the compiled raw `RunMhaH100` path after the concern that fixing
  `dV` may have moved the mismatch to `dK`.
- Command that produced the trusted result:
  - `source ./load_modules.sh && LEAN_CC=$PWD/scripts/lean_cc_wrapper.sh LEAN_CC_LINKER=bfd lake -R run runMhaH100Exe --dump-partials`
- Result:
  - `overall_ok=true`
  - `kernel_ref_ok=true`
  - `dq_ref_ok=true`
  - `dk_ref_ok=true`
  - `dv_ref_ok=true`
  - `dq_mae=0.000168`
  - `dk_mae=0.000166`
  - `dv_mae=0.000153`
- Interpretation:
  - the current 128x64 native raw kernel does not show a moved `dK` mismatch,
  - the earlier `dK`/`dV` confusion remains best explained by caller-side
    partial-buffer aliasing plus stale/overlapping build artifacts.
- Build note:
  - direct `lake -R build RunMhaH100` can still stall on this machine while
    linking `GenerateGpuKernels` into `.lake/build/bin`,
  - the supported `runMhaH100Exe` helper successfully avoided the bad
    `.lake/build/bin` artifact by relinking from the Lake trace to `/tmp`,
  - this still runs a compiled binary and does not use the interpreter.
- Linker experiment:
  - `ld.lld` from the Lean toolchain fails with the same `GLIBC_2.29` issue as
    Lean `clang`,
  - `ld.gold` exists in the module stack but fails here due an older
    `/cm/local/apps/gcc/9.2.0/lib64/libstdc++.so.6` missing
    `GLIBCXX_3.4.29`,
  - after forcing the GCCcore runtime library path, `gold` still fails with
    hidden-symbol errors (`_ZdlPvm`),
  - no linker-selector shim is kept in `scripts/lean_cc_wrapper.sh`; the
    known-working BFD path remains the default.
- TODO status:
  - [x] Confirm raw 128x64 `dK` did not regress after the aliasing fix.
  - [x] Remove the `gold` experiment after validating it is not reliable on
    this module stack.
  - [~] Fix the normal direct build stall so `lake -R build RunMhaH100` is
    sufficient without needing the trace relink fallback.
  - [x] Re-run 768x64 raw parity after the direct-build stall is addressed.
  - [ ] Run the first one-H100 benchmark rows through
    `RunFlashAttnBench`.

## 2026-04-22 Raw 768x64 Revalidation After Stale-Binary Check

- The apparent 768x64 `dK` / `dV` regression was reproduced only with a stale
  `.lake/build/bin/RunMhaH100Seq768` executable from before the current object
  files were built.
- Evidence:
  - the first `runMhaH100Seq768Exe --dump-partials` execution reported
    `dk_ref_ok=false` and `dv_ref_ok=false`,
  - that executable timestamp predated the source and generated C object,
  - a fresh relink from the current object files reported the correct partial
    dump and full parity.
- Trusted command after hardening the compiled-run path:
  - `source ./load_modules.sh && LEAN_CC=$PWD/scripts/lean_cc_wrapper.sh LEAN_CC_LINKER=bfd CUDA_VISIBLE_DEVICES=0 lake -R run runMhaH100Seq768Exe --dump-partials`
- Result:
  - `overall_ok=true`
  - `kernel_ref_ok=true`
  - `dq_ref_ok=true`
  - `dk_ref_ok=true`
  - `dv_ref_ok=true`
  - `dq_mae=0.000078`
  - `dk_mae=0.000077`
  - `dv_mae=0.000070`
- Build hygiene change:
  - `runBuiltExecutable` now treats a compiled executable as stale when the
    corresponding `.lake/build/ir/**/<Exe>.c.o.export` is newer than the ELF,
  - stale or invalid executables are relinked to `/tmp/tyr_relinked` before
    execution,
  - this keeps validation on compiled binaries while avoiding stale executable
    artifacts from looking like kernel math regressions.
- TODO status:
  - [x] Confirm raw 128x64 `dK` / `dV` parity.
  - [x] Confirm raw 768x64 `dK` / `dV` parity from fresh compiled objects.
  - [x] Add stale-binary detection to the compiled-run helper.
  - [~] Fix the underlying direct Lake final-link stall so the `/tmp` relink
    fallback stops being needed.
  - [ ] Continue with one-H100 `RunFlashAttnBench` forward/backward rows.

## 2026-04-22 Native Runtime Benchmark Path

- Added a narrow compiled C++ benchmark:
  - `cc/tools/bench_flash_attn.cpp`
  - `make -C cc bench-flash-attn TYR_GPU_CODEGEN_MODULE=Tyr.GPU.Kernels.MhaH100`
- Rationale:
  - avoids the Python `torch` / vendored LibTorch ABI mismatch,
  - avoids the current Lean `mkC10IoError` crash path in fwd+bwd benchmark
    rows,
  - still calls the real `tyr_ops::flash_attn_dispatch` runtime bridge.
- Switched the native bridge backward path to return native reduced `dVStack`:
  - previous bridge path reduced native `dKStack` but recomputed `dV` with
    PyTorch softmax/matmul,
  - fresh raw 128x64 and 768x64 validation now shows native `dVStack` parity,
    so the bridge now reduces `dVStack` symmetrically with `dKStack`.
- Command:
  - `source ./load_modules.sh && CUDA_VISIBLE_DEVICES=0 cc/build/tools/bench_flash_attn --case native_now --backend torch_sdpa,tyr_runtime --warmup 5 --iters 20 --repeats 3 --jsonl-out benchmarks/results/flash_attn_cpp_native_h100_native_dv.jsonl --jsonl-stdout`
- Result after the native-`dV` bridge change:
  - `native_dense_128x64`
    - `torch_sdpa`: `p50_ms=0.186584`
    - `tyr_runtime`: `p50_ms=0.197651`
    - `correctnessOk=true`
    - `speedupVsSdpaP50=0.944007`
  - `native_dense_768x64`
    - `torch_sdpa`: `p50_ms=0.188121`
    - `tyr_runtime`: `p50_ms=0.562412`
    - `correctnessOk=true`
    - `speedupVsSdpaP50=0.334489`
- Interpretation:
  - runtime fwd+bwd correctness is green for the current native 128x64 and
    768x64 routes,
  - native `dV` is no longer masked by a PyTorch recompute in the bridge,
  - Tyr is still slower than PyTorch SDPA on these one-H100 training rows,
    especially at 768x64, because the runtime route launches multiple kernels
    and writes/reduces stacked partials while SDPA uses a more fused backend.
- TODO status:
  - [x] Add compiled one-H100 C++ benchmark for runtime bridge vs SDPA.
  - [x] Use native reduced `dVStack` in the runtime bridge backward path.
  - [x] Produce first correctness+latency JSONL for 128x64 and 768x64.
  - [~] Rebuild and rerun the Lean `RunFlashAttnOp` parity executable against
    the updated static bridge.
  - [ ] Add forward-only inference timing rows.
  - [ ] Close the optimization gap versus PyTorch SDPA.

## 2026-04-22 ThunderKittens-Aligned Store-Add Accumulation

- Compared the current Tyr backward contract against ThunderKittens
  `mha_h100.cu`.
- Key finding:
  - TK accumulates `dK` / `dV` for a KV tile in registers across query tiles,
    then writes final gradients through `warp::tma::store_add_async` and
    `warp::tma::store_async_wait()`,
  - Tyr was still writing one q-major partial tile per `(qBlock, kvBlock)` and
    reducing with PyTorch/Torch tensor ops after the kernel,
  - that explains most of the benchmark gap attributed to partial reduction.
- Implementation moved in that direction:
  - `Tyr/GPU/Kernels/MhaH100.lean` now writes `dK` / `dV` with
    `storeGlobalAdd` into final zeroed gradient tensors,
  - the kernels wait on async TMA stores before reusing shared staging,
  - `cc/src/tyr_ops.cpp` now returns direct native `dK` / `dV` tensors and
    removes `reduce_stacked_partials`,
  - `RunMhaH100`, `RunMhaH100Seq768`, `RunMhaH100Train`, and the typed
    `Tyr.GPU.Ops.MhaH100` wrapper now use `contract=store_add_accum`.
- TODO state:
  - [x] Create an intermediate checkpoint commit before this riskier kernel
    contract change.
  - [x] Check TK source for whether the extra syncs are semantically necessary.
  - [x] Replace runtime bridge partial-stack outputs with direct gradient
    buffers.
  - [~] Finish normal Lake rebuild / generated CUDA refresh for the new
    store-add path.
  - [ ] Confirm generated CUDA contains `store_add_async` and
    `store_async_wait` for both 2-block and 12-block backward kernels.
  - [ ] Re-run raw 128x64 and 768x64 parity.
  - [ ] Re-run one-H100 C++ bridge benchmark and quantify whether removing the
    partial-stack reduction improves `tyr_runtime`.
  - [ ] If parity fails, compare generated C++ against TK's `compute_bwd_loop`
    and `kv_store` before touching math again.

## 2026-04-23 Store-Add Route Fixed and Benchmarked

- The store-add route failed with `an illegal memory access was encountered`
  inside `tkMhaH100Bwd2BlockPartials`.
- Diagnosis from comparing generated C++ to ThunderKittens:
  - [x] `warp::tma::store_add_async` must be issued by one warp per shared
    tile and followed by an async-store wait before shared-buffer reuse.
  - [x] Shared tiles used by TMA need the TK swizzle-alignment contract
    (`tma_swizzle_allocator` in TK, now `KITTENS_ALIGN_AS(1024)` in generated
    Tyr static shared declarations).
  - [x] `gl<..., st<...>>` parameters with embedded `CUtensorMap` descriptors
    must be `const __grid_constant__`, matching TK's descriptor-bearing
    globals.
  - [~] The current issuer policy is correct for the single-warpgroup generated
    MHA kernels; a future generalized backend needs explicit issuer policy
    rather than a hidden `warpid()==0` assumption.
- Implemented:
  - [x] Aligned generated `st<>` shared tile declarations.
  - [x] Emitted `const __grid_constant__` for TMA descriptor-bearing kernel
    parameters.
  - [x] Added `group<4>::sync(4)` around the H100 MHA dK/dV TMA store-add
    staging.
  - [x] Regenerated `cc/src/generated/Tyr_GPU_Kernels_MhaH100.cu`.
  - [x] Rebuilt `cc/build/tools/bench_flash_attn`.
- Validation:
  - [x] `CUDA_LAUNCH_BLOCKING=1` runtime route no longer faults for
    `native_dense_128x64`.
  - [x] compiled C++ bridge benchmark passes fwd+bwd correctness for
    `native_dense_128x64`.
  - [x] compiled C++ bridge benchmark passes fwd+bwd correctness for
    `native_dense_768x64`.
- Benchmark result:
  - `benchmarks/results/flash_attn_cpp_native_h100_store_add_gridconst.jsonl`
  - `native_dense_128x64`: Tyr `0.160957 ms`, PyTorch SDPA `0.147628 ms`,
    `speedupVsSdpaP50=0.91719`.
  - `native_dense_768x64`: Tyr `0.522411 ms`, PyTorch SDPA `0.177279 ms`,
    `speedupVsSdpaP50=0.339348`.
- Next TODO:
  - [~] Keep the current store-add route as the training-correct bridge path for
    fixed 128/768 x 64 rows.
  - [ ] Replace the q-block outer loop store-add pattern with a TK-like
    KV-centric backward kernel that accumulates dK/dV across query tiles before
    one final store-add.
  - [ ] Add forward-only inference benchmark rows for Qwen/Gemma-relevant
    shapes.
  - [ ] Add shape-specialized codegen coverage for head-dim 128, GQA/MQA, and
    longer sequence lengths.

## 2026-04-23 KV-Sweep Backward Stabilization

- Implemented the next H100 MHA backward checkpoint:
  - `tkMhaH100Bwd2BlockDq` / `tkMhaH100Bwd12BlockDq` compute `dQ` with the
    known-correct q-centric direct-store accumulation.
  - `tkMhaH100Bwd2BlockKvSweep` / `tkMhaH100Bwd12BlockKvSweep` compute `dK` /
    `dV` with a ThunderKittens-like KV-owned query sweep.
  - Runtime bridge, typed op wrappers, raw examples, and training example now
    launch direct `dQ` first, then the K/V sweep.
- Sync/TK comparison result:
  - [x] K/V ownership now matches the important TK structure: one KV tile per
    CTA, all query tiles swept, K/V accumulated in registers, final store-add.
  - [x] Generated TMA descriptor globals are `const __grid_constant__`.
  - [x] Generated TMA source shared tiles are 1024-byte aligned.
  - [x] K/V final writeback uses separate FP32 shared staging again.
  - [~] `dQ` is intentionally not using TK-style cross-KV TMA store-add yet.
    The fused attempt was nondeterministic under repeated runs.
  - [~] Sync is conservative raw `group<4>::sync(4)` around store-add staging;
    the general backend still needs a real async-handoff abstraction.
- Validation:
  - [x] Lean kernel module compiles.
  - [x] CUDA generation succeeds.
  - [x] Native C++ benchmark target builds.
  - [x] 128x64 launch-blocking bridge smoke passes.
  - [x] 768x64 launch-blocking bridge smoke passes.
  - [x] Repeated 128x64 diagnostics are deterministic after the split.
- Benchmark:
  - [x] Wrote `benchmarks/results/flash_attn_cpp_native_h100_dq_direct_kv_sweep.jsonl`.
  - `native_dense_128x64`: Tyr `0.329764 ms`, SDPA `0.231418 ms`,
    `correctnessOk=true`, `speedupVsSdpaP50=0.701769`.
  - `native_dense_768x64`: Tyr `1.33397 ms`, SDPA `0.348799 ms`,
    `correctnessOk=true`, `speedupVsSdpaP50=0.261476`.
- TODO state:
  - [x] Create a working, benchmarked KV-sweep K/V backward route.
  - [x] Keep the runtime bridge training-correct for the fixed native rows.
  - [~] Analyze unnecessary/too-granular syncs versus TK; current syncs are
    conservative but not yet a general pipeline model.
  - [ ] Make the TK-style q-gradient store-add path deterministic so the direct
    q-centric `dQ` pass can be removed.
  - [ ] Recover performance lost by the two-kernel backward split.
  - [ ] Add shape-specialized head-dim 128, causal, and GQA/MQA routes for
    Qwen/Gemma coverage.

## 2026-04-23 Async TMA DSL Parity Checkpoint

- Implemented and validated a first-class DSL async TMA/semaphore pass for H100
  MHA codegen.
- TODO status:
  - [x] Add typed semaphore init/wait phase support.
  - [x] Add typed TMA store commit/wait support.
  - [x] Add typed `group<N>::sync` and CTA `__syncthreads` primitives.
  - [x] Guard TMA load/expect with one issuing warp.
  - [x] Regenerate H100 MHA CUDA and confirm structural counts.
  - [x] Fix ptxas static shared overflow by reusing a dK/dV staging tile only
    after async-store wait.
  - [x] Rebuild `cc/build/tools/bench_flash_attn`.
  - [x] Benchmark 128x64 and 768x64 on one H100.
  - [~] Close sync parity: no `warp::sync` remains, but init uses conservative
    `__syncthreads` rather than TK's fully pipelined phase handoff.
  - [ ] Close math parity: generated kernels still emit `warp::mma` instead of
    TK `warpgroup::mm/mma`.
  - [ ] Close schedule parity: current route is split dQ + KV sweep, not TK's
    single pipelined backward kernel.
- Structural counts from regenerated `Tyr_GPU_Kernels_MhaH100.cu`:
  - `tma::load_async=44`, `init_semaphore=31`, `expect_bytes=31`, `wait=39`.
  - `warp::sync=0`, `group<4>::sync=24`, `store_commit_group=8`, `store_async_wait=8`.
  - `warp::mma=36`, `warpgroup::mma/mm=0`.
- Benchmark:
  - `benchmarks/results/flash_attn_cpp_native_h100_async_tma_dsl_parity.jsonl`
  - `native_dense_128x64`: Tyr `0.152307 ms`, SDPA `0.143101 ms`, gradients correct.
  - `native_dense_768x64`: Tyr `0.496581 ms`, SDPA `0.172443 ms`, gradients correct.
- Interpretation:
  - The async/TMA DSL pass recovers a large part of the previous regression
    (`768x64` improved from roughly `1.28 ms` to `0.50 ms`).
  - The remaining gap is not primarily dK/dV math correctness anymore; it is the
    missing TK WGMMA producer/consumer schedule and the extra split-backward
    launch structure.

## 2026-04-23 WGMMA Forward Parity Attempt

- Implementation status:
  - [x] Added IR/emitter support for `warpgroup::mm_*` and
    `warpgroup::mma_*` statements.
  - [x] Added typed primitives for `warpgroup::mm_ABt`,
    `warpgroup::mma_AB`, and related transpose/accumulate forms.
  - [x] Forward H100 MHA loops now call `warpgroupMmT` for scores and
    `warpgroupMma` for the probability/value accumulation.
  - [x] Lake rebuilt the compiled `GenerateGpuKernels` executable after the IR
    change; the expensive part was an avoidable generator link against TyrC,
    libtorch, and CUDA.
  - [x] Regenerated generated CUDA and counted WGMMA vs warp MMA calls.
  - [x] Rebuilt the native flash-attention benchmark target.
  - [x] Ran one-H100 correctness and timing for `native_now`.
- Generated-CUDA structural counts:
  - `warpgroup::mm_ABt=6`, `warpgroup::mma_AB=6`,
    `warpgroup::mma_async_wait=12`.
  - `warp::mma=24`, `warp::sync=0`, `group<4>::sync=18`.
  - `tma::load_async=44`, `store_commit_group=8`, `store_async_wait=8`.
- Benchmark:
  - [x] Wrote `benchmarks/results/flash_attn_cpp_native_h100_wgmma_forward_parity.jsonl`.
  - `native_dense_128x64`: Tyr `0.216733 ms`, SDPA `0.202832 ms`,
    gradients correct.
  - `native_dense_768x64`: Tyr `0.498877 ms`, SDPA `0.215072 ms`,
    gradients correct.
  - [~] Performance did not improve. ptxas reports WGMMA serialization from
    insufficient register resources, so simply swapping call names is not enough.
- ThunderKittens comparison notes:
  - [x] Forward target call shape is known: `warpgroup::mm_ABt` over Q/K shared
    tiles followed by `warpgroup::mma_AB` over P/V shared tiles.
  - [x] Backward target call shape is known: WGMMA score/dP, register/shared
    WGMMA dK/dV, then final KV-owned store.
  - [~] Tyr now models the forward WGMMA calls, but still feeds Q from a register
    tile and lacks TK's shared-Q, three-consumer/one-producer warpgroup role
    split.
  - [ ] Convert the KV sweep backward contractions to the TK WGMMA forms.
  - [ ] Decide whether the DSL needs explicit producer/consumer warpgroup roles
    before attempting fused dQ store-add again.

## 2026-04-23 Shared-Q WGMMA Experiment

- What I tried:
  - [x] Replaced the forward score matmul with a shared/shared
    `warpgroup::mm_ABt(scores, qShared, kShared)` shape in generated CUDA.
  - [x] Rebuilt the generated CUDA through `make -C cc bench-flash-attn`.
- Result:
  - [x] Compile failed in ThunderKittens at
    `warpgroup.cuh` with `static_assert(D::height == 1)`.
  - [x] The failure is structurally informative, not a random template issue:
    Tyr tried to write a 64x64 register tile, while TK's shared/shared WGMMA
    overload writes a 16x64 register subtile.
  - [x] Restored the production MHA forward loop to the previously validated
    register-Q WGMMA route and rebuilt the native benchmark target successfully.
  - [x] Added typed primitive names for the actual TK shape:
    `warpgroupMmSharedT64x16`, `warpgroupMmaSharedT64x16`, plus AB forms.
- Next required DSL/kernel work:
  - [ ] Represent 16-row WGMMA output subtiles in the H100 attention loop.
  - [ ] Add warpgroup-id/subtile ownership so four 16-row pieces can cover a
    64-row query tile.
  - [ ] Reattempt shared-Q forward only after the subtile ownership model exists.

## 2026-04-23 Forward WGMMA Register Cleanup

- Implemented:
  - [x] Removed unused forward K/V register tiles from the WGMMA route.
  - [x] Regenerated H100 MHA CUDA.
  - [x] Rebuilt `cc/build/tools/bench_flash_attn`.
  - [x] Re-ran one-H100 `native_now` benchmark.
- Benchmark:
  - [x] Wrote `benchmarks/results/flash_attn_cpp_native_h100_wgmma_forward_clean_regs.jsonl`.
  - `native_dense_128x64`: Tyr `0.208570 ms`, SDPA `0.191270 ms`,
    gradients correct.
  - `native_dense_768x64`: Tyr `0.493654 ms`, SDPA `0.196349 ms`,
    gradients correct.
- Status:
  - [x] Dead register declarations are gone from generated forward CUDA.
  - [~] WGMMA serialization warnings remain.
  - [ ] Performance still requires TK-style 16-row consumer subtiles and
    explicit warpgroup role scheduling.

## 2026-04-23 K/V Sweep WGMMA Checkpoint

- Implemented:
  - [x] Converted KV-sweep `dV += P @ dO` to register/shared
    `warpgroup::mma_AB`.
  - [x] Converted KV-sweep `dK += dS @ Q` to register/shared
    `warpgroup::mma_AB`.
  - [x] Reused the existing BF16 shared tile and reloaded Q before dK to avoid
    increasing static shared memory past the H100 ptxas limit hit earlier.
  - [x] Rebuilt `cc/build/tools/bench_flash_attn`.
- Structural counts:
  - `warpgroup::mm_ABt=6`, `warpgroup::mma_AB=10`,
    `warpgroup::mma_async_wait=16`.
  - `warp::mma=20`, `warp::sync=0`, `group<4>::sync=18`.
  - `tma::load_async=46`, `store_commit_group=8`, `store_async_wait=8`.
- Benchmark:
  - [x] Wrote `benchmarks/results/flash_attn_cpp_native_h100_bwd_kv_wgmma_reload_q.jsonl`.
  - `native_dense_128x64`: Tyr `0.196867 ms`, SDPA `0.186461 ms`,
    gradients correct.
  - `native_dense_768x64`: Tyr `0.471824 ms`, SDPA `0.188912 ms`,
    gradients correct.
- Status:
  - [x] Correctness remains green after WGMMA dK/dV.
  - [~] Runtime improves versus the cleaned forward-only WGMMA checkpoint.
  - [~] The Q reload is a deliberate temporary tradeoff to avoid adding another
    shared tile before the real TK producer/consumer schedule exists.
  - [ ] ptxas serialization warnings remain; next work must target TK's
    16-row consumer subtile/register schedule.
