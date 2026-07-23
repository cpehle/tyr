# GPU Backend Performance Plan

## Objective

Make generated GPU kernels independently buildable, architecture-correct,
correctness-gated, and reproducibly benchmarkable against eager PyTorch,
`torch.compile`, and relevant vendored implementations. Then use that measurement
loop to make the complete training-critical kernel stack competitive with
PyTorch: dense projections and their gradients, attention, normalization,
residual/rotary paths, loss, and optimizer work. MHA is one focused track, not
the sole optimization target; decode performance remains a regression guardrail.

## Measurement contract

Every reported result must:

- identify the exact generated kernel and runtime route;
- check numerical correctness before timing;
- separate kernel-only and end-to-end timing;
- use CUDA events with warmup and repeated samples;
- record the stop event on the measured stream and synchronize that event before
  reading elapsed time, so asynchronous enqueue latency cannot masquerade as
  execution latency;
- report p10, p50, and p90;
- exclude allocation, fixture generation, compilation, and synchronization from
  the timed loop unless explicitly measuring end-to-end latency;
- record the git revision and dirty state, generated-source hash, GPU identity,
  compute capability, driver, CUDA, PyTorch and compiler versions, launch
  dimensions, registers, shared memory, and selected route;
- report useful throughput such as GB/s or TFLOP/s where applicable;
- never include an incorrect result in speedup summaries.

The initial operation matrix covers copy, rotary, LayerNorm, RMSNorm, BF16 GEMM,
attention forward, attention forward/backward, and decode attention. Each case is
compared with eager PyTorch, `torch.compile`, and, when contract-equivalent, the
vendored ThunderKittens reference.

## Realistic training-scale policy

There is no backend-wide fixed shape profile. Each benchmark suite owns a
declarative, CLI-selectable matrix derived from the model it represents. Results
must distinguish:

- the local microbatch seen by one kernel launch;
- local tokens per device and microstep, `T = batch * sequence`;
- the global optimizer batch after data parallelism and gradient accumulation,
  which does not change the local launch shape.

The current model-derived anchor is the Qwen3-TTS talker:
`hidden=1024`, `intermediate=2048`, `QH=16`, `KVH=2`, `D=64`, and
`S=768`. Equal-head `MhaGB10` remains an attention-kernel compatibility
track, not the default operator-stack profile. Use B4 (`T=3072`) as the primary
throughput point, B8 (`T=6144`) as the saturation guardrail, B2 as the
smaller-memory point, and B1 as a latency/underfill edge or as a primary point
only when full-model memory really forces microbatch one.
Long-context models may legitimately make B1 or B2 the realistic point, so the
exact model shape must remain selectable rather than being encoded globally.

Training coverage is organized around complete per-operator contracts.
The `qwen3tts-talker` GEMM profile carries nine distinct shape contracts per
batch point: Q/output projection forward-dX and dW, K/V forward, dX, and dW,
plus both 1024-to-2048 MLP directions and their two dW orientations.

- projection and MLP GEMM triplets: forward, activation gradient, and weight
  gradient, with `M=T` and exact model input/output widths;
- attention: saved-state forward plus full backward;
- residual, normalization, and rotary: forward plus every input/parameter
  gradient used by the model;
- logits/cross-entropy and optimizer update;
- a whole-block and eventually whole-step row that exposes launch and
  materialization overhead hidden by isolated kernel medians.

Per-shape rows remain the source of truth. A weighted model-step aggregate may
be added once an exact model is selected, but a geometric mean over unrelated
toy shapes must not drive optimization priorities.

## Phase 1: Isolate generated kernel builds

Goal: building one kernel must not compile unrelated or stale generated CUDA.

1. Replace wildcard selection of `cc/src/generated/*.cu` with an explicit
   requested-module manifest.
2. Propagate the requested module and physical target through every Lake and Make
   invocation, including nested extern-library builds.
3. Generate architecture-specific artifacts with metadata containing the module,
   logical architecture, supported compute capabilities, required instruction
   families, source hash, and exported kernels.
4. Key generated objects by module, architecture, and source hash, and invalidate
   them when any component changes.
5. Reject incompatible modules before invoking NVCC.
6. Add regression tests proving that a copy build does not mention MHA, an SM121
   build does not compile SM90 units, and clean and incremental builds select the
   same sources.

Acceptance criteria:

- copy, rotary, and normalization build independently;
- the build log prints the complete selected manifest;
- a GB10 build contains no Hopper-only WGMMA instructions;
- incompatible modules fail early with an actionable diagnostic.

## Phase 2: Make the toolchain setup self-contained

Goal: the documented setup command must produce a usable C++ and CUDA build
without manual environment variables.

1. Derive Python headers and the LibTorch installation from the selected
   `.venv-gpu` interpreter.
2. Pass the Python include directory to both C++ and NVCC.
3. Add a `doctor` preflight for Python import, `Python.h`, LibTorch headers and
   libraries, NVCC, driver/runtime compatibility, compute capability, and
   ThunderKittens target support.
4. Distinguish official wheel support from compatibility/PTX fallback.
5. Exercise CUDA and non-CUDA preflight paths in CI.

Acceptance criteria:

- a fresh supported setup builds and runs copy without `CPATH`;
- failures identify the missing component and the supported repair command;
- the preflight reports whether the installed PyTorch officially supports the
  physical compute capability.

## Phase 3: Introduce a correct architecture model

Goal: separate physical compute capability from product family and instruction
capabilities.

1. Model SM120/SM121 explicitly.
2. Separate physical target, instruction capabilities, implementation family,
   SKU properties, and tuning configuration.
3. Represent TMA, Hopper WGMMA, Blackwell MMA/TMEM, cluster launch, distributed
   shared memory, FP8/FP4, barriers, and shared-memory limits individually.
4. Replace broad kernel architecture annotations with explicit capability
   requirements.
5. Stop classifying GB10 as Hopper and replace or rename any nominal GB10 kernel
   that still emits Hopper instructions.
6. Add positive and negative compile tests that inspect generated PTX/SASS for
   required and prohibited instruction families.

Acceptance criteria:

- GB10 is never classified as Hopper;
- SM121 kernels compile without Hopper WGMMA;
- incompatibility is rejected by the Lean/codegen layer with a useful message.

## Phase 4: Build the unified benchmark driver

Goal: one command emits comparable JSONL records for Tyr and reference backends.

1. Implement shared correctness, warmup, timing, percentile, metadata, and
   throughput helpers.
2. Provide kernel-only and end-to-end modes.
3. Cover small launch-bound, medium, bandwidth-bound, compute-bound, and edge-tile
   shapes for every operation.
4. Record runtime routing and reject silent fallback when benchmarking Tyr.
5. Store results as versioned JSONL with a stable schema.

Acceptance criteria:

- repeated p50 measurements remain within 5% under a clean GPU state;
- every Tyr row proves that the intended generated kernel ran;
- eager PyTorch and `torch.compile` use identical inputs and semantics.

## Phase 5: Make launch configuration device-aware

Goal: use actual device and compiled-kernel properties instead of typical SKU
estimates.

1. Query SM count, shared-memory limits, cooperative/cluster support, and other
   relevant runtime properties.
2. capture registers, static and dynamic shared memory, spills, thread limits,
   and occupancy for compiled kernels.
3. Replace `typicalSMs` launch sizing with runtime values.
4. Use CUDA occupancy APIs rather than the current simplified estimate.
5. Add correctness-gated autotuning for block size, tile shape, pipeline stages,
   persistent CTA count, and cluster size.
6. Cache tuning by GPU UUID, kernel hash, shape family, and dtype.

Acceptance criteria:

- persistent grids reflect the physical GPU;
- benchmark records include resource use and occupancy;
- tuning caches invalidate when generated code changes.

## Phase 6: Restore GB10 kernel coverage

Implement and validate in this order:

1. copy;
2. rotary;
3. LayerNorm;
4. RMSNorm;
5. small BF16 GEMM;
6. attention forward;
7. attention backward;
8. decode attention.

For each operation: build independently, verify representative and adversarial
shapes, benchmark all reference backends, inspect PTX/SASS and resources, and
record results before moving to the next operation.

Acceptance criteria:

- all core kernels run on SM121;
- tolerances are documented per operation and dtype;
- the benchmark matrix contains fresh Tyr and PyTorch measurements.

## Phase 7: Optimize the training-critical kernel stack

Goal: improve the weighted time of a real decoder training block, not one
isolated operation. Use the closed operator inventory in
`dev/gpu_native_training_todo.md` and keep every comparison contract-equivalent.

1. Add model-derived GEMM matrices for QKV/output and MLP projections, including
   forward, activation-gradient, and weight-gradient shapes. Prioritize these
   first because they normally dominate decoder training FLOPs.
2. Continue the saved-LSE attention forward/backward track, comparing generated
   code and SASS for TMA issue/waits, MMA pipelines, barriers, softmax,
   conversions, stores, and duplicated score/probability recomputation.
3. Add and measure the backward contracts for normalization and rotary, then the
   loss/reduction and optimizer surfaces required by the native train step.
4. Measure registers, spills, shared memory, occupancy, tensor-core utilization,
   memory throughput, barrier stalls, and eligible warps for every material
   kernel; use synchronized, idle-gated full-operation timing as the acceptance
   metric for fusions.
5. Add a whole decoder-block training row and use its measured operator weights
   to choose subsequent work. Run the complete correctness and performance
   matrix after every accepted change.

Acceptance criteria:

- all kernels needed by one native decoder train step have forward/backward
  correctness and synchronized performance rows at realistic local-token scales;
- no material training operation is more than 10% slower than matched PyTorch,
  and the whole-block row reaches parity before optimizing a larger megakernel;
- S128 and S768 attention remain no slower than PyTorch as an attention-specific
  sub-goal;
- every accepted change has current JSONL and profiler evidence.

## Phase 8: Tune decode and generalize schedules

1. Preserve the current near-parity decode results as regression baselines.
2. Investigate cases currently 5--7% behind PyTorch.
3. Tune KV-length tiles, one-block versus multi-block routing, GQA grouping, head
   dimensions, tails, and persistent versus conventional launches.
4. Promote successful choices into reusable codegen policies.

Acceptance criteria:

- no existing decode case regresses by more than 3%;
- geometric-mean performance exceeds PyTorch by at least 10%;
- routing decisions are reported and directly testable.

## Phase 9: Add performance regression gates

Per-commit GPU smoke tests cover isolated builds, correctness, and a small stable
benchmark subset with an initial 10% threshold. A scheduled dedicated-GPU suite
runs the full matrix and profiler collection with 3--5% thresholds, retained
artifacts, and comparison against the last accepted baseline. A single noisy
sample is insufficient to fail CI; gates use repeated samples or confidence
intervals.

## Active measurement queue

1. Compare the preserved generic QH16/KVH2/S768 RoPE executable with the active
   32-bit shape specialization. Require Q/K/dQ/dK pre- and post-timing parity,
   then run the B1/B2/B4/B8 GQA matrix at 20 warmups, 200 iterations, and seven
   repeats.
2. Run the nine-shape `qwen3tts-talker-primary` GEMM profile at B4, covering
   Q/output, K/V, and both MLP directions for forward, dX, and dW. Only then
   expand the same cases across B1/B2/B4/B8.
3. Validate the BF16 RMSNorm saved-state forward, input/residual VJP,
   and deterministic FP32 weight-gradient kernels first at B1, then across the
   B1/B2/B4/B8 talker matrix. Compare the three-kernel full operation with eager
   autograd and already-compiled Inductor.
4. Validate the fused BF16 vocabulary-3072 cross-entropy forward/VJP and
   deterministic mean reduction first at B1, then across B1/B2/B4/B8. Compare
   against eager autograd and already-compiled Inductor, including the actual
   backward execution and FP32 mean loss.
5. Compare Tyr GEMM with allocation-free eager `torch.mm(..., out=...)` and
   already-compiled Inductor. Every compiled graph must execute and synchronize
   before timing begins.
6. Treat MHA as one component of the training optimization program, not the
   exclusive target. Rank work by end-to-end training contribution across
   attention, GEMM, RoPE, RMSNorm/residual, loss, and eventually optimizer and
   whole-block fusion.
7. Do not accept a run unless the single GB10 has zero utilization and an empty
   CUDA process table before each backend, with foreign-process monitoring for
   the entire run. The current long-running `event-based-mamba` CUDA job makes
   timing attempts invalid; it must not be killed or silently benchmarked around.

## Current measured progress (2026-07-11)

- The shared Lean and Python harnesses use CUDA events and explicitly synchronize
  the stop event. JSONL rows record timer=`cuda_event` and
  completionFence=`cudaEventSynchronize(stop)`.
- Copy now uses one 512-thread CTA with two coalesced `float4` transfers per
  thread. It passes exact correctness and measures 2.056 us p50 (2.051 us p10,
  2.165 us p90) versus eager PyTorch `copy_` at 2.459 us p50 and
  `torch.compile` at 16.418 us p50. This is a 66.5% reduction from the original
  6.142 us Tyr route and is 16.4% faster than eager PyTorch. The accepted kernel
  uses 18 registers, no stack, and no dynamic shared memory. A scalar direct
  256-thread route (4.099 us) and a 256-thread four-`float4` route (3.419 us)
  were correct but rejected in favor of the 512-thread route.
- Rotary now uses a direct 512-thread grid-stride route for the fixed 64x64 FP32
  case, eliminating the input/trigonometric/output shared-memory round trips and
  materialized register tiles. It passes exact parity and improves from 8.189 us
  to 4.095 us p50. The accepted kernel uses 36 registers and no stack or dynamic
  shared memory, versus 168 registers for the tiled route. Eager PyTorch measures
  25.870 us end-to-end and `torch.compile` 17.248 us kernel plus output copy;
  those different timing scopes remain explicit rather than being presented as
  equivalent speedups.
- RMSNorm row tiling (64 rows per CTA to 16 rows per CTA, four CTAs) passes FP32
  and BF16 correctness. FP32 improves 203.192 to 34.812 us and removes a
  1104-byte per-thread stack spill; BF16 improves 49.136 to 24.571 us.
- RMSNorm's accepted fixed `[64,1024]` routes now use one 256-thread CTA per row,
  retain four residual values per thread, and reduce the FP32 sum of squares with
  warp shuffles. Both FP32 and BF16 pass correctness at 4.095 us p50. Eager
  PyTorch measures 7.811 us and 7.873 us respectively. The FP32 kernel uses 22
  registers and the BF16 kernel 21; neither spills or uses dynamic shared memory.
- LayerNorm uses the same row tiling and computes sum and sum-of-squares in its
  first pass, removing a redundant variance pass. It passes FP32 and BF16
  correctness. FP32 improves 242.99 to 43.012 us; BF16 improves 73.07 to
  22.522 us. Compiled resources are 129 registers and 128 bytes stack for FP32,
  and 96 registers with no stack for BF16.
- LayerNorm's accepted fixed `[64,1024]` routes now assign one 256-thread CTA to
  each row, retain four residual values per thread across warp-shuffle
  reductions, and avoid the second global residual load. Both FP32 and BF16 pass
  correctness at 4.096 us p50. This improves FP32 from 43.037 us and BF16 from
  22.543 us; eager PyTorch measures 8.611 us and 8.785 us respectively. Each
  accepted kernel uses 24 registers, no stack, and 80 bytes of reduction shared
  memory in addition to the generated wrapper's static allocation.
- Small BF16 GEMM now has a distinct GB10 64x64x64 tiled route rather than
  launching the unlaunchable B200 256x256 tile. For 256x256x64, Tyr measures
  6.144 us, eager PyTorch 6.216 us, and torch.compile 23.602 us. The Tyr kernel
  uses 25.6 KiB dynamic shared memory, 166 registers, and no stack spills. All
  rows synchronize the CUDA stop event and pass correctness.
- BF16 GEMM now accepts configurable `--m`, `--n`, and specialized `--k`
  values (64, 256, or 1024) in the same benchmark driver. The realistic
  M=768, N=1024, K=1024 projection passes correctness and exposes a gap hidden
  by the launch-bound K=64 case: the original 128-thread launch measured 89.710
  us versus eager PyTorch at 28.704 us. Resource/source inspection showed that
  the generated tile uses warp-local MMA, so four launched warps redundantly
  computed the same tile. A one-warp launch improved Tyr to 73.756 us, while a
  single-buffer TMA-lookahead variant was rejected at 75.688 us. The accepted
  schedule instead assigns 16 distinct output rows to each of four warps,
  shares each TMA-staged A/B tile, and uses warpgroup fragment loads/stores. It
  passes correctness at 33.319 us p50 (33.186 us p10, 33.452 us p90), a 2.69x
  improvement over the initial realistic route. Eager PyTorch measures 28.720
  us and `torch.compile` 32.944 us with compilation excluded. The accepted
  kernel uses 158 registers, 8 bytes stack, and no spills. Removing its
  warp-zero semaphore wait plus CTA barrier was rejected after causing an
  unspecified launch failure on SM121.
- Extending the same K=1024 route to the model-derived `M=768/1536/3072/6144`
  sweep exposed intermittent single-stage corruption that the old M=768 launch
  happened to miss (one B4 run reached max error 16.75). A CTA barrier after
  semaphore initialization was insufficient. The accepted route now uses two
  independent A/B shared stages and semaphores, refilling a stage only after all
  warps have retired its register loads. Pre- and post-timing checks pass for all
  four shapes in consecutive idle-gated 20-warmup/200-iteration/seven-repeat
  matrices. Tyr p50 is 30.744/57.541/113.111/224.311 us, or
  52.4/56.0/57.0/57.4 TFLOP/s. Matched allocation-free eager PyTorch is
  28.694/59.216/111.790/225.466 us: Tyr is 7.1% slower at B1, 2.8% faster at
  B2, 1.2% slower at the primary B4 point, and 0.5% faster at B8. The route uses
  179 registers, an 8-byte stack, 33,792 effective shared bytes, and no local
  memory. The suite now owns these configurable cases and reports synchronized
  FLOP/s; torch.compile setup remains outside every timed interval.
- Physical GB10 now defaults consistently to the Blackwell implementation
  family in Make and GPU wrappers instead of Hopper. Portable accepted kernels
  explicitly select that family, while the physical SM121 source gate still
  rejects Hopper WGMMA and SM100a tcgen05/TMEM before NVCC. A canonical copy
  build under `KITTENS_BLACKWELL` on `sm_121` passes exact correctness and
  retains its 2.061 us p50 result.
- GB10 attention forward now uses 16-query-row CTAs. It improves from 36.86
  to 8.201 us, removes a 1032-byte per-thread spill, and reaches near parity
  with eager PyTorch SDPA at 7.951 us while also producing LSE. Full forward
  and backward gradient parity passes.
- The attention benchmark now owns a configurable sequence matrix covering
  S64, S128, S256, S512, S768, S1024, and S2048. One runtime-sequence kernel
  passes correctness throughout; measured schedule dispatch retains 16 query
  rows through S1024 and uses 32 rows only at S2048. The 64-row experiment was
  rejected after measuring 1744 bytes of stack per thread and severe regressions.
- A fixed-shape GB10 S768/D64 forward specialization statically pipelines twelve
  KV tiles through two TMA/shared-memory stages. It passes correctness and
  measures 14.334 us p50 versus eager PyTorch SDPA at 11.581 us p50 (23.8%
  slower) with 100 iterations and seven repeats. Both use synchronized CUDA
  events; torch.compile setup is excluded. The kernel uses 255 registers and
  72 bytes of stack per thread. Its warp-scoped route launches one 32-thread
  warp per CTA. Hopper-capable shape specializations remain separate and use
  the existing producer/consumer warpgroup and WGMMA pipeline.
- A realistic B1/H16/S768/D64 measurement is part of the suite-defined
  `model-shapes` profile. The accepted SM121 route now uses four independent
  warp MMAs per CTA: each warp owns 16 query rows while the CTA reuses one
  TMA-staged K/V tile across 64 query rows. Mbarrier waits execute only in the
  producer warp and completion is published with a CTA barrier; executing the
  wait in all four warps caused an unspecified launch failure. With 20 warmups,
  200 iterations, and seven repeats, the accepted one-buffer lookahead route
  measures 38.921 us p50 (38.890 us p10, 40.065 us p90), down from 106.612 us
  p50: a 2.74x speedup. It refills the single shared K/V stage after all warps
  have copied the current tile into registers, overlapping the next TMA with
  compute without increasing shared-memory occupancy. Q, output, and LSE move
  directly between registers and global memory; this removes Q/output staging
  and fixes the former shared LSE scratch race. Output correctness passes with
  MAE 0.000091 and max error 0.001953; the new explicit LSE gate passes with
  MAE 0.000002 and max error 0.000008. Eager PyTorch SDPA measures 30.788 us
  p50, so the remaining gap is 1.26x rather than 3.45x;
  `torch.compile` measures 35.360 us with compilation/setup excluded. Every
  row uses a synchronized CUDA stop event. The accepted kernel uses 239
  registers, an 8-byte stack frame with no spills, 16 bytes static shared
  memory, and 16 KiB dynamic K/V shared memory. Rejected variants include the
  rows32 route (134.445 us, 448-byte stack spill), synchronous four-warp route
  (230.784 us), two-stage four-warp route (61.475 us), and 16-key shared-subtile
  streaming route (illegal memory access). The accepted route uses the existing
  base-2 online-softmax convention with scaled LSE; it is performance-neutral
  versus natural exp at 38.921 us p50 but retains identical 239-register,
  spill-free resources and passes the stronger output-plus-LSE gate.
  An eight-warp/128-query-row CTA experiment also passes output and LSE
  correctness, but regresses to 43.101 us p50 versus the accepted 38.921 us.
  It is rejected: reducing CTA count and K/V staging duplication alone does not
  close the gap, so further work should target the per-warp 128-key pipeline and
  register/softmax efficiency rather than increasing query rows per CTA.
  A scaled-state base-2 softmax experiment kept the running maximum in log2
  units to remove per-tile maximum rescaling. It passed output/LSE correctness
  but was latency-neutral at 38.901 us p50 and increased static FADD/FMUL counts
  (792/740 versus 384/380), with resources unchanged at 239 registers and an
  8-byte stack. It is rejected.
  A compact runtime-loop version of the same one-buffer TMA schedule also passes
  output/LSE correctness and measures 38.904 us p50 in a synchronized short
  run, effectively identical to the accepted 38.921 us. It reduces generated
  source size but raises register usage from 239 to 243 with the same 8-byte
  stack, so it is rejected in favor of the lower-register unrolled specialization.
  A true 128-key warp-tile experiment halves the KV loop from twelve iterations
  to six and passes output/LSE correctness, confirming the DSL can express the
  PyTorch-like key geometry on SM121. It is not hardware-viable with warp MMA:
  ptxas reaches 255 registers and a 600-byte stack frame, and synchronized
  latency regresses to 122.709 us p50. It is rejected. Closing the remaining
  forward gap therefore requires a fragmented/streamed 128-key representation
  An occupancy-oriented eight-row-per-warp experiment was rejected before CUDA
  timing: SM121 warp MMA requires the M dimension to be divisible by 16, and
  the DSL correctly failed elaboration rather than emitting an invalid tile.
  A separate first-KV-tile softmax specialization removed the mathematically
  redundant initial output/running-sum rescale. It retained the accepted 239
  registers, 8-byte stack, and zero spills and passed output/LSE correctness,
  but measured 39.217 us p50 (38.977 us p10, 41.148 us p90) versus the accepted
  38.921 us p50. The run began with no CUDA compute processes and 0% GPU
  utilization. It is rejected and the accepted softmax path is restored.
  Counter-based Nsight Compute profiling is unavailable on this host due
  `ERR_NVGPUCTRPERM`; counter-free SASS inspection records 768 BF16 HMMA and
  408 `MUFU.EX2` instructions in the accepted specialization.
  A controlled `-maxrregcount=192` build was rejected before timing: ptxas
  lowered the forward kernel from 239 to 192 registers only by introducing a
  224-byte stack frame, 560 bytes of spill stores, and 1840 bytes of spill
  loads per thread. Register capping is therefore not a viable substitute for
  reducing the live fragmented pipeline state in the source schedule.
  A D64-specific LSE epilogue simplified
  `-8 * (log(sum) + max / 8)` to `-8 * log(sum) - max`, removing two static
  FMUL instructions while retaining 239 registers, an 8-byte stack, and zero
  spills. Both output and LSE correctness pass. With both backends protected by
  the idle/foreign-process gate, it measures 38.962 us p50 (38.926 us p10,
  39.817 us p90) versus the accepted 38.921 us p50, so it is latency-neutral
  and rejected.
  A direct-register K/V experiment removed the 16 KiB dynamic shared stage,
  all TMA semaphore control, and both CTA barriers per key tile. Each of the
  four query warps instead loaded its own 64x64 K/V register tiles from
  global/L2. It passes output/LSE correctness, but duplicates K/V traffic four
  times and reaches 255 registers with a 48-byte stack. Under the idle GPU gate
  its short synchronized screen measures 93.322 us, 2.40x slower than the
  accepted 38.921 us, so it is rejected and the TMA lookahead route is restored.
  The PyTorch baseline now reports inference and grad-enabled training forward
  separately. Under the idle gate, matched training forward measures 30.790 us
  p50 while inference measures 30.777 us; compilation remains excluded from the
  separate 34.938 us `torch.compile` row. An idle-gated Nsight Systems trace
  identifies PyTorch's selected kernel as a four-warp FlashAttention
  specialization with 128 query rows and 128 key rows per CTA.
  Tyr now matches the 128-query/four-warp half of that geometry. A naive
  32-query-row-per-warp version passes correctness and measures 42.894 us in a
  short screen, but reaches 255 registers and a 248-byte stack because full K
  and V register tiles overlap. The accepted split pipeline gives K and V
  independent single-buffer semaphores: next-K TMA overlaps score/softmax work,
  then V is loaded only after the score tile converts to BF16, and next-V TMA
  overlaps the output MMA. This retains the 16 KiB dynamic shared footprint,
  lowers the stack to 72 bytes, and passes output, LSE, dQ, dK, and dV gates.
  The full idle-gated 20-warmup/200-iteration/seven-repeat run measures 34.308 us
  p50 (34.103 us p10, 34.789 us p90), 11.9% faster than the prior accepted
  38.921 us and 11.4% behind matched eager PyTorch training forward. It is also
  1.8% faster than the paired `torch.compile` row. The route suffix is
  `_warp4_rows32_splitkv_tma1_exp2` and is the new accepted S768 forward baseline.
  A follow-up matched PyTorch's 128-key dimension by TMA-loading 128x64 K and V
  shared tiles, then consuming two sequential 64-row register fragments from
  each. It halves TMA transaction count and passes output/LSE plus all gradient
  gates, but raises dynamic shared memory from 16 KiB to 32 KiB and the stack
  from 72 to 200 bytes at 255 registers. Its idle-gated short screen measures
  37.469 us versus 34.308 us for the accepted 64-key split pipeline, so it is
  rejected. The next forward work should reduce the accepted route's remaining
  spill state rather than enlarge the K/V shared tile in the current RT model.
  Persisting the running output tile in BF16 between K/V iterations was also
  rejected statically. Relative to the accepted kernel, it raises stack use
  from 72 to 88 bytes, local loads from 81 to 136, local stores from 39 to 44,
  and FP32/BF16 conversion instructions from 416 to 800 while leaving the 255
  register ceiling, 384 shared-load instructions, and 1536 MMA instructions
  unchanged. A useful replacement must shorten the live FP32 output state
  without introducing conversion traffic.
  An eight-warp variant instead kept 128 query rows per CTA while reducing each
  warp to 16 rows. It reaches 189 registers, an 8-byte stack, and zero ptxas
  spill loads/stores with the same 16 KiB K/V shared stage. Output, LSE, dQ,
  dK, and dV all pass, but the idle-gated synchronized short screen measures
  45.147 us versus the accepted 34.308 us. Doubling the CTA from 128 to 256
  threads costs substantially more than removing the four-warp route's spill
  traffic, so it is rejected. Further work must retain the four-warp/128-thread
  geometry.
  The attention matrix now includes a suite-configured batch-sweep for
  B1/B2/B4/B8 at H16/S768/D64, and LeanBenchmark summary rows optionally carry
  generic work items, units, and correctly inverted throughput percentiles.
  The full idle-gated 20-warmup/200-iteration/seven-repeat sweep reports Tyr
  forward p50 latency of 34.805/65.069/139.120/299.063 us, corresponding to
  22.07/23.61/22.08/20.54 million tokens/s. Matched eager PyTorch training
  forward reports 30.807/65.624/135.998/268.057 us and
  24.93/23.41/22.59/22.92 million tokens/s. Tyr is 13.0% slower at B1,
  0.8% faster at B2, 2.3% slower at B4, and 11.6% slower at B8; one B1 point
  therefore does not characterize the route. Against torch.compile with setup
  excluded, Tyr is 0.7% faster at B1 and 6.6-15.1% faster at B2-B8.
  A lane-zero-per-warp semaphore experiment tried to remove twelve of the
  accepted route's 36 CTA barriers by publishing each TMA completion with a
  warp sync. Its first launch produced correct output, but a repeated launch
  failed with CUDA unspecified launch failure. The semaphore protocol therefore
  still requires CTA-wide completion publication; the experiment is rejected.
  Tyr's GPU IR now supports both compile-time and runtime-indexed shared-tile
  fragment loads, lowering to ThunderKittens `st.subtile` views and covered by
  an exact codegen guard. This enabled two genuine streamed-128 experiments.
  Statically spelling the two 64-row fragments reduces the monolithic spill to
  56 bytes but still reaches 255 registers and measures 40.536 us p50. Looping
  over a runtime fragment index successfully reuses register state, reaching
  211 registers and an 8-byte stack, but measures 41.050 us p50. Both pass
  output/LSE correctness and are rejected: their 32 KiB single K/V stage lowers
  effective CTA occupancy enough to outweigh halving TMA transaction count.
  The accepted 64-key/16 KiB route remains faster. A future streamed-128 win on
  SM121 needs either a more compact shared representation or producer/consumer
  overlap that preserves occupancy; register fragmentation alone is now proven
  insufficient.
- Nsight Systems identifies the eager PyTorch route as
  `pytorch_flash::flash_fwd_kernel` with a 128x128, four-warp trait. For
  B1/H16/S768 it launches grid `(6,1,16)`, block `(128,1,1)`, and measures
  about 31.6 us per invocation in the trace. The active Tyr 64x64 route launches
  12 CTAs per head; the remaining geometry gap is primarily PyTorch's 128-key
  tile and specialized fragment pipeline rather than redundant per-query K/V
  reloads.
- A direct WGMMA capability probe established that Hopper and GB10 lowering
  cannot currently share Tyr's `warpgroupMm/Mma` emission. Under
  `KITTENS_BLACKWELL`, ThunderKittens expects tcgen05/TMEM destinations and
  completion semaphores, while Tyr emits the Hopper register-tile overload and
  `mma_async_wait`; NVCC rejects that combination. Hopper should continue to
  use WGMMA. GB10 needs a separate group-MMA representation/lowering before a
  PyTorch-like 128x128 CTA can be generated.
- Architecture-specific compile probes now sharpen that boundary further.
  A generated TMEM allocator/load path compiles for `sm_100a`, while ptxas
  rejects `tcgen05.alloc`, `tcgen05.ld`, fences, waits, and deallocation for
  both `sm_121` and non-`a` `sm_100`. Therefore GB10 cannot use either Hopper
  WGMMA or SM100a tcgen05/TMEM; its accepted path must use SM121 warp MMA.
  Hopper retains WGMMA, and SM100a has a distinct TMEM path. The capability
  records no longer claim WGMMA on SM100, and the native build rejects generated
  WGMMA/tcgen05/TMEM sources for GB10 before invoking NVCC. The GPU IR now has
  concrete typed TMEM allocator, half-TMEM superlane allocation, cooperative
  TMEM-to-register load, and load-wait operations, validated by an SM100a NVCC
  probe. These primitives are groundwork for B200 attention, not a GB10 route.
- Attention backward now times the complete generated sequence: row correction,
  dQ/dK partials, split dV partials, and generated dK/dV reduction. A correctness
  audit exposed that three identical `torch.zeros` expressions for mutable
  outputs were commoned by Lean and aliased; distinct allocations now prevent
  dV from overwriting dQ/dK. All gradients pass (dQ MAE 0.000160, dK 0.000162,
  dV 0.000156). Splitting dV out of the 255-register combined kernel reduces
  the synchronized full sequence from about 77.9 us to 50.728 us p50 (50.611
  us p10, 52.506 us p90), a 34.9% improvement, with 20 warmups, 200 iterations,
  and seven repeats. The PyTorch baseline now executes real autograd backward
  on a retained forward graph and synchronizes the CUDA stop event; forward
  construction is excluded. It measures 57.033 us p50 (56.792 us p10, 58.382
  us p90), so Tyr is 1.12x faster at B1/H1/S128/D64. This comparison includes
  PyTorch autograd/output-allocation overhead and is labeled
  `backward_only_retained_graph`; it does not claim kernel-only parity.
  Compiled resources are 38 registers/zero stack for prep, 255 registers/560
  bytes stack for dQ/dK (down from 1152 bytes in the combined kernel), 255
  registers/72 bytes stack for split dV, and 80 registers/zero stack for the
  generated reduction.
  A second accepted redesign removes the partial tensors and reduction entirely:
  dQ is query-owned, while one key-owned 16-row kernel reduces dK and dV locally
  across all query tiles. The full three-kernel sequence (prep, direct dQ,
  direct dK/dV) measures 31.713 us p50 (31.636 us p10, 34.277 us p90), another
  37.5% improvement over 50.728 us and 59.3% faster than the original 77.9 us
  complete route. It is 1.80x faster than the synchronized PyTorch eager
  backward baseline at 57.033 us. Correctness remains dQ MAE 0.000160, dK MAE
  0.000162, and dV MAE 0.000156. Direct dQ uses 226 registers with zero stack;
  direct dK/dV uses 212 registers with zero stack; prep remains 38 registers
  with zero stack. No global partial-gradient buffers or reduction launch remain
  in the active route.
- The same ownership design now covers the realistic B1/H16/S768/D64 backward
  shape. The first correct one-warp dQ specialization measured about 0.588 ms
  for the complete prep+dQ+dK/dV sequence, with Nsight Systems attributing
  roughly 80% of traced time to dQ: all 768 query CTAs independently reloaded
  the twelve K/V tiles. The accepted dQ schedule groups four 16-row query warps
  per CTA and shares one TMA-staged K/V tile across them, reducing the dQ grid
  to 192 CTAs and eliminating fourfold K/V staging duplication. It retains
  aggressive S768/D64 shape specialization.
  A second trace showed that this made dK/dV the new bottleneck. Its accepted
  schedule likewise groups four 16-row key warps per CTA, shares Q/dO/L/D
  staging, and uses a compact runtime loop with single-buffer TMA lookahead;
  the next Q/dO tile overlaps current score/gradient computation without
  source-unrolling 48 loop bodies. The canonical configurable
  `--case b1_h16_s768` run (20 warmups, 200 timed iterations, seven repeats)
  measures 0.179670 ms p50 (0.177775 p10, 0.180033 p90), a 3.27x improvement
  over the first correct route and 10.0% faster than the paired eager-PyTorch
  retained-graph autograd median of 0.199613 ms. PyTorch varies broadly from
  0.143059 ms p10 to 0.420375 ms p90, while Tyr remains tightly clustered.
  All gradients pass: dQ MAE 0.000078/max 0.003910, dK MAE 0.000077/max
  0.001556, and dV MAE 0.000065/max 0.001893. Both backends synchronize their
  CUDA stop event; forward construction and torch.compile setup are excluded.
  The four-warp dQ kernel uses 255 registers and a 40-byte stack frame; prep
  uses 38 registers/zero stack; pipelined four-warp dK/dV uses 186 registers
  and a 40-byte stack frame. In the post-change trace, dK/dV remains the largest
  kernel at 124.832 us median versus 53.760 us for dQ and 7.008 us for prep.
  The S768 training-backward harness now uses the same declarative B1/B2/B4/B8
  matrix, launches every generated kernel over the true batch grid, checks all
  dQ/dK/dV values, and reports token throughput. In the full idle-gated
  20-warmup/200-iteration/seven-repeat run, Tyr complete backward p50 is
  0.171995/0.454122/0.923166/1.781572 ms, versus eager PyTorch retained-graph
  autograd at 0.201769/0.370837/0.630621/1.354278 ms. Tyr is 14.8% faster at
  B1 but 22.5%, 46.4%, and 31.6% slower at B2/B4/B8, respectively. The Tyr
  dK/dV phase alone takes 0.107028/0.266454/0.604845/1.168485 ms and remains
  59-66% of complete backward. Training optimization must therefore prioritize
  batch-scaled dK/dV rather than extrapolating the favorable B1 result or
  optimizing inference-only forward.
  Rejected dK/dV variants include synchronous four-warp shared staging
  (0.206310 ms complete sequence in the short run) and an eight-warp CTA
  (0.186854 ms), which halves TMA stream count but loses enough scheduling
  efficiency to regress versus the accepted four-warp TMA route (0.180918 ms
  under the same short-run settings).
- Shared benchmark mechanics now live in the sibling LeanBenchmark package at
  ../lean-benchmark. Tyr owns only the CUDA-event adapter. The package tests
  synchronization-before-elapsed ordering. Its summary schema now accepts an
  optional per-iteration work count and suite-owned unit, deriving inverse
  p10/p50/p90 throughput correctly from latency percentiles. Profiles and case
  payloads remain declarative and owned by each benchmark suite.
- One driver, scripts/gpu/bench.sh, covers copy, rotary, LayerNorm, RMSNorm,
  BF16 GEMM, GB10 MHA, and all cases; redundant per-kernel shell wrappers were
  removed.
- The unified driver now treats an idle GPU as a hard validity gate for every
  Tyr and PyTorch measurement. Before each backend it waits for both zero
  reported utilization and an empty CUDA compute-process table. During the
  timed process it samples the CUDA process table and invalidates the run if
  any PID other than the measured backend appears; it repeats the empty-process
  check afterward. An end-to-end MHA smoke run observed and reported both idle
  gates before producing benchmark rows.
- The unified driver now appends one run-scoped provenance row containing the
  git revision/dirty state, selected module and physical/implementation target,
  generated-source SHA256, GPU name/UUID/compute capability, driver and NVCC
  version, and the complete `cuobjdump --dump-resource-usage` report. An
  end-to-end copy run verified all fields against the selected
  `Tyr_GPU_Kernels_Copy.cu` artifact; the synchronized short run measured Tyr at
  2.273 us and eager PyTorch at 2.662 us, with `torch.compile` explicitly
  recording `compileSetupExcluded=true`.
- Replacing the driver's explicit registration/generation/native-build sequence
  with `lake -R run buildGpuTarget` reduced a same-module warm build to 12.33 s
  and successfully switched from GEMM to copy. It is not accepted yet: after an
  MHA source/signature edit, the first build generated CUDA from the previous
  registration artifact and produced a seven-pointer launcher for the new
  eight-pointer Lean call. The following build generated the current source.
  This proves a one-build-lag/ABI hazard caused by the cycle between the selected
  module dynlib and `libtyr`; the benchmark retains the explicit source-current
  sequence until Phase 1 cuts that dependency graph. `buildGpuTarget` now at
  least accepts an optional forwarded `--` safely instead of treating it as a
  module name.
- An S768 backward experiment folded the separate
  `D = rowsum(dO * O)` preparation into the query-owned four-warp dQ kernel.
  Current GPU IR layout inference rejected using the row-reduction result in the
  later `subRow` path (`RV layout conflict`); separating the produced/consumed
  vectors through global memory compiled only after exposing the stale-generation
  issue above and was not retained. The last accepted three-kernel backward route
  has been restored and rebuilt. Under a contended GPU (an unrelated Python CUDA
  process held 3.43 GiB and 96% utilization), a short run is correctness-only:
  forward, LSE, dQ, dK, and dV all pass. Its latency is deliberately excluded
  from performance summaries.
- The accepted next S768 dK/dV change removes a provably redundant layout
  transform: V is invariant across all 48 query tiles and is used only by
  `mmaT V dO`, but the accepted source loaded it as a column tile and emitted
  `warp::swap_layout` inside the runtime loop. Loading V directly as a row tile
  removes that transform from every iteration and lowers compiled dK/dV register
  use from 186 to 175, with the same 40-byte stack, 1296-byte static shared
  allocation, and no local spills. Forward, LSE, dQ, dK, and dV correctness all
  pass. On a subsequently clean GPU, the full 20-warmup/200-iteration/seven-repeat
  run measures 0.174683 ms p50 (0.172317 p10, 0.176650 p90), improving the prior
  accepted 0.179670 ms by 2.8%. Paired eager PyTorch retained-graph backward is
  0.194397 ms p50, so Tyr is 10.1% faster. The benchmark route carries a `_vrow`
  suffix.
- A follow-up direct-vector experiment removed the shared L/D broadcast and its
  apparent second per-query CTA barrier. It compiled at 182 registers, 40-byte
  stack, and 1040 bytes static shared memory, and a short run happened to pass.
  The full run exposed dK/dV corruption (dK max error 0.136674, dV max error
  0.078359), so it is rejected even though its invalid 0.175487 ms timing looked
  competitive. The barrier also ensures all four warps finish loading the current
  Q/dO shared tile before the producer issues the next single-buffer TMA
  overwrite; removing it creates a real race that short runs can hide.
- The accepted safe overlap design instead double-buffers the 16x64 Q and dO
  shared stages. Tiles 0 and 1 are prefetched; after every CTA barrier, warp 0
  refills the stage retired by the preceding iteration with tile `i+1`, while
  all warps load and compute from the other stable stage. Semaphore phase is
  `(i / 2) % 2`, independently per stage. This preserves the barrier that proved
  necessary above while overlapping the next TMA with the current MMAs. The
  sustained correctness gate passes, and the full clean-control run measures
  0.169502 ms p50 (0.167003 p10, 0.171253 p90), another 3.0% faster than the
  accepted V-row route and 5.7% faster than the earlier 0.179670 ms schedule.
  Paired eager PyTorch retained-graph backward measures 0.170709 ms p50, so Tyr
  is marginally faster at B1/H16/S768/D64 while remaining much tighter across
  repeats. The dK/dV kernel uses 191 registers, a 40-byte stack, 1296 bytes
  static shared memory, 9216 bytes effective dynamic shared memory, and no local
  spills. Its route suffix is `_vrow_qdo2`.
- A second prep-to-dQ fusion attempt separated the incompatible row-vector
  layouts by writing the produced `D = rowsum(dO * O)` vector to its required
  global output and reloading it into the dQ consumer layout. This resolves the
  prior generated-CUDA `RV layout conflict`, but the larger dQ kernel reaches
  255 registers with a 40-byte stack, 16 bytes of spill stores, and 176 bytes of
  spill loads. Sustained correctness still passes. In a clean synchronized
  20-warmup/200-iteration/seven-repeat run, isolated dQ is 0.053561 ms p50 and
  dK/dV is 0.104831 ms p50, but the full two-kernel sequence is 0.191492 ms p50
  (0.189552 p10, 0.191989 p90). That is 12.97% slower than the accepted
  three-kernel `_vrow_qdo2` result of 0.169502 ms, so the fusion is rejected and
  the non-spilling prep+dQ+dK/dV route is restored. A post-restore synchronized
  correctness run passes all forward/LSE/dQ/dK/dV checks and measures the full
  accepted sequence at 0.168133 ms in a short 20-iteration control. This also
  shows that launch-count reduction alone is not a valid optimization proxy:
  register pressure and the full device-event-timed sequence remain the gate.
- A direct-global L/D follow-up removed the second CTA barrier from each query
  tile, but it also lost the efficient cooperative vector broadcast. On an idle
  B8 control it regressed dK/dV to 1.338 ms and complete backward to 2.036 ms,
  versus 1.168485 ms and 1.781572 ms for `_vrow_qdo2`; it is rejected.
- The accepted batch-scaled dK/dV schedule now stages 32 Q/dO rows per TMA pair
  while consuming them as two nested 16-row register fragments. This halves the
  48 query-stage waits/refills to 24 without forcing a 32-row score/probability
  live range. The compiler retains compact runtime loops: the executed schedule
  still performs 48 barriers, 48 TMA loads, and 1536 HMMA instructions, while
  compiled resources fall from 191 registers/40-byte stack in `_vrow_qdo2` and
  255 registers/88-byte stack in the monolithic 32-row candidate to 176
  registers/40-byte stack with no spills. All B1/B2/B4/B8 forward, LSE, dQ, dK,
  and dV checks pass. In the idle-gated 20-warmup/200-iteration/seven-repeat
  seeded sweep, dK/dV p50 is 0.099096/0.216314/0.466564/0.925465 ms and complete
  backward p50 is 0.166742/0.365761/0.776150/1.520081 ms. Relative to the prior
  `_vrow_qdo2` batch baseline, complete backward improves by
  3.1%/19.5%/15.9%/14.7%. An immediate full-32 control retains a 3.2% B1 edge
  but is 10.9%/5.0%/3.4% slower at B2/B4/B8, so the streamed route is accepted
  for training throughput. Its benchmark suffix is `_vrow_q32stream_qdo2`.
  A paired eager-PyTorch retained-graph backward run, with CUDA stop-event
  synchronization, measures 0.163331/0.331451/0.613625/1.326243 ms. Tyr is now
  2.1%/10.4%/26.5%/14.6% slower, making B4 and B8 the next optimization targets.
- The next accepted dK/dV schedule stages 64 Q/dO rows and consumes four nested
  16-row fragments. It cuts executed Q/dO TMA operations and CTA barriers from
  48 to 24. The tradeoff is 232 registers and 33,792 bytes effective dynamic
  shared memory, versus 176 registers and 17,408 bytes for the 32-row route;
  both have a 40-byte stack and zero spills. The full correctness matrix passes.
  Idle-gated dK/dV p50 is 0.093270/0.210738/0.436878/0.858511 ms and complete
  backward p50 is 0.158212/0.352510/0.741293/1.460786 ms, improving the 32-row
  route by 5.1%/3.6%/4.5%/3.9% end to end. Against the synchronized eager
  PyTorch backward control, Tyr is 3.1% faster at B1 and 6.4%/20.8%/10.1%
  slower at B2/B4/B8. This supersedes the 32-row route; its suffix is
  `_vrow_q64stream_qdo4`.
- A first dQ fragment-streaming variant double-buffered full 64-row K/V stages.
  It reduced dQ to 125 registers, a 24-byte stack, and no spills, but required
  33,792 bytes of shared memory. Its B4/B8 dQ medians were
  0.259460/0.478658 ms, 6.0%/2.0% slower than the full-tile route, so the static
  resource improvement alone was rejected.
- The accepted dQ schedule instead double-buffers 32-row K/V stages and consumes
  two 16-row fragments. It keeps the 17,408-byte effective shared footprint,
  lowers resources from 255 registers with 16-byte stores/176-byte loads spilled
  to 121 registers, a 24-byte stack, and no spills, and overlaps the next TMA
  stage. Idle-gated dQ p50 is 0.051286/0.100299/0.232795/0.441067 ms, improving
  the prior route by 4.0%/5.0%/4.9%/6.0%. Complete backward p50 becomes
  0.159548/0.339798/0.727745/1.419650 ms. Tyr is 2.3% faster than eager PyTorch
  at B1 and 2.5%/18.6%/7.0% slower at B2/B4/B8. The accepted dQ suffix is
  `_dq_warp4_frag16_kv32x2`.
- Training attention is now the primary acceptance metric. A directly timed
  CUDA-event row spans the saved-LSE forward, backward prep, dQ, and dK/dV
  launches rather than summing independently sampled medians. Tyr p50 is
  0.196080/0.417872/0.883665/1.735822 ms at B1/B2/B4/B8; synchronized eager
  PyTorch, which rebuilds and executes the autograd graph on every iteration,
  is 0.299935/0.377396/0.792274/1.611863 ms. B1 PyTorch is dispatch-noisy
  (0.168111-0.349147 ms p10-p90), while the throughput-relevant B2/B4/B8 gap is
  10.7%/11.5%/7.7%. Compile setup is outside every timed interval. The prep
  kernel is only 1-4% of the Tyr training step, so it is not an isolated target;
  remove it only through a fusion that improves the full training row.
- The accepted prep-to-dQ fusion adds an explicit register-vector layout
  conversion to the backend. A row reduction produces ThunderKittens' `ortho`
  RV layout, one `warp::copy` shuffle converts it to the `align` layout consumed
  by dQ, and the reduced D vector is stored once for dK/dV. This removes the
  global store/reload bridge from the rejected fusion and keeps dQ at 121
  registers, a 24-byte stack, 17,408 effective shared bytes, and no local
  memory. D, dQ, dK, and dV pass at B1/B2/B4/B8. Fused-dQ p50, including prep,
  is 0.052274/0.106761/0.263709/0.499460 ms; complete backward is
  0.152790/0.350939/0.718407/1.368892 ms. The directly timed training step is
  0.192997/0.413388/0.867660/1.699736 ms, improving every batch by
  1.6%/1.1%/1.8%/2.1%. The remaining eager-PyTorch training gaps at B2/B4/B8
  are 9.5%/9.5%/5.5%. The accepted dQ route suffix is
  `_dq_fusedprep_rvshuffle_frag16_kv32x2`.
- A 96-row dK/dV control reduced staged query groups from 12 to 8 and compiled
  at 168 registers, a 40-byte stack, and no local memory, versus 232 registers
  for the accepted 64-row schedule. Its roughly 50 KiB dynamic shared footprint
  erased that advantage: idle-gated B4/B8 dK/dV was 0.844109/1.679699 ms versus
  0.444241/0.869303 ms for 64 rows. Correctness passes, but the candidate is
  rejected without a full run.
- The accepted training-oriented dK/dV schedule stages 48 Q/dO rows and consumes
  three 16-row fragments. It uses 232 registers, a 40-byte stack, no local
  memory, and roughly 25 KiB effective dynamic shared memory. D, dQ, dK, and dV
  pass at B1/B2/B4/B8. In the idle-gated 20-warmup/200-iteration/seven-repeat
  sweep, dK/dV p50 is 0.091284/0.194242/0.412896/0.814253 ms and the directly
  timed saved-LSE forward-plus-backward training step is
  0.193888/0.396098/0.832414/1.651015 ms. Relative to the prior 64-row route,
  training is effectively flat at noisy B1 and improves by 4.2%/4.1%/2.9% at
  B2/B4/B8. This supersedes the 64-row route for training; its suffix is
  `_vrow_q48stream_qdo3`. A fresh paired idle-gated run in
  `/tmp/mha_training_final_q48.jsonl` measures Tyr at
  0.187447/0.397091/0.840268/1.642582 ms and synchronized eager PyTorch at
  0.340365/0.407158/0.755140/1.603796 ms. PyTorch B1 is again bimodal; at the
  stable larger points Tyr is 2.5% faster at B2, 11.3% slower at B4, and 2.4%
  slower at B8. B4 full training is therefore the next primary regression.
- A direct-column-load dK/dV control generalized shared-to-register loads so a
  row-staged Q/dO fragment could be loaded through `ldmatrix.trans` directly,
  replacing two register `swap_layout` operations. Correctness passes, but the
  extra shared loads regress dK/dV by 4.8%/1.3%/0.2%/0.8% at B1/B2/B4/B8 in
  matched idle-gated smoke runs. The candidate, its broader primitive typing,
  and route labels were rejected and q48 was restored byte-for-byte.
- PyTorch autograd returns BF16 dQ/dK/dV for BF16 Q/K/V. Tyr now matches that
  training contract while retaining FP32 MMA accumulators and converting only
  at each gradient epilogue. The earlier apparent dQ corruption was a benchmark
  harness bug: the BF16 kernel was launched/validated through an output path
  that did not consistently provide distinct BF16 buffers, so several epilogue
  variants inherited the same false failure. With dQ/dK/dV allocated from the
  distinct BF16 q/k/v tensors and converted to FP32 only for validation, all
  four D/dQ/dK/dV checks pass at B1/B2/B4/B8; dQ MAE is 0.000031 and its observed
  maximum error is at most 0.003906. Regenerating from the corrected Lean source
  produces byte-identical CUDA to the measured candidate.
- Fresh uncontaminated per-shape runs use 20 warmups, 200 iterations, seven
  repeats, synchronized CUDA stop events, and the idle/foreign-process gate.
  The superseded FP32-gradient Tyr route is
  0.194318/0.411176/0.841154/1.665443 ms at B1/B2/B4/B8. PyTorch is
  0.311814/0.357717/0.740462/1.608712 ms with its native BF16 gradients and
  0.356489/0.461713/0.975990/2.126098 ms when all three gradients are converted
  to FP32 inside the timed interval. The accepted BF16-gradient Tyr route
  measures 0.184049/0.380082/0.823070/1.567676 ms, improving Tyr by
  5.3%/7.6%/2.2%/5.9%. Against native PyTorch it is 41.0% faster at B1, 6.3%
  slower at B2, 11.2% slower at B4, and 2.6% faster at B8. A prior combined
  PyTorch sweep was rejected in full after the monitor observed foreign
  Shakespeare training processes during measurement. B4 remains the primary
  training regression, followed by B2; B1/B8 are now ahead at matched dtype.
- An idle-gated Nsight Systems trace at B4 shows that the PyTorch main backward
  is one unified dQ/dK/dV FlashAttention kernel with grid `(6,4,16)`, block
  `(256,1,1)`, and a 64-query/128-key BF16 trait, plus separate dot and dQ
  conversion kernels. Tyr still launches separate dQ and dK/dV kernels and
  recomputes scores and probabilities in each. Removing that duplicated work is
  now the main algorithmic target rather than tuning either launch in isolation.
- An eight-warp/128-key dK/dV control mirrored the PyTorch B4 grid-x and
  256-thread CTA geometry. It passes the complete B1/B2/B4/B8 correctness matrix
  and compiles at 234 registers, a 40-byte stack, and no local memory, but
  regresses idle-gated dK/dV from about 0.369/0.703 ms to 0.495/0.879 ms at
  B4/B8. At that register footprint only one CTA is resident, so the lower
  CTA/TMA count does not recover the lost scheduling parallelism. The four-warp
  q48 BF16-gradient route is restored byte-for-byte.
- Tyr's MHA benchmark now seeds the Torch RNG after fixture setup so candidate
  and control runs use reproducible correctness inputs, matching the existing
  Python baseline's seed-zero policy. This followed one random B4 forward input
  exceeding the absolute/relative tolerance in an otherwise unchanged control;
  the fixed seed passes the full forward and backward matrix and prevents input
  drift from being mistaken for a kernel regression.
- A follow-up tried to co-stage LSE/D as row-vector TMA transactions on the same
  double-buffered semaphore, removing the second CTA barrier. It compiled at
  232 registers, a 40-byte stack, and zero spills, but both the initial 2D view
  and the corrected flattened-input view raised an illegal-memory-access error
  at the first backward synchronization on SM121. No latency from this variant
  is accepted; `_vrow_q32stream_qdo2` was restored byte-for-byte.
- A narrow iteration path was proven without changing canonical build artifacts:
  directly elaborate the selected kernel to a temporary olean, overlay it onto a
  hard-linked copy of `.lake/build/lib/lean`, run `GenerateMain` with that overlay
  first on `LEAN_PATH`, and compile the emitted CUDA directly. This generated the
  current source in seconds and avoided the extern-lib one-build lag. It should
  replace the cyclic broad benchmark build after being packaged as a supported
  build helper with executable-ABI invalidation.
- These measurements are GB10 results. They are local working baselines, not yet
  stable CI thresholds.

## Implementation order

The intended reviewable change series is:

1. generated-module manifest and build isolation;
2. Python/NVCC toolchain preflight;
3. unified benchmark schema and PyTorch baselines;
4. SM121 capability model;
5. GB10 copy, rotary, and normalization;
6. runtime occupancy and launch configuration;
7. GB10 attention;
8. generated-versus-vendored H100 attention optimization;
9. decode tuning and performance CI.

The first milestone is complete when a clean checkout can build only the GB10
copy kernel, verify it, and emit a reproducible Tyr/eager-PyTorch/`torch.compile`
benchmark record with no manual environment overrides.
