# GPU Native Training Todo

This is the execution tracker for getting from the current ThunderKittens-based
GPU kernel surface to one narrow Tyr-native training path that no longer depends
on PyTorch for forward/backward math at runtime.

The intent is not to "remove PyTorch everywhere" first. The intent is:

1. expand the kernel/runtime surface until one closed training slice exists,
2. keep Tyr's existing PyTorch-backed runtime/model path as the first oracle during bring-up,
3. then run the same training slice in native-only mode.

## Goal

Ship one small decoder-style GPU training executable that can:

- allocate parameters and activations on GPU,
- run forward,
- run backward,
- update parameters,
- step for multiple iterations,
- decrease loss,
- avoid PyTorch math calls in native mode.

The first target should be intentionally small and deterministic:

- tiny decoder-style block,
- synthetic token batches,
- batch size `2`,
- sequence length `128`,
- `1` or `2` heads,
- head dimension `64`,
- dropout disabled,
- fixed seed,
- single GPU.

## Current State

What is already in reasonably good shape:

- codegen floor/family split is now explicit in behavior: emitted guard family is
  separate from the capability floor,
- LeanTest codegen suites are runnable through
  `./scripts/gpu/test_codegen_leantest.sh`,
- Tyr already has a substantial PyTorch-backed execution path under the
  `Tyr.Torch` / `torch.*` stack, including training-oriented code such as
  `Examples/GPT/Train.lean`,
- end-to-end PyTorch parity has already been exercised for:
  - `copy`,
  - `rotary`,
  - `flashattn`,
  - `mha_h100`,
- vendored ThunderKittens references can be used as a second oracle path,
- GB10-specific reduced MHA kernels now exist under `Tyr.GPU.Kernels.MhaGB10`.

What is not true yet:

- there is no closed native training slice,
- PyTorch is still doing too much fixture/oracle/runtime work in examples,
- the runtime layer above raw launchers is too thin,
- training-critical kernels are still missing or not connected,
- no tiny decoder train-step executable exists in `Examples/GPU/`.

## Non-Goals For The First Milestone

- full NanoChat or distributed training,
- checkpointing or dataset IO complexity,
- mixed-precision/autocast policy work,
- broad kernel-family expansion without a training dependency,
- replacing PyTorch as an oracle before the native path is stable.

## Verification Ladder

Use the cheapest verification path that still gives a trustworthy comparison:

1. Tyr-native GPU path vs Tyr's existing PyTorch-backed path.
2. Tyr-native GPU path vs direct PyTorch tensor math where needed.
3. Tyr-native GPU path vs vendored ThunderKittens reference kernels when the
   comparison is specifically about ThunderKittens fidelity rather than general
   model correctness.

This matters because the `Tyr.Torch` stack already gives:

- parameter structures,
- training-loop shapes,
- optimizer behavior,
- loss definitions,
- model wiring,
- a Lean-native way to compare tensors without immediately dropping into ad hoc
  Python harnesses.

So the first tiny decoder executable should prefer:

- shared parameter/init layout between native and Torch-backed Tyr modes,
- shared synthetic batches,
- shared logging/metrics structure,
- direct tensor-by-tensor comparisons inside Lean,

before introducing additional external oracle glue.

## Architecture Rules

- Keep capability floor and emitted/build family separate.
- Do not use `.SM100` as shorthand for "Blackwell-family".
- GB10-compatible reduced kernels should remain `.SM90` floor when they only
  require Hopper-level semantics, while selecting `.Blackwell` as the emitted
  family.
- Only add true `.SM100` kernels when they actually require Blackwell-only
  semantics such as TMEM, tcgen05/TMEM destinations, or 2-CTA cluster paths.

## Fusion Rule

Design the runtime and kernel surface so it can support megakernel-style
execution rather than forcing everything through many small launches.

Concretely:

- prefer fused kernels over decomposed launch chains when the dataflow is stable,
- do not let the runtime wrapper layer hard-code an op-by-op scheduler,
- keep intermediate buffers and workspace ownership explicit so they can later be
  internalized into larger fused kernels,
- treat "tiny decoder training" as a stepping stone toward larger fused
  train-step regions, not as the final architecture.

For the first milestone this does not mean "build one giant kernel immediately".
It means the interfaces should allow us to move in that direction without
rewriting the stack.

## `feat/cutile-ir` Consideration

There is now a separate `feat/cutile-ir` branch that introduces a substantial
TileIR stack:

- `Tyr/GPU/Codegen/TileIR/*`,
- a cutile-style frontend,
- dedicated TileIR tests/tooling.

This matters for the native-training plan because a more structured tile-level
IR may be a better long-term substrate for:

- larger fusion regions,
- megakernel-style train-step lowering,
- analysis/passes before backend emission,
- avoiding ad hoc launcher orchestration as the fused surface grows.

So before hardening too much of the native-training runtime layer, we should
evaluate whether the first fused train-step regions should target:

1. the current kernel/codegen path,
2. the TileIR/cutile path,
3. or a mixed strategy where the current path lands the first milestone and
   TileIR becomes the fusion-oriented follow-up.

For now, the correct constraint is:

- do not design the runtime wrapper layer in a way that would block a later
  TileIR-backed fused path.

## First Native Training Milestone

Create a narrow executable such as:

- `Examples/GPU/TrainTinyDecoder.lean`

This executable should support two modes:

- `--oracle`
  - runs Tyr kernels,
  - compares selected tensors and gradients against the existing Torch-backed
    Tyr path first, with direct PyTorch and vendored ThunderKittens as fallback
    oracle layers,
  - used for bring-up and debugging.
- `--native`
  - does not call PyTorch math,
  - uses only Tyr runtime wrappers and GPU kernels,
  - used to demonstrate real native training.

The first model slice should be:

- token embedding or a simplified input projection,
- one attention block,
- one output projection,
- cross-entropy loss,
- backward pass,
- SGD or AdamW update.

## Exit Criteria

We can call the first milestone complete when all of the following are true:

- one executable runs multiple train steps with no PyTorch math in native mode,
- loss decreases over a short deterministic run,
- per-parameter updates occur on GPU,
- the same executable has an oracle mode that can compare against PyTorch,
- LeanTest and script-driven regression coverage exists for the train step,
- GB10/Hopper family selection is explicit and tested for the selected kernels.

We should consider the design healthy only if the same runtime structure can
later absorb more fusion, for example:

- fused attention + residual + normalization regions,
- fused backward/update regions,
- reduced intermediate materialization between train-step stages.

## Dependency-Ordered Todo

### P0: Runtime Layer For Native Training

- [ ] Create a thin runtime wrapper layer above raw launchers.
- [ ] Centralize:
  - tensor allocation,
  - stream selection,
  - workspace management,
  - launch config calculation,
  - error reporting,
  - deterministic seed handling.
- [ ] Move reusable logic out of `Examples/GPU/Run*.lean` into shared helpers.
- [ ] Keep the runtime layer small and explicit; do not hide kernel selection
  behind an opaque framework.
- [ ] Keep the runtime layer megakernel-friendly:
  - allow a fused train-step wrapper to own workspace and launch sequencing,
  - avoid APIs that require every intermediate to become a public tensor,
  - avoid assuming every op is an independently launched unit.
- [ ] Keep the runtime layer compatible with a possible future TileIR-backed
  fused lowering path from `feat/cutile-ir`.

Suggested landing area:

- `Examples/GPU/Runtime.lean` or `Tyr/GPU/Runtime/*.lean`

### P1: Training-Critical Kernel Inventory

- [ ] Write down the exact closed operator set for the first train step.
- [ ] Classify each operator as:
  - existing and validated,
  - existing but not validated on hardware,
  - missing runtime wrapper,
  - missing kernel surface,
  - missing backward.

The first slice should require:

- [ ] input embedding or simplified token gather path,
- [ ] QKV projection GEMMs,
- [ ] output projection GEMM,
- [ ] rotary forward,
- [ ] rotary backward,
- [ ] MHA forward,
- [ ] MHA backward,
- [ ] residual add,
- [ ] layernorm forward,
- [ ] layernorm backward,
- [ ] logits projection,
- [ ] cross-entropy loss,
- [ ] parameter-gradient accumulation,
- [ ] optimizer update.
- [ ] For each item, mark whether it should remain standalone or be targeted for
  early fusion into a larger train-step region.

### P2: Loss And Reduction Surface

- [ ] Add native loss/reduction kernels needed for training.
- [ ] Prefer the smallest closed set over a general-purpose loss library.
- [ ] Start with:
  - softmax/logsumexp reduction pieces,
  - cross-entropy forward,
  - cross-entropy backward,
  - argmax/top-k only if required for debugging, not for the train path.

Likely landing area:

- `Tyr/GPU/Kernels/Loss.lean`
- `Examples/GPU/RunLoss*.lean`

### P3: Optimizer Surface

- [ ] Implement the smallest useful optimizer path first.
- [ ] Start with SGD if it reduces bring-up risk.
- [ ] Add AdamW only after the native train step is stable.
- [ ] Keep optimizer updates on GPU.
- [ ] Make the optimizer path reusable rather than embedding it directly into an
  example executable.

Likely landing area:

- `Tyr/GPU/Kernels/Optimizer.lean`
- `Examples/GPU/OptimizerRuntime.lean`

### P4: Tiny Decoder Forward-Only Bring-Up

- [ ] Create `Examples/GPU/TrainTinyDecoder.lean`.
- [ ] Start with forward-only execution and logging.
- [ ] Allow simplified input projection if embedding gather is not ready yet.
- [ ] Use fixed synthetic token batches and fixed parameter initialization.
- [ ] Make intermediate tensors easy to dump in oracle mode.
- [ ] Reuse Tyr-side parameter layouts and tensor conventions from the existing
  Torch-backed training/examples path whenever possible.

The first forward path should prove:

- [ ] parameter allocation,
- [ ] attention launch wiring,
- [ ] residual/layernorm wiring,
- [ ] logits output shape correctness,
- [ ] loss computation.
- [ ] The forward path should already identify the first fusion boundary, e.g.
  "attention + residual + norm" or "projection + loss prelude".

### P5: Tiny Decoder Backward Completion

- [ ] Wire native backward for the selected forward path.
- [ ] Ensure every trainable parameter receives a gradient.
- [ ] Do not fall back to PyTorch autograd in native mode.
- [ ] Keep oracle mode capable of comparing:
  - forward outputs,
  - loss,
  - selected activations,
  - selected parameter gradients.

### P6: Native Train-Step Loop

- [ ] Add repeated train-step execution in `--native` mode.
- [ ] Log:
  - step,
  - loss,
  - gradient norm,
  - update norm,
  - selected kernel family/target.
- [ ] Prove that the first few steps reduce loss on the synthetic task.
- [ ] Add a short smoke mode for CI/local regression.
- [ ] Add launch-count and workspace diagnostics so we can see whether the train
  step is converging toward a fused/megakernel-friendly shape rather than
  exploding into many tiny launches.

### P7: Oracle And Regression Coverage

- [ ] Add LeanTest coverage for the tiny decoder train step.
- [ ] Add a script-driven e2e runner similar to the current GPU parity scripts.
- [ ] Keep three lanes:
  - codegen/unit,
  - operator parity,
  - end-to-end train-step.
- [ ] Treat the train-step tests as first-class regression tests, not examples
  that happen to print metrics.

Suggested tests:

- [ ] Tyr-native vs Torch-backed Tyr forward parity,
- [ ] Tyr-native vs Torch-backed Tyr gradient parity,
- [ ] deterministic forward parity,
- [ ] deterministic loss parity,
- [ ] selected gradient parity,
- [ ] one-step parameter update parity,
- [ ] short-horizon native loss decrease,
- [ ] GB10/Hopper family selection sanity for the chosen kernel path.

### P8: Blackwell/GB10-Specific Follow-Ups

- [ ] Keep the first train-step path shared where possible.
- [ ] Add Blackwell-family specializations only where they materially improve the
  chosen training slice.
- [ ] The first candidates are:
  - reduced GB10 MHA path,
  - Blackwell GEMM path when used in the tiny decoder,
  - only then TMEM/tcgen05 refinements if the model slice needs them.

### P9: Megakernel Follow-Up

- [ ] After the first native train step is stable, identify the highest-value
  fusion boundary for a megakernel-style path.
- [ ] Start with one fused region rather than trying to collapse the entire
  train step at once.
- [ ] Candidate first regions:
  - attention + residual + layernorm forward,
  - loss + gradient seed generation,
  - optimizer update epilogue for a parameter shard.
- [ ] Ensure test coverage can compare fused and unfused paths on the same
  deterministic inputs.
- [ ] Decide whether the first fused region should stay on the current kernel
  path or move onto the TileIR/cutile path.

## Suggested File Targets

These are the most likely first landing points:

- `dev/gpu_native_training_todo.md`
- `Examples/GPU/TrainTinyDecoder.lean`
- `Examples/GPU/Runtime.lean`
- `Examples/GPU/TrainTinyDecoderOracle.lean`
- `Tests/TestGPUTinyDecoderTrain.lean`
- `Tests/RunTestGPUTinyDecoderTrain.lean`
- `scripts/gpu/test_tiny_decoder_train.sh`
- `Tyr/GPU/Kernels/Loss.lean`
- `Tyr/GPU/Kernels/Optimizer.lean`

## First Implementation Slice

The first slice should be deliberately narrow:

- [ ] create the tiny decoder executable shell,
- [ ] choose the minimal input path:
  - real embedding gather if ready,
  - otherwise a temporary dense input projection,
- [ ] identify the closest Torch-backed Tyr training/model path to mirror for
  parameter layout and loss wiring,
- [ ] wire forward-only attention + output projection + loss,
- [ ] record the first intended fusion boundary while doing the forward wiring,
- [ ] add oracle-mode comparison for forward and loss,
- [ ] only then add backward and optimizer update.

This is the highest-leverage order because it produces an end-to-end spine early
without pretending the whole training stack is already native.

## Immediate Next Steps

- [ ] finalize the operator inventory for the tiny decoder path,
- [ ] decide whether the first input path is embedding-gather or dense input projection,
- [ ] decide which existing Torch-backed Tyr path is the primary oracle
  reference for the tiny decoder slice,
- [ ] land the runtime wrapper layer used by both parity runners and training examples,
- [ ] create the first `TrainTinyDecoder` forward-only executable,
- [ ] add oracle-mode forward/loss comparison before native backward work.
