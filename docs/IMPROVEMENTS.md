# Improvement suggestions

Consolidated findings collected while writing the `docs/` guides. Every item
was verified against the source at the time of writing; a handful of the
highest-impact claims were additionally re-verified during consolidation.
These are suggestions, not committed work — each item has a tag, a precise
reference, and a one-line suggested action.

Tags: **[bug-risk]** likely wrong behavior · **[correctness]** misleading or
incomplete semantics · **[docs]** stale/missing documentation ·
**[testing]** missing coverage · **[cleanup]** dead code / duplication ·
**[performance]** avoidable cost.

## Core

- **[docs]** `Tyr/Torch.lean:20` module docstring references a `torch.nanoproof`
  submodule that does not exist. → Fix the docstring.
- **[docs]** `Tyr/Torch.lean:1442` `focal_loss` docstring points at a
  nonexistent `nn.cross_entropy_loss`. → Fix the reference.
- **[correctness]** Generic `Tensor.matmul` takes its result shape from the
  total `matmulShape` (`Tyr/Basic.lean:179`), which returns junk `#[]` on
  invalid input, so inner-dim mismatches compile and fail only inside
  libtorch. Only `mm`/`linear`/`bmm4d`-family ops check statically. → Route
  through the checked `TensorSpec.matmulShape?` or document the gap.
- **[correctness]** The broadcast `HMul (T s1) (T s2) (T s2)` instance captures
  bare numeric literals, so `w * 0.02` fails with `OfScientific (T …)`;
  `(0.02 : Float)` works. → Add an `OfScientific` path or document the idiom.
- **[docs]** `deriving TensorStruct` and even the `T s` leaf instance require
  `import Tyr.Module.Derive`; the `TensorStruct.lean` docstring never says so.
  → State the import requirement in the docstring.
- **[testing]** No dedicated test suite for the `TensorStruct` traversal class
  (existing suites cover only schema/flatten). → Add `Tests/TestTensorStruct.lean`.
- **[cleanup]** `class differentiable` (`Tyr/Torch.lean:293`) has a single
  in-file instance and is effectively dead. → Remove or wire it up.

## Serialization

- **[performance]** Elaboration-time SafeTensors introspection reads whole
  shards (`Tyr/SafeTensors/Schema.lean:272-273`). → Header-only reads.
- **[docs]** `safetensors_type_provider` failure is a generic message that
  swallows the underlying error (`TypeProvider.lean:628-632`). → Surface it.
- **[testing]** No end-to-end checkpoint round-trip test and no Hub test
  coverage; `Tests/TestCheckpoint.lean:69` writes the untracked
  `dummy_path.ckpt/` directory into the repo root. → Round-trip test + write
  to a temp dir.

## Data

- **[docs]** `Tyr/Data/Pretraining.lean:22-26` still claims the parquet FFI
  declarations are placeholders although `cc/src/tyr_parquet.cpp` implements
  them. → Update the comment.
- **[cleanup]** The `Tyr/Data.lean` umbrella omits `Download`/`HuggingFace`;
  `Tyr.lean` does not import `Tyr.Text`. → Align umbrellas with reality.

## Autodiff

- **[cleanup]** `registerGpuVJPRule` has zero call sites repo-wide, so the GPU
  VJP registry is always empty, and `Tyr.GPU.AD.init` is never invoked.
  → Wire the registry up or remove the scaffolding.
- **[correctness]** The placeholder/hybrid AD rule packs
  (`Tyr/AD/Rules/RulePackKStmt.lean:1469-1580`) are numerically meaningless.
  → Mark experimental prominently or remove.

## Modules and modular

- **[bug-risk]** `Conv2d.outShape` (`Tyr/Module/Conv2d.lean:33`) returns the
  *input* channel count instead of `out_channels`. Re-verified. → Fix.
- **[cleanup]** The `Tyr/Module.lean` umbrella omits `RMSNorm`; the `⊛` infix
  has zero call sites; `Affine`/`Conv2d` are legacy; no `NormedModule`
  instance exists for `RMSNorm`. → Umbrella fix + prune or adopt.

## Optimization

- **[correctness]** NorMuon silently ignores `weightDecay`/`wdMul`
  (`Tyr/Optim/NorMuon.lean:223-224`). → Implement or reject the arguments.
- **[bug-risk]** `stepMatrixSingle` silently resets state on shape change.
  → Fail loudly or document the reset.
- **[docs]** Grassmann `distance` docstring says geodesic; the implementation
  is chordal. → Make doc and code agree.
- **[cleanup]** Gradient clipping is a TODO (`Tyr/Optim.lean:299`).

## Distributed and pipeline

- **[bug-risk]** `Pipeline.spawnBackground` awaits the task inside the spawn
  (`← IO.wait task` at `Tyr/Pipeline.lean:162`), so "background" tasks are
  fully blocking. Re-verified. → Return the task handle without waiting.
- **[correctness]** `Sharding.fromFull`/`gather` ignore `shardDim` at slice
  time and `shardedShape` silently falls back to the unsharded shape
  (dim-0-only runtime). → Enforce or error on `shardDim ≠ 0`.
- **[docs]** `ChatSFT.trainLoop` consumes only `embeddingLr`;
  `matrixLr`/`unembeddingLr`/`weightDecay`/`gradClip` are dead config fields
  despite the file header's "dual optimizer" claim. → Implement or document.
- **[correctness]** GRPO `temperature`/`topK`/`topP` are plumbed but never
  applied; `grpoStepWithModelUpdate` requires batch size 1. → Apply or
  document the delegation contract.

## Models

- **[bug-risk]** `sampleForest` index normalization shadows `i`
  (`Tyr/Model/BranchingFlows.lean:709-710`): for a reversed pair `(5,3)` the
  second line reads the rebound `i`, collapsing the pair to `(3,3)` and
  merging a node with itself. Re-verified. → Compute `lo`/`hi` from the
  original pair.
- **[docs]** `Tyr/Model/Qwen.lean:4` header still says "Qwen3 model
  implementation for Flux text encoding" although the module is the
  Qwen3/Qwen2.5-Omni causal-LM substrate. → Update.
- **[cleanup]** VAE hardcodes `[1,128,16,16] → [1,3,256,256]` shapes and
  duplicates `forward`/`forward1` bodies. → Parameterize.
- **[testing]** No Gemma4, Flux, or VAE tests (only demo executables).

## GPU

- **[cleanup]** `#print_gpu_kernel` (`Codegen/Attribute.lean:786`) and
  `#print_kernel_ffi` (`Codegen/FFI.lean:212`) declare syntax but have no
  elaboration rules anywhere. → Implement or remove.
- **[docs]** `Kernel.family` doc comment mentions a nonexistent `GB10`
  constructor (`Codegen/IR.lean:475`); SM100 capability constants are marked
  "estimated". → Correct the comment; confirm constants.
- **[cleanup]** Dead `scaleMatchesKernel` (`Ops/FlashAttn.lean:66`);
  `mhaFwdDispatch` maps the `.tkMhaH100Decode` variant to portable SDPA
  (`Ops/MhaH100.lean:176-177`) — surprising dispatch asymmetry worth
  documenting or fixing.
- **[docs]** `Readme.md`'s "GPU Kernel Parity" section lists entrypoints
  (`copy`, `rotary`, `layernorm`, `flashattn`, `mha_h100`) that the actual
  `scripts/gpu/test_parity_suite.sh` does not wire up. → Sync with the script.
- **[docs]** The pre-existing `docs/gpu/thunderkittens-porting-status.md`
  links to machine-specific absolute paths (`/grid/zador/...`) that resolve
  nowhere else. → Convert to relative links.

## DiffEq

- **[correctness]** Marker-trait taxonomy looks off vs diffrax: `SRA1` is
  registered as a `StratonovichSolver` (`Solver/SRA1.lean:79`; diffrax treats
  SRA1 as Itô), and ODE solvers `Heun`/`Midpoint`/`Ralston`/`Bosh3` also carry
  `StratonovichSolver` instances; `ReversibleHeun` lacks a
  `ReversibleSolver` instance. Re-verified the instances. → Audit the traits.
- **[cleanup]** `Solution.ensureOkay` is a no-op; stale header comment at
  `Adjoint/Core.lean:14`. → Implement or remove; fix header.

## Event-skeleton

- **[cleanup]** Dead move kinds `belNoise`/`itoPQ`/`dropSmallTimingTerm`
  (constructed nowhere outside `Core.lean`). → Remove or implement.
- **[cleanup]** `Tyr.lean` does not re-export `Tyr.EventSkeleton`. → Decide
  whether the omission is intentional; document either way.

## MCTS

- **[bug-risk]** `MctxDag.expandEdge` evaluates `recurrentFn` *before* the
  `UNVISITED` check (`Tyr/MctxDag/Search.lean:168` vs `:178`), so the model is
  evaluated even when the edge target exists, and `setEdge` then overwrites
  the existing edge's reward/discount. Re-verified. → Reorder the check.
- **[correctness]** Hash-based pseudo-Dirichlet noise, ignored
  `_dirichletAlpha`, and a temperature argument that is an argmax no-op
  (`Tyr/Mctx/Policies.lean`). → Implement the real semantics or rename.
- **[docs]** Only the two umbrella files have module docstrings; the 14
  submodule files have none. → Add headers.

## Audio and inference

- **[performance]** KV-cache append clones the full cache buffer per token —
  `slice_scatter` "returns a copy" (`cc/src/tyr.cpp:2292`), contradicting the
  in-place claim in `Tyr/Inference/KVCache.lean`'s header. → True in-place
  update or honest docs.
- **[bug-risk]** Out-of-range `layerIdx` silently returns `newQ` as the
  attention output (`Tyr/Inference/KVCache.lean:154-155`). → Fail loudly.
- **[bug-risk]** Off-macOS audio stubs make `start` raise but `read`/`stop`
  silently no-op (`cc/src/apple_audio_input.mm:241-275`) — a capture loop that
  misses the `start` failure spins silently. → Consistent failure behavior.
- **[cleanup]** Nothing under `Tyr/Model/` imports `Tyr.Inference.KVCache`
  (only the decode harness and a NanoChat shim do); dead
  `lean_float_buffer_inhabited` in `cc/src/float_buffer.cpp`.

## FFI and build

- **[cleanup]** Dead files `cc/lxla.cpp` and `cc/plugin_init.cpp`; empty
  `HDRS :=` and undefined `XLA_INC_FLAGS` in three `cc/Makefile` recipes.
  → Remove the XLA remnants.
- **[docs]** `Readme.md` never mentions the `../lean-urdf-typeprovider`
  sibling-checkout path dependency (`lakefile.lean:261`) — a fresh clone
  without it cannot build. → Document (now covered in
  `docs/getting-started.md`).

## Testing and CI

- **[testing]** 14 orphaned test files (12 DiffEq parity suites,
  `TestModularBudget`, `TestNanoGPTCopy`) are compiled by no runner; 6 GPU
  example harnesses are likewise unreferenced. → Wire in or delete.
- **[testing]** Zero test imports for `Tyr.Audio`, `Hub`, `PRNG`, `RL`,
  `Widget`, `Log`, `Text`, `Inference`. → Add smoke tests.
- **[cleanup]** `parseArgs` is copy-pasted across ~11 test runners; empty
  `Tyr/TVMArith/` and `Tests/TVMArith/` directories ship with the repo.
  → Factor out; drop the empty dirs.

## Suggested priorities

Highest-impact items to tackle first:

1. `Pipeline.spawnBackground` blocks — "background" tasks are synchronous
   (`Tyr/Pipeline.lean:162`).
2. `sampleForest` self-merge via let-shadowing
   (`Tyr/Model/BranchingFlows.lean:709-710`).
3. `Conv2d.outShape` returns input channels (`Tyr/Module/Conv2d.lean:33`).
4. KV-cache full-buffer clone per appended token (`cc/src/tyr.cpp:2292`) —
   a per-token O(cache) cost on every decode step.
5. `MctxDag.expandEdge` model evaluation before the visited-edge check
   (`Tyr/MctxDag/Search.lean:168`).
6. Sharding silently ignoring `shardDim` — correctness trap for any non-zero
   shard dimension.
7. NorMuon dropping `weightDecay`/`wdMul` silently.
8. Generic `Tensor.matmul` accepting inner-dim mismatches at compile time
   (junk `#[]` from `matmulShape`).
9. DiffEq `StratonovichSolver` marker audit (SRA1 and friends vs diffrax).
10. Wire up or delete the empty GPU VJP registry and the numerically
    meaningless AD placeholder rule packs.
