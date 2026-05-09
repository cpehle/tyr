# Type-Safe Tensor Frontend — Tracking Device and DType in `T`

> **Status (branch `type-safe-tensor-meta`)**: Phase 0 (foundations + typed
> API surface) and Phase 1 (Module-layer typed wrappers + instances)
> complete. Phase 3 (FFI ingress Σ-types) partially complete for
> SafeTensors. Phase 2 (model-file migrations) is **incremental and
> ongoing** — one entry-point migration landed (`Qwen3ForCausalLM.embedTokensT`)
> as a template; bulk model-file migration remains follow-up work. See
> commit history on the branch for landed work and the bottom of this
> file for what's next.

## Context

Today `T : Shape → Type` (`Tyr/Basic.lean:107-108`) carries only shape at the
type level. `Device` and `DType` exist as runtime inductives
(`Tyr/Basic.lean:25-39, 101-105`) and are inspected via runtime introspection
helpers (`T.dtype`, `T.device`) plus ad-hoc casting helpers (`castLike`,
`toFloat'`, `toBFloat16'`, `T.to`). This pushes a class of bugs to runtime:

- Mixing dtypes — e.g. accidentally feeding fp32 activations to a bf16
  weight, or passing an int64 token tensor where a float embedding is
  expected.
- Mixing devices — e.g. CPU input + CUDA weight ⇒ hard PyTorch crash.
- Forgotten conversions during refactors — `.to(device)` / `.to(dtype)`
  silently no-op or crash if the target dtype/device differs from caller
  expectations.

`Tyr/Torch.lean` alone has ~175 `T #[…]` annotations and ~185 ops on `T`,
the model files reach 374–542 sites each, and the FFI surface is 310
`@[extern]` declarations (108 mention `T`). A phased migration is the only
realistic path.

## Goal

```lean
structure TensorMeta where
  device : Device
  dtype  : DType
deriving Repr, Inhabited, BEq, DecidableEq

def T (_m : TensorMeta) (_shape : Shape) : Type := TSpec.type
```

A single phantom `TensorMeta` index — erased at runtime, the underlying
handle is still the opaque `TSpec.type` — that makes device/dtype
mismatches a Lean type error.

Why one combined index, not two separate ones (`Device` and `DType`):

- The dominant op pattern is *metadata-preserving* (add, matmul, linear,
  cat, layernorm, silu, transpose, reshape). With one index, "same
  metadata in, same metadata out" reads off the page as a single shared
  variable. Two indices would force binding both even though they always
  travel together.
- Conversion ops become cleaner — output type is a record update
  (`{ m with device := target } s`) driven directly by the runtime arg,
  with no implicit-result-index + auto-`rfl` dance needed.
- Σ-types at FFI ingress collapse from `(dev : Device) × (dt : DType) ×
  T dev dt s` to `(m : TensorMeta) × T m s`.
- The one case where two indices would read more cleanly — "same dtype,
  different device" or the reverse — is genuinely rare (cross-device
  comms / async copy primitives) and pays a single `m.dtype = m'.dtype`
  proof obligation when it shows up.

Example before/after for the dominant pattern:

```lean
-- before
def linear {m n b : UInt64} (x : T #[b, m]) (M : T #[n, m]) : T #[b, n]

-- after
def linear {meta : TensorMeta} {m n b : UInt64}
    (x : T meta #[b, m]) (M : T meta #[n, m]) : T meta #[b, n]
```

The result type pins the metadata, so a downstream op that expects
`T cudaBF16 …` won't accept a CPU-resident or fp32 tensor without an
explicit conversion.

## Design

### Core type change (`Tyr/Basic.lean`)

```lean
structure TensorMeta where
  device : Device
  dtype  : DType
deriving Repr, Inhabited, BEq, DecidableEq

opaque TSpec : NonemptyType
def T (_m : TensorMeta) (_s : Shape) : Type := TSpec.type
```

Indices are erased — the underlying `TSpec.type` is unchanged. FFI
conversion (`fromTorchTensor` in `cc/src/tyr.cpp`) does not change. This
is a phantom-type refactor, not a runtime-representation change.

`Device` needs `DecidableEq` derived (currently only `BEq`); required so
`TensorMeta` can also derive it.

### Op signature patterns

Three patterns cover ~95% of ops:

1. **Metadata-preserving** (most ops). Single shared `{m : TensorMeta}`
   implicit threads through inputs and output:

   ```lean
   def matmul {m : TensorMeta} {s1 s2 : Shape}
       (a : T m s1) (b : T m s2) : T m (matmulShape s1 s2)
   ```

2. **Conversion ops** (`T.toDevice`, `T.toDType`, `castLike`,
   `toFloat'`, `toBFloat16'`). Output type is a record update of the
   input meta, computed directly from the runtime target arg:

   ```lean
   def T.toDevice {m : TensorMeta} {s : Shape}
       (self : T m s) (target : Device) : T { m with device := target } s

   def T.toDType {m : TensorMeta} {s : Shape}
       (self : T m s) (target : DType) : T { m with dtype := target } s
   ```

   No proof obligations, no auto-`rfl` trick — the type just depends on
   the explicit runtime arg.

3. **Mixed-meta ops** that legitimately accept different metadata.
   Most commonly: an integer index tensor with a separately-typed value
   tensor. These bind multiple metadata variables and pin specific
   fields where the contract requires:

   ```lean
   /-- Token IDs are int64, output keeps weight metadata. Caller is
       responsible for placing `ids` on the same device as `weight`. -/
   def embedding {m : TensorMeta} {V H batch seq : UInt64}
       (weight : T m #[V, H])
       (ids : T { m with dtype := .Int64 } #[batch, seq])
       : T m #[batch, seq, H]
   ```

   When dtype must differ but device must match, an `m.device = m'.device`
   premise is the canonical way to express it (see the rare cross-axis
   case below).

### Cross-axis ops (rare)

Cross-device copies and similar "preserve dtype, change device" (or
vice-versa) primitives need a projection equality:

```lean
def crossDeviceCopy {m m' : TensorMeta} {s : Shape}
    (h : m.dtype = m'.dtype) (x : T m s) : T m' s
```

In practice these ops are a small surface — distributed comm primitives,
async transfer wrappers — so the awkwardness is contained.

### Shape syntax stays the same

`#[b, m]` literals continue to elaborate to `Array UInt64`. `T meta #[b, m]`
reads naturally.

### No "any-meta" escape hatch

We deliberately do **not** introduce a `TensorMeta.Any` or sentinel value.
Truly polymorphic ops bind `{m : TensorMeta}` as an implicit, exactly like
shape-polymorphic ops today.

### Construction APIs

- `zeros`, `ones`, `rand`, `full` already take an explicit `Device`
  argument at runtime. They become:

  ```lean
  def zeros (s : Shape) (dev : Device) (dt : DType := .Float32)
      : IO (T { device := dev, dtype := dt } s)
  ```

  Or, more ergonomically:

  ```lean
  def zeros (s : Shape) (m : TensorMeta := { device := .CPU, dtype := .Float32 })
      : IO (T m s)
  ```

  The struct-arg form is cleaner once `TensorMeta` is the lingua franca,
  and call-sites can use `{ device := .CUDA 0, dtype := .BFloat16 }`
  literal syntax.

- `fromBlob`-style FFI ingress: `T.viewAs (m : TensorMeta) : T m s`
  performs a runtime check (does the underlying handle actually have
  these properties) and throws on mismatch. Used at FFI boundaries
  where the caller asserts a known metadata.

- `safetensorsLoad` returns the Σ-type form `(m : TensorMeta) × T m s`
  since metadata is genuinely unknown until the file header is parsed.
  Most consumers destructure once and pass `m` along.

### Erased indices ⇒ no runtime cost

Because `T` reduces to the same opaque handle regardless of `m`, the
generated C code is identical to today. No extra allocations, no extra
runtime checks. This is the cheap part of the refactor.

## Verbosity at user-facing sites

Per-line cost in model code (e.g. `Tyr/Model/Qwen35/Model.lean` with 374
`T #[…]` annotations):

```lean
-- before
let qh    : T #[batch, num_heads, 1, head_dim]      := nn.transpose_for_attention q

-- after
let qh    : T m #[batch, num_heads, 1, head_dim]    := nn.transpose_for_attention q
```

Each annotated line gains ~3 chars (`m `). That's the minimum-cost form;
two-index would be ~7 chars (`dev dt `).

Mitigations to keep model bodies readable:

1. **`section variable`** — declare `variable {m : TensorMeta}` once at
   the top of a model module so every `def` in scope picks it up
   implicitly. Body annotations still need `m`, but the `def` lines
   stay tidy.

2. **Audit redundant let-annotations.** Many existing
   `let foo : T #[...] := ...` annotations are pure documentation —
   Lean infers them. A pre-migration pass should strip the ones that
   aren't needed. The remaining annotations are the ones that actually
   pull weight; adding `m` to those is a smaller cost.

3. **Polymorphic abbreviation** for "I don't care which" cases:
   ```lean
   abbrev Tn (s : Shape) := ∀ {m : TensorMeta}, T m s
   ```
   Use sparingly — restrict to genuinely metadata-polymorphic helpers
   (e.g. shape-only utilities). In model code, prefer the explicit
   `T m s`.

## Migration Strategy

A single-PR rewrite is impractical (180+ ops in Torch.lean alone, 4-5k
`T #[…]` sites across the codebase, 310 FFI extern decls). Phased:

### Phase 0 — Foundations (1 PR)

- `Tyr/Basic.lean`: define `TensorMeta`, derive `DecidableEq` on `Device`,
  change `T` to `def T (_m : TensorMeta) (_s : Shape) : Type`.
- Mass-rewrite `Tyr/Torch.lean` (~185 ops, ~108 FFI decls touching `T`):
  add `{m : TensorMeta}` to every signature, replace `T s` → `T m s`.
  Mostly mechanical; do it in a single commit to avoid a long-lived
  half-typed state.
- Update conversion ops (`T.to`, `castLike`, `toFloat'`, `toBFloat16'`)
  to use the record-update output-type pattern.
- Constructors (`zeros`/`ones`/`rand`/`full`/`empty`) take `TensorMeta`
  and produce `T m s`.

### Phase 1 — Module layer (1 PR)

- `Tyr/Module/Core.lean`, `Tyr/Module/RMSNorm.lean`, `Tyr/Module/LayerNorm.lean`,
  `Tyr/Module/Linear.lean` — leaf modules. Pin parameter dtypes
  (typically bf16 for inference) by binding `m` and constraining its
  `dtype` field at module construction.

### Phase 2 — Model code (one PR per family, dependency order)

1. `Tyr/Model/Qwen/*.lean` and `Tyr/Model/Qwen35/*.lean` — large surface
   but the same pattern repeats. Most ops will get `{m : TensorMeta}`
   bound implicit args and propagate them through. Apply
   verbosity-mitigation #2 (drop redundant let-annotations) before
   adding `m`.
2. Remaining models: Gemma4, Whisper, KittenTTS, Qwen3ASR, Qwen3TTS.
3. `Tyr/Pipeline.lean`, `Tyr/AutoGrad.lean`, `Tyr/Optim.lean` — last,
   since they sit on top of all model code.

### Phase 3 — FFI ingress polish (1 PR)

- `safetensorsLoad`, `parquet` ingress, and other "metadata-from-file"
  paths return `(m : TensorMeta) × T m s`. Audit call sites — most can
  pin the metadata after inspecting the file header (load → check →
  `viewAs`).
- `TYR_DEVICE` env reads (`Tyr/Pipeline.lean:308`,
  `Tyr/Model/Whisper/Weights.lean:151`,
  `Tyr/Model/KittenTTS/Weights.lean:763`,
  `Tyr/Model/Qwen3ASR/Weights.lean:224`) flow into the constructed
  `TensorMeta` at top-level entry points.

### Phase 4 — Tighten polymorphic ops (1 PR)

- Replace `{m : TensorMeta}` "I-don't-care" implicits with concrete
  `dtype` constraints where a single value is the only reasonable
  choice (e.g. token IDs are always `.Int64`, attention masks are
  always `.Bool` or `.Int64`).
- Add lints / CI checks: forbid `{m : TensorMeta}` in signatures that
  should pin a specific dtype.

## Critical files

- `Tyr/Basic.lean:23-108` — `Shape`, `DType`, `Device`, `T` core
  definitions; add `TensorMeta`.
- `Tyr/Torch.lean` — ~185 ops, ~108 FFI decls; bulk of Phase 0 churn.
- `Tyr/Module/Core.lean` — `Module` type class & instances; needs a
  pass since modules carry parameters with specific metadata.
- `Tyr/Module/RMSNorm.lean`, `LayerNorm.lean`, `Linear.lean` — first
  Phase-1 targets.
- `cc/src/tyr.cpp:264-310, 295-311` — FFI dtype/device extraction; no
  changes required (indices are erased).
- `Tyr/Pipeline.lean:308`, `Tyr/Model/Whisper/Weights.lean:151`,
  `Tyr/Model/KittenTTS/Weights.lean:763`,
  `Tyr/Model/Qwen3ASR/Weights.lean:224` — `TYR_DEVICE` env-var reads;
  feed into Σ-type construction at boundaries.

## Risks

1. **Elaborator performance.** Adding one index to ~5000 type
   annotations may slow the build. Lean 4 dependent type elaboration
   is usually fast but watch `Tyr/Model/KittenTTS/Model.lean` (542
   sites) and `Tyr/Model/Qwen35/Model.lean` (374 sites).
2. **Record-update unification.** Conversion-op output types like
   `T { m with device := target } s` rely on Lean's record-update
   reducing during unification. If this elaborates poorly, fall back
   to a typeclass-style `MetaUpdate m target m'` approach. Likely
   fine, but worth monitoring.
3. **Σ-type ergonomics at FFI ingress.** `(m : TensorMeta) × T m s` is
   already cleaner than the two-index version, but consuming it still
   requires `let ⟨m, t⟩ := …`. Document the rule: Σ-type for genuinely
   unknown-at-compile-time, `viewAs` for "I know this is bf16."
4. **Mixed-precision training.** Once parameters are pinned to
   `{ dtype := .BFloat16 }`, gradients-in-fp32 plumbing needs explicit
   conversion sites. Phase 2 should land alongside `Tyr/AutoGrad.lean`
   updates that thread compute-dtype distinct from parameter-dtype.
5. **Legacy callers in `Examples/`** need updates per phase. CI builds
   executables, so this is caught automatically.

## Verification

After each phase:

```bash
lake -R build
lake -R test  # if test runner is hooked up
```

Phase 0 verification specifically:

```bash
# spot-check: every T usage now has {m : TensorMeta} in scope
rg -n "^\s*(def|theorem|example|lemma)\b.*T #\[" Tyr/Torch.lean | head
```

End-to-end smoke after Phase 2:

```bash
lake -R build Qwen35RunHF
./.lake/build/bin/Qwen35RunHF --prompt "Hello" --max-new-tokens 16
```

A successful run with the migrated code path proves the indices erase
correctly and the FFI boundary still works.

## Out of scope for this plan

- Tracking *requires_grad* at the type level (separate phantom).
- Tracking shape **constraints** (e.g. "even" or "divisible by 64") —
  doable with refined types but a different axis of safety.
- Compile-time shape checking that goes beyond what `Shape` already
  provides (the `matmulShape` style functions stay as today).

## Branch status — what landed on `type-safe-tensor-meta`

13 commits, all build-green, 12/12 tests pass.

### Phase 0: Foundations + typed API ✅

  - `Tyr/Basic.lean`: `TensorMeta` struct, `Tensor m s` phantom-typed
    alias for `TSpec.type`, `Tensor.unsafeOfT` / `Tensor.toT` adapters,
    `DecidableEq` on `Device`.
  - `Tyr/Tensor.lean`: ~50 ops covering constructors, conversion,
    arithmetic + scalar variants + HAdd/HSub/HMul/HDiv instances,
    activations (sigmoid/silu/softmax/rsqrt/relu/gelu/softplus/tanh/
    exp/log/abs/sqrt), reductions (sumDim/meanDim/argmax),
    shape (reshape/transpose/expand/cat/unsqueeze/squeeze),
    linalg (matmul/linear2d/linear3d/affine3d), slicing
    (slice/sliceScatter), attention transposes
    (transposeForAttention/transposeFromAttention), embedding (with
    Int64 ids constraint), SDPA family (sdpaGQA/sdpaGQAQKV/
    tyrFlashAttn4d), functional module ops (rmsNormWeighted/layerNorm),
    logical/comparison (eqScalar/whereSelect/logicalNot/logicalOr/anyB/
    fullInt).

### Phase 1: Module-layer typed wrappers ✅

  - `Tyr/Module/RMSNorm.lean`: forward2dT/3dT/4dT/5dT + typed Module
    instances
  - `Tyr/Module/LayerNorm.lean`: forward3dT + typed Module instance
  - `Tyr/Module/Linear.lean`: forward2dT/3dT + typed Module instances
  - `Tyr/Module/Affine.lean`: stepT
  - `Tyr/Module.lean`: fixed missing `import Tyr.Module.RMSNorm`

  The `m |> x` infix syntax dispatches correctly on typed inputs.

### Phase 3: FFI ingress (partial — SafeTensors only) ✅

  - `Tensor.loadSafeTensor`: caller-asserted-meta variant
  - `Tensor.loadSafeTensorAuto`: Σ-typed `(m : TensorMeta) × Tensor m s`
    with runtime device/dtype detection
  - `Tensor.viewAs`: cast at FFI boundaries

### Tests ✅

  `Tests/TestTensorTyped.lean` — 12 runtime tests + 2 compile-time
  `#check_failure` negative tests:

    - constructor metadata round-trip
    - arithmetic preserves metadata
    - `toFloat32` updates dtype index
    - shape / tensorMeta projections
    - `RMSNorm`/`Linear` typed forwards
    - activations preserve metadata
    - transpose updates shape index
    - attention transpose round-trip
    - typed `Module |> x` dispatch
    - end-to-end typed attention block (Q/K/V/reshape/transpose pipeline)
    - end-to-end typed MLP (`Examples/TypedTensor/MiniMLP.lean`)
    - mismatched-meta `add` rejected at compile time
    - non-Int64 ids rejected by `embedding` at compile time

### Phase 2: Model-file migrations (started, mostly remaining)

  - `Tyr/Model/Qwen3/Model.lean`: `embedTokensT` added as the first
    in-place model migration template. Legacy `embedTokens` kept for
    backwards compatibility.

  **Remaining**: ~50 model files / 4-5K `T #[…]` sites. Migration is
  per-file, follows the pattern from `embedTokensT`:

    1. `import Tyr.Tensor` in the model file.
    2. Add typed sibling methods with `T` suffix that delegate to the
       legacy `T s` impl via `Tensor.unsafeOfT` / `.toT`.
    3. Once all callers in a model are migrated, drop the legacy
       sibling.
    4. As intermediate `T s` site annotations are migrated to
       `Tensor m s`, the call sites pin metadata at the type level and
       Lean catches mismatches.

  The largest model files by `T #[…]` site count: KittenTTS (542),
  Gemma4 (538), Qwen35 (374). Plan ~1 PR per model family; expect
  elaborator-tuning passes when shape-arithmetic-heavy code (e.g.
  `cfg.head_dim / 2`) is exposed to the new index.

### Phase 4: Tighten polymorphic ops (not started)

  Once enough of the codebase is migrated, sweep for `{m : TensorMeta}`
  binders that should be pinned to specific dtype constraints:

    - all token-ID tensors → pin `dtype := .Int64`
    - position embeddings → typically pin device to match weights
    - attention masks → `.Bool` or `.Int64`
    - mixed-precision training: parameters at `.BFloat16`,
      gradients/optimizer state at `.Float32` — explicit conversion
      sites become type-checked.

  Add lints / CI checks: forbid `{m : TensorMeta}` in functions whose
  semantics require a specific dtype.
