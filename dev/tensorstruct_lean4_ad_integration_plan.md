# TensorStruct / Lean4 AD Integration Plan

This plan describes how to integrate Tyr's `LeanJaxpr` AD frontend tightly with
Lean4 elaboration, `TensorStruct`, and module-level typed APIs.

The goal is not to make `LeanJaxpr` the semantic IR for arbitrary Lean terms.
The goal is to make:

- Lean4 structures, modules, and elaboration the user-facing surface,
- `TensorStruct` the structural schema / flatten-unflatten layer,
- `LeanJaxpr` the flat tensor-program AD IR,
- structured gradients and pullbacks reconstruct back into native Lean values.

## Design Principles

- Keep structured Lean types at the API boundary.
- Keep `LeanJaxpr` flat and execution-oriented.
- Derive structural AD metadata from `TensorStruct`, not ad hoc field reflection in every frontend.
- Preserve `diff` / `static` / `frozen` participation consistently across classic autograd and elimination AD.
- Let elaboration produce AD artifacts once, instead of recovering them later from lowered IR whenever possible.

## Target User Experience

Desired surface:

```lean
structure MyModel where
  w : T #[m, n]
  b : T #[m]
  cfg : Static Config
  ema : Frozen #[m]
  deriving TensorStruct

@[autodiff frontend := graphax]
def loss (model : MyModel) (x : T #[batch, n]) : T #[] :=
  ...
```

Elaboration should:

1. derive a structural schema from the function signature,
2. flatten differentiable tensor leaves into a frontend view,
3. build and register direct `LeanJaxpr`,
4. register structured companion APIs such as `loss.grad`, `loss.vjp`, or `loss.linearize`,
5. reconstruct gradients back into `MyModel`-shaped values.

## Current State

Already available:

- `TensorStruct` traversal and derived instances:
  - [Tyr/TensorStruct.lean](/Users/pehle/dev/tyr/Tyr/TensorStruct.lean#L243)
  - [Tyr/Module/Derive.lean](/Users/pehle/dev/tyr/Tyr/Module/Derive.lean#L94)
- Module types already require `TensorStruct`:
  - [Tyr/Module/Core.lean](/Users/pehle/dev/tyr/Tyr/Module/Core.lean#L75)
- Existing AD parameter participation split (`diff` / `static` / `frozen`):
  - [Tyr/AutoGrad.lean](/Users/pehle/dev/tyr/Tyr/AutoGrad.lean#L46)
  - [Tyr/AD/JaxprLike/Core.lean](/Users/pehle/dev/tyr/Tyr/AD/JaxprLike/Core.lean#L165)
- Direct elaboration-time frontend `LeanJaxpr` registration:
  - [Tyr/AD/JaxprLike/Elab.lean](/Users/pehle/dev/tyr/Tyr/AD/JaxprLike/Elab.lean#L16)
  - [Tyr/AD/JaxprLike/HintRegistry.lean](/Users/pehle/dev/tyr/Tyr/AD/JaxprLike/HintRegistry.lean#L20)
  - [Tyr/AD/JaxprLike/Pipeline.lean](/Users/pehle/dev/tyr/Tyr/AD/JaxprLike/Pipeline.lean#L76)
- `TensorStruct`-aware schema metadata plus template-based flatten/rebuild:
  - [Tyr/AD/TensorStructSchema.lean](/Users/pehle/dev/tyr/Tyr/AD/TensorStructSchema.lean#L19)
  - [Tests/TestADTensorStructSchema.lean](/Users/pehle/dev/tyr/Tests/TestADTensorStructSchema.lean#L1)
  - [Tests/TestADTensorStructFlatten.lean](/Users/pehle/dev/tyr/Tests/TestADTensorStructFlatten.lean#L1)
- structured frontend signature metadata for stable leaf bindings, boundary reconstruction, and `JVar` construction:
  - [Tyr/AD/Frontend/Signature.lean](/Users/pehle/dev/tyr/Tyr/AD/Frontend/Signature.lean#L1)
  - [Tests/TestADFrontendSignature.lean](/Users/pehle/dev/tyr/Tests/TestADFrontendSignature.lean#L1)
- elaboration-time structured signature registration and validation for direct frontend `LeanJaxpr`s:
  - [Tyr/AD/JaxprLike/Elab.lean](/Users/pehle/dev/tyr/Tyr/AD/JaxprLike/Elab.lean#L16)
  - [Tyr/AD/JaxprLike/HintRegistry.lean](/Users/pehle/dev/tyr/Tyr/AD/JaxprLike/HintRegistry.lean#L20)
  - [Tyr/AD/JaxprLike/Pipeline.lean](/Users/pehle/dev/tyr/Tyr/AD/JaxprLike/Pipeline.lean#L76)
  - [Tests/TestADJaxprLike.lean](/Users/pehle/dev/tyr/Tests/TestADJaxprLike.lean#L680)
- first structured `grad` / `vjp` companion reconstruction helpers over frontend leaf buffers:
  - [Tyr/AD/Frontend/Companion.lean](/Users/pehle/dev/tyr/Tyr/AD/Frontend/Companion.lean#L1)
  - [Tests/TestADFrontendSignature.lean](/Users/pehle/dev/tyr/Tests/TestADFrontendSignature.lean#L153)
- first structured frontend runtime API over flat backend callbacks:
  - [Tyr/AD/Frontend/API.lean](/Users/pehle/dev/tyr/Tyr/AD/Frontend/API.lean#L1)
  - [Tests/TestADFrontendSignature.lean](/Users/pehle/dev/tyr/Tests/TestADFrontendSignature.lean#L295)
- elaboration-time synthesis of structured frontend companion declarations when a registered frontend bundle provides a runtime frontend constant:
  - [Tyr/AD/JaxprLike/Elab.lean](/Users/pehle/dev/tyr/Tyr/AD/JaxprLike/Elab.lean#L1)
  - [Tests/TestADJaxprLikeElabFixture.lean](/Users/pehle/dev/tyr/Tests/TestADJaxprLikeElabFixture.lean#L113)
  - [Tests/TestADJaxprLike.lean](/Users/pehle/dev/tyr/Tests/TestADJaxprLike.lean#L712)
- a first definition-site frontend derivation helper for ordinary Lean defs in a restricted supported subset:
  - [Tyr/AD/Frontend/Elab.lean](/Users/pehle/dev/tyr/Tyr/AD/Frontend/Elab.lean#L1)
  - [Tests/TestADFrontendElab.lean](/Users/pehle/dev/tyr/Tests/TestADFrontendElab.lean#L1)

Still missing:

- elaborators that emit direct `LeanJaxpr` automatically from normal high-level definitions,
- unified user-facing AD attributes over this schema-aware path,
- synthesized structured `grad` / `vjp` / `linearize` declarations over executed AD results.

## Architecture

### 1. Structural Schema Layer

Add a schema layer above `LeanJaxpr` and below user-facing Lean definitions.

Proposed core types:

```lean
structure TensorLeafPath where
  segments : Array String

inductive LeafRole where
  | diff
  | static
  | frozen

structure TensorLeafSpec where
  path : TensorLeafPath
  shape : Option (Array Nat)
  dtype : Option String
  role : LeafRole

structure TensorStructSchema where
  typeName : Name
  leaves : Array TensorLeafSpec
```

Purpose:

- provide stable leaf ordering,
- record path names such as `model.w` or `model.block_3.attn.q_proj.weight`,
- preserve static/frozen participation at the structural boundary,
- drive flattening, reconstruction, diagnostics, and gradient formatting.

Recommended implementation location:

- new file: `Tyr/AD/TensorStructSchema.lean`

### 2. Flatten / Unflatten Boundary

Build a schema-aware boundary API for any `[TensorStruct α]`.

Needed capabilities:

- flatten only tensor leaves into ordered leaf arrays,
- keep `Static` leaves in schema metadata but not executable leaf buffers,
- optionally expose `Frozen` leaves as forward-only leaves,
- reconstruct `α` from a leaf array using the original structure.

Current implementation shape:

```lean
class TensorStructFlatten (α : Type) where
  flattenAt : α → TensorLeafPath → TensorLeafSelection → Array TensorLeafValue
  rebuildAt :
    α →
    TensorLeafPath →
    TensorLeafSelection →
    Array TensorLeafValue →
    StateT Nat (Except String) α
```

With user-facing helpers:

```lean
namespace TensorStructFlatten
def flatten (x : α) (selection := .diffAndFrozen) : Array TensorLeafValue
def rebuildFrom (template : α) (leaves : Array TensorLeafValue)
  (selection := .diffAndFrozen) : Except String α
end TensorStructFlatten
```

`TensorLeafValue` is now an existential runtime tensor wrapper instead of an
elaboration-only `Expr` payload:

```lean
structure TensorLeafValue where
  role : TensorLeafRole
  payload : Sigma T
```

This keeps the boundary runtime-usable while still checking typed tensor shapes
during reconstruction.

Recommended touch points:

- new file: `Tyr/AD/TensorStructSchema.lean`
- later helpers in `Tyr/TensorStruct.lean`

### 3. Signature-Level AD Schema

Add a function-boundary schema object for AD frontends.

Proposed type:

```lean
structure FrontendADSignature where
  inputs : Array TensorStructSchema
  outputs : Array TensorStructSchema
  params : Array TensorStructSchema
```

This should drive:

- flattening of structured inputs,
- binding of flat Jaxpr variables,
- reconstruction of structured outputs and cotangents,
- stable diagnostics when a frontend says "leaf 7" or "vertex 12".

Recommended implementation location:

- new file: `Tyr/AD/Frontend/Signature.lean`

### 4. Elaboration Integration

Extend the current direct registration hook into a true frontend pipeline.

Near-term elaboration surface:

```lean
@[autodiff frontend := graphax]
def f ...
```

or:

```lean
def f ...
attribute [ad_frontend MyNs.fFrontend] f
```

Elaboration responsibilities:

1. inspect the declaration type,
2. derive `TensorStructSchema` for inputs/outputs/parameters,
3. flatten structured arguments into frontend leaves,
4. build direct `LeanJaxpr`,
5. register it through the existing direct frontend registry,
6. optionally synthesize companion declarations (`grad`, `vjp`, `jvp`, debug dumps).

Prefer a single named frontend bundle applied once at the declaration boundary,
for example `attribute [ad_frontend frontendSpec] f`, rather than separate
signature/jaxpr attachments or large inline payloads.
Do not duplicate function meaning by hand-written frontend IR except for tests
and low-level bootstrapping.

Recommended touch points:

- extend [Tyr/AD/JaxprLike/Elab.lean](/Users/pehle/dev/tyr/Tyr/AD/JaxprLike/Elab.lean#L16)
- new elaboration helper module such as `Tyr/AD/Frontend/Elab.lean`

### 5. Structured Gradient Reconstruction

Once execution returns flat cotangents, reconstruct them back to the original
`TensorStruct` shapes.

Required behavior:

- differentiable leaves receive cotangents,
- static leaves are preserved unchanged or zeroed structurally, depending on API,
- frozen leaves are present but marked non-updatable,
- gradient output type mirrors the original structured input type where appropriate.

Example:

```lean
def loss.grad : MyModel → T #[batch, n] → MyModel
```

or more explicitly:

```lean
structure GradResult (α : Type) where
  primal : Option Expr
  grad : α
```

Implementation note:

- reuse `TensorStruct` reconstruction rather than inventing a parallel record builder.

### 6. Unify With Existing AD APIs

The new frontend should not become a separate AD island.

Unification goals:

- keep `ParamKind.diff/static/frozen` and `DiffParticipation.diff/static/frozen` aligned,
- allow `@[autodiff]` to choose backend or frontend mode instead of branching into unrelated systems,
- make structured `grad` / `jvp` / `vjp` available regardless of whether the backend is:
  - classic IR autograd,
  - `LeanJaxpr` + elimination,
  - future sparse forward mode,
  - future checkpointed reverse mode.

Recommended approach:

- define one high-level user attribute family,
- route backend-specific generation behind the scenes,
- keep `LeanJaxpr` as a shared backend IR rather than the user API.

## Execution Plan

### Phase 0: Schema Prototype

Goal:

- establish the structural metadata layer without changing execution yet.

Work:

- add `TensorLeafPath`, `LeafRole`, `TensorLeafSpec`, `TensorStructSchema`,
- add schema derivation for:
  - plain `T s`,
  - `Static α`,
  - `Frozen s`,
  - structures with derived `TensorStruct`,
  - `Option`, `Array`, `Vector`, tuples.

Status:

- implemented in [Tyr/AD/TensorStructSchema.lean](/Users/pehle/dev/tyr/Tyr/AD/TensorStructSchema.lean#L19)

Acceptance:

- a derived structure such as a small module can produce a stable schema,
- tests verify leaf ordering and static/frozen classification.

Suggested files:

- `Tyr/AD/TensorStructSchema.lean`
- `Tests/TestADTensorStructSchema.lean`

### Phase 1: Flatten / Unflatten

Goal:

- make the schema executable for AD boundaries.

Work:

- implement flatten/unflatten over schema-derived structures,
- add validation that flattened leaf counts match schema counts,
- add round-trip tests on nested module structs.

Status:

- partially implemented in [Tyr/AD/TensorStructSchema.lean](/Users/pehle/dev/tyr/Tyr/AD/TensorStructSchema.lean#L99)
- covered by [Tests/TestADTensorStructFlatten.lean](/Users/pehle/dev/tyr/Tests/TestADTensorStructFlatten.lean#L1)

Acceptance:

- `flatten` followed by `unflatten` round-trips representative parameter trees,
- path/shape metadata stays stable.

Suggested files:

- `Tyr/AD/TensorStructSchema.lean`
- `Tests/TestADTensorStructFlatten.lean`

### Phase 2: Signature-Aware Direct Frontend IR

Goal:

- connect structured Lean boundaries to direct `LeanJaxpr`.

Work:

- add `FrontendADSignature`,
- add helpers mapping flattened leaves to `JVar`s with `VarMeta`,
- extend elaboration helpers to build direct `LeanJaxpr` from structured signatures.

Status:

- mostly implemented:
  - [Tyr/AD/Frontend/Signature.lean](/Users/pehle/dev/tyr/Tyr/AD/Frontend/Signature.lean#L1)
  - [Tyr/AD/JaxprLike/Elab.lean](/Users/pehle/dev/tyr/Tyr/AD/JaxprLike/Elab.lean#L16)
  - [Tyr/AD/JaxprLike/HintRegistry.lean](/Users/pehle/dev/tyr/Tyr/AD/JaxprLike/HintRegistry.lean#L20)
  - [Tyr/AD/JaxprLike/Pipeline.lean](/Users/pehle/dev/tyr/Tyr/AD/JaxprLike/Pipeline.lean#L76)
- covered by:
  - [Tests/TestADFrontendSignature.lean](/Users/pehle/dev/tyr/Tests/TestADFrontendSignature.lean#L1)
  - [Tests/TestADJaxprLike.lean](/Users/pehle/dev/tyr/Tests/TestADJaxprLike.lean#L680)

Acceptance:

- a simple structured function can register direct `LeanJaxpr` without using `FnBody` recovery,
- imported declarations preserve their frontend Jaxpr registration.

Remaining gap:

- automatic frontend tracing/elaboration still needs to go beyond the current restricted single-primitive subset and produce the direct `LeanJaxpr` for general structured definitions.

Suggested files:

- `Tyr/AD/Frontend/Signature.lean`
- `Tyr/AD/JaxprLike/Elab.lean`
- `Tests/TestADJaxprLike.lean`

### Phase 3: Structured AD Companions

Goal:

- expose native Lean-facing AD APIs over structured values.

Work:

- generate structured `grad`, `vjp`, and `linearize` companions,
- reconstruct cotangents back to structured output types,
- integrate static/frozen handling with zero/non-updatable policy.

Status:

- companion reconstruction helpers are now in place:
  - [Tyr/AD/Frontend/Companion.lean](/Users/pehle/dev/tyr/Tyr/AD/Frontend/Companion.lean#L1)
  - [Tests/TestADFrontendSignature.lean](/Users/pehle/dev/tyr/Tests/TestADFrontendSignature.lean#L153)
- a runtime structured frontend API now exists for backend-agnostic `call`, `linearize`, `vjp`, and scalar-loss `grad` over flat leaf callbacks:
  - [Tyr/AD/Frontend/API.lean](/Users/pehle/dev/tyr/Tyr/AD/Frontend/API.lean#L1)
  - [Tests/TestADFrontendSignature.lean](/Users/pehle/dev/tyr/Tests/TestADFrontendSignature.lean#L295)
- `attribute [ad_frontend frontendSpec] f` can now synthesize `f.frontend`, `f.linearize`, `f.vjp`, `f.valueAndGrad`, and `f.grad` when `frontendSpec` includes a runtime frontend constant:
  - [Tyr/AD/JaxprLike/Elab.lean](/Users/pehle/dev/tyr/Tyr/AD/JaxprLike/Elab.lean#L1)
  - [Tests/TestADJaxprLike.lean](/Users/pehle/dev/tyr/Tests/TestADJaxprLike.lean#L712)
- `Tyr/AD/Frontend/Elab.lean` now derives direct frontend `LeanJaxpr` from ordinary Lean defs in the first supported subset: one registered primitive call over tensor-typed parameters, without manual fixture `LeanJaxpr` literals.
- remaining gap: elaboration still does not derive those runtime frontend constants automatically from ordinary high-level definitions.
- residual limitation: executing registration-heavy smoke checks under `lean --run` currently trips a Lean IR interpreter assertion, so the new executable coverage is restricted to derivation itself rather than full env-mutation tests.

Acceptance:

- a structure-typed model function can produce structure-typed gradients,
- gradient APIs compose with `Module` and `TensorStruct`-based training code.

Suggested files:

- new frontend AD API module, e.g. `Tyr/AD/Frontend/API.lean`
- `Tyr/AutoGrad.lean`
- tests under `Tests/TestAutoGrad.lean` and new frontend-specific suites

### Phase 4: Module And Training Integration

Goal:

- make the new path useful in real Tyr model code.

Work:

- add helpers for `[Module M In Out]` and `[ModuleCtx ...]`,
- expose parameter-typed gradient results for module training loops,
- integrate with existing `TensorStruct.grads` / `zeroGrads` semantics where sensible.

Acceptance:

- a `Module` instance with derived `TensorStruct` can use the new frontend path without manual leaf plumbing,
- example training code can choose backend while keeping the same structured parameter type.

Suggested files:

- `Tyr/Module/Core.lean`
- `Tyr/TensorStruct.lean`
- example model files

### Phase 5: Graphax / AlphaGrad Frontend Specialization

Goal:

- use the new structured frontend layer to improve current parity work.

Work:

- let Graphax-like frontends emit direct `LeanJaxpr` with structural metadata from elaboration,
- let AlphaGrad task materialization derive fixed slots from structured boundaries and partitions,
- eliminate hand-wired `FnBodyLoweringHints` for common traced source cases.

Acceptance:

- structural exactness no longer depends on manual hint injection,
- traced tasks preserve partitions and structural metadata automatically.

## Testing Strategy

Add tests in three layers.

Schema tests:

- leaf ordering for nested structures,
- `Static` / `Frozen` classification,
- shape metadata preservation.

Frontend elaboration tests:

- attribute elaboration produces direct `LeanJaxpr`,
- imported modules preserve registrations,
- invalid schemas fail at elaboration with precise errors.

Structured AD tests:

- structured gradient result types match input/module shapes,
- static fields do not receive gradients,
- frozen fields are preserved but not updated,
- elimination backend and classic autograd agree on supported overlap cases.

## Main Risks

Risk 1:
- `TensorStruct` today is a traversal API, not yet a schema API.

Mitigation:
- add schema support as a new layer instead of overloading the meaning of `map`/`fold`.

Risk 2:
- elaboration-time tracing of arbitrary Lean definitions may be too ambitious initially.

Mitigation:
- begin with restricted frontend-annotated definitions and explicit supported primitive sets.

Risk 3:
- flatten/unflatten for dynamic containers (`Array`, nested optional branches) may need stronger runtime validation.

Mitigation:
- define stable rules for dynamic container support and reject unsupported shapes early.

Risk 4:
- structured AD APIs could diverge from existing `AutoGrad`.

Mitigation:
- make frontend/schema-aware APIs a thin layer over shared participation metadata and backend selection.

## Recommended Immediate Next Step

Implement Phase 3 next:

- synthesize declaration-level structured `grad` / `vjp` APIs on top of the
  existing companion reconstruction helpers,
- then teach elaboration to derive/register direct `LeanJaxpr` automatically
  from normal structured definitions instead of fixture-style registrations.

The next bottleneck is no longer schema or signature plumbing; it is user-facing
structured AD companion generation and automatic frontend IR emission.
