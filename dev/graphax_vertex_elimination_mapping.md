# Graphax-Style Vertex Elimination in Tyr (LeanJaxpr-First)

This document defines a concrete implementation path to bring Graphax-like cross-country elimination into Tyr, updated with project decisions:

- Restructure around a Jaxpr-like IR in Tyr (not statement transposition-first).
- Expose custom elimination orders as a user-facing API early.
- Do not fallback to classic backend at runtime when elimination backend is selected.
- Build full sparse algebra (not hybrid as the end state).
- Include communication/distributed cost and planning in elimination.
- Prioritize correctness and coverage before optimization work.
- Keep custom order semantics close to Graphax (vertex-id based).
- Make constraint policies first-class in AlphaGrad optimization (not Tyr-only).

---

## Decision Baseline

## D1: Jaxpr-like core IR is the center of the design

Rationale:
- Graphax is built around traced equation graphs (`jaxpr` eqns + per-primitive local Jacobians).
- Tyr already has IR forms (`FnBody` and `KStmt`) that are close to SSA and can be normalized into an eqn graph.

## D2: Custom elimination order is user-facing

Rationale:
- This is a core value prop of cross-country elimination.
- Needed for research workflows (AlphaGrad-style or hand-designed schedules).

## D3: No runtime fallback when elimination backend is requested

Rationale:
- Forces coverage and correctness discipline.
- Avoids silent behavior drift across backends.

Note:
- Differential comparison against current backend is still required in tests, but not as runtime fallback.

## D4: Full sparse algebra is required

Rationale:
- Hybrid materialization can be useful for bootstrapping, but final design must preserve sparse structure through compose/add/eliminate.

## D5: Communication-aware elimination is in scope

Rationale:
- Tyr already has distributed primitives (`allReduce`, `reduceScatter`, `allGather`) and sharding abstractions.
- Elimination order should optimize not just FLOPs but also collective traffic.

## D6: Correctness-first implementation priority

Rationale:
- Correctness is the gating objective for each phase.
- Performance work is layered after invariant checks, differential tests, and coverage guarantees.

This is not treated as an optional tradeoff in planning; optimization milestones depend on correctness milestones.

## D7: Graphax-style custom order format

Rationale:
- Staying close to Graphax preserves transferability of known orders and tooling.
- Vertex-index order is simple, explicit, and deterministic once LeanJaxpr is normalized.

Decision:
- User custom order is an array of vertex IDs corresponding to LeanJaxpr equation vertices.
- Use Graphax-style numbering convention: 1-based indices in topological equation order.

## D8: Constraint policy is an optimization objective in AlphaGrad

Rationale:
- AlphaGrad currently optimizes in action space while Graphax/Tyr execute vertex orders.
- Constraint-guided optimization should influence the sampled/eliminated sequence directly, not be a post-processing pass.

Decision:
- Keep AlphaGrad policy action space compatible (`0..n-1`), but inject constraint feasibility into action masking during optimization.
- Keep Graphax/Tyr execution interface compatible (`1..n`) for explicit vertex order.
- Provide lossless adapters between the two spaces.

---

## 1) Build a LeanJaxpr-like AD IR

The key restructuring is to insert a canonical graph IR layer between existing Tyr IR and AD lowering.

Add modules:
- `Tyr/AD/JaxprLike/Core.lean`
- `Tyr/AD/JaxprLike/FromFnBody.lean`
- `Tyr/AD/JaxprLike/FromKStmt.lean`
- `Tyr/AD/JaxprLike/Validate.lean`

## 1.1 Proposed core structures

```lean
abbrev JVarId := Nat

structure JVar where
  id : JVarId
  ty : IRType
  meta : VarMeta

structure JEqn where
  op : OpName
  invars : Array JVar
  outvars : Array JVar
  params : OpParams
  source : SourceRef

structure LeanJaxpr where
  constvars : Array JVar
  invars : Array JVar
  eqns : Array JEqn
  outvars : Array JVar
```

`VarMeta` must retain AD-relevant information:
- static/frozen/diff participation (`ParamKind`-compatible)
- shape/layout/sharding metadata
- device and dtype
- alias group id (for repeated-arg accumulation correctness)

## 1.2 Normalization constraints

To make elimination deterministic and robust, enforce:
- explicit A-normalized equations
- explicit multi-result op handling
- explicit side-effect boundaries
- topological ordering
- unique var IDs

For `FnBody` and `KStmt`, this means canonicalizing into pure dataflow eqns where possible and marking non-diff/effectful ops as non-eliminable boundaries.

## 1.3 Why this maps well to Graphax

Graphax relies on:
- eqn-level primitive registry
- graph edges from input var to output var carrying local Jacobians
- vertex elimination over this graph

A normalized LeanJaxpr gives the same shape of problem in Tyr.

---

## 2) Local Jacobian extraction from LeanJaxpr eqns

Add modules:
- `Tyr/AD/JaxprLike/Rules.lean`
- `Tyr/AD/JaxprLike/RuleRegistry.lean`
- `Tyr/AD/JaxprLike/RuleCheck.lean`

## 2.1 Rule contract

For each differentiable equation:
- compute primal output metadata
- produce one local Jacobian map per differentiable input

```lean
abbrev LocalJacRule :=
  JEqn -> RuleContext ->
  Except RuleError (Array LocalJacEdge)
```

`LocalJacEdge`:
- `src : JVarId`
- `dst : JVarId`
- `map : SparseLinearMap`

## 2.2 Strict coverage (no fallback)

When elimination backend is selected:
- every differentiable op in LeanJaxpr must have a local-jac rule
- if not, fail with a precise coverage error listing op + source location

No fallback to classic transpose backend.

## 2.3 Rule reuse strategy

Do not duplicate rule logic blindly.

Bridge existing registries:
- reuse symbolic derivative formulas already encoded in `JVPRule`/`VJPRule` where structurally possible
- add explicit local-jac registrations for GPU ops where existing transpose logic is currently procedural

---

## 3) Full sparse algebra for edge maps

Add modules:
- `Tyr/AD/Sparse/Dim.lean`
- `Tyr/AD/Sparse/Map.lean`
- `Tyr/AD/Sparse/Transform.lean`
- `Tyr/AD/Sparse/Compose.lean`
- `Tyr/AD/Sparse/Add.lean`
- `Tyr/AD/Sparse/Validate.lean`

## 3.1 Core data model

Mirror Graphax concepts, adapted to Tyr:
- sparse/dense dimensions with pairing metadata
- deferred transforms (reshape/transpose/slice/broadcast/squeeze/concat/cast)
- symbolic shape and sharding metadata on maps

```lean
structure SparseLinearMap where
  outDims : Array OutDim
  inDims  : Array InDim
  payload : SparsePayload
  preTransforms  : Array JacTransform
  postTransforms : Array JacTransform
  shardInfo : ShardMeta
```

`SparsePayload` should support:
- diagonal payloads
- block-sparse payloads
- Kronecker-style identity/factored structures
- dense payloads as a valid special case

## 3.2 Required operations

- `compose : SparseLinearMap -> SparseLinearMap -> Except AlgebraError SparseLinearMap`
- `add : SparseLinearMap -> SparseLinearMap -> Except AlgebraError SparseLinearMap`
- `applyTransform` / `applyInverseTransform`
- `coalesce` (canonical sparse form)
- `estimateFlops`
- `estimateBytesLocal`
- `estimateBytesComm`

## 3.3 Materialization policy

Because full sparse algebra is required:
- materialization is allowed only as explicit transformation for debug/verification or final export.
- elimination internals should preserve sparse forms.

---

## 4) Graph build + ordered elimination

Add modules:
- `Tyr/AD/Elim/Graph.lean`
- `Tyr/AD/Elim/Order.lean`
- `Tyr/AD/Elim/Eliminate.lean`
- `Tyr/AD/Elim/Cost.lean`

## 4.1 Graph structure

```lean
structure ElimGraph where
  forward  : HashMap JVarId (HashMap JVarId SparseLinearMap)
  backward : HashMap JVarId (HashMap JVarId SparseLinearMap)
  inputs   : Array JVarId
  outputs  : Array JVarId
  eliminable : Array JVarId
```

Build rule:
- for each local Jacobian edge `u -> v`, insert same map in `forward[u][v]` and `backward[v][u]`.

## 4.2 Vertex elimination kernel

For vertex `v`:
- for each incoming edge `(i -> v)` and outgoing edge `(v -> o)`:
  - `candidate = compose(map(v->o), map(i->v))`
  - if `(i -> o)` exists: `candidate = add(candidate, existing)`
  - write `(i -> o)`
- remove obsolete edges around `v` (except retained outputs as configured)

This is directly aligned with Graphax elimination semantics.

## 4.3 Order support

Expose as user-facing API:
- `fwd`
- `rev`
- `custom [vertex ids]` (Graphax-style 1-based LeanJaxpr vertex ids)
- `constraints { ... }` (hard and soft order constraints)
- `alphagrad { actionSeq0 := [...], constraints := ... }`
- `heuristic markowitz`
- `heuristic commAware`

Validation rules:
- custom order must include every eliminable vertex exactly once
- cannot include non-eliminable boundary vertices
- provide human-readable errors with vertex/source mapping
- include explicit diagnostics for index base and out-of-range IDs

## 4.4 Recommended order-policy representation (AlphaGrad-compatible)

Represent policy with explicit ID spaces and adapters:

```lean
abbrev ActionId0 := Nat   -- AlphaGrad action id, expected in [0, n-1]
abbrev VertexId1 := Nat   -- Graphax/Tyr vertex id, expected in [1, n]

structure ConstraintSpec where
  hardPrecedence : Array (VertexId1 × VertexId1) -- u must be before v
  softPrecedence : Array (VertexId1 × VertexId1 × Float)
  groups : Array (Array VertexId1) -- optional grouped ordering hints
  commHints : Array CommHint

inductive OrderPolicy where
| explicitVertex (order1 : Array VertexId1)
| constrainedVertex (base : Option (Array VertexId1)) (c : ConstraintSpec)
| alphaGradAction (actions0 : Array ActionId0) (c : Option ConstraintSpec)
| heuristic (name : String)
```

Normalization pipeline:
1. Parse policy and validate ID domain (`0..n-1` for actions, `1..n` for vertices).
2. Convert all forms to canonical `VertexId1` where needed for elimination execution.
3. Compile `ConstraintSpec` into a step function:
`feasible(t, eliminated, candidateVertex) -> Bool`.
4. Run elimination only with feasible candidates.
5. If infeasible at any step, fail with strict diagnostic (no fallback order).

Why this should be the default:
- Preserves Graphax transferability for explicit orders.
- Preserves AlphaGrad training compatibility in action space.
- Makes constraints part of optimization state/action feasibility, not a separate late-stage filter.

## 4.5 AlphaGrad integration details (constraint-first)

Current AlphaGrad/Graphax boundary behavior (confirmed in code):
- AlphaGrad environment step uses `vertex = action + 1` for elimination.
- Runtime reward path converts action trace to Graphax order via `[a+1 for a in act_seq]`.
- Markowitz order generation uses vertex IDs in `1..n`.

Integration contract:
1. AlphaGrad policy emits action logits over `ActionId0`.
2. Runtime mask becomes:
`mask = eliminatedMask OR NOT constraintFeasible(action+1, state)`.
3. Sampling/argmax occurs only over feasible actions.
4. Collected trajectories remain action-native (`ActionId0`) for PPO/A0 training compatibility.
5. Before elimination execution in Tyr/Graphax path, convert once:
`VertexId1 = ActionId0 + 1`.
6. Constraint violations are impossible by construction if masking is correct; still assert during replay for correctness.

Constraint objective extensions in AlphaGrad:
- Hard constraints: enforced by masking (probability mass = 0 on invalid actions).
- Soft constraints: add reward penalty or auxiliary loss term.
- Communication-aware term: include `bytes + latency * collectives` estimate in per-step reward/cost model.

This keeps constraint policies inside AlphaGrad optimization while preserving exact compatibility with Graphax-style execution order.

## 4.6 Indexing evidence and interoperability notes

Evidence from current upstream code:
- AlphaGrad action semantics are explicitly `0..n-1` with `vertex = action + 1`.
- AlphaGrad runtime reward path converts action sequences with `[a+1 for a in act_seq]` before calling Graphax `jacve`.
- AlphaGrad Markowitz routines generate vertex IDs from `1..n`.
- Graphax order checking/enumeration uses equation indices starting at `1`, and elimination indexes equations as `eqns[vertex-1]`.

Interoperability note:
- Treat `ActionId0` and `VertexId1` as different types in APIs.
- Disallow implicit integer reuse between the two spaces.
- Always cross the boundary through explicit adapters (`+1`, `-1`) with domain checks.

Observed implementation hazard to account for in adapter logic:
- AlphaGrad runtime currently stores `act_seq` in a zero-initialized array and uses `act_seq > 0` as a fill sentinel. This can lose information for action `0`. Tyr-side adapter code should avoid this ambiguity by using explicit length counters or `-1` as an empty sentinel when replaying action traces.

---

## 5) Communication-aware elimination

Add modules:
- `Tyr/AD/Elim/CommModel.lean`
- `Tyr/AD/Elim/DistributedLower.lean`

## 5.1 Why communication belongs in elimination

In distributed settings, edge composition/addition can imply:
- cross-rank reduction of cotangents
- gather/scatter for sharded parameter maps
- ownership transfers for partial Jacobian blocks

Elimination order should minimize both compute and communication.

## 5.2 Map-level communication metadata

Attach to each map:
- owner set / partitioning
- sharding axis
- replication state
- estimated remote dependencies

## 5.3 Communication cost model

For an elimination candidate edge update, estimate:
- local flops
- local memory movement
- collective bytes and collective count
- latency-weighted objective:

`score = alpha * flops + beta * local_bytes + gamma * collective_bytes + delta * collective_count`

`commAware` order minimizes this score incrementally.

## 5.4 Lowering to existing distributed primitives

Use existing Tyr distributed interfaces for generated communication steps:
- `reduceScatter`
- `allGather`
- `allReduce`
- list variants where needed for owner-based partitioning

No hidden fallback: if required collective pattern is unsupported, backend fails with explicit diagnostic.

---

## 6) Lowering eliminated maps back to executable code

Add modules:
- `Tyr/AD/Elim/LowerFnBody.lean`
- `Tyr/AD/Elim/LowerKStmt.lean`

Output targets:
- Lean IR (`FnBody`) for compiler-path AD integration
- GPU IR (`KStmt`) for kernel-path AD integration

Lowering stages:
1. finalize remaining maps (apply deferred transforms as needed)
2. schedule sparse kernels / sparse primitive ops
3. insert communication ops according to map shard metadata
4. emit cotangent accumulation code per requested inputs

---

## 7) API surface

Add user-level controls (exact naming TBD):
- `@[autodiff, backend := elimination]`
- `@[autodiff, backend := elimination, order := rev]`
- `@[autodiff, backend := elimination, order := custom [..]]`
- runtime equivalent options for GPU kernels

Also add debugging controls:
- dump LeanJaxpr
- dump elimination graph
- dump elimination trace with per-step cost
- dump generated communication plan

---

## 8) Phased implementation plan (updated)

## Phase 0: LeanJaxpr skeleton and strict coverage checks

- Implement LeanJaxpr core + normalization from `FnBody`/`KStmt`
- Implement rule registry and strict missing-rule failure
- No elimination yet

Deliverable:
- any eligible function can be converted to LeanJaxpr and validated for rule coverage
- correctness gates required before moving to later phases

## Phase 1: Sparse algebra core

- Implement `SparseLinearMap` types and invariants
- Implement compose/add/transform operations for arithmetic + broadcast/reduction subset
- Property tests for algebraic correctness

Deliverable:
- sparse map algebra passes standalone tests

## Phase 2: Elimination engine + user custom order

- Implement graph build and elimination loop
- Expose user-facing `custom` order API using Graphax-style 1-based vertex IDs
- Implement `constraints` policy and strict feasibility checker
- Implement AlphaGrad action-to-vertex adapters (`+1` and `-1`)
- Implement `fwd` and `rev`

Deliverable:
- elimination backend runs end-to-end for arithmetic subset with strict no-fallback behavior and constraint-aware order execution

## Phase 3: Lowering integration (FnBody + KStmt)

- Lower eliminated maps into executable AD code
- Differential tests against runtime autograd for supported ops

Deliverable:
- elimination backend usable in real training/test loops for supported op coverage

## Phase 4: Communication-aware elimination

- Add sharding/owner metadata on maps
- Add communication cost model and comm-aware order heuristic
- Route communication-aware score into AlphaGrad reward/objective path
- Generate distributed collectives in lowering

Deliverable:
- measurable communication-aware improvement on sharded workloads

## Phase 5: Full op coverage and sparse-first kernels

- Expand primitive rule coverage
- remove dense-special assumptions
- optimize sparse kernels and coalescing

Deliverable:
- broad model coverage under strict elimination backend

---

## 9) Testing and verification

1. Structural tests
- LeanJaxpr conversion preserves dependency structure
- var uniqueness, topological order, and boundary constraints

2. Algebra tests
- compose/add associativity-like invariants where applicable
- transform inversion properties
- sparse consistency checks after each operation

3. AD differential tests
- compare generated gradients to runtime autograd on supported ops
- compare custom order results vs `fwd`/`rev` for consistency

4. Distributed tests
- single-rank equivalence to distributed rank-aggregated result
- communication plan reproducibility and determinism

5. Failure-mode tests
- missing-rule strict failure diagnostics
- invalid custom order diagnostics
- unsupported collective pattern diagnostics

---

## 10) Key risks and mitigation

Risk: LeanJaxpr conversion from effectful IR is messy.
- Mitigation: explicit non-eliminable boundaries and strict validation.

Risk: sparse algebra implementation complexity.
- Mitigation: phased algebra completeness with strong property tests from phase 1 onward.

Risk: distributed planning adds nontrivial complexity.
- Mitigation: first implement local elimination end-to-end, then layer communication model in phase 4.

Risk: no-fallback policy slows adoption.
- Mitigation: clear coverage diagnostics and incremental op rollout with explicit backend gating.

---

## Concrete next patch set

1. Add LeanJaxpr core types + `FnBody` conversion skeleton.
2. Add strict local-jac rule registry + missing-rule checker.
3. Add elimination `OrderPolicy` ADT with `custom`, `constraints`, and AlphaGrad adapters.
4. Add policy normalization + strict feasibility diagnostics (index base, duplicates, infeasible constraints).
5. Add sparse map type skeleton and invariants (no dense fallback codepath).
