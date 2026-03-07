# Graphax-Style Elimination: Issue Sequence

This issue stack translates the plan in `dev/graphax_vertex_elimination_mapping.md` into implementable work items.

Status:
- Issue 1 is now bootstrapped in code (new `Tyr/AD` hierarchy + order-policy/adapter scaffolding).
- Remaining issues are ordered by dependency and correctness risk.

Progress update (2026-03-06):
- Issue 1: completed (module scaffolding + adapters + validation + tests wired).
- Issue 2: skeleton completed (`FromFnBody` conservative lowering).
- Issue 2: advanced further (special-op lowering kinds and arity checks now include communication/structural alias families).
- Issue 3: advanced (lowering now covers unary/binary plus reductions, broadcasts, transforms, slice/concat, outer, and scans with strict unsupported diagnostics for the remainder).
- Issue 4: largely completed for core invariants (deterministic vertex order utilities + topological/output availability checks).
- Issue 5: partially advanced (local-Jac rule registry + strict coverage checker + richer source metadata diagnostics).
- Issue 6: completed (new sparse core types in `Tyr/AD/Sparse/{Dim,Map,Transform}.lean`).
- Issue 7: completed for core algebra (`Validate` + `Add` + `Compose` with strict checks, no dense fallback).
- Issue 8: advanced from skeleton to checked sparse elimination (`Eliminate` now uses sparse `compose`/`add` and surfaces errors deterministically).
- Issue 9: partially advanced (constraint feasibility module added).
- Issue 10: advanced further (soft/comm objective terms now shape non-terminal AlphaGrad priors and values in addition to transition rewards, while preserving strict `true = invalid` mask semantics).
- Issue 11: partially advanced (communication cost abstractions added).
- Issue 11: advanced further (group/hint communication terms now affect heuristic policy scoring, not only cumulative episode accounting).
- Issue 15: advanced (`LeanJaxpr -> LocalJacEdge` extraction path, `buildAndExtract*`, and elimination adapters from jaxpr/kstmt).
- Issue 15: advanced further (`buildAndExtractFromFnBody` parity tests now cover mixed no-grad/communication/structural/dot-general alias extraction).
- Issue 16: advanced (KStmt rule-pack now includes all-lowered placeholder registration helpers, including structural ops beyond unary/binary).
- Issue 16: advanced further (all-supported explicit semantics pack added and parity-facing extraction tests moved off placeholder/hybrid behavior).
- Issue 17: advanced (constant-Jac + symbolic unary/binary semantics aligned with Graphax/AlphaGrad overlap: add/sub/mul/div/max/min and core unary set).
- Issue 20: advanced further (source-path structural alias rules now cover `transpose`/`reshape`/`squeeze`/`broadcast_in_dim`/`slice`/`concatenate`/`convert_element_type` with parity tests).
- Issue 20: advanced further (dynamic/update source-path aliases now cover `dynamic_slice`/`dynamic_update_slice` plus gather/scatter family with explicit non-diff index handling in parity tests).
- Issue 20: advanced further (key structural KStmt rules now emit exact sparse payload entries when shape metadata is available: `broadcast`, `reduce_sum`, `transpose`, `sliceRows`, `sliceCols`, `concatCols`, `cumsum`).
- Issue 21: advanced (parity rule-pack now includes explicit no-grad/control rules for `stop_gradient` and `iota`, plus dedicated dot-general semantics and extraction tests).
- Issue 21: advanced further (Graphax/JAX alias coverage now includes extra unary/binary primitives and communication-collective unary aliases, with parity tests gating extraction behavior).
- Issue 21: advanced further (source-path reduction/control aliases now include `reduce_*` and `select_n` semantics with parity tests).
- Issue 21: advanced further (source-path alias coverage now includes `_p` primitive variants and additional AlphaGrad-relevant primitives: `select`, `slice_in_dim`, `pad`, and `dynamic_update_index_in_dim` with explicit local-Jac semantics/tests).
- Issue 21: parity baseline completed for higher-order control aliases (`scan`/`cond`) with deterministic control-flow local-Jac rules (no fallback): `cond` ignores predicate input and propagates data inputs; `scan` propagates all data/carry inputs.
- Issue 21: advanced further (`scan`/`cond` control-flow rules are now metadata-aware and multi-output aware, partitioning predicate/carry/data paths using typed op params).
- Issue 12: advanced (`LeanJaxpr -> KStmt` lowering now covers `dot_general` and `mma` families for representable `mm`/outer-like contraction patterns).
- Issue 2: advanced further (`FnBody -> LeanJaxpr` now canonicalizes dot-general aliases and records typed control-flow counts for `scan`/`cond`).
- Issue 21: advanced in AlphaGrad example integration path (`Examples/AlphaGradPort/Tasks`) by switching all non-`RoeFlux_1d` task materialization to strict all-semantics rule registration and asserting non-semantic edge extraction as hard failure.
- Issue 21: advanced further with explicit support-matrix gate tests that enforce declared op-family registration for the parity pack, including `scan`/`cond` control-flow aliases under strict no-fallback semantics.
- Issue 22: advanced (action-space compatibility layer now supports full-vertex and explicit restricted action tables, with root/recurrent mask-semantics diagnostics and strict `ActionId0 -> VertexId1` boundary checks).
- Issue 23: advanced (DAG keying now canonicalizes sparse maps via shape + sorted sparse entries, removing `repr`-string dependence for transposition identity).
- DAG-MCTS execution is now policy-selectable (`alphaZero` and `gumbelMuZero`) through a shared DAG search entrypoint.

Immediate next priorities (2026-03-07):
1. Extend dot-general lowering beyond current representable `mm`/outer-like contraction subsets (including richer contract/batch patterns).
2. Expand exact sparse payload coverage to remaining structural primitives and alias paths.
3. Move `scan`/`cond` from metadata-aware dependency routing to full subjaxpr/body-aware local-Jac semantics.

## Issue 1: Bootstrap LeanJaxpr + Policy Modules

Goal:
- Create initial module layout for LeanJaxpr-like IR and elimination order policy.

Scope:
- Add `Tyr/AD/JaxprLike/Core.lean`.
- Add `Tyr/AD/JaxprLike/Validate.lean`.
- Add `Tyr/AD/Elim/OrderPolicy.lean`.
- Add `Tyr/AD/Elim/AlphaGradAdapter.lean`.
- Add umbrella imports `Tyr/AD.lean`, `Tyr/AD/JaxprLike.lean`, `Tyr/AD/Elim.lean`.

Acceptance:
- Modules compile.
- No behavior change in existing `Tyr.AutoGrad`.
- Explicit `ActionId0 <-> VertexId1` adapter API exists with domain checks.

## Issue 2: `FnBody -> LeanJaxpr` Conversion Skeleton

Goal:
- Introduce deterministic normalization entrypoint from Lean IR `FnBody`.

Scope:
- New module `Tyr/AD/JaxprLike/FromFnBody.lean`.
- Define conversion context and placeholder handling for unsupported/effectful statements.

Acceptance:
- Conversion produces `LeanJaxpr` for a minimal arithmetic subset.
- Unsupported constructs fail with source-aware diagnostics.

Depends on:
- Issue 1.

## Issue 3: `KStmt -> LeanJaxpr` Conversion Skeleton

Goal:
- Normalize GPU IR (`KStmt`) into the same LeanJaxpr-like representation.

Scope:
- New module `Tyr/AD/JaxprLike/FromKStmt.lean`.
- Preserve op names/params required by local-Jacobian rule dispatch.

Acceptance:
- Minimal GPU op subset round-trips into valid `LeanJaxpr`.
- Shared validation pipeline runs for both conversion fronts.

Depends on:
- Issue 1.

## Issue 4: Validation Hardening + Deterministic Vertex Assignment

Goal:
- Make normalization robust enough for elimination planning.

Scope:
- Enforce unique IDs, topological equation order, and boundary/non-eliminable markings.
- Add deterministic `eqn index -> vertexId1` assignment rules.

Acceptance:
- Property-style tests for deterministic vertex numbering.
- Duplicate/missing vertex diagnostics include equation/source metadata.

Depends on:
- Issues 2 and 3.

## Issue 5: Local Jacobian Rule Registry for LeanJaxpr Eqns

Goal:
- Build rule dispatch for per-equation local Jacobian edges.

Scope:
- New modules:
- `Tyr/AD/JaxprLike/Rules.lean`
- `Tyr/AD/JaxprLike/RuleRegistry.lean`
- `Tyr/AD/JaxprLike/RuleCheck.lean`
- Bridge from existing `JVPRule`/`VJPRule` where possible.
- Strict missing-rule failure (no fallback).

Acceptance:
- Rule coverage checker reports missing ops before elimination.
- Arithmetic baseline ops have local-Jacobian rules.

Depends on:
- Issues 2, 3, and 4.

## Issue 6: Sparse Linear Map Core Types

Goal:
- Introduce sparse-first Jacobian edge representation.

Scope:
- New modules:
- `Tyr/AD/Sparse/Dim.lean`
- `Tyr/AD/Sparse/Map.lean`
- `Tyr/AD/Sparse/Transform.lean`
- Invariants for shape compatibility and transform composition.

Acceptance:
- Sparse map constructors/invariants compile.
- Basic sanity tests on transform legality.

Depends on:
- Issue 5.

## Issue 7: Sparse Algebra Ops (`compose`, `add`, `coalesce`)

Goal:
- Implement elimination-critical sparse algebra primitives.

Scope:
- New modules:
- `Tyr/AD/Sparse/Compose.lean`
- `Tyr/AD/Sparse/Add.lean`
- `Tyr/AD/Sparse/Validate.lean`
- Include local/comm cost estimators used later by schedulers.

Acceptance:
- Algebra tests for compatibility and structural correctness.
- No implicit dense fallback path.

Depends on:
- Issue 6.

## Issue 8: Elimination Graph Build + Kernel

Goal:
- Implement Graphax-style elimination engine on LeanJaxpr edges.

Scope:
- New modules:
- `Tyr/AD/Elim/Graph.lean`
- `Tyr/AD/Elim/Eliminate.lean`
- Build forward/backward adjacency from local Jacobian edges.
- Implement vertex elimination update rules.

Acceptance:
- End-to-end elimination runs on arithmetic subset.
- Structural invariants preserved after each elimination step.

Depends on:
- Issues 5 and 7.

## Issue 9: Order Policy Execution + Constraint Feasibility

Goal:
- Execute explicit, constrained, and AlphaGrad-derived policies with strict checks.

Scope:
- Extend `Tyr/AD/Elim/OrderPolicy.lean`.
- Add `Tyr/AD/Elim/ConstraintFeasibility.lean`.
- Compile hard constraints into step-level feasibility predicate.
- Enforce no-fallback behavior on infeasible policy states.

Acceptance:
- Invalid IDs, duplicates, and infeasible constraints produce deterministic errors.
- `ActionId0 -> VertexId1` execution path is fully covered by tests.

Depends on:
- Issues 1 and 8.

## Issue 10: AlphaGrad Constraint-Integrated Optimization Path

Goal:
- Make constraints first-class in AlphaGrad optimization, not post-processing.

Scope:
- Inject constraint feasibility into action masking.
- Keep trajectory storage in `ActionId0`.
- Convert to `VertexId1` only at elimination boundary.
- Add hard/soft constraint objective components.

Acceptance:
- Constraint violations are prevented by masking.
- Replay-time assertions pass for action->vertex conversion.
- Existing AlphaGrad policy training shape remains compatible.

Depends on:
- Issue 9.

## Issue 11: Communication-Aware Scheduling and Costing

Goal:
- Add distributed communication terms to elimination and AlphaGrad objectives.

Scope:
- New modules:
- `Tyr/AD/Elim/CommModel.lean`
- `Tyr/AD/Elim/Cost.lean`
- Integrate comm-aware scoring (`bytes`, `collectives`, latency terms).
- Connect score into constrained policy ranking and AlphaGrad reward path.

Acceptance:
- Communication plan is deterministic and inspectable.
- Comm-aware order differs from flop-only baseline on sharded cases.

Depends on:
- Issues 8, 9, and 10.

## Issue 12: Lowering to `FnBody`/`KStmt` + Differential Tests

Goal:
- Close the loop from elimination output to executable AD code.

Scope:
- New modules:
- `Tyr/AD/Elim/LowerFnBody.lean`
- `Tyr/AD/Elim/LowerKStmt.lean`
- Differential tests vs existing autograd for supported op set.

Acceptance:
- Elimination backend produces gradients matching baseline within tolerance.
- Unsupported ops fail early via rule coverage checks.

Depends on:
- Issues 8, 9, and 11.

## Cross-Cutting Test Issues

Issue 13: Property and Differential Test Suite
- Add tests for invariants, order validation, adapter correctness, sparse algebra, and distributed equivalence.
- Can be developed incrementally starting after Issue 4.

Issue 14: Diagnostics and Developer Tooling
- Add debug dumps: LeanJaxpr, elimination graph, elimination trace, communication plan.
- Start after Issue 8 and expand through Issue 12.

## Additional Discovered Issues

## Issue 15: Local-Jacobian Rule Execution + Edge Extraction

Goal:
- Provide the missing `LeanJaxpr -> Array LocalJacEdge` bridge by executing registered local-Jac rules equation-by-equation.

Scope:
- New module `Tyr/AD/JaxprLike/Extract.lean`.
- Aggregate source-aware rule-execution errors with equation indices.
- Extend pipeline with `buildAndExtract*` entrypoints.

Acceptance:
- For covered ops, extraction returns deterministic edges.
- Missing/malformed rules fail with source metadata, without fallback.

Depends on:
- Issue 5.

## Issue 16: Built-In Rule-Pack Bootstrapping

Goal:
- Seed registry coverage for the currently supported lowering front (`FromKStmt` unary/binary ops).

Scope:
- New module `Tyr/AD/JaxprLike/RulePackKStmt.lean`.
- Canonical op-name helpers aligned with `FromKStmt`.
- Batch registration helpers for placeholder and future real rules.

Acceptance:
- `buildAndExtractFromKStmts` passes coverage for unary/binary subsets after pack registration.
- Rule-pack behavior is tested and deterministic.

Depends on:
- Issues 3, 5, and 15.

## Issue 17: Real Local-Jacobian Semantics + Shape Propagation

Goal:
- Replace identity-like placeholders with op-specific sparse Jacobian entries and shape metadata.

Scope:
- Add real local-Jac rules for arithmetic/unary/binary primitives.
- Thread `inDim?`/`outDim?` and sparse entries from lowering metadata.
- Validate generated maps against sparse invariants before elimination.

Acceptance:
- Eliminator consumes non-placeholder sparse maps on baseline arithmetic graphs.
- Shape mismatches are caught at rule construction/extraction time.

Depends on:
- Issues 6, 7, and 16.

## Issue 18: Order-Policy Execution Layer

Goal:
- Add a strict execution boundary from `NormalizedOrderPolicy` to concrete elimination runs.

Scope:
- Deterministic policy interpreter for explicit, constrained, and AlphaGrad-derived orders.
- Hook step feasibility checks and constraint diagnostics into execution.
- Expose one-shot `policy -> elimination result` API.

Acceptance:
- Invalid/infeasible policy states fail before mutating elimination state.
- Policy execution is fully test-covered across ID-space adapters.

Depends on:
- Issues 8, 9, and 10.

## Issue 19: Deterministic Edge Canonicalization for DAG/Hashing

Goal:
- Remove reliance on `map.repr` strings for structural identity in DAG keys and diagnostics.

Scope:
- Canonical edge serialization based on `(src,dst,entries,shape)` tuples.
- Update DAG keying and trace outputs to use canonical edge descriptors.

Acceptance:
- Equivalent sparse maps with different `repr` strings hash to identical keys.
- DAG transposition behavior remains deterministic under canonicalized edges.

Depends on:
- Issues 7, 8, and 10.

## Issue 20: Graphax/AlphaGrad Structural-Op Parity in `FromKStmt`

Goal:
- Keep `KStmt -> LeanJaxpr` lowering aligned with the structural op families used by Graphax/AlphaGrad graph builders.

Scope:
- Preserve canonical lowering and metadata for: `reduce`, `reduceAccum`, `broadcast`, `binaryBroadcast`, `transpose`, `swapLayout`, `convert`, `sliceRows`, `sliceCols`, `concatCols`, `outer`, `cumsum`, and `cumprod`.
- Add explicit unsupported diagnostics for remaining `KStmt` constructors with source indices.
- Maintain deterministic var ordering and output inference under mixed structural + arithmetic statements.

Acceptance:
- Mixed structural/arithmetic programs lower without coverage regressions.
- Unsupported `KStmt` constructors remain hard failures with source-aware diagnostics.
- Deterministic lowering tests cover representative structural op paths.

Depends on:
- Issues 3, 4, and 15.

## Issue 21: Graphax/AlphaGrad Coverage Matrix as Rule-Pack Gate

Goal:
- Turn support parity into an executable gate instead of a narrative-only target.

Scope:
- Maintain `dev/graphax_alphagrad_support_matrix.md` as the source of truth for op-level status.
- Add CI-facing coverage tests that check:
  - unary/binary semantic rule presence for the declared overlap set,
  - explicit semantic rule presence for declared KStmt + Graphax/AlphaGrad parity op families (including communication, dynamic/update, and higher-order control aliases),
  - explicit deterministic control-flow alias rule presence for `scan`/`cond` aliases,
  - strict failure for undeclared ops (no fallback).
- Ensure tests exercise both explicit all-supported KStmt registration and Graphax/AlphaGrad parity-pack registration helpers.

Acceptance:
- Matrix and tests remain synchronized (no silently drifting support claims).
- Coverage failures identify op names and lowering source metadata.
- `scan`/`cond` alias families are explicitly gated as registered no-fallback control-flow rules.
- No runtime fallback path is introduced to satisfy parity tests.

Depends on:
- Issues 5, 16, and 17.

## Issue 22: AlphaGrad Action-Space and Mask-Semantics Compatibility Layer

Goal:
- Match AlphaGrad action/mask behavior precisely while preserving Tyr’s strict constraint semantics.

Scope:
- Support both action-space conventions found in AlphaGrad code paths:
  - full-vertex logits with invalid masking,
  - intermediate-count action sizing.
- Add explicit mask-semantics diagnostics (`true = invalid`) at root and recurrent calls.
- Add replay/trace validation utilities to prevent sentinel ambiguity and index-base drift.
- Document/encode the `action0 -> vertex1` boundary as the only conversion point.

Acceptance:
- Action/mask compatibility tests cover both sizing conventions and constraint masks.
- Invalid actions fail deterministically with actionable diagnostics.
- 0-based action traces and 1-based elimination orders round-trip losslessly.

Depends on:
- Issues 9, 10, and 18.

## Issue 23: DAG Canonicalization Follow-Through for Sparse-Map Semantics

Goal:
- Complete the move away from `repr`-dependent DAG identity under richer semantic maps.

Scope:
- Extend canonical edge keys to include sparse entries and shape fields, not just textual repr.
- Add deterministic sorting/canonicalization for edge-key generation.
- Validate DAG transposition behavior under semantically equivalent maps with differing repr strings.

Acceptance:
- Equivalent maps produce identical DAG keys regardless of repr text.
- Non-equivalent maps remain distinguishable.
- DAG search behavior remains deterministic across runs.

Depends on:
- Issues 19 and 21.
