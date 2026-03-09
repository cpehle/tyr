# Graphax Parity Checklist

This checklist tracks the remaining work to reach practical Graphax parity in Tyr's elimination AD path.

Scope:
- Match Graphax's elimination semantics and supported primitive behavior closely enough that Graphax-style orders, local-Jac extraction, and elimination execution transfer cleanly.
- Keep Tyr-only extensions (`scan`, `cond`, richer action-space abstractions, alternative rewards) out of the Graphax parity definition unless Graphax itself depends on them.

Non-goals:
- Matching Graphax's unsupported-primitive surface by adding Tyr regressions for every unsupported case.
- Counting Tyr-only `scan` / `cond` support as a parity requirement, since upstream Graphax hard-fails on unknown primitives rather than executing those higher-order bodies.

## Current Status

Already in place:
- LeanJaxpr extraction and sparse elimination graph construction.
- Graphax/JAX alias coverage for the main arithmetic, reduction, structural, no-grad, communication, and dynamic-update families.
- Exact sparse payloads for key shape-aware structural aliases: `reshape`, `squeeze`, `broadcast_in_dim`, `slice`, `slice_in_dim`, `transpose`, `concatenate`, and `pad` when explicit pad extents are present, including the `FnBody` path via `FnBodyLoweringHints`.
- `FnBody` lowering now records a first-class `sourceOp` on lowered equations, so canonicalized `dot_general` / `scan` / `cond` still retain their Graphax/JAX frontend primitive identity in semantic tags and diagnostics.
- Declaration-driven lowering can now consume registered per-decl `FnBodyLoweringHints`, so higher-level Tyr frontends can attach source metadata once and reuse the standard `buildFromDecl` / `buildAndExtractFromDecl` path.
- `buildFromDecl` can now also prefer elaboration-registered direct frontend AD bundles (for example `attribute [ad_frontend frontendSpec] f`) over `FnBody` recovery, so higher-level Tyr frontends are no longer forced to round-trip through lowered Lean IR when they can emit high-level AD IR directly.
- Strict no-fallback rule-pack gating for the declared Graphax parity op surface.
- LeanJaxpr-derived elimination graphs now carry explicit `inputs`, `outputs`, and `eliminable` partitions.
- The higher-level order-policy surface now resolves Graphax-style `forward` / `reverse`, filtered explicit custom orders, and AlphaGrad action orders against that eliminable set, and the Jaxpr/KStmt execution path can run policies directly.
- `dot_general` lowering now accepts canonical `mm` / `outer`-like cases even when unit batch axes appear in arbitrary positions, not only as leading dimensions, including transpose variants that still normalize to a single `KStmt.mm`.
- Supported-subset parity regressions now compare Graphax/JAX structural alias payloads directly against the executable KStmt path for overlapping exact cases (`broadcast_in_dim`, `transpose`, `slice_in_dim`, `concatenate`).

Still blocking full parity:
- Structural local-Jac behavior is not yet exact for every transform/reduction path Graphax models explicitly. Remaining gaps are mostly automatic traced/elaborated-frontend population of richer source metadata, whether via direct `LeanJaxpr` emission or the existing per-decl hint path (especially traced shape/extent params for `pad` and similar ops), plus the reduction/value-dependent cases that still stay on the conservative semantic path.
- Executable lowering for `dot_general` still covers only the representable `mm` / `outer`-like subset; richer contraction layouts still stop at semantic extraction.

## P0: Core Graphax Semantics

1. Add explicit `inputs`, `outputs`, and `eliminable` graph partitions.
Goal:
- Represent the same boundary distinctions Graphax uses when deciding which vertices may be removed and what `fwd` / `rev` should enumerate.

Work:
- Extend the elimination graph metadata to track input/output/intermediate/eliminable classes directly.
- Stop inferring "everything is eliminable" from equation count alone.
- Carry those partitions through validation, order normalization, and elimination execution.

Acceptance:
- A graph can distinguish non-eliminable boundary vertices from eliminable interior vertices.
- Validation rejects explicit orders that include non-eliminable vertices.
- The partition information is visible in diagnostics and tests.

Likely touch points:
- `Tyr/AD/JaxprLike/Core.lean`
- `Tyr/AD/JaxprLike/VertexOrder.lean`
- `Tyr/AD/Elim/Graph.lean`
- `Tyr/AD/Elim/Eliminate.lean`

2. Add Graphax-style order presets and filtered order validation.
Goal:
- Make `fwd`, `rev`, and explicit custom orders behave like Graphax rather than only accepting raw total orders over equation indices.

Status:
- Completed for the current direct Graphax path: partitioned graphs expose deterministic forward/reverse eliminable orders, complete-order validation rejects non-eliminable or incomplete orders, and higher-level `OrderPolicy` normalization/execution now resolves `forward`, `reverse`, explicit custom orders, and AlphaGrad action orders against the graph's explicit eliminable set.

Work:
- Add user-facing order presets for forward and reverse eliminable traversal.
- Validate explicit orders against the eliminable set rather than only shape/duplicate checks.
- Preserve Graphax's 1-based vertex convention in all explicit-order APIs.

Acceptance:
- `fwd` expands to the Graphax-compatible eliminable forward order.
- `rev` expands to the reverse eliminable order.
- Explicit orders over eliminable vertices round-trip through normalization without renumbering drift.

Likely touch points:
- `Tyr/AD/JaxprLike/VertexOrder.lean`
- `Tyr/AD/Elim/OrderPolicy.lean`
- `Tyr/AD/Elim/FromJaxpr.lean`

## P1: Primitive And Local-Jac Exactness

3. Finish exact structural payload coverage for the remaining Graphax transform surface.
Goal:
- Replace semantic tags with exact sparse entries wherever the transform is linear and shape metadata makes the mapping unambiguous.

Work:
- Add exact payload builders for the remaining structural aliases and source-path variants that are still representable.
- Wire real traced/elaborated frontends to populate either direct `LeanJaxpr` registrations or `FnBodyLoweringHints` automatically so traced `jax.lax.pad` / `Graphax.pad` equations hit the exact sparse payload path without manual metadata injection.
- Audit higher-rank transform cases to separate "exactly representable" from "must remain semantic".
- Keep ambiguous/value-dependent cases on the conservative semantic path.

Acceptance:
- The declared Graphax structural alias families are either exact or explicitly documented as intentionally semantic-only.
- Extraction regressions assert concrete sparse entries for the new exact cases.
- No dense fallback is introduced.

Likely touch points:
- `Tyr/AD/JaxprLike/RulePackKStmt.lean`
- `Tests/TestADJaxprLikeExtract.lean`
- `dev/graphax_alphagrad_support_matrix.md`

4. Extend Graphax-facing `dot_general` representability and lowering.
Goal:
- Cover more of Graphax's contraction surface in executable Tyr lowering, not only the current `mm` / `outer` subset.

Work:
- Support additional contraction layouts beyond the newly accepted arbitrary-position unit-batch `mm` / `outer` cases and transpose variants.
- Distinguish "extractable local-Jac semantics" from "lowerable back to executable KStmt" in diagnostics.
- Add tests for accepted and intentionally rejected layouts.

Acceptance:
- More Graphax-style `dot_general` cases lower successfully.
- Unsupported layouts fail with precise representability diagnostics instead of generic rejection.
- Tests cover both the new accepted layouts and the remaining strict failures.

Likely touch points:
- `Tyr/AD/JaxprLike/FromFnBody.lean`
- `Tyr/AD/Elim/LowerKStmt.lean`
- `Tests/TestADJaxprLikeExtract.lean`

## P2: Quality And Verification

5. Add Graphax differential parity tests on the supported subset.
Goal:
- Turn "looks aligned" into executable evidence.

Status:
- Started for the overlapping exact structural subset: CI now checks that Graphax/JAX alias extraction and the executable KStmt path agree on sparse payload dimensions and entries for `broadcast_in_dim`, `transpose`, `slice_in_dim`, and `concatenate`. Remaining work is to extend that fixture set across more primitive/order families and, where practical, against external Graphax-produced references.

Work:
- Build a fixture set of Graphax-supported primitive graphs and explicit orders.
- Compare extracted edge structure, eliminable sets, and elimination-order behavior against the Graphax reference semantics.
- Keep unsupported Graphax cases out of the parity fixture set.

Acceptance:
- CI has a Graphax parity suite over the supported primitive subset.
- Failures identify the primitive/order family that drifted.

6. Tighten parity docs around true blockers versus Tyr-only extensions.
Goal:
- Keep future work focused on actual Graphax parity rather than mixing it with Tyr roadmap items.

Work:
- Keep the support matrix and issue sequence aligned with this checklist.
- Continue marking `scan` / `cond` as Tyr extensions, not Graphax parity blockers.

Acceptance:
- The support matrix, issue sequence, and this checklist describe the same Graphax target.

## Recommended Execution Order

1. Add graph partitions.
2. Add `fwd` / `rev` and filtered explicit-order validation.
3. Finish exact structural payload coverage.
4. Extend `dot_general` lowering.
5. Add differential Graphax parity tests.
