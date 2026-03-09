# Tyr AD Improvement Plan

This is a broader AD roadmap for Tyr beyond the current Graphax/AlphaGrad elimination port.

Focused companion plan:

- `dev/tensorstruct_lean4_ad_integration_plan.md` for the Lean4 elaboration +
  `TensorStruct` + direct `LeanJaxpr` frontend integration path.

## Principles

- Keep strict no-fallback behavior when an advanced AD backend is explicitly selected.
- Reuse one canonical AD IR and metadata model across `FnBody`, `KStmt`, and higher-level model code.
- Prefer exact sparse/local-linear semantics when representable; use structured semantic tags only when exact payloads are genuinely unavailable.
- Separate three concerns cleanly:
  1. graph extraction,
  2. local derivative semantics,
  3. backend-specific execution (`JVP`/`VJP`, elimination, checkpointed reverse, search-guided scheduling).

## 1. Canonical AD IR And Metadata

Goal:
- Make `LeanJaxpr` the shared AD substrate instead of a port-specific side path.

Work:
- Expand `VarMeta` so shape/layout/sharding/alias/value-availability metadata is propagated from `FnBody`, `KStmt`, and model frontends.
- Add explicit `inputs`, `outputs`, and `eliminable` partitions to the elimination graph so Graphax-style `fwd`/`rev` and filtered-order semantics are first-class.
- Preserve more source-level structural parameters (`broadcast_dimensions`, concat axes, slice windows, padding specs, gather/scatter metadata) directly in `OpParams`.

Payoff:
- One normalization pipeline can feed elimination AD, classic reverse-mode, forward-mode, and diagnostic tools.

## 2. Local Derivative Coverage

Goal:
- Move from symbolic coverage to execution-grade local derivative semantics.

Work:
- Finish exact sparse payload coverage for remaining linear structural primitives and source-path aliases.
- Add richer `dot_general` metadata and representability checks, then support more contract/batch layouts in executable lowering.
- Bridge existing `JVPRule` / `VJPRule` logic into the local-Jac registry so primitive formulas are not duplicated.
- Add explicit value-dependent tags or payload builders for nonlinear structured ops where the Jacobian depends on primal values.

Payoff:
- Elimination, sparse forward-mode, and reverse accumulation all benefit from the same primitive library.

## 3. Higher-Order And Interprocedural AD

Goal:
- Stop treating `scan` / `cond` / closures as dependency-only approximations.

Work:
- Extend `FromFnBody` to retain body/subjaxpr information for higher-order control.
- Interpret `scan` / `cond` with branch/body-aware local-Jac rules instead of metadata-only routing.
- Add support for nested calls and effect boundaries so differentiation can cross module-level helper functions safely.
- Make checkpointing/recomputation policies explicit at the IR level.

Payoff:
- Tyr gets robust AD on real training/inference programs instead of only flattened primitive graphs.

## 4. Backend Strategy

Goal:
- Treat elimination AD as one backend in a larger Tyr AD stack.

Work:
- Expose backend selection at the user API: classic reverse, forward/JVP, sparse elimination, recursive checkpoint, comm-aware elimination.
- Add backend capability checks and precise unsupported diagnostics before execution.
- Add Graphax-style order presets (`fwd`, `rev`, explicit vertex order) and AlphaGrad-compatible search modes.
- Add optional exact AlphaGrad-style rewards (`nops`, runtime proxy) alongside Tyr’s current heuristic/comm-aware objective.

Payoff:
- Users can choose the AD backend that matches the workload instead of being locked into one strategy.

## 5. Search, Scheduling, And Policies

Goal:
- Make learned and heuristic elimination policies a real Tyr subsystem, not only a port example.

Work:
- Preserve AlphaGrad compatibility as an adapter mode, but keep Tyr’s explicit `actionVertices` table as the canonical action-space abstraction.
- Add graph transforms/baselines such as pre-eliminations, graph cleanup/compression, and stronger Markowitz-style heuristics.
- Upgrade the current example policy stack from small MLP features to graph-aware encoders with masked attention over eliminable/output vertices.
- Support shared-shape multi-task training and transfer across elimination tasks.

Payoff:
- Research workflows can compare hand-designed orders, heuristics, and learned policies inside one runtime.

## 6. Validation And CI

Goal:
- Make AD coverage and correctness measurable.

Work:
- Keep support matrices as executable coverage gates.
- Add differential tests against existing Tyr autograd on the supported subset.
- Add property tests for sparse-map algebra, graph invariants, and order-policy legality.
- Add benchmark suites for compile time, elimination cost, runtime, and memory/communication estimates.

Payoff:
- AD work stops regressing silently as the op surface and backend count grow.

## 7. Suggested Execution Order

Near term:
- Finish exact structural alias/reduction coverage.
- Add explicit eliminable/input/output partitions and Graphax-style order presets.
- Extend `dot_general` lowering coverage.

Mid term:
- Implement body-aware `scan` / `cond`.
- Unify local-Jac rules with existing `JVP` / `VJP` registries.
- Add exact AlphaGrad reward modes and stronger policy baselines.

Long term:
- Make `LeanJaxpr` the common substrate for all Tyr AD backends.
- Add graph-aware learned schedulers and distributed/communication-aware elimination planning.
- Expose backend selection and diagnostics in the main Tyr user API.
