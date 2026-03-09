# AlphaGrad Parity Checklist

This checklist tracks the remaining work to reach practical AlphaGrad parity in Tyr's elimination-planning stack.

Scope:
- Match AlphaGrad's environment semantics, task materialization, reward modes, search interface, and training/runtime behavior closely enough that AlphaGrad-style experiments can be reproduced in Tyr.
- Preserve Tyr's stricter diagnostics where possible, but make any intentional semantic differences explicit.

Non-goals:
- Treating AlphaGrad parity as only an action-space adapter problem.
- Requiring Tyr to keep AlphaGrad quirks unless they materially affect reproduced behavior.

## Current Status

Already in place:
- Explicit `ActionId0 <-> VertexId1` adapters with checked conversions.
- Support for full-vertex and explicit restricted action tables.
- Root/recurrent invalid-mask diagnostics with strict `true = invalid` semantics.
- Partition-aware fixed-slot compatibility for tasks that carry explicit graph partitions: action space stays full-width, output/non-eliminable slots are masked invalid, and rollout length/termination now count eliminable vertices rather than raw slot width.
- Selectable reward modes for Tyr-native heuristic rewards, exact AlphaGrad base-env `-nops`, and a terminal-only `-total_nops` proxy for runtime-style comparisons.
- A runnable AlphaGrad-style environment, MCTS integration, and example policy-training pipelines.
- AlphaGrad port example tasks now default to the `-nops` reward mode, and logs report the selected reward mode explicitly.

Still blocking full parity:
- Tyr does not yet mirror AlphaGrad's exact fixed-slot action/output masking behavior across all paths; it is now available for partition-aware task graphs, but traced-source and joint-task paths still need to populate those partitions consistently.
- Most tasks are still materialized from Tyr-side `KStmt`/proxy definitions rather than traced JAX/Graphax callables.
- Runtime-timing reward integration, graph transforms/baselines, and policy architecture still differ substantially from AlphaGrad.

## P0: Environment And Task Semantics

1. Match AlphaGrad's exact action-slot and output-mask behavior.
Goal:
- Reproduce AlphaGrad's action-space sizing and masking conventions exactly where parity matters.

Status:
- Partially complete. Tyr now supports the key fixed-slot compatibility shape for partition-aware task graphs: `numVertices` remains the full action width, outputs/non-eliminable vertices are root/recurrent-invalid from the start, and episode length follows the eliminable partition. Remaining work is to carry the same semantics through traced task materialization, joint-task embedding, and any paths that still only see unpartitioned edge sets.

Work:
- Mirror the remaining padded/fixed-slot action behavior used in AlphaGrad policy and search code paths.
- Preserve AlphaGrad-compatible output masking and invalid-action masking semantics over the full vertex/action surface, including traced-source and joint-task materialization paths.
- Keep Tyr's explicit `actionVertices` table as the implementation mechanism, but expose an exact AlphaGrad-compatible mode.

Acceptance:
- Full-vertex and intermediate-count AlphaGrad sizing conventions are both reproducible.
- AlphaGrad-style invalid masks and output masks match expected behavior on fixture graphs.
- Action traces round-trip without sentinel ambiguity or off-by-one drift.

Likely touch points:
- `Tyr/AD/Elim/OrderPolicy.lean`
- `Tyr/AD/Elim/AlphaGradAdapter.lean`
- `Tyr/AD/Elim/AlphaGradMctx.lean`

2. Add a traced JAX/Graphax task frontend for AlphaGrad parity runs.
Goal:
- Stop relying mostly on hand-written Tyr proxy tasks when claiming AlphaGrad parity.

Work:
- Materialize tasks from traced JAX/Graphax callables, preserving the same output/input reuse behavior AlphaGrad's interpreter uses.
- Keep the existing Tyr task fixtures, but separate them from the exact-parity task path.
- Add import-time checks so unsupported extracted semantics fail early.

Acceptance:
- Core AlphaGrad benchmark tasks can be built from traced source definitions, not only local proxies.
- Output-reused-as-input cases match AlphaGrad's graph construction behavior.
- Task materialization failures are explicit about the unsupported primitive or metadata gap.

Likely touch points:
- `Examples/AlphaGradPort/Tasks.lean`
- `Tyr/AD/JaxprLike/FromFnBody.lean`
- parity-fixture scripts/docs under `dev/`

3. Add exact AlphaGrad reward modes.
Goal:
- Make AlphaGrad comparisons fair by supporting the same base reward definitions.

Status:
- Base reward-mode parity is now in place: Tyr supports exact stepwise `-nops`, retains the existing heuristic/communication-aware reward as a separate mode, and exposes a clearly labeled terminal-only `-total_nops` proxy for runtime-style comparisons. Remaining work here is direct `jacve`/timing integration if exact runtime reward parity is required.

Work:
- Add direct runtime/timing reward integration if exact `jacve`-style terminal reward parity is required.
- Keep the current heuristic/communication-aware reward separate from the parity modes.

Acceptance:
- Reward mode is selectable and visible in logs/config.
- The same action trace can be evaluated under AlphaGrad-parity and Tyr-native reward modes.
- Tests cover the shipped reward formulas (`-nops`, heuristic, and terminal-only proxy).

Likely touch points:
- `Tyr/AD/Elim/AlphaGradMctx.lean`
- `Examples/AlphaGradPort/A0Train.lean`
- `Examples/AlphaGradPort/PolicyTrain.lean`

## P1: AlphaGrad Search And Training Stack

4. Add AlphaGrad graph transforms and baselines.
Goal:
- Match the preprocessing and heuristic baseline tools AlphaGrad uses before or alongside learning.

Work:
- Port or reimplement `compress`, `clean`, `embed`, `safe_preeliminations`, and exact `minimal_markowitz`.
- Keep transform outputs compatible with Tyr's elimination graph representation and action masks.
- Compare Tyr-native heuristics against the AlphaGrad-compatible baselines rather than replacing one with the other.

Acceptance:
- AlphaGrad-style preprocessing can run over supported tasks end-to-end.
- Minimal-Markowitz and related baselines are available in the port examples.
- Tests show transform invariants and deterministic output on fixture graphs.

Likely touch points:
- `Tyr/AD/Elim/AlphaGradMctx.lean`
- `Examples/AlphaGradPort/PolicySweep.lean`
- new transform modules if the logic outgrows `AlphaGradMctx`

5. Upgrade the policy/model stack toward AlphaGrad's transformer setup.
Goal:
- Reduce the architecture gap between Tyr's example policies and AlphaGrad's learned agent.

Work:
- Add a graph-aware or transformer-style policy with output-aware masking/attention.
- Preserve the small MLP as a lightweight baseline, but stop treating it as the parity path.
- Align MCTS/root/recurrent data flow with AlphaGrad's history-stacked search inputs where required.

Acceptance:
- Tyr has an AlphaGrad-parity policy mode separate from the baseline MLP.
- Search/training code can consume the richer observation/model format without changing environment semantics.
- Training examples document which mode is "baseline" versus "parity".

Likely touch points:
- `Examples/AlphaGradPort/PolicyTrain.lean`
- `Examples/AlphaGradPort/A0Train.lean`

6. Add exact or explicitly compatible tree-search semantics.
Goal:
- Close the remaining gap between Tyr's AlphaZero/MuZero-style search loop and AlphaGrad's recurrent expansion behavior.

Work:
- Audit root/recurrent masking, history stacking, rollout-length handling, and child-prior masking behavior.
- Decide whether to replicate AlphaGrad's pre-step masking quirk exactly or keep Tyr's cleaner behavior behind a documented compatibility flag.
- Add deterministic fixtures for search-policy equivalence on small graphs.

Acceptance:
- Search semantics that materially affect chosen actions are either matched exactly or flagged as intentional compatibility deviations.
- Regression tests cover the selected compatibility mode.

## P2: Experiment Reproducibility

7. Add shared-task multi-run evaluation and reproducibility checks.
Goal:
- Make AlphaGrad parity claims rest on repeated experiments rather than one-off example runs.

Work:
- Add sweeps over the traced AlphaGrad benchmark set with fixed seeds and explicit reward/policy modes.
- Record comparable metrics: reward, chosen order, elimination cost, and runtime proxy.
- Keep the evaluation harness separate from exploratory Tyr-native policy experiments.

Acceptance:
- There is one documented AlphaGrad-parity evaluation path for supported tasks.
- Results are reproducible enough to catch semantic regressions in CI or scheduled runs.

8. Keep exact-parity quirks isolated behind explicit compatibility modes.
Goal:
- Avoid contaminating Tyr's cleaner abstractions when AlphaGrad has historical edge cases.

Work:
- Gate AlphaGrad-only quirks behind named compatibility settings.
- Document any preserved quirks, especially around masking and trace replay.

Acceptance:
- Default Tyr modes remain clean.
- AlphaGrad parity mode is explicit and test-covered.

## Recommended Execution Order

1. Match exact action-slot and output-mask semantics.
2. Add traced task materialization.
3. Finish runtime-timing reward parity if needed.
4. Port graph transforms and baselines.
5. Upgrade the policy/search stack.
6. Add reproducibility runs and compatibility-mode documentation.
