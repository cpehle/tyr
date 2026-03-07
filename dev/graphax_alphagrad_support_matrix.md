# Graphax / AlphaGrad / Tyr Support Matrix (AD Elimination)

This document tracks implemented support parity for elimination-oriented AD.

## 1) Primitive and Operation Coverage

Legend:
- `explicit`: op-specific local-Jac behavior (exact or symbolic)
- `transform`: deferred structural-Jac behavior
- `placeholder`: registered but identity-like placeholder behavior
- `missing`: not lowered/registered in current Tyr path

### 1.1 Unary ops

| Family | Graphax | AlphaGrad | Tyr (current) |
|---|---|---|---|
| Core unary math (`neg`, `abs`, `exp`, `log`, `sqrt`, `square`, `sin`, `cos`, `tanh`, `logistic`) | `explicit` | `explicit` (sparsity-type based) | `explicit` symbolic/constant for KStmt unary set |
| Extra unary (`log1p`, `asin`, `acos`, `atan`, `tan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`, `erf`, `integer_pow`) | `explicit` | partial | `explicit` symbolic aliases in LeanJaxpr parity pack; still `missing` in current KStmt lowering vocabulary |
| Tyr-specific unary (`relu`, `gelu`, `silu`, `swish`, `recip`, `rsqrt`, init ops) | n/a | partial (`rsqrt`) | `explicit` symbolic/constant |

Tyr note:
- Constant-Jac exact rules: `Copy`, `Neg`, `Zero`, `One`, `PosInfty`, `NegInfty`.
- Symbolic scalar-Jac rules: `Exp`, `Exp2`, `Log`, `Log2`, `Abs`, `Relu`, `Sqrt`, `Rsqrt`, `Tanh`, `FastTanh`, `Sigmoid`, `Gelu`, `Silu`, `Swish`, `Sin`, `Cos`, `Recip`, `Square`.

### 1.2 Binary ops

| Family | Graphax | AlphaGrad | Tyr (current) |
|---|---|---|---|
| `add`, `sub`, `mul`, `div`, `max`, `min` | `explicit` | `explicit` (sparsity-type based) | `explicit` (add/sub exact constants, others symbolic) |
| `pow`, `atan2`, `eq`, `gt`, `lt`, `add_any` | `explicit` | partial/explicit | LeanJaxpr parity aliases now cover `pow`, `atan2`, `add_any`, and zero-edge semantics for `eq/gt/lt`; still `missing` in current KStmt binary vocabulary |

### 1.3 Reductions and contraction

| Family | Graphax | AlphaGrad | Tyr (current) |
|---|---|---|---|
| `reduce_sum`, `reduce_max`, `reduce_min` | `explicit` | `explicit` (sparsity-type based) | explicit structured semantics in KStmt reduction rules plus LeanJaxpr alias pack (`jax.lax.reduce_*`, `Graphax.reduce_*`) |
| `dot_general` | `explicit` | `explicit` | explicit LeanJaxpr `dotGeneral` local-Jac rule (plus `outer` structural rule for current KStmt lowering) |
| `prod`-style reductions | not first-class in Graphax core list | partial | lowered (`reduce .Prod`, `cumprod`) with explicit structured semantic tags |

### 1.4 Structural transforms

| Family | Graphax | AlphaGrad | Tyr (current) |
|---|---|---|---|
| `transpose` | `transform` | structural mapping | lowered + explicit structured semantic tag |
| `reshape`, `squeeze`, `broadcast`, `slice`, `slice_in_dim`, `pad`, `concat`, `convert_element_type` | `transform` / custom local rules | structural mapping | explicit structured semantics in both KStmt lowering subset (`broadcast`, `binaryBroadcast`, `sliceRows`, `sliceCols`, `concatCols`, `convert`, `swapLayout`) and LeanJaxpr alias pack (`reshape`, `squeeze`, `broadcast_in_dim`, `slice`, `slice_in_dim`, `pad`, `concatenate`, `convert_element_type`) |
| dynamic slice/update (`dynamic_slice`, `dynamic_update_slice`, `dynamic_update_index_in_dim`, gather/scatter family) | partial | present | explicit LeanJaxpr alias semantics (base/update edges; index inputs treated non-diff), including `_p` alias variants |

### 1.5 Special / no-grad / control primitives

| Family | Graphax | AlphaGrad | Tyr (current) |
|---|---|---|---|
| `stop_gradient`, `iota`, `device_put`, `pjit` handling | explicit no-edge behavior or special-cased | partial (`iota`, `stop_gradient`) | explicit no-edge rules for all four in LeanJaxpr rule pack (including Graphax alias names) |
| branch/control (`select_n`, `select`) | explicit masking-style elemental rule | partial | explicit select-data edge rules (selector excluded) in LeanJaxpr alias pack, including `_p` alias forms |
| higher-order control (`scan`, `cond`) | no elemental rules; hard failure | present in training loops | explicit deterministic control-flow alias rules in parity pack: `cond` ignores predicate input and propagates data inputs; `scan` propagates all data/carry inputs (no fallback) |
| communication collectives (`all_gather`, `all_to_all`, `reduce_scatter`, `collective_permute`, `psum/pmean/pmax/pmin`) | limited / ad-hoc (`pjit` hack, sparse transform paths) | partial | explicit LeanJaxpr unary semantic alias rules (dependency-preserving, no fallback) |
| unknown primitive behavior | hard failure (`NotImplementedError`) | registry-key lookup failure | hard coverage failure (no fallback) |

## 2) Order and Action Semantics

| System | Action space | Elimination space | Mapping |
|---|---|---|---|
| Graphax | n/a (order given as vertices) | 1-based vertices | direct |
| AlphaGrad | 0-based actions | 1-based vertices | `vertex = action + 1` |
| Tyr | 0-based `ActionId0` in optimization; 1-based `VertexId1` in elimination | 1-based vertices | checked adapters (`+1` / `-1`) with domain validation |

Tyr status:
- Adapter boundary is explicit and validated.
- Constraint feasibility is injected in action masking before transition.
- Action-space compatibility supports:
  - full-vertex action/logit sizing (`vertex = action + 1`),
  - explicit restricted action-space tables (`ActionId0 -> VertexId1`) for intermediate-only paths.
- Root/recurrent mask diagnostics enforce `true = invalid` semantics.

## 3) Constraint Masking and Fallback Semantics

| Behavior | Graphax | AlphaGrad | Tyr (current) |
|---|---|---|---|
| Hard constraints in optimizer loop | n/a | mostly mask-driven invalid-action suppression | integrated into masking + transition checks |
| Infeasible action handling | n/a | environment may no-op/skip in some paths | deterministic violation + penalty (no fallback) |
| No-rule fallback | none | registry-dependent failures | none (strict coverage checks) |

Tyr design choice:
- Keep strict no-fallback semantics for elimination backend selection.
- Keep constraints as part of optimization (not post-processing).

## 4) Current Tyr Status

Implemented now:
- `FromKStmt` lowering expanded beyond unary/binary to structural ops needed for Graphax/AlphaGrad-style elimination graphs.
- Rule-pack registration now has:
  - legacy bootstrap placeholder/hybrid packs (kept for transitional bring-up only),
  - all-supported explicit semantics pack (no placeholder fallback),
  - unary/binary semantic pack aligned with Graphax/AlphaGrad overlap.
- structural/reduction semantic pack using structured (typed) sparse-map tags.
- Coverage/extraction tests now cover arithmetic, structural, Graphax alias, communication alias, dot-general, `select`/`pad`, and dynamic update aliases (including `_p` forms) in `FnBody -> LeanJaxpr -> extract` source paths.
- A declared-coverage gate now validates matrix claims directly:
  - `registerKStmtAllSupportedSemanticsRules` must cover `allKStmtSupportedOpNames`,
  - `registerGraphaxAlphaGradParityRules` must cover the declared Graphax/AlphaGrad parity op families,
  - higher-order control aliases (`allHigherOrderControlAliasOpNames`, including `scan`/`cond`) must be covered by explicit deterministic control-flow rules (strict no-fallback behavior).
- AlphaGrad task materialization for all non-`RoeFlux_1d` tasks (`Perceptron`, `Encoder`, `RobotArm_6DOF`, `BlackScholes_Jacobian`, `HumanHeartDipole`, `PropaneCombustion`) now uses strict all-semantics registration and rejects non-semantic extracted edges.

Still missing for closer parity:
- Full numeric structural Jacobian payloads for every structural primitive. Current status: exact sparse-entry payloads are now emitted for key linear structural/reduction ops (`broadcast`, `binaryBroadcast`, `reduce_sum`, `transpose`, `sliceRows`, `sliceCols`, `concatCols`, `cumsum`) when shape metadata is available, and source-path structural aliases now do the same for shape-aware `broadcast_in_dim`, `slice`/`slice_in_dim`, `transpose`, and `concatenate` cases; nonlinear/value-dependent structural ops still use semantic tags.
- Full subjaxpr/body interpretation for higher-order/control primitives. Current status: `scan`/`cond` rules are now metadata-aware (predicate/carry/data partitioning and multi-output routing), but still conservative dependency semantics rather than branch/body jaxpr execution.
- Dot-general backend representability beyond canonical 2D `mm`/`outer` forms. Current status: `LowerKStmt` now also accepts dot-general specs with any number of leading unit batch axes (canonicalized to `KStmt.mm`), but still rejects non-leading/non-unit batch patterns and richer contraction layouts.
- Separate execution checklists now live in:
  - `dev/graphax_parity_checklist.md`
  - `dev/alphagrad_parity_checklist.md`

## 5) Immediate Next Priorities

1. Complete structural exact-payload coverage for remaining structural ops and alias paths (especially reductions, `pad`, and higher-rank/value-dependent transforms where representable).
2. Upgrade `scan`/`cond` from metadata-aware dependency routing to full subjaxpr/body-aware local-Jac semantics.
3. Extend dot-general lowering beyond `mm`/outer-like subsets for additional contract/batch patterns that can map to executable backends.
