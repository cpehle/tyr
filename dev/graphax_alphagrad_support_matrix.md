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
| Extra unary (`log1p`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`, `erf`, `integer_pow`) | `explicit` | partial | `missing` in current KStmt lowering vocabulary |
| Tyr-specific unary (`relu`, `gelu`, `silu`, `swish`, `recip`, `rsqrt`, init ops) | n/a | partial (`rsqrt`) | `explicit` symbolic/constant |

Tyr note:
- Constant-Jac exact rules: `Copy`, `Neg`, `Zero`, `One`, `PosInfty`, `NegInfty`.
- Symbolic scalar-Jac rules: `Exp`, `Exp2`, `Log`, `Log2`, `Abs`, `Relu`, `Sqrt`, `Rsqrt`, `Tanh`, `FastTanh`, `Sigmoid`, `Gelu`, `Silu`, `Swish`, `Sin`, `Cos`, `Recip`, `Square`.

### 1.2 Binary ops

| Family | Graphax | AlphaGrad | Tyr (current) |
|---|---|---|---|
| `add`, `sub`, `mul`, `div`, `max`, `min` | `explicit` | `explicit` (sparsity-type based) | `explicit` (add/sub exact constants, others symbolic) |
| `pow`, `atan2`, `eq`, `gt`, `lt`, `add_any` | `explicit` | partial/explicit | `missing` in current KStmt binary vocabulary |

### 1.3 Reductions and contraction

| Family | Graphax | AlphaGrad | Tyr (current) |
|---|---|---|---|
| `reduce_sum`, `reduce_max`, `reduce_min` | `explicit` | `explicit` (sparsity-type based) | lowered from KStmt (`reduce`, `reduceAccum`) with explicit structured semantic tags |
| `dot_general` | `explicit` | `explicit` | explicit LeanJaxpr `dotGeneral` local-Jac rule (plus `outer` structural rule for current KStmt lowering) |
| `prod`-style reductions | not first-class in Graphax core list | partial | lowered (`reduce .Prod`, `cumprod`) with explicit structured semantic tags |

### 1.4 Structural transforms

| Family | Graphax | AlphaGrad | Tyr (current) |
|---|---|---|---|
| `transpose` | `transform` | structural mapping | lowered + explicit structured semantic tag |
| `reshape`, `squeeze`, `broadcast`, `slice`, `concat`, `convert_element_type` | `transform` / custom local rules | structural mapping | lowered subset: `broadcast`, `binaryBroadcast`, `sliceRows`, `sliceCols`, `concatCols`, `convert`, `swapLayout`; explicit structured semantic tags |
| dynamic slice/update | partial | present | `missing` in current KStmt lowering |

### 1.5 Special / no-grad / control primitives

| Family | Graphax | AlphaGrad | Tyr (current) |
|---|---|---|---|
| `stop_gradient`, `iota`, `device_put`, `pjit` handling | explicit no-edge behavior or special-cased | partial (`iota`, `stop_gradient`) | explicit no-edge rules for `stop_gradient`/`iota` in LeanJaxpr rule pack; `device_put`/`pjit` still missing |
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
  - unary/binary placeholder pack,
  - extended structural placeholder pack,
  - all-supported placeholder pack,
  - unary/binary hybrid semantic pack,
  - unary/binary semantic pack aligned with Graphax/AlphaGrad overlap.
- structural/reduction semantic pack using structured (typed) sparse-map tags.
- Coverage/extraction tests now cover arithmetic, structural, Graphax alias, and dot-general lowering paths.

Still missing for closer parity:
- Full numeric structural Jacobian payloads (current structural/reduction rules are explicit semantic tags, not dense/value materialization).
- `device_put` / `pjit`-style control primitives in LeanJaxpr lowering/rule packs.

## 5) Immediate Next Priorities

1. Add declared support tests that gate matrix claims (op-by-op parity checks).
2. Add `device_put` / `pjit` no-edge/control primitive lowering and explicit rule coverage.
3. Expand dot-general lowering coverage in source paths (beyond manual LeanJaxpr construction).
4. Upgrade structural semantic tags into exact sparse payload constructors where feasible.
