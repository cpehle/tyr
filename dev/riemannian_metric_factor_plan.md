# Riemannian Metric-Factor Optimizer Proposal

This note proposes a Tyr-native implementation of recursive metric-factor propagation for layerwise Riemannian optimization.

The target mathematical recursion is:

```text
L_L = L_o
L_{α-1} = L_α A_α
K_α = L_α B_α
G^{(α)} = K_α^T K_α + D_α
Δw_α = -η (G^{(α)})^{-1} g_α
```

with `A_α = d_z f_α`, `B_α = d_w f_α`, and `D_α` a positive definite layer metric, ideally diagonal or block-diagonal.

## Why this belongs in Tyr

Tyr already has the three pieces this design needs:

- `Tyr.Manifolds.Basic` and `Tyr.Manifolds.Optimizer` provide the manifold boundary:
  tangent/cotangent types, retractions, and optional non-Riemannian dual maps.
- `Tyr.Modular.*` already models layerwise composition and per-module sensitivity budgets.
- `Tyr.AD.JaxprLike.*` and `Tyr.AD.Elim.*` already carry local Jacobian structure.

The missing piece is a first-class representation of a propagated metric factor.

The implementation should treat the propagated object as a metric factor, not a Cholesky factor. At the output layer it may be represented by a Cholesky factor, but after pullback it will in general be rectangular and no longer triangular.

## Main constraint from the current optimizer stack

The current matrix optimizer interfaces in `Tyr/Optim/DualOptimizer.lean` and `Tyr/Optim/ManifoldMuon.lean` are param-local:

- they receive `(param, grad, state)`,
- they do not receive cached activations,
- they do not receive local Jacobians,
- they do not receive a backward-propagated output metric factor.

That means the recursive factor pass cannot be implemented as a new `GradientTransformation` or as a drop-in branch inside the current `DualOptimizer` fast path.

Phase 1 should therefore introduce a forward-aware optimizer path that sits between:

1. module forward evaluation with cached local linearization data, and
2. the final parameter update/retraction step.

This is the same architectural reason K-FAC-style methods usually need more than raw parameter gradients.

## Proposed file layout

### 1. `Tyr/Modular/MetricFactor.lean`

New core representation for Euclidean-valued coframe factors.

Suggested starting point:

```lean
namespace Tyr.Modular

structure MetricFactor (rank dim : UInt64) where
  matrix : torch.T #[rank, dim]

structure DiagonalMass (dim : UInt64) where
  diag : torch.T #[dim]

def pullback {r inDim outDim : UInt64}
    (L : MetricFactor r outDim)
    (A : torch.T #[outDim, inDim]) : MetricFactor r inDim :=
  { matrix := torch.nn.mm L.matrix A }

def gram {r dim : UInt64} (K : MetricFactor r dim) : torch.T #[dim, dim] :=
  torch.nn.mm (torch.nn.transpose2d K.matrix) K.matrix

def solveWoodbury {r dim : UInt64}
    (D : DiagonalMass dim)
    (K : MetricFactor r dim)
    (g : torch.T #[dim]) : torch.T #[dim] := ...

end Tyr.Modular
```

Notes:

- Keep this object Euclidean and explicit. Do not try to embed it into `DifferentiableManifold`.
- Phase 1 should support only diagonal `D_α`, because that is exactly where the Woodbury win is.
- For matrix parameters, flatten to a vector at the solver boundary. User-facing modules can keep matrix-shaped weights.

### 2. `Tyr/Manifolds/Embedded.lean`

Minimal extension of the existing manifold stack so a geometry-aware optimizer can turn an ambient update into a tangent update before retracting.

Suggested interface:

```lean
namespace Tyr.AD

class EmbeddedManifold (M : Type) [DifferentiableManifold M] where
  Ambient : Type
  toAmbient : M → Ambient
  projectAmbientTangent : {x : M} → Ambient → Tangent x

end Tyr.AD
```

Why add this:

- `DifferentiableManifold` already gives `retract`, but it does not expose a generic ambient-to-tangent projection.
- The existing concrete manifolds already have the needed projections:
  `StiefelTangent.project`, `GrassmannTangent.project`, `OrthogonalTangent.fromAmbient`, `HyperbolicTangent.project`.
- This keeps the new optimizer on top of the current manifold design instead of forking it.

This also lets `Tyr.Modular.Manifold.MatrixManifoldCarrier` become a thin convenience wrapper over the more general manifold interface later.

### 3. `Tyr/Modular/RiemannianModule.lean`

New layer-local interface for modules that can expose the two local linear maps needed by the recursion.

Suggested shape:

```lean
namespace Tyr.Modular

structure LocalLinearization (inDim paramDim outDim : UInt64) where
  A : torch.T #[outDim, inDim]
  B : torch.T #[outDim, paramDim]

class RiemannianModule
    (M : Type) (Input Cache : Type)
    (inDim paramDim outDim : UInt64) where
  forwardCached : M → Input → (torch.T #[outDim] × Cache)
  linearization : M → Cache → LocalLinearization inDim paramDim outDim
  paramMass : M → DiagonalMass paramDim
  applyAmbientUpdate : M → torch.T #[paramDim] → Float → M

end Tyr.Modular
```

Phase-1 instances should be explicit and small:

- `torch.Linear`
- `Tyr.Modular.Manifold.ManifoldLinear`
- simple pointwise nonlinearities if needed for an MLP proof-of-concept
- optional `LayerNorm` once the basic path is stable

For `ManifoldLinear`, `applyAmbientUpdate` should:

1. reshape the ambient vector update back to matrix form,
2. project it to the tangent space using `EmbeddedManifold.projectAmbientTangent`,
3. retract with `DifferentiableManifold.retract`,
4. update bias in Euclidean space.

This reuses the current manifold support directly. It also avoids re-normalizing the direction through `DualMapGeometry.dualMapStep`, which would interfere with the Woodbury-derived step magnitude.

### 4. `Tyr/Optim/RiemannianSGD.lean`

New forward-aware optimizer that consumes a sequential tape of `RiemannianModule`s.

Suggested responsibilities:

- build the output metric factor `L_o`,
- run the backward recursion on `λ_α` and `L_α`,
- compute `K_α = L_α B_α`,
- apply the Woodbury solve,
- emit per-layer diagnostics.

Suggested diagnostics:

```lean
structure LayerStepDiagnostics where
  layerIndex : Nat
  gradientNorm : Float
  factorRank : UInt64
  innerSolveResidual : Float
  innerConditionEstimate : Float
  updateNorm : Float
```

### 5. `Tyr/Modular/OutputMetric.lean`

Phase-1 output metric choices should stay simple:

- identity metric,
- diagonal loss Hessian approximation,
- Gauss-Newton diagonal for classification/regression heads.

Suggested interface:

```lean
class OutputMetricFactor (Y : Type) (outDim rank : UInt64) where
  factor : Y → MetricFactor rank outDim
```

This should remain separate from `DifferentiableManifold`. The output metric is an optimization choice, not a property every output type must own.

## Training-loop integration

## Phase 1: do not start in NanoChat

The right first integration target is `Examples/GPT/Train.lean`, not `Examples/NanoChat/ModdedTrain.lean`.

Reason:

- `Examples/GPT/Train.lean` is already a small explicit training loop.
- It avoids distributed-update complexity.
- It gives a clean place to validate that factor recursion matches dense Jacobian baselines.
- This aligns with the existing direction in `dev/issues.md` that new optimizer plumbing should land on a small GPT-like path before the full NanoChat stack.

Suggested new example:

- `Examples/GPT/RiemannianTrain.lean`

This example should train a tiny 2-layer MLP or small GPT block on a tiny task and expose:

- identity vs Gauss-Newton output metric,
- Euclidean `Linear` vs `StiefelLinear`,
- dense inverse check vs Woodbury check on tiny dimensions.

## Core step structure

The intended backward pass is:

```text
forward:
  y, tape := model.forwardCached x
  L := outputMetricFactor y
  λ := dℓ/dy

backward for α = L .. 1:
  lin := tape[α]
  Aα := lin.A
  Bα := lin.B
  gα := Bα^T λα
  Kα := L Bα
  Δwα := woodbury(Dα, Kα, gα)
  paramsα := applyAmbientUpdate paramsα Δwα η
  λ := Aα^T λ
  L := L Aα
```

This is the exact place where the manuscript insight pays off: no global `∂y/∂w^{(α)}` tensor is needed.

## How this reuses current manifold support

This proposal reuses the existing manifold stack at the parameter boundary.

For Euclidean parameters:

- the ambient update is the tangent update,
- `retract` is identity.

For `Stiefel`, `Grassmann`, `Orthogonal`, `Hyperbolic`:

- use the existing manifold value types from `Tyr.Manifolds.*`,
- use their existing ambient-to-tangent projection functions,
- use their existing `retract`/`exp` implementations.

This keeps the metric-factor logic independent of the manifold family. The metric-factor pass computes an ambient update; the manifold layer decides how to realize that update on the constraint manifold.

## Phase 2: generic graph path via existing local Jacobian support

Once the explicit sequential path works, generalize using the existing AD graph stack:

- `Tyr.AD.JaxprLike.Rules`
- `Tyr.AD.Elim.Graph`
- `Tyr.AD.Sparse.*`

The phase-2 idea is:

1. lower a forward pass to local Jacobian edges,
2. propagate `MetricFactor` backward over those edges,
3. materialize per-parameter `K` factors only for trainable leaves.

This is a natural fit for Tyr because the local Jacobian infrastructure already exists. It is a better long-term fit than hardcoding every architecture as a handwritten sequential recursion.

## Interaction with the current modular norm stack

The new Riemannian path should not replace `Tyr.Modular.Norm` or `Tyr.Modular.Budget`.

Instead:

- keep `NormedModule` and `sequentialDownstreamScales` as learning-rate budget logic,
- multiply the base Riemannian step size by the same downstream sensitivity budget,
- reuse the existing `NormedModule` instances for `ManifoldLinear`.

So the phase-1 update for a layer becomes:

```text
η_α = η_base * modularBudget_α
Δw_α = -η_α (K_α^T K_α + D_α)^{-1} g_α
```

This gives a clean story: modular norms choose the scale, the metric factor chooses the geometry.

## What should not be done in phase 1

Do not:

- try to retrofit this directly into `torch.Optim.GradientTransformation`,
- try to make every `Module` automatically a `RiemannianModule`,
- start with NanoChat distributed parity,
- require full graph lowering before a first working path,
- conflate metric factors with Cholesky factors in the API.

Each of those choices would make the first implementation larger and less clear.

## Minimal milestone stack

### RM01: core factor algebra

Files:

- `Tyr/Modular/MetricFactor.lean`
- `Tests/TestMetricFactor.lean`

Acceptance:

- `pullback` recursion matches explicit matrix multiplication.
- Woodbury solve matches dense inverse on tiny shapes.

### RM02: embedded manifold bridge

Files:

- `Tyr/Manifolds/Embedded.lean`
- updates in `Tyr/Manifolds.lean`
- `Tests/TestEmbeddedManifold.lean`

Acceptance:

- ambient update projection works for Euclidean, Stiefel, Grassmann, Orthogonal, Hyperbolic.

### RM03: explicit riemannian modules

Files:

- `Tyr/Modular/RiemannianModule.lean`
- instances for `torch.Linear` and `Tyr.Modular.Manifold.ManifoldLinear`
- `Tests/TestRiemannianModule.lean`

Acceptance:

- `K_α` built from recursion matches explicit `L_o J^{(α)}` on 2-layer linear network.

### RM04: forward-aware optimizer

Files:

- `Tyr/Optim/RiemannianSGD.lean`
- `Examples/GPT/RiemannianTrain.lean`
- `Tests/TestRiemannianSGD.lean`

Acceptance:

- tiny model trains,
- metric solve stays finite,
- manifold-constrained weights remain on-manifold after each step.

### RM05: AD graph generalization

Files:

- bridge code in `Tyr/AD/JaxprLike/*`
- graph propagation code in new `Tyr/Modular/RiemannianGraph.lean`

Acceptance:

- factor propagation works from local Jacobian edges without handwritten layer recursion.

## Bottom line

The right Tyr implementation is:

- a new metric-factor layer in `Tyr.Modular`,
- a thin extension of `Tyr.Manifolds` for ambient-to-tangent projection,
- a forward-aware optimizer path separate from the current param-local optimizer stack,
- and a first end-to-end integration on the small GPT example before any NanoChat work.

That uses the current manifold support where it is strongest: tangent projection, retraction, and manifold-native parameter wrappers. It keeps the new geometry pass focused on the actual missing object: the recursively propagated metric factor.
