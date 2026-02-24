/-
  Tyr/Manifolds.lean

  Re-exports all manifold implementations.

  Available manifolds:
  - DifferentiableManifold: Base typeclass for manifolds with tangent/cotangent bundles
  - EuclideanSpace: Flat manifolds (Float, Tensor, etc.)
  - Stiefel n p: Matrices with orthonormal columns (St(n,p) = {X : XᵀX = I})
  - Orthogonal n: Orthogonal group (O(n) = {Q : QᵀQ = QQᵀ = I})
  - Grassmann n p: p-dimensional subspaces of ℝⁿ
  - Hyperbolic n: Hyperbolic space H^n in the hyperboloid model
-/

import Tyr.Manifolds.Basic
import Tyr.Manifolds.Stiefel
import Tyr.Manifolds.Orthogonal
import Tyr.Manifolds.Grassmann
import Tyr.Manifolds.Hyperbolic

/-!
# `Tyr.Manifolds`

Manifold abstraction entrypoint re-exporting supported geometric manifolds and their utilities.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

