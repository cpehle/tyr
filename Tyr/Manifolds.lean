/-
  Tyr/Manifolds.lean

  Re-exports all manifold implementations.

  Available manifolds:
  - DifferentiableManifold: Base typeclass for manifolds with tangent/cotangent bundles
  - EuclideanSpace: Flat manifolds (Float, Tensor, etc.)
  - Stiefel n p: Matrices with orthonormal columns (St(n,p) = {X : XᵀX = I})
  - Orthogonal n: Orthogonal group (O(n) = {Q : QᵀQ = QQᵀ = I})
  - Grassmann n p: p-dimensional subspaces of ℝⁿ
-/

import Tyr.Manifolds.Basic
import Tyr.Manifolds.Stiefel
import Tyr.Manifolds.Orthogonal
import Tyr.Manifolds.Grassmann
