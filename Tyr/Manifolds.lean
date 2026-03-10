import Tyr.Manifolds.Basic
import Tyr.Manifolds.Embedded
import Tyr.Manifolds.Optimizer
import Tyr.Manifolds.Sphere
import Tyr.Manifolds.Stiefel
import Tyr.Manifolds.Orthogonal
import Tyr.Manifolds.Grassmann
import Tyr.Manifolds.Oblique
import Tyr.Manifolds.Positive
import Tyr.Manifolds.PoincareBall
import Tyr.Manifolds.PositiveDefinite
import Tyr.Manifolds.PSD
import Tyr.Manifolds.FixedRank
import Tyr.Manifolds.Hyperbolic

/-!
# Tyr.Manifolds

`Tyr.Manifolds` is the umbrella import for manifold geometry support in Tyr.
It re-exports the core manifold interfaces plus every concrete manifold family
currently shipped under `Tyr/Manifolds`, so geometry-aware optimization and
constrained parameterization code can depend on a single import.

## Major Components

- [`Basic`](##Tyr.Manifolds.Basic): foundational manifold/typeclass definitions.
- [`Embedded`](##Tyr.Manifolds.Embedded): embedded-manifold interface for ambient/tangent projections.
- [`Optimizer`](##Tyr.Manifolds.Optimizer): geometry-aware optimizer interfaces, including dual-map/Finsler hooks.
- [`Sphere`](##Tyr.Manifolds.Sphere): unit-norm vectors in Euclidean space.
- [`Stiefel`](##Tyr.Manifolds.Stiefel): orthonormal-column matrix manifold geometry.
- [`Orthogonal`](##Tyr.Manifolds.Orthogonal): orthogonal-group manifold utilities.
- [`Grassmann`](##Tyr.Manifolds.Grassmann): subspace-valued manifold representation.
- [`Oblique`](##Tyr.Manifolds.Oblique): matrices with unit-norm columns.
- [`Positive`](##Tyr.Manifolds.Positive): elementwise-positive matrices.
- [`PositiveDefinite`](##Tyr.Manifolds.PositiveDefinite): symmetric positive definite matrix geometry.
- [`PSD`](##Tyr.Manifolds.PSD): fixed-rank PSD factor and elliptope manifolds.
- [`FixedRank`](##Tyr.Manifolds.FixedRank): embedded fixed-rank matrix manifold geometry.
- [`PoincareBall`](##Tyr.Manifolds.PoincareBall): hyperbolic geometry in the ball model.
- [`Hyperbolic`](##Tyr.Manifolds.Hyperbolic): hyperboloid-model hyperbolic geometry.

## Scope

Use this module when you want full manifold support through one import.
Geometry-specialized code can import individual manifold modules directly.
-/
