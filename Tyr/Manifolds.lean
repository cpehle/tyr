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

- [`Basic`](./Manifolds/Basic.html): foundational manifold/typeclass definitions.
- [`Embedded`](./Manifolds/Embedded.html): embedded-manifold interface for ambient/tangent projections.
- [`Optimizer`](./Manifolds/Optimizer.html): geometry-aware optimizer interfaces, including dual-map/Finsler hooks.
- [`Sphere`](./Manifolds/Sphere.html): unit-norm vectors in Euclidean space.
- [`Stiefel`](./Manifolds/Stiefel.html): orthonormal-column matrix manifolds.
- [`Orthogonal`](./Manifolds/Orthogonal.html): orthogonal-group manifold utilities.
- [`Grassmann`](./Manifolds/Grassmann.html): subspace-valued manifold representation.
- [`Oblique`](./Manifolds/Oblique.html): matrices with unit-norm columns.
- [`Positive`](./Manifolds/Positive.html): elementwise-positive matrices.
- [`PositiveDefinite`](./Manifolds/PositiveDefinite.html): symmetric positive definite matrix geometry.
- [`PSD`](./Manifolds/PSD.html): fixed-rank PSD factor and elliptope manifolds.
- [`FixedRank`](./Manifolds/FixedRank.html): embedded fixed-rank matrix manifolds.
- [`PoincareBall`](./Manifolds/PoincareBall.html): hyperbolic geometry in the ball model.
- [`Hyperbolic`](./Manifolds/Hyperbolic.html): hyperboloid-model hyperbolic geometry.

## Scope

Use this module when you want full manifold support through one import.
Geometry-specialized code can import individual manifold modules directly.
-/
