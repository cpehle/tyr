import Tyr.Manifolds.Basic
import Tyr.Manifolds.Optimizer
import Tyr.Manifolds.Stiefel
import Tyr.Manifolds.Orthogonal
import Tyr.Manifolds.Grassmann
import Tyr.Manifolds.Hyperbolic

/-!
# Tyr.Manifolds

`Tyr.Manifolds` is the umbrella import for manifold geometry support in Tyr.
It re-exports core manifold interfaces and concrete manifold families used by
geometry-aware optimization and constrained parameterization workflows.

## Major Components

- `Basic`: foundational manifold/typeclass definitions.
- `Optimizer`: geometry-aware optimizer interfaces (including dual-map/Finsler hooks).
- `Stiefel`: orthonormal-column matrix manifolds.
- `Orthogonal`: orthogonal-group manifold utilities.
- `Grassmann`: subspace-valued manifold representation.
- `Hyperbolic`: hyperboloid-model hyperbolic geometry.

## Scope

Use this module when you want full manifold support through one import.
Geometry-specialized code can import individual manifold modules directly.
-/
