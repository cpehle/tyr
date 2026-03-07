import Tyr.Manifolds.Basic

namespace Tyr.AD

open DifferentiableManifold

/--
Cartesian product manifold instance.

Implements the standard product manifold on Lean pairs.
-/
instance prodManifold [DifferentiableManifold M] [DifferentiableManifold N] :
    DifferentiableManifold (M × N) where
  Tangent x := (DifferentiableManifold.Tangent x.1) × (DifferentiableManifold.Tangent x.2)
  Cotangent x := (DifferentiableManifold.Cotangent x.1) × (DifferentiableManifold.Cotangent x.2)
  zeroTangent x := (DifferentiableManifold.zeroTangent x.1, DifferentiableManifold.zeroTangent x.2)
  zeroCotangent x := (DifferentiableManifold.zeroCotangent x.1, DifferentiableManifold.zeroCotangent x.2)
  addTangent := by
    intro x a b
    exact (DifferentiableManifold.addTangent a.1 b.1, DifferentiableManifold.addTangent a.2 b.2)
  addCotangent := by
    intro x a b
    exact (DifferentiableManifold.addCotangent a.1 b.1, DifferentiableManifold.addCotangent a.2 b.2)
  scaleTangent s v :=
    (DifferentiableManifold.scaleTangent s v.1, DifferentiableManifold.scaleTangent s v.2)
  sharp := by
    intro x g
    exact (DifferentiableManifold.sharp g.1, DifferentiableManifold.sharp g.2)
  flat := by
    intro x v
    exact (DifferentiableManifold.flat v.1, DifferentiableManifold.flat v.2)
  exp x v :=
    (DifferentiableManifold.exp x.1 v.1, DifferentiableManifold.exp x.2 v.2)
  retract x v :=
    (DifferentiableManifold.retract x.1 v.1, DifferentiableManifold.retract x.2 v.2)

end Tyr.AD
