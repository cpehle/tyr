/-
  Tyr/Manifolds/Optimizer.lean

  Generic manifold-optimizer primitives with explicit dual-map support.

  This file extends the manifold stack with a light-weight interface for
  non-Riemannian/Finsler-style optimization where the steepest direction need
  not be the metric sharp-map direction.
-/
import Tyr.Manifolds.Basic
import Tyr.Manifolds.Stiefel

namespace Tyr.AD

open DifferentiableManifold

/--
A geometry interface for constrained steepest descent with an explicit dual map.

`dualMap` can differ from the Riemannian `sharp`; this is the hook used by
non-Riemannian/norm-constrained optimizers.
-/
class DualMapGeometry (M : Type) [DifferentiableManifold M] where
  /-- Tangent-space norm used to measure step size. -/
  tangentNorm : {x : M} → Tangent x → Float
  /-- Cotangent-space dual norm used to measure gradient size. -/
  cotangentNorm : {x : M} → Cotangent x → Float
  /-- Steepest-descent dual map from cotangent to tangent space. -/
  dualMap : {x : M} → Cotangent x → Tangent x

namespace DualMapGeometry

/-- Steepest direction induced by the geometry's dual map. -/
def steepestDirection [DifferentiableManifold M] [DualMapGeometry M]
    {x : M} (g : Cotangent x) : Tangent x :=
  DualMapGeometry.dualMap g

/-- Apply a dual-map step then retract back to the manifold. -/
def dualMapStep [DifferentiableManifold M] [DualMapGeometry M]
    (x : M) (g : Cotangent x) (lr : Float) : M :=
  let dir := steepestDirection (x := x) g
  let step := scaleTangent (-lr) dir
  retract x step

/-- Optional diagnostics for geometry-aware updates. -/
structure StepDiagnostics where
  tangentNorm : Float
  cotangentNorm : Float
  deriving Repr, Inhabited

/-- Apply a dual-map step and return basic norm diagnostics. -/
def dualMapStepWithDiagnostics [DifferentiableManifold M] [DualMapGeometry M]
    (x : M) (g : Cotangent x) (lr : Float) : M × StepDiagnostics :=
  let dir := steepestDirection (x := x) g
  let step := scaleTangent (-lr) dir
  let x' := retract x step
  let stats : StepDiagnostics := {
    tangentNorm := DualMapGeometry.tangentNorm (x := x) dir
    cotangentNorm := DualMapGeometry.cotangentNorm (x := x) g
  }
  (x', stats)

end DualMapGeometry

/--
Float dual-map geometry using sign descent.

This intentionally differs from Euclidean gradient descent and acts as a tiny
non-Riemannian example: the step direction saturates to {-1, 0, 1}.
-/
instance floatSignDualMap : DualMapGeometry Float where
  tangentNorm := by
    intro x v
    let _ := x
    simpa using (Float.abs v)
  cotangentNorm := by
    intro x g
    let _ := x
    simpa using (Float.abs g)
  dualMap := by
    intro x g
    let _ := x
    let g' : Float := g
    simpa using (if g' > 0.0 then (1.0 : Float) else if g' < 0.0 then (-1.0 : Float) else (0.0 : Float))

/--
Stiefel geometry adapter with spectral/nuclear norms and projected dual map.

The dual map is implemented as tangent projection followed by unit Frobenius
normalization as a practical approximation.
-/
instance stiefelDualMapGeometry (m n : UInt64) : DualMapGeometry (Stiefel m n) where
  tangentNorm := fun {x} (v : StiefelTangent m n) =>
    let _ := x
    torch.linalg.spectralNorm v.vec
  cotangentNorm := fun {x} (g : StiefelTangent m n) =>
    let _ := x
    torch.linalg.nuclearNorm g.vec
  dualMap := fun {x} (g : StiefelTangent m n) =>
    let projected := StiefelTangent.project x g.vec
    let fnorm := torch.linalg.frobeniusNorm projected.vec
    if fnorm == 0.0 then
      projected
    else
      StiefelTangent.smul (1.0 / fnorm) projected

end Tyr.AD
