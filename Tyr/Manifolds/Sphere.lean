import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

/-- Real sphere S^{n-1} represented as unit-norm vectors in `R^n`. -/
structure Sphere (n : UInt64) where
  coords : T #[n]

namespace Sphere

private def dot {n : UInt64} (a b : T #[n]) : Float :=
  torch.nn.item (torch.nn.sumAll (torch.mul a b))

/-- Project ambient vector to the unit sphere. -/
def project (n : UInt64) (v : T #[n]) : Sphere n :=
  let norm := torch.linalg.l2Norm v
  let eps := 1e-12
  if norm <= eps then
    ⟨torch.ones #[n] * (1.0 / Float.sqrt n.toFloat)⟩
  else
    ⟨torch.mul_scalar v (1.0 / norm)⟩

/-- Random point on the sphere. -/
def random (n : UInt64) : IO (Sphere n) := do
  let v ← torch.randn #[n]
  pure (project n v)

end Sphere

/-- Tangent vector at a sphere point. -/
structure SphereTangent (n : UInt64) where
  vec : T #[n]

namespace SphereTangent

/-- Orthogonal projection to the tangent space at `x`. -/
def project (x : Sphere n) (v : T #[n]) : SphereTangent n :=
  let xDotV := Sphere.dot x.coords v
  let correction := torch.mul_scalar x.coords xDotV
  ⟨torch.sub v correction⟩

/-- Zero tangent vector. -/
def zero (n : UInt64) : SphereTangent n := ⟨torch.zeros #[n]⟩

/-- Add tangent vectors. -/
def add (a b : SphereTangent n) : SphereTangent n :=
  ⟨torch.add a.vec b.vec⟩

/-- Scale tangent vector. -/
def smul (s : Float) (v : SphereTangent n) : SphereTangent n :=
  ⟨torch.mul_scalar v.vec s⟩

end SphereTangent

instance sphereManifold (n : UInt64) : DifferentiableManifold (Sphere n) where
  Tangent _ := SphereTangent n
  Cotangent _ := SphereTangent n
  zeroTangent _ := SphereTangent.zero n
  zeroCotangent _ := SphereTangent.zero n
  addTangent a b := SphereTangent.add a b
  addCotangent a b := SphereTangent.add a b
  scaleTangent s v := SphereTangent.smul s v
  sharp := id
  flat := id
  exp x v :=
    let nv := torch.linalg.l2Norm v.vec
    if nv <= 1e-12 then x else
      let dir := torch.mul_scalar v.vec (1.0 / nv)
      let c := Float.cos nv
      let s := Float.sin nv
      let y := torch.add (torch.mul_scalar x.coords c) (torch.mul_scalar dir s)
      Sphere.project n y
  retract x v :=
    Sphere.project n (torch.add x.coords v.vec)

end Tyr.AD
