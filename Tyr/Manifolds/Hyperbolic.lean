/-
  Tyr/Manifolds/Hyperbolic.lean

  Hyperbolic manifold H^n in the hyperboloid model.
-/
import Tyr.Manifolds.Basic
import Tyr.Torch

namespace Tyr.AD

open torch
open DifferentiableManifold

/-! ## Hyperbolic Manifold H^n (hyperboloid model)

We model hyperbolic space as a two-sheeted hyperboloid in Minkowski space:

  H^n = { x in R^{n+1} : <x,x>_L = -1, x_{n+1} > 0 }

where <x,y>_L = sum_{i=1..n} x_i y_i - x_{n+1} y_{n+1}.
-/

-- Local helpers for tensor operations
@[extern "lean_torch_cat2"]
private opaque cat2 {s1 s2 : Shape} (t1 : @& T s1) (t2 : @& T s2) (dim : Int) : T #[]

private def cat1d (t1 : T #[n]) (t2 : T #[m]) : T #[n + m] :=
  torch.reshape (cat2 t1 t2 0) #[n + m]

private def expandScalar (t : T #[]) (s : Shape) : T s :=
  torch.full s (torch.nn.item t)

/-- Hyperbolic space H^n represented as an (n+1)-vector on the hyperboloid. -/
structure Hyperbolic (n : UInt64) where
  /-- Ambient coordinates in R^{n+1} -/
  coords : T #[n + 1]

namespace Hyperbolic

/-- Minkowski inner product (signature (n,1)) returning a scalar tensor. -/
def minkowskiInnerT {n : UInt64} (x y : T #[n + 1]) : T #[] :=
  let spatialX := torch.data.slice x (dim := 0) (start := 0) (len := n)
  let spatialY := torch.data.slice y (dim := 0) (start := 0) (len := n)
  let timeX := torch.data.slice x (dim := 0) (start := n) (len := 1)
  let timeY := torch.data.slice y (dim := 0) (start := n) (len := 1)
  let spatial := torch.nn.sumAll (torch.mul spatialX spatialY)
  let time := torch.nn.sumAll (torch.mul timeX timeY)
  torch.sub spatial time

/-- Minkowski inner product as Float. -/
def minkowskiInner {n : UInt64} (x y : T #[n + 1]) : Float :=
  torch.nn.item (minkowskiInnerT x y)

/-- Project an ambient vector to the hyperboloid (future sheet).
    Keeps the spatial part and recomputes the time coordinate. -/
def project (n : UInt64) (v : T #[n + 1]) : Hyperbolic n :=
  let spatial := torch.data.slice v (dim := 0) (start := 0) (len := n)
  let spatialSq := torch.nn.sumAll (torch.mul spatial spatial)
  let one := torch.full #[] 1.0
  let t := torch.nn.sqrt (torch.add spatialSq one)
  let tVec := torch.reshape t #[1]
  let coords := cat1d spatial tVec
  ⟨coords⟩

/-- Normalize a time-like vector to the hyperboloid using Minkowski norm. -/
def normalize (n : UInt64) (v : T #[n + 1]) : Hyperbolic n :=
  let inner := minkowskiInnerT v v
  let negInner := torch.mul_scalar inner (-1.0)
  let invNorm := torch.nn.rsqrt negInner
  let scale := expandScalar invNorm #[n + 1]
  let coords := torch.mul v scale
  ⟨coords⟩

/-- Random point on H^n via random spatial coordinates and derived time coord. -/
def random (n : UInt64) : IO (Hyperbolic n) := do
  let spatial ← torch.randn #[n]
  let spatialSq := torch.nn.sumAll (torch.mul spatial spatial)
  let one := torch.full #[] 1.0
  let t := torch.nn.sqrt (torch.add spatialSq one)
  let tVec := torch.reshape t #[1]
  let coords := cat1d spatial tVec
  return ⟨coords⟩

/-- Origin point (0, ..., 0, 1). -/
def origin (n : UInt64) : Hyperbolic n :=
  let spatial := torch.zeros #[n]
  let t := torch.full #[1] 1.0
  let coords := cat1d spatial t
  ⟨coords⟩

end Hyperbolic

/-- Tangent vector on H^n, represented in ambient coordinates. -/
structure HyperbolicTangent (n : UInt64) where
  /-- Tangent vector in R^{n+1} -/
  vec : T #[n + 1]

namespace HyperbolicTangent

/-- Zero tangent vector. -/
def zero (n : UInt64) : HyperbolicTangent n :=
  ⟨torch.zeros #[n + 1]⟩

/-- Add two tangent vectors. -/
def add (v w : HyperbolicTangent n) : HyperbolicTangent n :=
  ⟨torch.add v.vec w.vec⟩

/-- Scale a tangent vector. -/
def smul (s : Float) (v : HyperbolicTangent n) : HyperbolicTangent n :=
  ⟨torch.mul_scalar v.vec s⟩

/-- Project an ambient vector to the tangent space at X (Minkowski-orthogonal). -/
def project (X : Hyperbolic n) (V : T #[n + 1]) : HyperbolicTangent n :=
  let innerXV := Hyperbolic.minkowskiInnerT X.coords V
  let innerXX := Hyperbolic.minkowskiInnerT X.coords X.coords
  let negInnerXV := torch.mul_scalar innerXV (-1.0)
  let alpha := torch.nn.div negInnerXV innerXX
  let scale := expandScalar alpha #[n + 1]
  let correction := torch.mul X.coords scale
  ⟨torch.add V correction⟩

/-- Apply the Minkowski metric (signature (n,1)) to a vector: flips last coord. -/
def applyMetric {n : UInt64} (v : T #[n + 1]) : T #[n + 1] :=
  let spatial := torch.data.slice v (dim := 0) (start := 0) (len := n)
  let time := torch.data.slice v (dim := 0) (start := n) (len := 1)
  let negTime := torch.mul_scalar time (-1.0)
  cat1d spatial negTime

end HyperbolicTangent

/-- Hyperbolic manifold as a DifferentiableManifold.
    Uses a simple projection retraction onto the hyperboloid. -/
instance hyperbolicManifold (n : UInt64) : DifferentiableManifold (Hyperbolic n) where
  Tangent _ := HyperbolicTangent n
  Cotangent _ := HyperbolicTangent n
  zeroTangent _ := HyperbolicTangent.zero n
  zeroCotangent _ := HyperbolicTangent.zero n
  addTangent v w := HyperbolicTangent.add v w
  addCotangent v w := HyperbolicTangent.add v w
  scaleTangent s v := HyperbolicTangent.smul s v
  sharp v := ⟨HyperbolicTangent.applyMetric v.vec⟩
  flat v := ⟨HyperbolicTangent.applyMetric v.vec⟩
  exp X V :=
    let ambient := torch.add X.coords V.vec
    Hyperbolic.normalize n ambient
  retract X V :=
    let ambient := torch.add X.coords V.vec
    Hyperbolic.normalize n ambient

end Tyr.AD
