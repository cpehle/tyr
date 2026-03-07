import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

/-- Symmetric positive definite manifold SPD(n). -/
structure SymmetricPositiveDefinite (n : UInt64) where
  matrix : T #[n, n]

namespace SymmetricPositiveDefinite

private def symm {n : UInt64} (A : T #[n, n]) : T #[n, n] :=
  torch.mul_scalar (torch.add A (torch.nn.transpose2d A)) 0.5

/-- Project ambient matrix to SPD via `exp(sym(A))`. -/
def project (n : UInt64) (A : T #[n, n]) : SymmetricPositiveDefinite n :=
  ⟨torch.linalg.matrix_exp (symm A)⟩

/-- Random SPD point using exponential parameterization. -/
def random (n : UInt64) : IO (SymmetricPositiveDefinite n) := do
  let A ← torch.randn #[n, n]
  pure (project n A)

/-- Affine-invariant inspired distance proxy using `log(inv(A) B)`. -/
def distance (A B : SymmetricPositiveDefinite n) : Float :=
  let rel := torch.nn.mm (torch.linalg.inv A.matrix) B.matrix
  let logRel := torch.linalg.matrix_log rel
  torch.linalg.frobeniusNorm logRel

end SymmetricPositiveDefinite

structure SPDTangent (n : UInt64) where
  vec : T #[n, n]

namespace SPDTangent

def zero (n : UInt64) : SPDTangent n := ⟨torch.zeros #[n, n]⟩
def add (a b : SPDTangent n) : SPDTangent n := ⟨torch.add a.vec b.vec⟩
def smul (s : Float) (v : SPDTangent n) : SPDTangent n := ⟨torch.mul_scalar v.vec s⟩

def project (x : SymmetricPositiveDefinite n) (V : T #[n, n]) : SPDTangent n :=
  let _ := x
  ⟨torch.mul_scalar (torch.add V (torch.nn.transpose2d V)) 0.5⟩

end SPDTangent

instance spdManifold (n : UInt64) : DifferentiableManifold (SymmetricPositiveDefinite n) where
  Tangent _ := SPDTangent n
  Cotangent _ := SPDTangent n
  zeroTangent _ := SPDTangent.zero n
  zeroCotangent _ := SPDTangent.zero n
  addTangent a b := SPDTangent.add a b
  addCotangent a b := SPDTangent.add a b
  scaleTangent s v := SPDTangent.smul s v
  sharp := id
  flat := id
  exp x v :=
    let lx := torch.linalg.matrix_log x.matrix
    SymmetricPositiveDefinite.project n (torch.add lx v.vec)
  retract x v :=
    let lx := torch.linalg.matrix_log x.matrix
    SymmetricPositiveDefinite.project n (torch.add lx v.vec)

end Tyr.AD
