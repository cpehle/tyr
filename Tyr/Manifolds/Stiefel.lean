/-
  Tyr/Manifolds/Stiefel.lean

  Stiefel manifold St(n, p): n×p matrices with orthonormal columns.
-/
import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch (T)
open DifferentiableManifold

/-! ## Stiefel Manifold St(n, p)

The Stiefel manifold St(n, p) consists of n×p matrices with orthonormal columns:
  St(n, p) = { X ∈ ℝⁿˣᵖ : XᵀX = Iₚ }

Special cases:
- St(n, n) = O(n), the orthogonal group
- St(n, 1) = Sⁿ⁻¹, the unit sphere

The tangent space at X is the horizontal space:
  TₓSt(n,p) = { Z ∈ ℝⁿˣᵖ : XᵀZ + ZᵀX = 0 }
            = { Z : XᵀZ is skew-symmetric }
-/

/-- Stiefel manifold St(n, p): n×p matrices with orthonormal columns.
    Constraint: XᵀX = Iₚ (p×p identity) -/
structure Stiefel (n p : UInt64) where
  /-- The underlying n×p matrix -/
  matrix : T #[n, p]

namespace Stiefel

/-- Create a Stiefel element from the first p columns of a random matrix via QR.
    Uses QR decomposition of a random n×p matrix. -/
def random (n p : UInt64) : IO (Stiefel n p) := do
  let mat ← torch.randn #[n, p]
  -- QR decomposition gives Q with orthonormal columns
  let (Q, _R) := torch.linalg.qr_reduced mat
  return ⟨Q⟩

/-- Identity element for St(n, n) = O(n) -/
def identity (n : UInt64) : Stiefel n n :=
  ⟨torch.eye n⟩

/-- Project a general matrix to the Stiefel manifold using QR decomposition.
    The Q factor of QR has orthonormal columns. -/
def project (n p : UInt64) (mat : T #[n, p]) : Stiefel n p :=
  let (Q, _R) := torch.linalg.qr_reduced mat
  ⟨Q⟩

end Stiefel

/-- Tangent vector on the Stiefel manifold.
    Z ∈ TₓSt(n,p) satisfies: XᵀZ + ZᵀX = 0 (XᵀZ is skew-symmetric) -/
structure StiefelTangent (n p : UInt64) where
  /-- The tangent vector as an n×p matrix -/
  vec : T #[n, p]

namespace StiefelTangent

/-- Zero tangent vector -/
def zero (n p : UInt64) : StiefelTangent n p :=
  ⟨torch.zeros #[n, p]⟩

/-- Add two tangent vectors -/
def add (v w : StiefelTangent n p) : StiefelTangent n p :=
  ⟨torch.add v.vec w.vec⟩

/-- Scale a tangent vector -/
def smul (s : Float) (v : StiefelTangent n p) : StiefelTangent n p :=
  ⟨torch.mul_scalar v.vec s⟩

/-- Project an ambient space vector to the tangent space at X.
    Π_X(Z) = Z - X sym(XᵀZ)
    where sym(A) = (A + Aᵀ)/2 -/
def project (X : Stiefel n p) (Z : T #[n, p]) : StiefelTangent n p :=
  -- X^T @ Z : [p, p]
  let Xt := torch.nn.transpose2d X.matrix
  let XtZ := torch.nn.mm Xt Z
  -- sym(X^T Z) = (X^T Z + (X^T Z)^T) / 2
  let sym := torch.mul_scalar (torch.add XtZ (torch.nn.transpose2d XtZ)) 0.5
  -- Z - X @ sym(X^T Z)
  let projected := torch.sub Z (torch.nn.mm X.matrix sym)
  ⟨projected⟩

end StiefelTangent

/-- Stiefel manifold as a DifferentiableManifold.
    Uses QR retraction: R(X, Z) = qr(X + Z).Q -/
instance stiefelManifold (n p : UInt64) : DifferentiableManifold (Stiefel n p) where
  Tangent _ := StiefelTangent n p
  Cotangent _ := StiefelTangent n p
  zeroTangent _ := StiefelTangent.zero n p
  zeroCotangent _ := StiefelTangent.zero n p
  addTangent v w := StiefelTangent.add v w
  addCotangent v w := StiefelTangent.add v w
  scaleTangent s v := StiefelTangent.smul s v
  sharp := id
  flat := id
  exp X Z :=
    -- QR retraction: R(X, Z) = qr(X + Z).Q
    let newMat := torch.add X.matrix Z.vec
    let (Q, _R) := torch.linalg.qr_reduced newMat
    ⟨Q⟩

end Tyr.AD
