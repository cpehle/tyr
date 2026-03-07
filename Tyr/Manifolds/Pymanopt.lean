import Tyr.Manifolds.Product
import Tyr.Manifolds.Sphere
import Tyr.Manifolds.Oblique
import Tyr.Manifolds.Positive
import Tyr.Manifolds.PoincareBall
import Tyr.Manifolds.PositiveDefinite
import Tyr.Manifolds.PSD
import Tyr.Manifolds.FixedRank

/-!
# Tyr.Manifolds.Pymanopt

Real-valued manifold ports inspired by the manifold catalog in `../pymanopt`.

Currently included:
- Product manifold (pair instance)
- Sphere
- Oblique
- Positive (elementwise)
- Poincare ball
- Symmetric positive definite
- PSD fixed-rank and elliptope factor geometries
- Fixed-rank embedded matrices

Complex-valued pymanopt manifolds (unitary, complex circle, complex Grassmann,
complex PSD/HPD variants) are not ported here because the current typed tensor
stack is real-valued.
-/
