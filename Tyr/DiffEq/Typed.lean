/-
  Tyr/DiffEq/Typed.lean

  DiffEq instances for the spec-graded typed tensor `Tensor (σ : StaticSpec)`,
  so neural ODEs/SDEs can be written against the typed facade: solver states
  keep their shape/dtype/device claims through the solve, and `validate` can
  audit them at the boundaries.

  All instances delegate to the untyped `T` instances; the solver-internal
  arithmetic (add/sub/scale, elementwise max/div, RMS norms) is
  spec-preserving, so the phantom index is carried through unchanged. The
  float-scalar ops used by controllers keep Float32-class dtypes fixed (the
  scalar paths used here never change a floating dtype).
-/
import Tyr.DiffEq.Types
import Tyr.Typed.Tensor

namespace torch
namespace DiffEq

instance {σ : StaticSpec} : Inhabited (Tensor σ) :=
  ⟨Tensor.assumeSpec default⟩

instance {σ : StaticSpec} : DiffEqSpace (Tensor σ) where
  add a b := Tensor.assumeSpec (DiffEqSpace.add a.raw b.raw)
  sub a b := Tensor.assumeSpec (DiffEqSpace.sub a.raw b.raw)
  scale s a := Tensor.assumeSpec (DiffEqSpace.scale s a.raw)

instance {σ : StaticSpec} : DiffEqElem (Tensor σ) where
  abs a := Tensor.assumeSpec (DiffEqElem.abs a.raw)
  max a b := Tensor.assumeSpec (DiffEqElem.max a.raw b.raw)
  addScalar s a := Tensor.assumeSpec (DiffEqElem.addScalar s a.raw)
  div a b := Tensor.assumeSpec (DiffEqElem.div a.raw b.raw)

instance {σ : StaticSpec} : DiffEqSeminorm (Tensor σ) where
  rms a := DiffEqSeminorm.rms a.raw

end DiffEq
end torch
