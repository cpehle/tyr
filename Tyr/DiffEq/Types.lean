import Tyr.TensorStruct

namespace torch
namespace DiffEq

/-! ## Core Types

Lightweight aliases and algebraic structure for differential equation solvers.
-/

abbrev Time := Float
abbrev Scalar := Float

/-! ## Backend-agnostic interfaces

These typeclasses are intentionally minimal to allow non-Torch backends.
Provide instances for your backend's state/control types.
-/

/-- Vector-space-like operations for state/control trees. -/
class DiffEqSpace (α : Type) where
  add : α → α → α
  sub : α → α → α
  scale : Scalar → α → α

/-! Optional seminorm used by adaptive controllers and error estimates. -/
class DiffEqSeminorm (α : Type) where
  rms : α → Scalar

instance [TensorStruct α] : DiffEqSpace α where
  add := TensorStruct.add
  sub := TensorStruct.sub
  scale s x := TensorStruct.scale x s

instance : DiffEqSpace Float where
  add := (· + ·)
  sub := (· - ·)
  scale s x := s * x

instance : DiffEqSeminorm Float where
  rms x := Float.abs x

/-! Element-wise operations used for adaptive step size control. -/
class DiffEqElem (α : Type) where
  abs : α → α
  max : α → α → α
  addScalar : Scalar → α → α
  div : α → α → α

instance : DiffEqElem Float where
  abs := Float.abs
  max := max
  addScalar s x := x + s
  div := (· / ·)

private def tensorMaximum {s : Shape} (a b : T s) : T s :=
  where_ (gt a b) a b

instance {s : Shape} : DiffEqElem (T s) where
  abs := nn.abs
  max := tensorMaximum
  addScalar s x := add_scalar x s
  div := nn.div

private def rmsTensor {s : Shape} (t : T s) : Float :=
  let sq := mul t t
  let mean := nn.meanAll sq
  let root := nn.sqrt mean
  nn.item root

instance {s : Shape} : DiffEqSeminorm (T s) where
  rms := rmsTensor

instance (priority := 50) [TensorStruct α] : DiffEqElem α where
  abs := TensorStruct.map (fun t => nn.abs t)
  max := TensorStruct.zipWith tensorMaximum
  addScalar s := TensorStruct.map (fun t => add_scalar t s)
  div := TensorStruct.zipWith nn.div

instance (priority := 50) [TensorStruct α] : DiffEqSeminorm α where
  rms x := TensorStruct.fold (fun t acc => max acc (DiffEqSeminorm.rms t)) 0.0 x

@[inline] def axpy [DiffEqSpace α] (a : Scalar) (x y : α) : α :=
  DiffEqSpace.add (DiffEqSpace.scale a x) y

end DiffEq
end torch
