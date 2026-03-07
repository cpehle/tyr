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

namespace DiffEqArithmetic

/-- `+` operator instance derived from `DiffEqSpace`. Use via `local instance`. -/
def hAddInst [DiffEqSpace α] : HAdd α α α where
  hAdd := DiffEqSpace.add

/-- `-` operator instance derived from `DiffEqSpace`. Use via `local instance`. -/
def hSubInst [DiffEqSpace α] : HSub α α α where
  hSub := DiffEqSpace.sub

/-- Left scalar multiplication `a * x` derived from `DiffEqSpace.scale`. -/
def hMulInst [DiffEqSpace α] : HMul Scalar α α where
  hMul := DiffEqSpace.scale

/-- SMul (`a • x`) instance derived from `DiffEqSpace.scale`. -/
def smulInst [DiffEqSpace α] : SMul Scalar α where
  smul := DiffEqSpace.scale

end DiffEqArithmetic

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

instance [DiffEqSpace α] [DiffEqSpace β] : DiffEqSpace (α × β) where
  add a b := (DiffEqSpace.add a.1 b.1, DiffEqSpace.add a.2 b.2)
  sub a b := (DiffEqSpace.sub a.1 b.1, DiffEqSpace.sub a.2 b.2)
  scale s x := (DiffEqSpace.scale s x.1, DiffEqSpace.scale s x.2)

instance [DiffEqSpace α] : DiffEqSpace (Fin n → α) where
  add f g := fun i => DiffEqSpace.add (f i) (g i)
  sub f g := fun i => DiffEqSpace.sub (f i) (g i)
  scale s f := fun i => DiffEqSpace.scale s (f i)

instance : DiffEqSeminorm Float where
  rms x := Float.abs x

instance [DiffEqSeminorm α] [DiffEqSeminorm β] : DiffEqSeminorm (α × β) where
  rms x := max (DiffEqSeminorm.rms x.1) (DiffEqSeminorm.rms x.2)

instance [DiffEqSeminorm α] : DiffEqSeminorm (Fin n → α) where
  rms x :=
    (List.finRange n).foldl
      (fun acc i => max acc (DiffEqSeminorm.rms (x i)))
      0.0

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

instance [DiffEqElem α] [DiffEqElem β] : DiffEqElem (α × β) where
  abs x := (DiffEqElem.abs x.1, DiffEqElem.abs x.2)
  max x y := (DiffEqElem.max x.1 y.1, DiffEqElem.max x.2 y.2)
  addScalar s x := (DiffEqElem.addScalar s x.1, DiffEqElem.addScalar s x.2)
  div x y := (DiffEqElem.div x.1 y.1, DiffEqElem.div x.2 y.2)

instance [DiffEqElem α] : DiffEqElem (Fin n → α) where
  abs x := fun i => DiffEqElem.abs (x i)
  max x y := fun i => DiffEqElem.max (x i) (y i)
  addScalar s x := fun i => DiffEqElem.addScalar s (x i)
  div x y := fun i => DiffEqElem.div (x i) (y i)

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
