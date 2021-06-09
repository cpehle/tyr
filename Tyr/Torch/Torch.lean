namespace torch
def Shape := Array UInt64
constant TSpec : PointedType
def T (s : Shape) : Type :=  TSpec.type
instance (s : Shape) : Inhabited (T s) := {
  default := TSpec.val
}

@[extern "lean_torch_to_string"] constant T.toString {s : Shape} (t : @& T s) : String
@[extern "lean_torch_tensor_print"] constant T.print {s : Shape} (t : @& T s) : IO Unit

-- arange: Returns a tensor with a sequence of integers,
-- empty: Returns a tensor with uninitialized values,
-- eye: Returns an identity matrix,
-- full: Returns a tensor filled with a single value,
-- linspace: Returns a tensor with values linearly spaced in some interval,
-- logspace: Returns a tensor with values logarithmically spaced in some interval,
-- ones: Returns a tensor filled with all ones,
@[extern "lean_torch_ones"] constant ones (s : Shape) : T s
-- rand: Returns a tensor filled with values drawn from a uniform distribution on [0, 1).
-- randint: Returns a tensor with integers randomly drawn from an interval,
-- zeros: Returns a tensor filled with all zeros
@[extern "lean_torch_zeros"] constant zeros (s : Shape) : T s
-- randn: Returns a tensor filled with values drawn from a unit normal distribution,
@[extern "lean_torch_randn"] constant randn (s : Shape) : IO (T s)

instance (shape : Shape) : Inhabited (T shape) := ⟨zeros shape⟩

@[extern "lean_torch_tensor_add"] constant add {s : Shape} (t t' : T s) : T s
@[extern "lean_torch_tensor_sub"] constant sub {s : Shape} (t t' : T s) : T s

instance {shape : Shape} : Add (T shape) where
  add x y := add x y

instance {shape : Shape} : Sub (T shape) where
  sub x y := sub x y

end torch

def main : IO Unit := do
  let x <- torch.randn #[5,5]
  let y := torch.zeros #[5,5]
  let z := (x + y)
  let a := (x - x)
  x.print
  y.print
  (torch.ones #[5,5]).print
  pure ()