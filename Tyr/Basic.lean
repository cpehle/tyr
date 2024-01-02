namespace torch

abbrev Shape := Array UInt64

inductive DType where
| UInt8
| Int8
| Int16
| Int32
| Int64
| Float16
| Float32
| Float64

inductive Device where
| CUDA : Nat → Device
| CPU

opaque TSpec : NonemptyType
def T (_ : Shape) : Type :=  TSpec.type
instance (s : Shape) : Nonempty (T s) :=
  TSpec.property

@[extern "lean_torch_to_string"] opaque T.toString {s : Shape} (t : @& T s) : String
@[extern "lean_torch_tensor_print"] opaque T.print {s : Shape} (t : @& T s) : IO Unit

instance {s : Shape} : ToString (T s) where
  toString t := t.toString

def T.shape {s : Shape} (t : T s) : Shape := s

instance {S : Shape} : Repr (T s) where
  reprPrec t _ := t.toString