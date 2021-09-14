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

constant TSpec : PointedType
def T (s : Shape) : Type :=  TSpec.type
instance (s : Shape) : Inhabited (T s) := {
  default := TSpec.val
}

@[extern "lean_torch_to_string"] constant T.toString {s : Shape} (t : @& T s) : String
@[extern "lean_torch_tensor_print"] constant T.print {s : Shape} (t : @& T s) : IO Unit

instance {s : Shape} : ToString (T s) where
  toString t := t.toString

constant T.shape {s : Shape} (t : T s) : Shape := s

instance {S : Shape} : Repr (T s) where
  reprPrec t _ := t.toString