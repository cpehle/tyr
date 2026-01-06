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
| MPS  -- Metal Performance Shaders for Apple Silicon
  deriving Repr, Inhabited, BEq

opaque TSpec : NonemptyType
def T (_ : Shape) : Type :=  TSpec.type

/-! ## Shape Manipulation Helpers -/

/-- Compute output shape for unsqueeze: insert a dimension of size 1 at position `dim` -/
def unsqueezeShape (s : Shape) (dim : Nat) : Shape :=
  if dim > s.size then s
  else s[:dim].toArray ++ #[1] ++ s[dim:].toArray

/-- Compute output shape for squeeze: remove dimension at `dim` if it has size 1 -/
def squeezeShape (s : Shape) (dim : Nat) : Shape :=
  if h : dim < s.size then
    if s[dim]'h = 1 then s[:dim].toArray ++ s[dim+1:].toArray else s
  else s

/-- Compute output shape for transpose: swap dimensions dim0 and dim1 -/
def transposeShape (s : Shape) (dim0 dim1 : Nat) : Shape :=
  if h0 : dim0 < s.size then
    if h1 : dim1 < s.size then
      let v0 := s[dim0]'h0
      let v1 := s[dim1]'h1
      let s' := s.set (Fin.mk dim0 h0) v1
      have h1' : dim1 < s'.size := by rw [Array.size_set]; exact h1
      s'.set (Fin.mk dim1 h1') v0
    else s
  else s

/-- Compute output shape for reduction along a dimension -/
def reduceShape (s : Shape) (dim : Nat) (keepdim : Bool) : Shape :=
  if h : dim < s.size then
    if keepdim then s.set (Fin.mk dim h) 1
    else s[:dim].toArray ++ s[dim+1:].toArray
  else s

/-- Replace dimension at `dim` with a new size -/
def replaceAtDim (s : Shape) (dim : Nat) (newSize : UInt64) : Shape :=
  if h : dim < s.size then s.set (Fin.mk dim h) newSize else s

/-- Broadcast two batch shapes (everything except last 2 dims) following PyTorch rules -/
private def broadcastBatchShapes (s1 s2 : Shape) : Shape :=
  let n1 := s1.size
  let n2 := s2.size
  let maxLen := max n1 n2
  Array.ofFn fun (i : Fin maxLen) =>
    let idx1 := if i.val < maxLen - n1 then none else some (i.val - (maxLen - n1))
    let idx2 := if i.val < maxLen - n2 then none else some (i.val - (maxLen - n2))
    match idx1, idx2 with
    | none, none => 1
    | some j, none => s1.getD j 1
    | none, some j => s2.getD j 1
    | some j1, some j2 =>
      let d1 := s1.getD j1 1
      let d2 := s2.getD j2 1
      if d1 = 1 then d2 else if d2 = 1 then d1 else max d1 d2

/-- Compute output shape for matrix multiplication following PyTorch broadcasting rules.
    - 1D @ 1D: dot product -> scalar []
    - 2D @ 2D: [m,k] @ [k,n] -> [m,n]
    - 1D @ 2D: [k] @ [k,n] -> [n]
    - 2D @ 1D: [m,k] @ [k] -> [m]
    - ND @ ND: broadcast batch dims, matmul last 2 dims -/
def matmulShape (s1 s2 : Shape) : Shape :=
  match s1.size, s2.size with
  | 0, _ => #[]  -- scalar, invalid but return empty
  | _, 0 => #[]  -- scalar, invalid but return empty
  | 1, 1 => #[]  -- [k] @ [k] -> scalar (dot product)
  | 1, 2 => #[s2.getD 1 0]  -- [k] @ [k,n] -> [n]
  | 2, 1 => #[s1.getD 0 0]  -- [m,k] @ [k] -> [m]
  | 2, 2 => #[s1.getD 0 0, s2.getD 1 0]  -- [m,k] @ [k,n] -> [m,n]
  | 1, n2 =>
    -- [k] @ [..., k, n] -> [..., n]
    let batch := s2[:n2-2].toArray
    batch ++ #[s2.getD (n2 - 1) 0]
  | n1, 1 =>
    -- [..., m, k] @ [k] -> [..., m]
    let batch := s1[:n1-2].toArray
    batch ++ #[s1.getD (n1 - 2) 0]
  | n1, n2 =>
    -- General case: broadcast batch dims, matmul last 2
    let batch1 := s1[:n1-2].toArray
    let batch2 := s2[:n2-2].toArray
    let batchOut := broadcastBatchShapes batch1 batch2
    let m := s1.getD (n1 - 2) 0
    let n := s2.getD (n2 - 1) 0
    batchOut ++ #[m, n]
instance (s : Shape) : Nonempty (T s) :=
  TSpec.property

@[extern "lean_torch_to_string"] opaque T.toString {s : Shape} (t : @& T s) : String
@[extern "lean_torch_tensor_print"] opaque T.print {s : Shape} (t : @& T s) : IO Unit

/-! ## Tensor Metadata Extraction (for visualization widgets) -/

/-- Get the runtime shape of a tensor (useful for widgets/visualization) -/
@[extern "lean_torch_get_shape"]
opaque T.runtimeShape {s : Shape} (t : @& T s) : Array UInt64

/-- Get the dtype of a tensor as a string -/
@[extern "lean_torch_get_dtype"]
opaque T.dtype {s : Shape} (t : @& T s) : String

/-- Get the device of a tensor as a Device enum -/
@[extern "lean_torch_get_device_enum"]
opaque T.device {s : Shape} (t : @& T s) : Device

/-- Get the device of a tensor as a string (e.g., "cpu", "cuda:0", "mps") -/
@[extern "lean_torch_get_device"]
opaque T.deviceStr {s : Shape} (t : @& T s) : String

/-- Get tensor values as a flat array of floats (up to maxElements).
    Values are converted to Float for uniform access. -/
@[extern "lean_torch_get_values"]
opaque T.getValues {s : Shape} (t : @& T s) (maxElements : UInt64 := 1000) : FloatArray

/-- Get tensor statistics as a JSON string: {min, max, mean, std} -/
@[extern "lean_torch_get_stats"]
opaque T.stats {s : Shape} (t : @& T s) : String

instance {s : Shape} : ToString (T s) where
  toString t := t.toString

def T.shape {s : Shape} (_t : T s) : Shape := s

instance {s : Shape} : Repr (T s) where
  reprPrec t _ := t.toString