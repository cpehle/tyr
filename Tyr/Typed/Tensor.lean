import Tyr.Torch
import Tyr.Typed.Device

/-!
# Typed tensor facade

`DTensor shape dtype` is a lightweight typed view over Tyr's existing `T shape`.
It does not replace the runtime tensor representation.  Checked constructors
validate runtime dtype metadata when crossing from raw tensors into the typed
facade; `assumeDType` is available for code that already established the
invariant externally.
-/

namespace torch

structure DTensor (shape : Shape) (dtype : DType) where
  private mk ::
  raw : T shape

abbrev TT (shape : Shape) (dtype : DType) : Type :=
  DTensor shape dtype

abbrev SomeDTensor :=
  Sigma fun spec : TensorSpec => DTensor spec.shape spec.dtype

namespace DTensor

def assumeDType {shape : Shape} {dtype : DType} (raw : T shape) : DTensor shape dtype :=
  .mk raw

def toTensor {shape : Shape} {dtype : DType} (tensor : DTensor shape dtype) : T shape :=
  tensor.raw

instance {shape : Shape} {dtype : DType} : CoeOut (DTensor shape dtype) (T shape) where
  coe := toTensor

def spec {shape : Shape} {dtype : DType} (_tensor : DTensor shape dtype) : TensorSpec :=
  { shape, dtype }

def dtype {shape : Shape} {dtype : DType} (_tensor : DTensor shape dtype) : DType :=
  dtype

def shape {shape : Shape} {dtype : DType} (_tensor : DTensor shape dtype) : Shape :=
  shape

def actualSpec {shape : Shape} {dtype : DType} (tensor : DTensor shape dtype) : TensorSpec :=
  { shape := tensor.raw.runtimeShape, dtype := tensor.raw.dtype }

def ofTensor? {shape : Shape} (dtype : DType) (raw : T shape) : Option (DTensor shape dtype) :=
  if raw.dtype == dtype then
    some (.mk raw)
  else
    none

def ofTensor {shape : Shape} (dtype : DType) (raw : T shape) : Except String (DTensor shape dtype) := do
  TensorSpec.checkDType dtype raw.dtype
  pure (.mk raw)

def ofTensorWithContract {shape : Shape} (contract : TensorContract) (raw : T shape) :
    Except String (DTensor shape contract.spec.dtype) := do
  let actual : TensorSpec := { shape := raw.runtimeShape, dtype := raw.dtype }
  contract.check actual raw.device
  if h : contract.spec.shape = shape then
    pure (h ▸ .mk raw)
  else
    .error s!"Expected static shape {reprStr contract.spec.shape}, got {reprStr shape}"

def toSome {shape : Shape} {dtype : DType} (tensor : DTensor shape dtype) : SomeDTensor :=
  ⟨tensor.spec, tensor⟩

def add {shape : Shape} {lhsDType rhsDType : DType}
    (lhs : DTensor shape lhsDType) (rhs : DTensor shape rhsDType) :
    DTensor shape (DType.promote lhsDType rhsDType) :=
  .mk (torch.add lhs.raw rhs.raw)

def sub {shape : Shape} {lhsDType rhsDType : DType}
    (lhs : DTensor shape lhsDType) (rhs : DTensor shape rhsDType) :
    DTensor shape (DType.promote lhsDType rhsDType) :=
  .mk (torch.sub lhs.raw rhs.raw)

def mul {shape : Shape} {lhsDType rhsDType : DType}
    (lhs : DTensor shape lhsDType) (rhs : DTensor shape rhsDType) :
    DTensor shape (DType.promote lhsDType rhsDType) :=
  .mk (torch.mul lhs.raw rhs.raw)

def toFloat32 {shape : Shape} {dtype : DType} (tensor : DTensor shape dtype) :
    DTensor shape .Float32 :=
  .mk (torch.toFloat' tensor.raw)

def toBFloat16 {shape : Shape} {dtype : DType} (tensor : DTensor shape dtype) :
    DTensor shape .BFloat16 :=
  .mk (torch.toBFloat16' tensor.raw)

end DTensor

end torch
