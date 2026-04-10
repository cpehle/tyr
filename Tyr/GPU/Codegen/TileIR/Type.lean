/-!
# Tyr.GPU.Codegen.TileIR.Type

Lean-side representation of NVIDIA CUDA TileIR types.

The focus here is the actual public TileIR surface, not Tyr's legacy C++ kernel
IR. The first backend-facing milestone is:

- represent TileIR modules directly in Lean,
- render canonical TileIR MLIR text,
- and feed that text to NVIDIA's TileIR toolchain when it is installed.
-/

namespace Tyr.GPU.Codegen.TileIR

/-- Scalar element types used by public TileIR syntax. -/
inductive ScalarType where
  | i1
  | i8
  | i16
  | i32
  | i64
  | u8
  | u16
  | u32
  | u64
  | f16
  | bf16
  | f32
  | f64
  | index
  deriving Repr, Inhabited, BEq, DecidableEq

namespace ScalarType

def render : ScalarType → String
  | .i1 => "i1"
  | .i8 => "i8"
  | .i16 => "i16"
  | .i32 => "i32"
  | .i64 => "i64"
  | .u8 => "ui8"
  | .u16 => "ui16"
  | .u32 => "ui32"
  | .u64 => "ui64"
  | .f16 => "f16"
  | .bf16 => "bf16"
  | .f32 => "f32"
  | .f64 => "f64"
  | .index => "index"

def isIntegral : ScalarType → Bool
  | .i1 | .i8 | .i16 | .i32 | .i64
  | .u8 | .u16 | .u32 | .u64
  | .index => true
  | _ => false

def isFloat : ScalarType → Bool
  | .f16 | .bf16 | .f32 | .f64 => true
  | _ => false

def bitWidth? : ScalarType → Option Nat
  | .i1 => some 1
  | .i8 | .u8 => some 8
  | .i16 | .u16 | .f16 | .bf16 => some 16
  | .i32 | .u32 | .f32 | .index => some 32
  | .i64 | .u64 | .f64 => some 64

end ScalarType

/-- A TileIR dimension may be statically known or dynamic (`?`). -/
inductive ShapeDim where
  | static (value : Nat)
  | dynamic
  deriving Repr, Inhabited, BEq, DecidableEq

namespace ShapeDim

def render : ShapeDim → String
  | .static value => toString value
  | .dynamic => "?"

def toNat? : ShapeDim → Option Nat
  | .static value => some value
  | .dynamic => none

end ShapeDim

/-- Tile element types can be scalars or pointer scalars. -/
inductive ElemType where
  | scalar (ty : ScalarType)
  | ptr (ty : ScalarType)
  deriving Repr, Inhabited, BEq, DecidableEq

namespace ElemType

def renderEmbedded : ElemType → String
  | .scalar ty => ty.render
  | .ptr ty => s!"ptr<{ty.render}>"

end ElemType

structure TensorViewType where
  elem : ScalarType
  shape : Array ShapeDim
  strides : Array ShapeDim
  deriving Repr, Inhabited, BEq, DecidableEq

namespace TensorViewType

private def renderShape (shape : Array ShapeDim) : String :=
  String.intercalate "x" <| shape.toList.map ShapeDim.render

private def renderStrides (strides : Array ShapeDim) : String :=
  String.intercalate ", " <| strides.toList.map ShapeDim.render

def renderEmbedded (ty : TensorViewType) : String :=
  s!"tensor_view<{renderShape ty.shape}x{ty.elem.render}, strides=[{renderStrides ty.strides}]>"

end TensorViewType

structure PartitionViewType where
  tileShape : Array Nat
  tensor : TensorViewType
  dimMap : Array Nat := #[]
  deriving Repr, Inhabited, BEq, DecidableEq

namespace PartitionViewType

private def renderTileShape (shape : Array Nat) : String :=
  String.intercalate "x" <| shape.toList.map toString

private def renderDimMap (dims : Array Nat) : String :=
  String.intercalate ", " <| dims.toList.map toString

def renderEmbedded (ty : PartitionViewType) : String :=
  let dimMapPart :=
    if ty.dimMap.isEmpty then
      ""
    else
      s!", dim_map=[{renderDimMap ty.dimMap}]"
  s!"partition_view<tile=({renderTileShape ty.tileShape}), {ty.tensor.renderEmbedded}{dimMapPart}>"

end PartitionViewType

/-- Public TileIR value/parameter/result types. -/
inductive TileType where
  | ptr (elem : ScalarType)
  | tile (shape : Array ShapeDim) (elem : ElemType)
  | tensorView (desc : TensorViewType)
  | partitionView (desc : PartitionViewType)
  | token
  deriving Repr, Inhabited, BEq, DecidableEq

namespace TileType

private def renderShape (shape : Array ShapeDim) : String :=
  String.intercalate "x" <| shape.toList.map ShapeDim.render

def renderEmbedded : TileType → String
  | .ptr elem => s!"ptr<{elem.render}>"
  | .tile shape elem =>
      if shape.isEmpty then
        s!"tile<{elem.renderEmbedded}>"
      else
        s!"tile<{renderShape shape}x{elem.renderEmbedded}>"
  | .tensorView desc => desc.renderEmbedded
  | .partitionView desc => desc.renderEmbedded
  | .token => "token"

def render (ty : TileType) : String :=
  s!"!cuda_tile.{ty.renderEmbedded}"

def literalScalar? : TileType → Option ScalarType
  | .tile _ (.scalar ty) => some ty
  | _ => none

def elemType? : TileType → Option ElemType
  | .tile _ elem => some elem
  | _ => none

def staticShape? : TileType → Option (Array Nat)
  | .tile shape _ => shape.mapM ShapeDim.toNat?
  | .tensorView desc => desc.shape.mapM ShapeDim.toNat?
  | .partitionView desc => some desc.tileShape
  | .ptr _ => some #[]
  | .token => none

def rank? (ty : TileType) : Option Nat :=
  ty.staticShape?.map (·.size)

end TileType

def staticShape (dims : Array Nat) : Array ShapeDim :=
  dims.map ShapeDim.static

def scalarTileTy (elem : ScalarType) : TileType :=
  .tile #[] (.scalar elem)

def tileTy (elem : ScalarType) (dims : Array Nat) : TileType :=
  .tile (staticShape dims) (.scalar elem)

def ptrTy (elem : ScalarType) : TileType :=
  .ptr elem

def ptrTileTy (elem : ScalarType) : TileType :=
  .tile #[] (.ptr elem)

def tensorViewTy (elem : ScalarType) (shape strides : Array ShapeDim) : TileType :=
  .tensorView { elem, shape, strides }

def partitionViewTy (tileShape : Array Nat) (tensor : TensorViewType) (dimMap : Array Nat := #[]) : TileType :=
  .partitionView { tileShape, tensor, dimMap }

instance : ToString ScalarType where
  toString := ScalarType.render

instance : ToString ShapeDim where
  toString := ShapeDim.render

instance : ToString ElemType where
  toString := ElemType.renderEmbedded

instance : ToString TileType where
  toString := TileType.render

end Tyr.GPU.Codegen.TileIR
