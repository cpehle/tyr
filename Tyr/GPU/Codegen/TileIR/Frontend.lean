import Lean
import Lean.Parser.Do
import Tyr.GPU.Codegen.TileIR.Builder
import Tyr.GPU.Codegen.TileIR.Render

/-!
# Tyr.GPU.Codegen.TileIR.Frontend

TileIR frontend support built on Lean 4.29's extensible `do` elaborator.

The outer module/entry forms stay thin, but TileIR expression statements are now
handled by a custom `doLet` elaborator for `EntryM` blocks instead of rewriting
entire `do` sequences with macros.
-/

namespace Tyr.GPU.Codegen.TileIR

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Elab.Do
open Lean.Elab.Command
open Lean.Meta
open Lean.Parser.Command
open Lean.Parser.Term

declare_syntax_cat tileirBinder
syntax "(" ident " : " term ")" : tileirBinder

declare_syntax_cat tileirLoopRange
syntax "(" term " to " term "," " step " term ")" : tileirLoopRange

declare_syntax_cat tileirLoopCarry
syntax "(" ident " := " term ")" : tileirLoopCarry

declare_syntax_cat tileirIndexList
syntax "[" sepBy(term, ", ") "]" : tileirIndexList

declare_syntax_cat tileirTermTuple
syntax (name := tileirTermTupleEmpty) "(" ")" : tileirTermTuple
syntax (name := tileirTermTupleSingleton) "(" term ",)" : tileirTermTuple
syntax (name := tileirTermTupleTerms) "(" sepBy1(term, ", ") ")" : tileirTermTuple

declare_syntax_cat tileirTupleArg
syntax ident " := " tileirTermTuple : tileirTupleArg

declare_syntax_cat tileirTermArg
syntax ident " := " term : tileirTermArg

syntax (name := tileirModuleTerm) "tileir_module " str " do " doSeq : term
syntax (name := tileirGlobalTerm) "global " ident " : " term " := " term : term
syntax (name := tileirEntryTerm) "entry " str (ppSpace tileirBinder)* " do " doSeq : term

syntax (name := tileirConstIntTerm) "const_int " term " : " term : term
syntax (name := tileirConstFloatTerm) "const_float " term " : " term : term
syntax (name := tileirConstBoolTerm) "const_bool " term " : " term : term
syntax (name := tileirLoadPtrTkoTerm) "load_ptr_tko " term ", " term " : " term : term
syntax (name := tileirGetGlobalTerm) "get_global " ident " : " term : term
syntax (name := tileirBroadcastTerm) "broadcast " term " : " term : term
syntax (name := tileirReshapeTerm) "reshape " term " : " term : term
syntax (name := tileirOffsetTerm) "offset " term ", " term " : " term : term
syntax (name := tileirMakeTensorViewTerm) "make_tensor_view " term " : " term : term
syntax (name := tileirMakePartitionViewTerm) "make_partition_view " term " : " term : term
syntax (name := tileirMmafTerm) "mmaf " term ", " term ", " term : term
syntax (name := tileirMmaiTerm) "mmai " term ", " term ", " term : term
syntax (name := tileirMaxfTerm) "maxf " term ", " term : term
syntax (name := tileirMinfTerm) "minf " term ", " term : term
syntax (name := tileirFor1Term)
  "for_tile " ident
  " in " tileirLoopRange
  " carrying " tileirLoopCarry
  " do " doSeq : term
syntax (name := tileirStorePtrTkoTerm) "store_ptr_tko " term ", " term ", " term : term
syntax (name := tileirPrintTkoTerm) "print_tko " str : term
syntax (name := tileirContinueTerm) "continue_tile " term : term
syntax (name := tileirBreakTerm) "break_tile " term : term
syntax (name := tileirKernelDefCmd)
  "@[" "tileir_kernel" "]"
  "def " declId optDeclSig
  " := " "do " doSeq : command

syntax (name := ctCommentTerm) "ct.comment " str : term
syntax (name := ctBidTerm) "ct.bid " term : term
syntax (name := ctNumBlocksTerm) "ct.num_blocks " term : term
syntax (name := ctCDivTerm) "ct.cdiv" "(" term ", " term ")" : term
syntax (name := ctRange1Term) "ct.range" "(" term ")" : term
syntax (name := ctRange2Term) "ct.range" "(" term ", " term ")" : term
syntax (name := ctRange3Term) "ct.range" "(" term ", " term ", " term ")" : term
syntax (name := ctStaticRange1Term) "ct.static_range" "(" term ")" : term
syntax (name := ctStaticRange2Term) "ct.static_range" "(" term ", " term ")" : term
syntax (name := ctStaticRange3Term) "ct.static_range" "(" term ", " term ", " term ")" : term
syntax (name := ctStaticIterTerm) "ct.static_iter" "(" term ")" : term
syntax (name := ctIotaTerm) "ct.iota" " : " term : term
syntax (name := ctArangeTerm) "ct.arange" "(" term ")" : term
syntax (name := ctArangeDtypeTerm) "ct.arange" "(" term ", " ident " := " term ")" : term
syntax (name := ctLoadTypedTerm) "ct.load " term " : " term : term
syntax (name := ctLoadIndexedTerm)
  "ct.load " term ", " tileirTupleArg ", " tileirTupleArg : term
syntax (name := ctLoadIndexedScalarIndexTerm)
  "ct.load " term ", " ident " := " term ", " ident " := " tileirTermTuple : term
syntax (name := ctLoadPositionalTerm)
  "ct.load" "(" term ", " tileirTermTuple ", " tileirTermTuple ")" : term
syntax (name := ctGatherTerm) "ct.gather" "(" term ", " term ")" : term
syntax (name := ctLoadViewTerm) "ct.load_view " term ", " tileirIndexList " : " term : term
syntax (name := ctLoadGlobalTerm) "ct.load_global " ident " : " term : term
syntax (name := ctFullKwTerm) "ct.full" "(" tileirTermTuple ", " term ", " ident " := " term ")" : term
syntax (name := ctFullPosTerm) "ct.full" "(" tileirTermTuple ", " term ", " term ")" : term
syntax (name := ctFullKwShapeTerm) "ct.full" "(" term ", " term ", " ident " := " term ")" : term
syntax (name := ctFullPosShapeTerm) "ct.full" "(" term ", " term ", " term ")" : term
syntax (name := ctFullLikeKwTerm) "ct.full_like" "(" term ", " term ", " ident " := " term ")" : term
syntax (name := ctFullLikePosTerm) "ct.full_like" "(" term ", " term ", " term ")" : term
syntax (name := ctZerosKwTerm) "ct.zeros" "(" tileirTermTuple ", " ident " := " term ")" : term
syntax (name := ctZerosPosTerm) "ct.zeros" "(" tileirTermTuple ", " term ")" : term
syntax (name := ctZerosKwShapeTerm) "ct.zeros" "(" term ", " ident " := " term ")" : term
syntax (name := ctZerosPosShapeTerm) "ct.zeros" "(" term ", " term ")" : term
syntax (name := ctZerosLikeTerm) "ct.zeros_like" "(" term ")" : term
syntax (name := ctOnesKwTerm) "ct.ones" "(" tileirTermTuple ", " ident " := " term ")" : term
syntax (name := ctOnesPosTerm) "ct.ones" "(" tileirTermTuple ", " term ")" : term
syntax (name := ctOnesKwShapeTerm) "ct.ones" "(" term ", " ident " := " term ")" : term
syntax (name := ctOnesPosShapeTerm) "ct.ones" "(" term ", " term ")" : term
syntax (name := ctOnesLikeTerm) "ct.ones_like" "(" term ")" : term
syntax (name := ctAsTypeTerm) "ct.astype" "(" term ", " term ")" : term
syntax (name := ctCatTerm) "ct.cat" "(" tileirTermTuple ", " term ")" : term
syntax (name := ctBroadcastTerm) "ct.broadcast " term " : " term : term
syntax (name := ctBroadcastToTerm) "ct.broadcast_to" "(" term ", " tileirTermTuple ")" : term
syntax (name := ctBroadcastToShapeTerm) "ct.broadcast_to" "(" term ", " term ")" : term
syntax (name := ctReshapeTerm) "ct.reshape " term " : " term : term
syntax (name := ctReshapeShapeTerm) "ct.reshape" "(" term ", " tileirTermTuple ")" : term
syntax (name := ctPermuteTerm) "ct.permute" "(" term ", " tileirTermTuple ")" : term
syntax (name := ctTransposeTerm) "ct.transpose" "(" term ")" : term
syntax (name := ctExtractIndexedTerm)
  "ct.extract " term ", " tileirTupleArg ", " tileirTupleArg : term
syntax (name := ctExtractPositionalTerm)
  "ct.extract" "(" term ", " tileirTermTuple ", " tileirTermTuple ")" : term
syntax (name := ctOffsetTerm) "ct.offset " term ", " term " : " term : term
syntax (name := ctTensorViewTerm) "ct.tensor_view " term " : " term : term
syntax (name := ctPartitionViewTerm) "ct.partition_view " term " : " term : term
syntax (name := ctMmaTerm) "ct.mma " term ", " term ", " term : term
syntax (name := ctMatmulTerm) "ct.matmul " term ", " term ", " term : term
syntax (name := ctMmaiTerm) "ct.mmai " term ", " term ", " term : term
syntax (name := ctMaxTerm) "ct.max " term ", " term : term
syntax (name := ctMinTerm) "ct.min " term ", " term : term
syntax (name := ctMaximumTerm) "ct.maximum " term ", " term : term
syntax (name := ctMinimumTerm) "ct.minimum " term ", " term : term
syntax (name := ctEqualTerm) "ct.equal " term ", " term : term
syntax (name := ctNotEqualTerm) "ct.not_equal " term ", " term : term
syntax (name := ctLessTerm) "ct.less " term ", " term : term
syntax (name := ctLessEqualTerm) "ct.less_equal " term ", " term : term
syntax (name := ctGreaterTerm) "ct.greater " term ", " term : term
syntax (name := ctGreaterEqualTerm) "ct.greater_equal " term ", " term : term
syntax (name := ctWhereTerm) "ct.where" "(" term ", " term ", " term ")" : term
syntax (name := ctAssertTerm) "ct.assert " term ", " str : term
syntax (name := ctStoreTerm) "ct.store " term ", " term : term
syntax (name := ctStoreIndexedTerm)
  "ct.store " term ", " tileirTupleArg ", " tileirTermArg : term
syntax (name := ctStoreIndexedScalarIndexTerm)
  "ct.store " term ", " ident " := " term ", " ident " := " term : term
syntax (name := ctStorePositionalTerm)
  "ct.store" "(" term ", " tileirTermTuple ", " term ")" : term
syntax (name := ctScatterTerm) "ct.scatter" "(" term ", " term ", " term ")" : term
syntax (name := ctStoreViewTerm) "ct.store_view " term ", " tileirIndexList ", " term : term
syntax (name := ctPrintTerm) "ct.print " str : term
syntax (name := ctStaticEvalTerm) "ct.static_eval " term : term
syntax (name := ctStaticAssertTerm) "ct.static_assert " term : term
syntax (name := ctStaticAssertMsgTerm) "ct.static_assert " term ", " term : term
syntax (name := ctExpTerm) "ct.exp " term : term
syntax (name := ctExp2Term) "ct.exp2 " term : term
syntax (name := ctLogTerm) "ct.log " term : term
syntax (name := ctSqrtTerm) "ct.sqrt " term : term
syntax (name := ctRsqrtTerm) "ct.rsqrt " term : term
syntax (name := ctAbsTerm) "ct.abs " term : term
syntax (name := ctNegTerm) "ct.neg " term : term

abbrev FrontendM := StateT Nat MacroM

namespace ct

universe u

abbrev Const (α : Type u) := α

abbrev i1 : ScalarType := .i1
abbrev bool_ : ScalarType := .i1
abbrev i8 : ScalarType := .i8
abbrev int8 : ScalarType := .i8
abbrev i16 : ScalarType := .i16
abbrev int16 : ScalarType := .i16
abbrev i32 : ScalarType := .i32
abbrev int32 : ScalarType := .i32
abbrev i64 : ScalarType := .i64
abbrev int64 : ScalarType := .i64
abbrev u8 : ScalarType := .u8
abbrev uint8 : ScalarType := .u8
abbrev u16 : ScalarType := .u16
abbrev uint16 : ScalarType := .u16
abbrev u32 : ScalarType := .u32
abbrev uint32 : ScalarType := .u32
abbrev u64 : ScalarType := .u64
abbrev uint64 : ScalarType := .u64
abbrev f16 : ScalarType := .f16
abbrev float16 : ScalarType := .f16
abbrev bf16 : ScalarType := .bf16
abbrev f32 : ScalarType := .f32
abbrev float32 : ScalarType := .f32
abbrev f64 : ScalarType := .f64
abbrev float64 : ScalarType := .f64
abbrev «index» : ScalarType := .index

def Scalar (elem : ScalarType) : TileType :=
  scalarTileTy elem

def Tile (elem : ScalarType) (dims : _root_.Array Nat) : TileType :=
  tileTy elem dims

def Ptr (elem : ScalarType) : TileType :=
  ptrTy elem

def PtrTile (elem : ScalarType) : TileType :=
  ptrTileTy elem

def Array (elem : ScalarType) : TileType :=
  ptrTileTy elem

def TensorView (elem : ScalarType) (dims strides : _root_.Array ShapeDim) : TileType :=
  tensorViewTy elem dims strides

def PartitionView (tileDims : _root_.Array Nat) (tensor : TensorViewType) (dimMap : _root_.Array Nat := #[]) : TileType :=
  partitionViewTy tileDims tensor dimMap

abbrev Token : TileType := .token

structure Val (ty : TileType) where
  raw : Tyr.GPU.Codegen.TileIR.Value
  deriving Repr, Inhabited, BEq, DecidableEq

instance : CoeOut (Val ty) Tyr.GPU.Codegen.TileIR.Value where
  coe := Val.raw

class FloatScalarTy (dtype : ScalarType) : Prop where

instance : FloatScalarTy .f16 where
instance : FloatScalarTy .bf16 where
instance : FloatScalarTy .f32 where
instance : FloatScalarTy .f64 where

class IntegralScalarTy (dtype : ScalarType) : Prop where

instance : IntegralScalarTy .i1 where
instance : IntegralScalarTy .i8 where
instance : IntegralScalarTy .i16 where
instance : IntegralScalarTy .i32 where
instance : IntegralScalarTy .i64 where
instance : IntegralScalarTy .u8 where
instance : IntegralScalarTy .u16 where
instance : IntegralScalarTy .u32 where
instance : IntegralScalarTy .u64 where
instance : IntegralScalarTy .index where

class FloatValueTy (ty : TileType) : Prop where

instance (priority := high) {dtype : ScalarType} [FloatScalarTy dtype] :
    FloatValueTy (ct.Scalar dtype) where

instance (priority := high) {dtype : ScalarType} {shape : _root_.Array Nat} [FloatScalarTy dtype] :
    FloatValueTy (ct.Tile dtype shape) where

instance {dtype : ScalarType} [FloatScalarTy dtype] :
    FloatValueTy (scalarTileTy dtype) where

instance {dtype : ScalarType} {shape : _root_.Array Nat} [FloatScalarTy dtype] :
    FloatValueTy (tileTy dtype shape) where

private def scalarTypeOf! : TileType → ScalarType
  | .ptr elem => elem
  | .tile _ (.scalar elem) => elem
  | .tile _ (.ptr elem) => elem
  | .tensorView desc => desc.elem
  | .partitionView desc => desc.tensor.elem
  | .token => panic! "TileIR tokens do not have a dtype"

private def elemTypeOf! (ty : TileType) : ElemType :=
  match ty.elemType? with
  | some elem => elem
  | none => panic! s!"TileIR value type {ty.render} does not have an element type"

private def staticShapeOf! (ty : TileType) : _root_.Array Nat :=
  match ty.staticShape? with
  | some shape => shape
  | none => panic! s!"TileIR value type {ty.render} does not have a statically known shape"

private def tileWithElem (elem : ElemType) (shape : _root_.Array Nat) : TileType :=
  .tile (staticShape shape) elem

private def tileWithScalar (dtype : ScalarType) (shape : _root_.Array Nat) : TileType :=
  tileWithElem (.scalar dtype) shape

private def tileLike (ty : TileType) (shape : _root_.Array Nat) : TileType :=
  tileWithElem (elemTypeOf! ty) shape

private def castTy (ty : TileType) (dtype : ScalarType) : TileType :=
  tileWithScalar dtype (staticShapeOf! ty)

private def permuteNatShape (shape perm : _root_.Array Nat) : _root_.Array Nat :=
  perm.map fun axis => shape[axis]!

private def catNatShape (lhs rhs : _root_.Array Nat) (axis : Nat) : _root_.Array Nat :=
  Id.run do
    let mut out := lhs
    if axis < lhs.size && axis < rhs.size then
      out := out.set! axis (lhs[axis]! + rhs[axis]!)
    out

def param (name : String) (ty : TileType) : Val ty :=
  { raw := Tyr.GPU.Codegen.TileIR.arg name ty }

def ofRaw {ty : TileType} (raw : Tyr.GPU.Codegen.TileIR.Value) : Val ty :=
  { raw }

private def ofEntry {ty : TileType}
    (action : Tyr.GPU.Codegen.TileIR.EntryM Tyr.GPU.Codegen.TileIR.Value)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) := do
  pure { raw := ← action }

instance {ty : TileType} : Tyr.GPU.Codegen.TileIR.FillValue (Val ty) where
  build hint shape value dtype :=
    Tyr.GPU.Codegen.TileIR.fill hint shape value.raw dtype

namespace Val

def dtype {ty : TileType} (_ : Val ty) : ScalarType :=
  scalarTypeOf! ty

def shape {ty : TileType} (_ : Val ty) : _root_.Array Nat :=
  staticShapeOf! ty

def rank {ty : TileType} (value : Val ty) : Nat :=
  value.shape.size

def ndim {ty : TileType} (value : Val ty) : Nat :=
  rank value

def astype {ty : TileType} (value : Val ty) (dtype : ScalarType)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (castTy ty dtype)) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.astype "astype" value.raw dtype

def «reshape» {ty : TileType} (value : Val ty) (shape : _root_.Array Nat)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (tileLike ty shape)) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.reshapeLike "reshape" value.raw shape

def permute {elem : ScalarType} {shape : _root_.Array Nat}
    (value : Val (Tile elem shape))
    (axes : _root_.Array Nat)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem (permuteNatShape shape axes))) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.permute "permute" value.raw axes

def transpose {elem : ScalarType} {m n : Nat}
    (value : Val (Tile elem #[m, n]))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem #[n, m])) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.permute "transpose" value.raw #[1, 0]

def matmul {aElem bElem cElem : ScalarType} {m k n : Nat}
    [FloatScalarTy aElem] [FloatScalarTy bElem] [FloatScalarTy cElem]
    (a : Val (Tile aElem #[m, k]))
    (b : Val (Tile bElem #[k, n]))
    (c : Val (Tile cElem #[m, n]))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile cElem #[m, n])) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.mmaf "matmul" a.raw b.raw c.raw (some (Tile cElem #[m, n]))

def extract {elem : ScalarType} {shape : _root_.Array Nat}
    (value : Val (Tile elem shape))
    (indices : _root_.Array Tyr.GPU.Codegen.TileIR.Value)
    (resultShape : _root_.Array Nat)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem resultShape)) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.extract "extract" value.raw indices resultShape

end Val

def astype {ty : TileType} (value : Val ty) (dtype : ScalarType)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (castTy ty dtype)) :=
  value.astype dtype

def reshapeVal {ty : TileType} (value : Val ty) (shape : _root_.Array Nat)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (tileLike ty shape)) :=
  value.«reshape» shape

def permute {elem : ScalarType} {shape : _root_.Array Nat}
    (value : Val (Tile elem shape))
    (axes : _root_.Array Nat)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem (permuteNatShape shape axes))) :=
  value.permute axes

def transpose {elem : ScalarType} {m n : Nat}
    (value : Val (Tile elem #[m, n]))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem #[n, m])) :=
  value.transpose

def extract {elem : ScalarType} {shape : _root_.Array Nat}
    (value : Val (Tile elem shape))
    (indices : _root_.Array Tyr.GPU.Codegen.TileIR.Value)
    (resultShape : _root_.Array Nat)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem resultShape)) :=
  value.extract indices resultShape

def constInt (hint : String) (value : Int) (ty : TileType := Scalar i32)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.constInt hint value ty

def constFloat (hint : String) (value : Float) (ty : TileType := Scalar f32)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.constFloat hint value ty

def constBool (hint : String) (value : Bool) (ty : TileType := Scalar i1)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.constBool hint value ty

def bid (axis : Nat) : Tyr.GPU.Codegen.TileIR.EntryM (Val (Scalar i32)) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.tileBlockId axis

def numBlocks (axis : Nat) : Tyr.GPU.Codegen.TileIR.EntryM (Val (Scalar i32)) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.numTileBlocks axis

def cdiv (lhs rhs : Nat) : Nat :=
  if rhs == 0 then
    0
  else
    (lhs + rhs - 1) / rhs

private def staticRangeAux (fuel start stop stepVal : Nat) : List Nat :=
  match fuel with
  | 0 => []
  | fuel' + 1 =>
      if stepVal == 0 then
        []
      else if start < stop then
        start :: staticRangeAux fuel' (start + stepVal) stop stepVal
      else
        []

def staticRange1 (stop : Nat) : List Nat :=
  List.range stop

def staticRange2 (start stop : Nat) : List Nat :=
  staticRangeAux (stop - start + 1) start stop 1

def staticRange3 (start stop stepVal : Nat) : List Nat :=
  staticRangeAux (stop - start + 1) start stop stepVal

def staticIter {α : Type u} (xs : α) : α :=
  xs

def iotaAs (hint : String) (ty : TileType)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.iota hint ty

def arange (hint : String) (size : Nat) (dtype : ScalarType := i32)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile dtype #[size])) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.iota hint (Tile dtype #[size])

def loadAs (hint : String) {srcTy : TileType} (ptr : Val srcTy) (resultTy : TileType)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val resultTy) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.load hint ptr.raw resultTy

def loadIndexed {elem : ScalarType}
    (hint : String)
    (base : Val (Array elem))
    (indices : _root_.Array Tyr.GPU.Codegen.TileIR.Value)
    (shape : _root_.Array Nat)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem shape)) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.loadTiled hint base.raw indices shape

def gather {elem idxElem : ScalarType} {shape : _root_.Array Nat}
    (hint : String)
    (base : Val (Array elem))
    (idx : Val (Tile idxElem shape))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem shape)) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.gather hint base.raw idx.raw

def loadViewAs {viewTy : TileType}
    (hint : String)
    (view : Val viewTy)
    (indices : _root_.Array Tyr.GPU.Codegen.TileIR.Value)
    (resultTy : TileType)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val resultTy) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.loadView hint view.raw indices resultTy

def loadGlobalAs (hint globalName : String) (resultTy : TileType)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val resultTy) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.loadGlobal hint globalName resultTy

def full [Tyr.GPU.Codegen.TileIR.FillValue α]
    (hint : String)
    (shape : _root_.Array Nat)
    (value : α)
    (dtype : ScalarType)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile dtype shape)) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.full hint shape value dtype

def zeros (hint : String) (shape : _root_.Array Nat) (dtype : ScalarType)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile dtype shape)) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.zeros hint shape dtype

def ones (hint : String) (shape : _root_.Array Nat) (dtype : ScalarType)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile dtype shape)) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.ones hint shape dtype

def fullLike [Tyr.GPU.Codegen.TileIR.FillValue α] {likeDtype : ScalarType} {shape : _root_.Array Nat}
    (hint : String)
    (like : Val (Tile likeDtype shape))
    (value : α)
    (dtype : ScalarType)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile dtype shape)) := do
  let _ := like
  full hint shape value dtype

def zerosLike {dtype : ScalarType} {shape : _root_.Array Nat}
    (hint : String)
    (like : Val (Tile dtype shape))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile dtype shape)) := do
  let _ := like
  zeros hint shape dtype

def onesLike {dtype : ScalarType} {shape : _root_.Array Nat}
    (hint : String)
    (like : Val (Tile dtype shape))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile dtype shape)) := do
  let _ := like
  ones hint shape dtype

class BroadcastInput (srcTy : TileType) (dtype : ScalarType) : Prop where

instance {dtype : ScalarType} : BroadcastInput (Scalar dtype) dtype where

instance {dtype : ScalarType} : BroadcastInput (Tile dtype #[]) dtype where

def broadcastTo {srcTy : TileType} {dtype : ScalarType} [BroadcastInput srcTy dtype]
    (hint : String)
    (value : Val srcTy)
    (shape : _root_.Array Nat)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile dtype shape)) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.broadcast hint value.raw (Tile dtype shape)

def broadcastAs {srcTy : TileType}
    (hint : String)
    (value : Val srcTy)
    (dstTy : TileType)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val dstTy) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.broadcast hint value.raw dstTy

def reshapeAs {srcTy : TileType}
    (hint : String)
    (value : Val srcTy)
    (dstTy : TileType)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val dstTy) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.reshape hint value.raw dstTy

def offsetAs {ptrTy : TileType}
    (hint : String)
    (ptr : Val ptrTy)
    (idx : Tyr.GPU.Codegen.TileIR.Value)
    (dstTy : TileType)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val dstTy) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.offset hint ptr.raw idx dstTy

def tensorViewAs {ptrTy : TileType}
    (hint : String)
    (base : Val ptrTy)
    (desc : TensorViewType)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (.tensorView desc)) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.makeTensorView hint base.raw desc

def partitionViewAs {srcTy : TileType}
    (hint : String)
    (src : Val srcTy)
    (desc : PartitionViewType)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (.partitionView desc)) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.makePartitionView hint src.raw desc

def unary {ty : TileType}
    (hint : String)
    (op : UnaryOp)
    (value : Val ty)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.unary hint op value.raw (some ty)

def copy {ty : TileType} (value : Val ty) : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  unary "copy" .copy value

def exp {ty : TileType} [FloatValueTy ty]
    (value : Val ty) : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  unary "exp" .exp value

def exp2 {ty : TileType} [FloatValueTy ty]
    (value : Val ty) : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  unary "exp2" .exp2 value

def log {ty : TileType} [FloatValueTy ty]
    (value : Val ty) : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  unary "log" .log value

def sqrt {ty : TileType} [FloatValueTy ty]
    (value : Val ty) : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  unary "sqrt" .sqrt value

def rsqrt {ty : TileType} [FloatValueTy ty]
    (value : Val ty) : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  unary "rsqrt" .rsqrt value

def abs {ty : TileType} (value : Val ty) : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  unary "abs" .abs value

def neg {ty : TileType} (value : Val ty) : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  unary "neg" .neg value

def binary {ty : TileType}
    (hint : String)
    (op : BinaryOp)
    (lhs rhs : Val ty)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.binary hint op lhs.raw rhs.raw (some ty)

def add {ty : TileType} [FloatValueTy ty]
    (lhs rhs : Val ty) : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  binary "add" .addf lhs rhs

def sub {ty : TileType} [FloatValueTy ty]
    (lhs rhs : Val ty) : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  binary "sub" .subf lhs rhs

def mul {ty : TileType} [FloatValueTy ty]
    (lhs rhs : Val ty) : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  binary "mul" .mulf lhs rhs

def div {ty : TileType} [FloatValueTy ty]
    (lhs rhs : Val ty) : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  binary "div" .divf lhs rhs

def max {ty : TileType} [FloatValueTy ty]
    (lhs rhs : Val ty) : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  binary "max" .maxf lhs rhs

def min {ty : TileType} [FloatValueTy ty]
    (lhs rhs : Val ty) : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) :=
  binary "min" .minf lhs rhs

instance {ty : TileType} [FloatValueTy ty] :
    HAdd (Val ty) (Val ty) (Tyr.GPU.Codegen.TileIR.EntryM (Val ty)) where
  hAdd := add

instance {ty : TileType} [FloatValueTy ty] :
    HSub (Val ty) (Val ty) (Tyr.GPU.Codegen.TileIR.EntryM (Val ty)) where
  hSub := sub

instance {ty : TileType} [FloatValueTy ty] :
    HMul (Val ty) (Val ty) (Tyr.GPU.Codegen.TileIR.EntryM (Val ty)) where
  hMul := mul

instance {ty : TileType} [FloatValueTy ty] :
    HDiv (Val ty) (Val ty) (Tyr.GPU.Codegen.TileIR.EntryM (Val ty)) where
  hDiv := div

private def liftTileScalarBinary
    {elem : ScalarType}
    {shape : _root_.Array Nat}
    [FloatScalarTy elem]
    (hint : String)
    (op : Val (Tile elem shape) → Val (Tile elem shape) →
      Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem shape)))
    (lhs : Val (Tile elem shape))
    (rhs : Val (Scalar elem))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem shape)) := do
  let rhsTile ← broadcastTo s!"{hint}_rhs" rhs shape
  op lhs rhsTile

private def liftScalarTileBinary
    {elem : ScalarType}
    {shape : _root_.Array Nat}
    [FloatScalarTy elem]
    (hint : String)
    (op : Val (Tile elem shape) → Val (Tile elem shape) →
      Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem shape)))
    (lhs : Val (Scalar elem))
    (rhs : Val (Tile elem shape))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem shape)) := do
  let lhsTile ← broadcastTo s!"{hint}_lhs" lhs shape
  op lhsTile rhs

instance {elem : ScalarType} {shape : _root_.Array Nat} [FloatScalarTy elem] :
    HAdd (Val (Tile elem shape)) (Val (Scalar elem))
      (Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem shape))) where
  hAdd := liftTileScalarBinary "add" (fun lhs rhs => binary "add" .addf lhs rhs)

instance {elem : ScalarType} {shape : _root_.Array Nat} [FloatScalarTy elem] :
    HAdd (Val (Scalar elem)) (Val (Tile elem shape))
      (Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem shape))) where
  hAdd := liftScalarTileBinary "add" (fun lhs rhs => binary "add" .addf lhs rhs)

instance {elem : ScalarType} {shape : _root_.Array Nat} [FloatScalarTy elem] :
    HSub (Val (Tile elem shape)) (Val (Scalar elem))
      (Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem shape))) where
  hSub := liftTileScalarBinary "sub" (fun lhs rhs => binary "sub" .subf lhs rhs)

instance {elem : ScalarType} {shape : _root_.Array Nat} [FloatScalarTy elem] :
    HSub (Val (Scalar elem)) (Val (Tile elem shape))
      (Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem shape))) where
  hSub := liftScalarTileBinary "sub" (fun lhs rhs => binary "sub" .subf lhs rhs)

instance {elem : ScalarType} {shape : _root_.Array Nat} [FloatScalarTy elem] :
    HMul (Val (Tile elem shape)) (Val (Scalar elem))
      (Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem shape))) where
  hMul := liftTileScalarBinary "mul" (fun lhs rhs => binary "mul" .mulf lhs rhs)

instance {elem : ScalarType} {shape : _root_.Array Nat} [FloatScalarTy elem] :
    HMul (Val (Scalar elem)) (Val (Tile elem shape))
      (Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem shape))) where
  hMul := liftScalarTileBinary "mul" (fun lhs rhs => binary "mul" .mulf lhs rhs)

instance {elem : ScalarType} {shape : _root_.Array Nat} [FloatScalarTy elem] :
    HDiv (Val (Tile elem shape)) (Val (Scalar elem))
      (Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem shape))) where
  hDiv := liftTileScalarBinary "div" (fun lhs rhs => binary "div" .divf lhs rhs)

instance {elem : ScalarType} {shape : _root_.Array Nat} [FloatScalarTy elem] :
    HDiv (Val (Scalar elem)) (Val (Tile elem shape))
      (Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem shape))) where
  hDiv := liftScalarTileBinary "div" (fun lhs rhs => binary "div" .divf lhs rhs)

def cat {elem : ScalarType} {lhsShape rhsShape : _root_.Array Nat}
    (hint : String)
    (lhs : Val (Tile elem lhsShape))
    (rhs : Val (Tile elem rhsShape))
    (axis : Nat)
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile elem (catNatShape lhsShape rhsShape axis))) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.cat hint lhs.raw rhs.raw axis

private def compareFloatMode : ComparisonPredicate → FloatCompareMode
  | .notEqual => .unordered
  | _ => .ordered

private def signednessOf : ScalarType → Option Signedness
  | .i1 | .i8 | .i16 | .i32 | .i64 => some .signed
  | .u8 | .u16 | .u32 | .u64 | .index => some .unsigned
  | _ => none

def compareTileValues {dtype : ScalarType} {shape : _root_.Array Nat}
    (hint : String)
    (pred : ComparisonPredicate)
    (lhs rhs : Val (Tile dtype shape))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile i1 shape)) := do
  if dtype.isFloat then
    ofEntry <|
      Tyr.GPU.Codegen.TileIR.cmpf hint pred (compareFloatMode pred) lhs.raw rhs.raw (some (Tile i1 shape))
  else
    let some signedness := signednessOf dtype
      | throw s!"TileIR comparisons do not support scalar element type {dtype.render}"
    ofEntry <|
      Tyr.GPU.Codegen.TileIR.cmpi hint pred lhs.raw rhs.raw signedness (some (Tile i1 shape))

def compareRightScalar {dtype : ScalarType} {shape : _root_.Array Nat}
    (hint : String)
    (pred : ComparisonPredicate)
    (lhs : Val (Tile dtype shape))
    (rhs : Val (Scalar dtype))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile i1 shape)) := do
  let rhsTile ← broadcastTo s!"{hint}_rhs" rhs shape
  compareTileValues hint pred lhs rhsTile

def compareLeftScalar {dtype : ScalarType} {shape : _root_.Array Nat}
    (hint : String)
    (pred : ComparisonPredicate)
    (lhs : Val (Scalar dtype))
    (rhs : Val (Tile dtype shape))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile i1 shape)) := do
  let lhsTile ← broadcastTo s!"{hint}_lhs" lhs shape
  compareTileValues hint pred lhsTile rhs

def equal {dtype : ScalarType} {shape : _root_.Array Nat}
    (lhs rhs : Val (Tile dtype shape))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile i1 shape)) :=
  compareTileValues "equal" .equal lhs rhs

def notEqual {dtype : ScalarType} {shape : _root_.Array Nat}
    (lhs rhs : Val (Tile dtype shape))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile i1 shape)) :=
  compareTileValues "not_equal" .notEqual lhs rhs

def less {dtype : ScalarType} {shape : _root_.Array Nat}
    (lhs rhs : Val (Tile dtype shape))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile i1 shape)) :=
  compareTileValues "less" .lessThan lhs rhs

def lessEqual {dtype : ScalarType} {shape : _root_.Array Nat}
    (lhs rhs : Val (Tile dtype shape))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile i1 shape)) :=
  compareTileValues "less_equal" .lessThanOrEqual lhs rhs

def greater {dtype : ScalarType} {shape : _root_.Array Nat}
    (lhs rhs : Val (Tile dtype shape))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile i1 shape)) :=
  compareTileValues "greater" .greaterThan lhs rhs

def greaterEqual {dtype : ScalarType} {shape : _root_.Array Nat}
    (lhs rhs : Val (Tile dtype shape))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile i1 shape)) :=
  compareTileValues "greater_equal" .greaterThanOrEqual lhs rhs

def compareSurface
    (hint : String)
    (pred : ComparisonPredicate)
    {dtype : ScalarType}
    {shape : _root_.Array Nat}
    (lhs : Val (Tile dtype shape))
    (rhs : Val (Scalar dtype))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile i1 shape)) :=
  compareRightScalar hint pred lhs rhs

def compareSurfaceLeft
    (hint : String)
    (pred : ComparisonPredicate)
    {dtype : ScalarType}
    {shape : _root_.Array Nat}
    (lhs : Val (Scalar dtype))
    (rhs : Val (Tile dtype shape))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile i1 shape)) :=
  compareLeftScalar hint pred lhs rhs

def selectSameShape {dtype : ScalarType} {shape : _root_.Array Nat}
    (hint : String)
    (cond : Val (Tile i1 shape))
    (valIfTrue valIfFalse : Val (Tile dtype shape))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile dtype shape)) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.select hint cond.raw valIfTrue.raw valIfFalse.raw

def selectWithScalarBranches {dtype : ScalarType} {shape : _root_.Array Nat}
    (hint : String)
    (cond : Val (Tile i1 shape))
    (valIfTrue valIfFalse : Val (Scalar dtype))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile dtype shape)) := do
  let valIfTrue ← broadcastTo s!"{hint}_true" valIfTrue shape
  let valIfFalse ← broadcastTo s!"{hint}_false" valIfFalse shape
  selectSameShape hint cond valIfTrue valIfFalse

def selectWithTrueScalar {dtype : ScalarType} {shape : _root_.Array Nat}
    (hint : String)
    (cond : Val (Tile i1 shape))
    (valIfTrue : Val (Scalar dtype))
    (valIfFalse : Val (Tile dtype shape))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile dtype shape)) := do
  let valIfTrue ← broadcastTo s!"{hint}_true" valIfTrue shape
  selectSameShape hint cond valIfTrue valIfFalse

def selectWithFalseScalar {dtype : ScalarType} {shape : _root_.Array Nat}
    (hint : String)
    (cond : Val (Tile i1 shape))
    (valIfTrue : Val (Tile dtype shape))
    (valIfFalse : Val (Scalar dtype))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile dtype shape)) := do
  let valIfFalse ← broadcastTo s!"{hint}_false" valIfFalse shape
  selectSameShape hint cond valIfTrue valIfFalse

def selectWithScalarCond {dtype : ScalarType} {shape : _root_.Array Nat}
    (hint : String)
    (cond : Val (Scalar i1))
    (valIfTrue valIfFalse : Val (Tile dtype shape))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile dtype shape)) := do
  let cond ← broadcastTo s!"{hint}_cond" cond shape
  selectSameShape hint cond valIfTrue valIfFalse

def whereVal {dtype : ScalarType} {shape : _root_.Array Nat}
    (hint : String)
    (cond : Val (Tile i1 shape))
    (valIfTrue valIfFalse : Val (Tile dtype shape))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile dtype shape)) :=
  selectSameShape hint cond valIfTrue valIfFalse

def select {dtype : ScalarType} {shape : _root_.Array Nat}
    (hint : String)
    (cond : Val (Tile i1 shape))
    (valIfTrue valIfFalse : Val (Tile dtype shape))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile dtype shape)) :=
  whereVal hint cond valIfTrue valIfFalse

def mma {aElem bElem cElem : ScalarType} {m k n : Nat}
    [FloatScalarTy aElem] [FloatScalarTy bElem] [FloatScalarTy cElem]
    (hint : String)
    (a : Val (Tile aElem #[m, k]))
    (b : Val (Tile bElem #[k, n]))
    (c : Val (Tile cElem #[m, n]))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile cElem #[m, n])) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.mmaf hint a.raw b.raw c.raw (some (Tile cElem #[m, n]))

def matmul {aElem bElem cElem : ScalarType} {m k n : Nat}
    [FloatScalarTy aElem] [FloatScalarTy bElem] [FloatScalarTy cElem]
    (a : Val (Tile aElem #[m, k]))
    (b : Val (Tile bElem #[k, n]))
    (c : Val (Tile cElem #[m, n]))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile cElem #[m, n])) :=
  mma "matmul" a b c

def mmaiVal {aElem bElem cElem : ScalarType} {m k n : Nat}
    [IntegralScalarTy aElem] [IntegralScalarTy bElem] [IntegralScalarTy cElem]
    (hint : String)
    (a : Val (Tile aElem #[m, k]))
    (b : Val (Tile bElem #[k, n]))
    (c : Val (Tile cElem #[m, n]))
    : Tyr.GPU.Codegen.TileIR.EntryM (Val (Tile cElem #[m, n])) :=
  ofEntry <| Tyr.GPU.Codegen.TileIR.mmai hint a.raw b.raw c.raw .signed .signed (some (Tile cElem #[m, n]))

def store {elem : ScalarType} {shape : _root_.Array Nat}
    (ptr : Val (PtrTile elem))
    (value : Val (Tile elem shape))
    : Tyr.GPU.Codegen.TileIR.EntryM Unit :=
  Tyr.GPU.Codegen.TileIR.store ptr.raw value.raw

def storeIndexed {elem : ScalarType} {shape : _root_.Array Nat}
    (base : Val (Array elem))
    (indices : _root_.Array Tyr.GPU.Codegen.TileIR.Value)
    (value : Val (Tile elem shape))
    : Tyr.GPU.Codegen.TileIR.EntryM Unit :=
  Tyr.GPU.Codegen.TileIR.storeTiled base.raw indices value.raw

def scatter {elem idxElem : ScalarType} {shape : _root_.Array Nat}
    (base : Val (Array elem))
    (idx : Val (Tile idxElem shape))
    (value : Val (Tile elem shape))
    : Tyr.GPU.Codegen.TileIR.EntryM Unit :=
  Tyr.GPU.Codegen.TileIR.scatter base.raw idx.raw value.raw

def storeView {viewTy valTy : TileType}
    (view : Val viewTy)
    (indices : _root_.Array Tyr.GPU.Codegen.TileIR.Value)
    (value : Val valTy)
    : Tyr.GPU.Codegen.TileIR.EntryM Unit :=
  Tyr.GPU.Codegen.TileIR.storeView view.raw indices value.raw

def comment (text : String) : Tyr.GPU.Codegen.TileIR.EntryM Unit :=
  Tyr.GPU.Codegen.TileIR.comment text

def print (text : String) : Tyr.GPU.Codegen.TileIR.EntryM Unit :=
  Tyr.GPU.Codegen.TileIR.print text

class AssertableTy (ty : TileType) : Prop where

instance : AssertableTy (Scalar i1) where

instance {shape : _root_.Array Nat} : AssertableTy (Tile i1 shape) where

def assert {ty : TileType} [AssertableTy ty]
    (cond : Val ty)
    (message : String)
    : Tyr.GPU.Codegen.TileIR.EntryM Unit :=
  Tyr.GPU.Codegen.TileIR.assert cond.raw message

def staticAssert (cond : Bool) (mkMessage : Unit → String := fun _ => "")
    : Tyr.GPU.Codegen.TileIR.EntryM Unit :=
  Tyr.GPU.Codegen.TileIR.staticAssert cond mkMessage

def if1 {ty : TileType}
    (cond : Val (Scalar i1))
    (thenBranch elseBranch : Tyr.GPU.Codegen.TileIR.EntryM (Val ty))
    (hint : String := "if")
    : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) := do
  let raw ← Tyr.GPU.Codegen.TileIR.if1
    cond.raw
    (do pure (← thenBranch).raw)
    (do pure (← elseBranch).raw)
    hint
  pure { raw }

def continue1 {ty : TileType} (value : Val ty) : Tyr.GPU.Codegen.TileIR.LoopResult :=
  Tyr.GPU.Codegen.TileIR.continue1 value.raw

def break1 {ty : TileType} (value : Val ty) : Tyr.GPU.Codegen.TileIR.LoopResult :=
  Tyr.GPU.Codegen.TileIR.break1 value.raw

def for1 {ivTy ty : TileType}
    (lower upper stepVal : Val ivTy)
    (init : Val ty)
    (body : Val ivTy → Val ty → Tyr.GPU.Codegen.TileIR.EntryM Tyr.GPU.Codegen.TileIR.LoopResult)
    (ivHint : String := "iv")
    (hint : String := "loop")
    : Tyr.GPU.Codegen.TileIR.EntryM (Val ty) := do
  let raw ← Tyr.GPU.Codegen.TileIR.for1
    lower.raw
    upper.raw
    stepVal.raw
    init.raw
    (fun iv running => body { raw := iv } { raw := running })
    ivHint
    hint
  pure { raw }

end ct

private def strLit (s : String) : TSyntax `term :=
  ⟨Syntax.mkStrLit s⟩

private def strSyntax (s : String) : TSyntax `str :=
  ⟨Syntax.mkStrLit s⟩

private def frontendErrorAt {α} (stx : Syntax) (msg : String) : FrontendM α :=
  fun _ => Macro.throwErrorAt stx msg

private def freshTempIdent (ref : Syntax) (base : Name := `tileir_tmp) : FrontendM Ident := do
  let n ← get
  set (n + 1)
  let name ← MonadQuotation.addMacroScope (base.appendIndexAfter n)
  pure <| mkIdentFrom ref name

private def mkDoSeq (elems : Array (TSyntax `doElem)) : MacroM (TSyntax ``doSeq) :=
  `(doSeq| $[$elems:doElem]*)

private def mkBindDoElem (lhs : TSyntax `term) (rhs : TSyntax `term) : MacroM (TSyntax `doElem) := do
  let rhsElem : TSyntax `doElem ← `(doElem| $rhs:term)
  `(doElem| let $lhs:term ← $rhsElem:doElem)

private def mkI32ConstBind (id : Ident) (hint : String) (value : Nat) : MacroM (TSyntax `doElem) := do
  let ty : TSyntax `term ←
    `(Tyr.GPU.Codegen.TileIR.scalarTileTy Tyr.GPU.Codegen.TileIR.ScalarType.i32)
  let valueTerm : TSyntax `term := ⟨Syntax.mkNumLit (toString value)⟩
  let rhs : TSyntax `term ← `(Tyr.GPU.Codegen.TileIR.ct.constInt $(strLit hint) $valueTerm $ty)
  mkBindDoElem id rhs

private def wrapModernDo (body : TSyntax `term) : FrontendM (TSyntax `term) :=
  `(set_option backward.do.legacy false in $body)

private def wrapModernDoSeq (seq : TSyntax ``doSeq) : FrontendM (TSyntax `term) := do
  wrapModernDo (← `(do $seq))

private def unaryOpCtor : UnaryOp → TSyntax `term
  | .copy => mkIdent ``Tyr.GPU.Codegen.TileIR.UnaryOp.copy
  | .exp => mkIdent ``Tyr.GPU.Codegen.TileIR.UnaryOp.exp
  | .exp2 => mkIdent ``Tyr.GPU.Codegen.TileIR.UnaryOp.exp2
  | .log => mkIdent ``Tyr.GPU.Codegen.TileIR.UnaryOp.log
  | .sqrt => mkIdent ``Tyr.GPU.Codegen.TileIR.UnaryOp.sqrt
  | .rsqrt => mkIdent ``Tyr.GPU.Codegen.TileIR.UnaryOp.rsqrt
  | .abs => mkIdent ``Tyr.GPU.Codegen.TileIR.UnaryOp.abs
  | .neg => mkIdent ``Tyr.GPU.Codegen.TileIR.UnaryOp.neg

private def binaryOpCtor : BinaryOp → TSyntax `term
  | .addf => mkIdent ``Tyr.GPU.Codegen.TileIR.BinaryOp.addf
  | .subf => mkIdent ``Tyr.GPU.Codegen.TileIR.BinaryOp.subf
  | .mulf => mkIdent ``Tyr.GPU.Codegen.TileIR.BinaryOp.mulf
  | .divf => mkIdent ``Tyr.GPU.Codegen.TileIR.BinaryOp.divf
  | .maxf => mkIdent ``Tyr.GPU.Codegen.TileIR.BinaryOp.maxf
  | .minf => mkIdent ``Tyr.GPU.Codegen.TileIR.BinaryOp.minf

private def comparisonPredCtor : ComparisonPredicate → TSyntax `term
  | .equal => mkIdent ``Tyr.GPU.Codegen.TileIR.ComparisonPredicate.equal
  | .notEqual => mkIdent ``Tyr.GPU.Codegen.TileIR.ComparisonPredicate.notEqual
  | .lessThan => mkIdent ``Tyr.GPU.Codegen.TileIR.ComparisonPredicate.lessThan
  | .lessThanOrEqual => mkIdent ``Tyr.GPU.Codegen.TileIR.ComparisonPredicate.lessThanOrEqual
  | .greaterThan => mkIdent ``Tyr.GPU.Codegen.TileIR.ComparisonPredicate.greaterThan
  | .greaterThanOrEqual => mkIdent ``Tyr.GPU.Codegen.TileIR.ComparisonPredicate.greaterThanOrEqual

private def hintFromPattern (pat : TSyntax `term) : String :=
  match pat with
  | `(term| $id:ident) => id.getId.toString
  | `(term| { value := $valueId:ident, token := $_tok:ident }) => valueId.getId.toString
  | `(term| _) => "v"
  | _ => "v"

private def unpackLoopRange (stx : TSyntax `tileirLoopRange) :
    TSyntax `term × TSyntax `term × TSyntax `term :=
  (⟨stx.raw[1]⟩, ⟨stx.raw[3]⟩, ⟨stx.raw[6]⟩)

private def unpackLoopCarry (stx : TSyntax `tileirLoopCarry) :
    Ident × TSyntax `term :=
  (⟨stx.raw[1]⟩, ⟨stx.raw[3]⟩)

private def unpackIndexList (stx : TSyntax `tileirIndexList) : Array (TSyntax `term) :=
  match stx with
  | `(tileirIndexList| [$indices:term,*]) => indices.getElems
  | _ => #[]

private def unpackTermTuple (stx : TSyntax `tileirTermTuple) : Array (TSyntax `term) :=
  match stx.raw.getKind with
  | ``tileirTermTupleEmpty => #[]
  | ``tileirTermTupleSingleton => #[⟨stx.raw[1]⟩]
  | ``tileirTermTupleTerms => stx.raw[1].getSepArgs.map (fun arg => ⟨arg⟩)
  | _ => #[]

private def unpackTupleArg (stx : TSyntax `tileirTupleArg) : Name × TSyntax `tileirTermTuple :=
  match stx with
  | `(tileirTupleArg| $name:ident := $value:tileirTermTuple) => (name.getId, value)
  | _ => panic! "invalid tileirTupleArg syntax"

private def unpackTermArg (stx : TSyntax `tileirTermArg) : Name × TSyntax `term :=
  match stx with
  | `(tileirTermArg| $name:ident := $value:term) => (name.getId, value)
  | _ => panic! "invalid tileirTermArg syntax"

mutual

private partial def ensureValueTerm
    (stx : TSyntax `term)
    : FrontendM ((TSyntax `term) × Array (TSyntax `doElem)) := do
  match stx with
  | `(term| ($inner)) =>
      ensureValueTerm inner
  | `(term| $id:ident) =>
      pure (id, #[])
  | _ =>
      match ← expandTileExpr? stx "_tmp" with
      | some action =>
          let tmp ← freshTempIdent stx
          let bind ← liftM <| mkBindDoElem tmp action
          pure (tmp, #[bind])
      | none =>
          pure (stx, #[])

private partial def wrapWithPrep
    (prep : Array (TSyntax `doElem))
    (body : TSyntax `term)
    : FrontendM (TSyntax `term) := do
  if prep.isEmpty then
    pure body
  else
    let bodyElem : TSyntax `doElem ← `(doElem| $body:term)
    let seq ← liftM <| mkDoSeq (prep.push bodyElem)
    wrapModernDoSeq seq

private partial def ensureValueTerms
    (terms : Array (TSyntax `term))
    : FrontendM ((Array (TSyntax `term)) × Array (TSyntax `doElem)) := do
  let mut values : Array (TSyntax `term) := #[]
  let mut prep : Array (TSyntax `doElem) := #[]
  for term in terms do
    let (value, valuePrep) ← ensureValueTerm term
    values := values.push value
    prep := prep ++ valuePrep
  pure (values, prep)

private partial def expandUnaryExpr
    (_hint : String)
    (op : UnaryOp)
    (src : TSyntax `term)
    : FrontendM (TSyntax `term) := do
  let (src, prep) ← ensureValueTerm src
  let body ←
    match op with
    | .copy => `(Tyr.GPU.Codegen.TileIR.ct.copy $src)
    | .exp => `(Tyr.GPU.Codegen.TileIR.ct.exp $src)
    | .exp2 => `(Tyr.GPU.Codegen.TileIR.ct.exp2 $src)
    | .log => `(Tyr.GPU.Codegen.TileIR.ct.log $src)
    | .sqrt => `(Tyr.GPU.Codegen.TileIR.ct.sqrt $src)
    | .rsqrt => `(Tyr.GPU.Codegen.TileIR.ct.rsqrt $src)
    | .abs => `(Tyr.GPU.Codegen.TileIR.ct.abs $src)
    | .neg => `(Tyr.GPU.Codegen.TileIR.ct.neg $src)
  wrapWithPrep prep body

private partial def expandBinaryExpr
    (_hint : String)
    (op : BinaryOp)
    (lhs rhs : TSyntax `term)
    : FrontendM (TSyntax `term) := do
  let (lhs, lhsPrep) ← ensureValueTerm lhs
  let (rhs, rhsPrep) ← ensureValueTerm rhs
      let body ←
        match op with
        | .addf => `(HAdd.hAdd $lhs $rhs)
        | .subf => `(HSub.hSub $lhs $rhs)
        | .mulf => `(HMul.hMul $lhs $rhs)
        | .divf => `(HDiv.hDiv $lhs $rhs)
        | .maxf => `(Tyr.GPU.Codegen.TileIR.ct.max $lhs $rhs)
        | .minf => `(Tyr.GPU.Codegen.TileIR.ct.min $lhs $rhs)
  wrapWithPrep (lhsPrep ++ rhsPrep) body

private partial def expandFillExpr
    (hint : String)
    (shape : TSyntax `tileirTermTuple)
    (value dtype : TSyntax `term)
    : FrontendM (TSyntax `term) := do
  let shapeTerms := unpackTermTuple shape
  let body ← `(Tyr.GPU.Codegen.TileIR.ct.full $(strLit hint) #[$shapeTerms,*] $value $dtype)
  pure body

private partial def expandExtractExpr
    (src : TSyntax `term)
    (indices : Array (TSyntax `term))
    (shape : Array (TSyntax `term))
    : FrontendM (TSyntax `term) := do
  let (src, srcPrep) ← ensureValueTerm src
  let (indices, indexPrep) ← ensureValueTerms indices
  let body ← `(Tyr.GPU.Codegen.TileIR.ct.extract $src #[$indices,*] #[$shape,*])
  wrapWithPrep (srcPrep ++ indexPrep) body

private partial def expandTileExpr?
    (stx : TSyntax `term)
    (hint : String)
    : FrontendM (Option (TSyntax `term)) := withRef stx do
  if stx.raw.getKind == ``ctLoadIndexedTerm then
    let `(ctLoadIndexedTerm| ct.load $ptr:term, $arg1, $arg2) := stx
      | Macro.throwErrorAt stx "invalid indexed ct.load syntax"
    let arg1 : TSyntax `tileirTupleArg := ⟨arg1.raw⟩
    let arg2 : TSyntax `tileirTupleArg := ⟨arg2.raw⟩
    let (name1, value1) := unpackTupleArg arg1
    let (name2, value2) := unpackTupleArg arg2
    let (indicesTuple, shapeTuple) ←
      if name1 == `index && name2 == `shape then
        pure (value1, value2)
      else if name1 == `shape && name2 == `index then
        pure (value2, value1)
      else
        frontendErrorAt stx "indexed ct.load expects exactly `index := ...` and `shape := ...` arguments"
    let (ptr, ptrPrep) ← ensureValueTerm ptr
    let (indices, indexPrep) ← ensureValueTerms (unpackTermTuple indicesTuple)
    let shapeTerms := unpackTermTuple shapeTuple
    let body ← `(Tyr.GPU.Codegen.TileIR.ct.loadIndexed $(strLit hint) $ptr #[$indices,*] #[$shapeTerms,*])
    pure <| some (← wrapWithPrep (ptrPrep ++ indexPrep) body)
  else if stx.raw.getKind == ``ctLoadIndexedScalarIndexTerm then
    let `(ctLoadIndexedScalarIndexTerm| ct.load $ptr:term, $name1:ident := $index:term, $name2:ident := $shape:tileirTermTuple) := stx
      | Macro.throwErrorAt stx "invalid scalar-index ct.load syntax"
    unless name1.getId == `index && name2.getId == `shape do
      frontendErrorAt stx "scalar-index ct.load expects `index := ...` followed by `shape := ...`"
    let (ptr, ptrPrep) ← ensureValueTerm ptr
    let (index, indexPrep) ← ensureValueTerm index
    let shapeTerms := unpackTermTuple shape
    let body ← `(Tyr.GPU.Codegen.TileIR.ct.loadIndexed $(strLit hint) $ptr #[$index] #[$shapeTerms,*])
    pure <| some (← wrapWithPrep (ptrPrep ++ indexPrep) body)
  else if stx.raw.getKind == ``ctLoadPositionalTerm then
    let `(ctLoadPositionalTerm| ct.load($ptr:term, $indices:tileirTermTuple, $shape:tileirTermTuple)) := stx
      | Macro.throwErrorAt stx "invalid positional ct.load syntax"
    let (ptr, ptrPrep) ← ensureValueTerm ptr
    let (indices, indexPrep) ← ensureValueTerms (unpackTermTuple indices)
    let shapeTerms := unpackTermTuple shape
    let body ← `(Tyr.GPU.Codegen.TileIR.ct.loadIndexed $(strLit hint) $ptr #[$indices,*] #[$shapeTerms,*])
    pure <| some (← wrapWithPrep (ptrPrep ++ indexPrep) body)
  else if stx.raw.getKind == ``ctStoreIndexedTerm then
    let `(ctStoreIndexedTerm| ct.store $ptr:term, $arg1, $arg2) := stx
      | Macro.throwErrorAt stx "invalid indexed ct.store syntax"
    let arg1 : TSyntax `tileirTupleArg := ⟨arg1.raw⟩
    let arg2 : TSyntax `tileirTermArg := ⟨arg2.raw⟩
    let (tupleName, indicesTuple) := unpackTupleArg arg1
    let (valueName, value) := unpackTermArg arg2
    if tupleName != `index || valueName != `tile then
      frontendErrorAt stx "indexed ct.store expects `index := ...` and `tile := ...` arguments"
    let (ptr, ptrPrep) ← ensureValueTerm ptr
    let (indices, indexPrep) ← ensureValueTerms (unpackTermTuple indicesTuple)
    let (value, valuePrep) ← ensureValueTerm value
    let body ← `(Tyr.GPU.Codegen.TileIR.ct.storeIndexed $ptr #[$indices,*] $value)
    pure <| some (← wrapWithPrep (ptrPrep ++ indexPrep ++ valuePrep) body)
  else if stx.raw.getKind == ``ctStoreIndexedScalarIndexTerm then
    let `(ctStoreIndexedScalarIndexTerm| ct.store $ptr:term, $name1:ident := $index:term, $name2:ident := $value:term) := stx
      | Macro.throwErrorAt stx "invalid scalar-index ct.store syntax"
    unless name1.getId == `index && name2.getId == `tile do
      frontendErrorAt stx "scalar-index ct.store expects `index := ...` followed by `tile := ...`"
    let (ptr, ptrPrep) ← ensureValueTerm ptr
    let (index, indexPrep) ← ensureValueTerm index
    let (value, valuePrep) ← ensureValueTerm value
    let body ← `(Tyr.GPU.Codegen.TileIR.ct.storeIndexed $ptr #[$index] $value)
    pure <| some (← wrapWithPrep (ptrPrep ++ indexPrep ++ valuePrep) body)
  else if stx.raw.getKind == ``ctStorePositionalTerm then
    let `(ctStorePositionalTerm| ct.store($ptr:term, $indices:tileirTermTuple, $value:term)) := stx
      | Macro.throwErrorAt stx "invalid positional ct.store syntax"
    let (ptr, ptrPrep) ← ensureValueTerm ptr
    let (indices, indexPrep) ← ensureValueTerms (unpackTermTuple indices)
    let (value, valuePrep) ← ensureValueTerm value
    let body ← `(Tyr.GPU.Codegen.TileIR.ct.storeIndexed $ptr #[$indices,*] $value)
    pure <| some (← wrapWithPrep (ptrPrep ++ indexPrep ++ valuePrep) body)
  else match stx with
  | `(tileirConstIntTerm| const_int $value:term : $ty:term) =>
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.constInt $(strLit hint) $value $ty))
  | `(tileirConstFloatTerm| const_float $value:term : $ty:term) =>
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.constFloat $(strLit hint) $value $ty))
  | `(tileirConstBoolTerm| const_bool $value:term : $ty:term) =>
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.constBool $(strLit hint) $value $ty))
  | `(ctCommentTerm| ct.comment $msg:str) =>
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.comment $msg))
  | `(ctBidTerm| ct.bid $axis:term) =>
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.bid $axis))
  | `(ctNumBlocksTerm| ct.num_blocks $axis:term) =>
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.numBlocks $axis))
  | `(ctIotaTerm| ct.iota : $ty:term) =>
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.iotaAs $(strLit hint) $ty))
  | `(ctArangeTerm| ct.arange($size:term)) =>
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.arange $(strLit hint) $size))
  | `(ctArangeDtypeTerm| ct.arange($size:term, $argName:ident := $dtype:term)) =>
      if argName.getId != `dtype then
        frontendErrorAt stx "ct.arange only supports a `dtype := ...` keyword argument"
      else
        pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.arange $(strLit hint) $size $dtype))
  | `(ctLoadTypedTerm| ct.load $ptr:term : $ty:term) =>
      let (ptr, prep) ← ensureValueTerm ptr
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.loadAs $(strLit hint) $ptr $ty)
      pure <| some (← wrapWithPrep prep body)
  | `(ctGatherTerm| ct.gather($base:term, $idx:term)) =>
      let (base, basePrep) ← ensureValueTerm base
      let (idx, idxPrep) ← ensureValueTerm idx
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.gather $(strLit hint) $base $idx)
      pure <| some (← wrapWithPrep (basePrep ++ idxPrep) body)
  | `(ctLoadViewTerm| ct.load_view $view:term, $indices:tileirIndexList : $ty:term) =>
      let (view, viewPrep) ← ensureValueTerm view
      let (indices, indexPrep) ← ensureValueTerms (unpackIndexList indices)
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.loadViewAs $(strLit hint) $view #[$indices,*] $ty)
      pure <| some (← wrapWithPrep (viewPrep ++ indexPrep) body)
  | `(ctLoadGlobalTerm| ct.load_global $name:ident : $ty:term) =>
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.loadGlobalAs $(strLit hint) $(strLit name.getId.toString) $ty))
  | `(ctFullKwTerm| ct.full($shape:tileirTermTuple, $value:term, $argName:ident := $dtype:term)) =>
      if argName.getId != `dtype then
        frontendErrorAt stx "ct.full only supports a `dtype := ...` keyword argument"
      else
        pure <| some (← expandFillExpr hint shape value dtype)
  | `(ctFullPosTerm| ct.full($shape:tileirTermTuple, $value:term, $dtype:term)) =>
      pure <| some (← expandFillExpr hint shape value dtype)
  | `(ctFullKwShapeTerm| ct.full($shape:term, $value:term, $argName:ident := $dtype:term)) =>
      if argName.getId != `dtype then
        frontendErrorAt stx "ct.full only supports a `dtype := ...` keyword argument"
      else
        pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.full $(strLit hint) $shape $value $dtype))
  | `(ctFullPosShapeTerm| ct.full($shape:term, $value:term, $dtype:term)) =>
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.full $(strLit hint) $shape $value $dtype))
  | `(ctFullLikeKwTerm| ct.full_like($like:term, $value:term, $argName:ident := $dtype:term)) =>
      if argName.getId != `dtype then
        frontendErrorAt stx "ct.full_like only supports a `dtype := ...` keyword argument"
      else
        let (like, prep) ← ensureValueTerm like
        let body ← `(Tyr.GPU.Codegen.TileIR.ct.fullLike $(strLit hint) $like $value $dtype)
        pure <| some (← wrapWithPrep prep body)
  | `(ctFullLikePosTerm| ct.full_like($like:term, $value:term, $dtype:term)) =>
      let (like, prep) ← ensureValueTerm like
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.fullLike $(strLit hint) $like $value $dtype)
      pure <| some (← wrapWithPrep prep body)
  | `(ctZerosKwTerm| ct.zeros($shape:tileirTermTuple, $argName:ident := $dtype:term)) =>
      if argName.getId != `dtype then
        frontendErrorAt stx "ct.zeros only supports a `dtype := ...` keyword argument"
      else
        let shapeTerms := unpackTermTuple shape
        pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.zeros $(strLit hint) #[$shapeTerms,*] $dtype))
  | `(ctZerosPosTerm| ct.zeros($shape:tileirTermTuple, $dtype:term)) =>
      let shapeTerms := unpackTermTuple shape
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.zeros $(strLit hint) #[$shapeTerms,*] $dtype))
  | `(ctZerosKwShapeTerm| ct.zeros($shape:term, $argName:ident := $dtype:term)) =>
      if argName.getId != `dtype then
        frontendErrorAt stx "ct.zeros only supports a `dtype := ...` keyword argument"
      else
        pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.zeros $(strLit hint) $shape $dtype))
  | `(ctZerosPosShapeTerm| ct.zeros($shape:term, $dtype:term)) =>
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.zeros $(strLit hint) $shape $dtype))
  | `(ctZerosLikeTerm| ct.zeros_like($like:term)) =>
      let (like, prep) ← ensureValueTerm like
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.zerosLike $(strLit hint) $like)
      pure <| some (← wrapWithPrep prep body)
  | `(ctOnesKwTerm| ct.ones($shape:tileirTermTuple, $argName:ident := $dtype:term)) =>
      if argName.getId != `dtype then
        frontendErrorAt stx "ct.ones only supports a `dtype := ...` keyword argument"
      else
        let shapeTerms := unpackTermTuple shape
        pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.ones $(strLit hint) #[$shapeTerms,*] $dtype))
  | `(ctOnesPosTerm| ct.ones($shape:tileirTermTuple, $dtype:term)) =>
      let shapeTerms := unpackTermTuple shape
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.ones $(strLit hint) #[$shapeTerms,*] $dtype))
  | `(ctOnesKwShapeTerm| ct.ones($shape:term, $argName:ident := $dtype:term)) =>
      if argName.getId != `dtype then
        frontendErrorAt stx "ct.ones only supports a `dtype := ...` keyword argument"
      else
        pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.ones $(strLit hint) $shape $dtype))
  | `(ctOnesPosShapeTerm| ct.ones($shape:term, $dtype:term)) =>
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.ones $(strLit hint) $shape $dtype))
  | `(ctOnesLikeTerm| ct.ones_like($like:term)) =>
      let (like, prep) ← ensureValueTerm like
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.onesLike $(strLit hint) $like)
      pure <| some (← wrapWithPrep prep body)
  | `(ctAsTypeTerm| ct.astype($src:term, $dtype:term)) =>
      let (src, prep) ← ensureValueTerm src
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.astype $src $dtype)
      pure <| some (← wrapWithPrep prep body)
  | `(ctCatTerm| ct.cat($tiles:tileirTermTuple, $dim:term)) =>
      let tileTerms := unpackTermTuple tiles
      unless tileTerms.size == 2 do
        frontendErrorAt stx "ct.cat expects exactly two tile operands"
      let (tiles, prep) ← ensureValueTerms tileTerms
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.cat $(strLit hint) $(tiles[0]!) $(tiles[1]!) $dim)
      pure <| some (← wrapWithPrep prep body)
  | `(ctStaticEvalTerm| ct.static_eval $e:term) =>
      expandTileExpr? e hint
  | `(tileirLoadPtrTkoTerm| load_ptr_tko $ptr:term, $tok:term : $ty:term) =>
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.loadPtrTko $(strLit hint) $ptr $tok $ty))
  | `(tileirGetGlobalTerm| get_global $name:ident : $ty:term) =>
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.getGlobal $(strLit hint) $(strLit name.getId.toString) $ty))
  | `(ctBroadcastTerm| ct.broadcast $src:term : $ty:term) =>
      let (src, prep) ← ensureValueTerm src
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.broadcastAs $(strLit hint) $src $ty)
      pure <| some (← wrapWithPrep prep body)
  | `(ctBroadcastToTerm| ct.broadcast_to($src:term, $shape:tileirTermTuple)) =>
      let (src, prep) ← ensureValueTerm src
      let shapeTerms := unpackTermTuple shape
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.broadcastTo $(strLit hint) $src #[$shapeTerms,*])
      pure <| some (← wrapWithPrep prep body)
  | `(ctBroadcastToShapeTerm| ct.broadcast_to($src:term, $shape:term)) =>
      let (src, prep) ← ensureValueTerm src
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.broadcastTo $(strLit hint) $src $shape)
      pure <| some (← wrapWithPrep prep body)
  | `(tileirBroadcastTerm| broadcast $src:term : $ty:term) =>
      let (src, prep) ← ensureValueTerm src
      let body ← `(Tyr.GPU.Codegen.TileIR.broadcast $(strLit hint) $src $ty)
      pure <| some (← wrapWithPrep prep body)
  | `(ctReshapeTerm| ct.reshape $src:term : $ty:term) =>
      let (src, prep) ← ensureValueTerm src
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.reshapeAs $(strLit hint) $src $ty)
      pure <| some (← wrapWithPrep prep body)
  | `(ctReshapeShapeTerm| ct.reshape($src:term, $shape:tileirTermTuple)) =>
      let (src, prep) ← ensureValueTerm src
      let shapeTerms := unpackTermTuple shape
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.reshapeVal $src #[$shapeTerms,*])
      pure <| some (← wrapWithPrep prep body)
  | `(ctPermuteTerm| ct.permute($src:term, $perm:tileirTermTuple)) =>
      let (src, prep) ← ensureValueTerm src
      let permTerms := unpackTermTuple perm
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.permute $src #[$permTerms,*])
      pure <| some (← wrapWithPrep prep body)
  | `(ctTransposeTerm| ct.transpose($src:term)) =>
      let (src, prep) ← ensureValueTerm src
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.transpose $src)
      pure <| some (← wrapWithPrep prep body)
  | `(ctExtractIndexedTerm| ct.extract $src:term, $arg1, $arg2) =>
      let arg1 : TSyntax `tileirTupleArg := ⟨arg1.raw⟩
      let arg2 : TSyntax `tileirTupleArg := ⟨arg2.raw⟩
      let (name1, value1) := unpackTupleArg arg1
      let (name2, value2) := unpackTupleArg arg2
      let (indicesTuple, shapeTuple) ←
        if name1 == `index && name2 == `shape then
          pure (value1, value2)
        else if name1 == `shape && name2 == `index then
          pure (value2, value1)
        else
          frontendErrorAt stx "ct.extract expects exactly `index := ...` and `shape := ...` arguments"
      pure <| some (← expandExtractExpr src (unpackTermTuple indicesTuple) (unpackTermTuple shapeTuple))
  | `(ctExtractPositionalTerm| ct.extract($src:term, $indices:tileirTermTuple, $shape:tileirTermTuple)) =>
      pure <| some (← expandExtractExpr src (unpackTermTuple indices) (unpackTermTuple shape))
  | `(tileirReshapeTerm| reshape $src:term : $ty:term) =>
      let (src, prep) ← ensureValueTerm src
      let body ← `(Tyr.GPU.Codegen.TileIR.reshape $(strLit hint) $src $ty)
      pure <| some (← wrapWithPrep prep body)
  | `(ctOffsetTerm| ct.offset $ptr:term, $idx:term : $ty:term) =>
      let (ptr, ptrPrep) ← ensureValueTerm ptr
      let (idx, idxPrep) ← ensureValueTerm idx
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.offsetAs $(strLit hint) $ptr $idx $ty)
      pure <| some (← wrapWithPrep (ptrPrep ++ idxPrep) body)
  | `(tileirOffsetTerm| offset $ptr:term, $idx:term : $ty:term) =>
      let (ptr, ptrPrep) ← ensureValueTerm ptr
      let (idx, idxPrep) ← ensureValueTerm idx
      let body ← `(Tyr.GPU.Codegen.TileIR.offset $(strLit hint) $ptr $idx $ty)
      pure <| some (← wrapWithPrep (ptrPrep ++ idxPrep) body)
  | `(ctTensorViewTerm| ct.tensor_view $base:term : $desc:term) =>
      let (base, prep) ← ensureValueTerm base
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.tensorViewAs $(strLit hint) $base $desc)
      pure <| some (← wrapWithPrep prep body)
  | `(tileirMakeTensorViewTerm| make_tensor_view $base:term : $desc:term) =>
      let (base, prep) ← ensureValueTerm base
      let body ← `(Tyr.GPU.Codegen.TileIR.makeTensorView $(strLit hint) $base $desc)
      pure <| some (← wrapWithPrep prep body)
  | `(ctPartitionViewTerm| ct.partition_view $src:term : $desc:term) =>
      let (src, prep) ← ensureValueTerm src
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.partitionViewAs $(strLit hint) $src $desc)
      pure <| some (← wrapWithPrep prep body)
  | `(tileirMakePartitionViewTerm| make_partition_view $src:term : $desc:term) =>
      let (src, prep) ← ensureValueTerm src
      let body ← `(Tyr.GPU.Codegen.TileIR.makePartitionView $(strLit hint) $src $desc)
      pure <| some (← wrapWithPrep prep body)
  | `(ctMmaTerm| ct.mma $a:term, $b:term, $c:term) =>
      let (a, aPrep) ← ensureValueTerm a
      let (b, bPrep) ← ensureValueTerm b
      let (c, cPrep) ← ensureValueTerm c
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.mma $(strLit hint) $a $b $c)
      pure <| some (← wrapWithPrep (aPrep ++ bPrep ++ cPrep) body)
  | `(ctMatmulTerm| ct.matmul $a:term, $b:term, $c:term) =>
      let (a, aPrep) ← ensureValueTerm a
      let (b, bPrep) ← ensureValueTerm b
      let (c, cPrep) ← ensureValueTerm c
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.matmul $a $b $c)
      pure <| some (← wrapWithPrep (aPrep ++ bPrep ++ cPrep) body)
  | `(tileirMmafTerm| mmaf $a:term, $b:term, $c:term) =>
      let (a, aPrep) ← ensureValueTerm a
      let (b, bPrep) ← ensureValueTerm b
      let (c, cPrep) ← ensureValueTerm c
      let body ← `(Tyr.GPU.Codegen.TileIR.mmaf $(strLit hint) $a $b $c)
      pure <| some (← wrapWithPrep (aPrep ++ bPrep ++ cPrep) body)
  | `(ctMmaiTerm| ct.mmai $a:term, $b:term, $c:term) =>
      let (a, aPrep) ← ensureValueTerm a
      let (b, bPrep) ← ensureValueTerm b
      let (c, cPrep) ← ensureValueTerm c
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.mmaiVal $(strLit hint) $a $b $c)
      pure <| some (← wrapWithPrep (aPrep ++ bPrep ++ cPrep) body)
  | `(tileirMmaiTerm| mmai $a:term, $b:term, $c:term) =>
      let (a, aPrep) ← ensureValueTerm a
      let (b, bPrep) ← ensureValueTerm b
      let (c, cPrep) ← ensureValueTerm c
      let body ← `(Tyr.GPU.Codegen.TileIR.mmai $(strLit hint) $a $b $c)
      pure <| some (← wrapWithPrep (aPrep ++ bPrep ++ cPrep) body)
  | `(ctMaxTerm| ct.max $lhs:term, $rhs:term) =>
      pure <| some (← expandBinaryExpr hint .maxf lhs rhs)
  | `(ctMaximumTerm| ct.maximum $lhs:term, $rhs:term) =>
      pure <| some (← expandBinaryExpr hint .maxf lhs rhs)
  | `(tileirMaxfTerm| maxf $lhs:term, $rhs:term) =>
      pure <| some (← expandBinaryExpr hint .maxf lhs rhs)
  | `(ctMinTerm| ct.min $lhs:term, $rhs:term) =>
      pure <| some (← expandBinaryExpr hint .minf lhs rhs)
  | `(ctMinimumTerm| ct.minimum $lhs:term, $rhs:term) =>
      pure <| some (← expandBinaryExpr hint .minf lhs rhs)
  | `(tileirMinfTerm| minf $lhs:term, $rhs:term) =>
      pure <| some (← expandBinaryExpr hint .minf lhs rhs)
  | `(ctAssertTerm| ct.assert $cond:term, $msg:str) =>
      let (cond, prep) ← ensureValueTerm cond
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.assert $cond $msg)
      pure <| some (← wrapWithPrep prep body)
  | `(ctStaticAssertTerm| ct.static_assert $cond:term) =>
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.staticAssert $cond))
  | `(ctStaticAssertMsgTerm| ct.static_assert $cond:term, $msg:term) =>
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.staticAssert $cond (fun _ => toString $msg)))
  | `(tileirFor1Term| for_tile $iv:ident in $range:tileirLoopRange carrying $carrySpec:tileirLoopCarry do $body:doSeq) =>
      let (lower, upper, stepVal) := unpackLoopRange range
      let (carry, init) := unpackLoopCarry carrySpec
      let (lower, lowerPrep) ← ensureValueTerm lower
      let (upper, upperPrep) ← ensureValueTerm upper
      let (stepVal, stepPrep) ← ensureValueTerm stepVal
      let (init, initPrep) ← ensureValueTerm init
      let loopBody ← wrapModernDoSeq body
      let loop ←
        `(Tyr.GPU.Codegen.TileIR.ct.for1
            $lower
            $upper
            $stepVal
            $init
            (fun $iv:ident => fun $carry:ident => $loopBody)
            $(strLit iv.getId.toString)
            $(strLit hint))
      pure <| some (← wrapWithPrep (lowerPrep ++ upperPrep ++ stepPrep ++ initPrep) loop)
  | `(tileirPrintTkoTerm| print_tko $msg:str) =>
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.printTko $msg))
  | `(ctStoreTerm| ct.store $ptr:term, $value:term) =>
      let (ptr, ptrPrep) ← ensureValueTerm ptr
      let (value, valuePrep) ← ensureValueTerm value
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.store $ptr $value)
      pure <| some (← wrapWithPrep (ptrPrep ++ valuePrep) body)
  | `(ctScatterTerm| ct.scatter($base:term, $idx:term, $value:term)) =>
      let (base, basePrep) ← ensureValueTerm base
      let (idx, idxPrep) ← ensureValueTerm idx
      let (value, valuePrep) ← ensureValueTerm value
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.scatter $base $idx $value)
      pure <| some (← wrapWithPrep (basePrep ++ idxPrep ++ valuePrep) body)
  | `(ctStoreViewTerm| ct.store_view $view:term, $indices:tileirIndexList, $value:term) =>
      let (view, viewPrep) ← ensureValueTerm view
      let (indices, indexPrep) ← ensureValueTerms (unpackIndexList indices)
      let (value, valuePrep) ← ensureValueTerm value
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.storeView $view #[$indices,*] $value)
      pure <| some (← wrapWithPrep (viewPrep ++ indexPrep ++ valuePrep) body)
  | `(tileirStorePtrTkoTerm| store_ptr_tko $ptr:term, $value:term, $tok:term) =>
      let (ptr, ptrPrep) ← ensureValueTerm ptr
      let (value, valuePrep) ← ensureValueTerm value
      let (tok, tokPrep) ← ensureValueTerm tok
      let body ← `(Tyr.GPU.Codegen.TileIR.storePtrTko $ptr $value $tok)
      pure <| some (← wrapWithPrep (ptrPrep ++ valuePrep ++ tokPrep) body)
  | `(ctPrintTerm| ct.print $msg:str) =>
      pure <| some (← `(Tyr.GPU.Codegen.TileIR.ct.print $msg))
  | `(tileirContinueTerm| continue_tile $value:term) =>
      let (value, prep) ← ensureValueTerm value
      let body ← `(pure (Tyr.GPU.Codegen.TileIR.ct.continue1 $value))
      pure <| some (← wrapWithPrep prep body)
  | `(tileirBreakTerm| break_tile $value:term) =>
      let (value, prep) ← ensureValueTerm value
      let body ← `(pure (Tyr.GPU.Codegen.TileIR.ct.break1 $value))
      pure <| some (← wrapWithPrep prep body)
  | `(term| if $cond then do $thenSeq else do $elseSeq) =>
      let (cond, condPrep) ← ensureValueTerm cond
      let thenBranch ← wrapModernDoSeq thenSeq
      let elseBranch ← wrapModernDoSeq elseSeq
      let body ← `(Tyr.GPU.Codegen.TileIR.ct.if1 $cond $thenBranch $elseBranch $(strLit hint))
      pure <| some (← wrapWithPrep condPrep body)
  | `(term| if $cond then $thenTerm else $elseTerm) =>
      pure <| some (← `(if $cond then $thenTerm else $elseTerm))
  | `(term| exp $src) =>
      pure <| some (← expandUnaryExpr hint .exp src)
  | `(ctExpTerm| ct.exp $src:term) =>
      pure <| some (← expandUnaryExpr hint .exp src)
  | `(term| exp2 $src) =>
      pure <| some (← expandUnaryExpr hint .exp2 src)
  | `(ctExp2Term| ct.exp2 $src:term) =>
      pure <| some (← expandUnaryExpr hint .exp2 src)
  | `(term| log $src) =>
      pure <| some (← expandUnaryExpr hint .log src)
  | `(ctLogTerm| ct.log $src:term) =>
      pure <| some (← expandUnaryExpr hint .log src)
  | `(term| sqrt $src) =>
      pure <| some (← expandUnaryExpr hint .sqrt src)
  | `(ctSqrtTerm| ct.sqrt $src:term) =>
      pure <| some (← expandUnaryExpr hint .sqrt src)
  | `(term| rsqrt $src) =>
      pure <| some (← expandUnaryExpr hint .rsqrt src)
  | `(ctRsqrtTerm| ct.rsqrt $src:term) =>
      pure <| some (← expandUnaryExpr hint .rsqrt src)
  | `(term| abs $src) =>
      pure <| some (← expandUnaryExpr hint .abs src)
  | `(ctAbsTerm| ct.abs $src:term) =>
      pure <| some (← expandUnaryExpr hint .abs src)
  | `(term| copy $src) =>
      pure <| some (← expandUnaryExpr hint .copy src)
  | `(term| neg $src) =>
      pure <| some (← expandUnaryExpr hint .neg src)
  | `(ctNegTerm| ct.neg $src:term) =>
      pure <| some (← expandUnaryExpr hint .neg src)
  | `(term| $lhs + $rhs) =>
      pure <| some (← expandBinaryExpr hint .addf lhs rhs)
  | `(term| $lhs - $rhs) =>
      pure <| some (← expandBinaryExpr hint .subf lhs rhs)
  | `(term| $lhs * $rhs) =>
      pure <| some (← expandBinaryExpr hint .mulf lhs rhs)
  | `(term| $lhs / $rhs) =>
      pure <| some (← expandBinaryExpr hint .divf lhs rhs)
  | _ =>
      pure none

end

private def expandCtRange
    (stx : TSyntax `term)
    : FrontendM ((TSyntax `term × TSyntax `term × TSyntax `term) × Array (TSyntax `doElem)) := do
  let kind := stx.raw.getKind
  if kind == ``ctRange1Term then
    let upper : TSyntax `term := ⟨stx.raw[2]⟩
    let (upper, upperPrep) ← ensureValueTerm upper
    let lowerId ← freshTempIdent stx `tileir_range_lower
    let stepId ← freshTempIdent stx `tileir_range_step
    let lowerBind ← liftM <| mkI32ConstBind lowerId "range_lower" 0
    let stepBind ← liftM <| mkI32ConstBind stepId "range_step" 1
    pure ((lowerId, upper, stepId), #[lowerBind, stepBind] ++ upperPrep)
  else if kind == ``ctRange2Term then
    let lower : TSyntax `term := ⟨stx.raw[2]⟩
    let upper : TSyntax `term := ⟨stx.raw[4]⟩
    let (lower, lowerPrep) ← ensureValueTerm lower
    let (upper, upperPrep) ← ensureValueTerm upper
    let stepId ← freshTempIdent stx `tileir_range_step
    let stepBind ← liftM <| mkI32ConstBind stepId "range_step" 1
    pure ((lower, upper, stepId), lowerPrep ++ upperPrep ++ #[stepBind])
  else if kind == ``ctRange3Term then
    let lower : TSyntax `term := ⟨stx.raw[2]⟩
    let upper : TSyntax `term := ⟨stx.raw[4]⟩
    let stepVal : TSyntax `term := ⟨stx.raw[6]⟩
    let (lower, lowerPrep) ← ensureValueTerm lower
    let (upper, upperPrep) ← ensureValueTerm upper
    let (stepVal, stepPrep) ← ensureValueTerm stepVal
    pure ((lower, upper, stepVal), lowerPrep ++ upperPrep ++ stepPrep)
  else
    frontendErrorAt stx "TileIR `for` loops expect `ct.range(stop)`, `ct.range(start, stop)`, or `ct.range(start, stop, step)`"

private def isCtRuntimeRangeSyntax (stx : TSyntax `term) : Bool :=
  let kind := stx.raw.getKind
  kind == ``ctRange1Term || kind == ``ctRange2Term || kind == ``ctRange3Term

private def reduceExprFully (e : Expr) : TermElabM Expr := do
  instantiateMVars (← Lean.Meta.reduceAll (← instantiateMVars e))

private def decodeNatExpr? (e : Expr) : TermElabM (Option Nat) := do
  pure (← reduceExprFully e).rawNatLit?

private partial def decodeListExpr?
    (decodeElem : Expr → TermElabM (Option α))
    (e : Expr)
    : TermElabM (Option (List α)) := do
  let e ← reduceExprFully e
  let fn := e.getAppFn
  let args := e.getAppArgs
  if fn.isConstOf ``List.nil then
    pure (some [])
  else if fn.isConstOf ``List.cons then
    if args.size == 3 then
      let some head := ← decodeElem args[1]!
        | pure none
      let some tail := ← decodeListExpr? decodeElem args[2]!
        | pure none
      pure (some (head :: tail))
    else
      pure none
  else
    pure none

private partial def decodeArrayExpr?
    (decodeElem : Expr → TermElabM (Option α))
    (e : Expr)
    : TermElabM (Option (_root_.Array α)) := do
  let e ← reduceExprFully e
  let fn := e.getAppFn
  let args := e.getAppArgs
  if fn.isConstOf ``Array.mk || fn.isConstOf ``List.toArray then
    let listArg := args[1]!
    return (← decodeListExpr? decodeElem listArg).map List.toArray
  else
    pure none

private def decodeScalarTypeExpr? (e : Expr) : TermElabM (Option ScalarType) := do
  let e ← reduceExprFully e
  let fn := e.getAppFn
  let args := e.getAppArgs
  if !args.isEmpty then
    pure none
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ScalarType.i1 then
    pure (some .i1)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ScalarType.i8 then
    pure (some .i8)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ScalarType.i16 then
    pure (some .i16)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ScalarType.i32 then
    pure (some .i32)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ScalarType.i64 then
    pure (some .i64)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ScalarType.u8 then
    pure (some .u8)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ScalarType.u16 then
    pure (some .u16)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ScalarType.u32 then
    pure (some .u32)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ScalarType.u64 then
    pure (some .u64)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ScalarType.f16 then
    pure (some .f16)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ScalarType.bf16 then
    pure (some .bf16)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ScalarType.f32 then
    pure (some .f32)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ScalarType.f64 then
    pure (some .f64)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ScalarType.index then
    pure (some .index)
  else
    pure none

private def decodeShapeDimExpr? (e : Expr) : TermElabM (Option ShapeDim) := do
  let e ← reduceExprFully e
  let fn := e.getAppFn
  let args := e.getAppArgs
  if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ShapeDim.dynamic then
    pure (some .dynamic)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ShapeDim.static then
    let some value := ← decodeNatExpr? args[0]!
      | pure none
    pure (some (.static value))
  else
    pure none

private def decodeElemTypeExpr? (e : Expr) : TermElabM (Option ElemType) := do
  let e ← reduceExprFully e
  let fn := e.getAppFn
  let args := e.getAppArgs
  if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ElemType.scalar then
    let some ty := ← decodeScalarTypeExpr? args[0]!
      | pure none
    pure (some (.scalar ty))
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ElemType.ptr then
    let some ty := ← decodeScalarTypeExpr? args[0]!
      | pure none
    pure (some (.ptr ty))
  else
    pure none

private partial def decodeTileTypeExpr? (e : Expr) : TermElabM (Option TileType) := do
  let e ← reduceExprFully e
  let fn := e.getAppFn
  let args := e.getAppArgs
  if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ct.Scalar then
    let some elem := ← decodeScalarTypeExpr? args[0]!
      | pure none
    pure (some <| scalarTileTy elem)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ct.Tile then
    let some elem := ← decodeScalarTypeExpr? args[0]!
      | pure none
    let some dims := ← decodeArrayExpr? decodeNatExpr? args[1]!
      | pure none
    pure (some <| tileTy elem dims)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ct.Ptr ||
      fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ct.Array then
    let some elem := ← decodeScalarTypeExpr? args[0]!
      | pure none
    pure (some <| ptrTy elem)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ct.PtrTile then
    let some elem := ← decodeScalarTypeExpr? args[0]!
      | pure none
    pure (some <| ptrTileTy elem)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.scalarTileTy then
    let some elem := ← decodeScalarTypeExpr? args[0]!
      | pure none
    pure (some <| scalarTileTy elem)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.tileTy then
    let some elem := ← decodeScalarTypeExpr? args[0]!
      | pure none
    let some dims := ← decodeArrayExpr? decodeNatExpr? args[1]!
      | pure none
    pure (some <| tileTy elem dims)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ptrTy then
    let some elem := ← decodeScalarTypeExpr? args[0]!
      | pure none
    pure (some <| ptrTy elem)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ptrTileTy then
    let some elem := ← decodeScalarTypeExpr? args[0]!
      | pure none
    pure (some <| ptrTileTy elem)
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.TileType.ptr then
    let some elem := ← decodeScalarTypeExpr? args[0]!
      | pure none
    pure (some (.ptr elem))
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.TileType.tile then
    let some shape := ← decodeArrayExpr? decodeShapeDimExpr? args[0]!
      | pure none
    let some elem := ← decodeElemTypeExpr? args[1]!
      | pure none
    pure (some (.tile shape elem))
  else if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.TileType.token then
    pure (some .token)
  else
    pure none

private structure SurfaceValueTypeInfo where
  typeExpr : Expr
  elem? : Option ScalarType
  shapeExpr? : Option Expr
  isScalar : Bool

private def shapeExprIsEmpty (e : Expr) : TermElabM Bool := do
  let e ← reduceExprFully e
  let fn := e.getAppFn
  let args := e.getAppArgs
  if fn.isConstOf ``Tyr.GPU.Codegen.TileIR.staticShape then
    let inner ← reduceExprFully args[0]!
    let innerFn := inner.getAppFn
    let innerArgs := inner.getAppArgs
    if innerFn.isConstOf ``Array.mk || innerFn.isConstOf ``List.toArray then
      match ← decodeListExpr? (fun _ => pure (some ())) innerArgs[1]! with
      | some elems => pure elems.isEmpty
      | none => pure false
    else
      pure false
  else if fn.isConstOf ``Array.mk || fn.isConstOf ``List.toArray then
    match ← decodeListExpr? (fun _ => pure (some ())) args[1]! with
    | some elems => pure elems.isEmpty
    | none => pure false
  else
    pure false

private def renderSurfaceTypeInfo (info : SurfaceValueTypeInfo) : TermElabM String := do
  pure (toString (← ppExpr info.typeExpr))

private def sameSurfaceShape (lhs rhs : SurfaceValueTypeInfo) : TermElabM Bool := do
  if lhs.isScalar != rhs.isScalar then
    pure false
  else if lhs.isScalar then
    pure true
  else
    match lhs.shapeExpr?, rhs.shapeExpr? with
    | some lhsShape, some rhsShape => isDefEq lhsShape rhsShape
    | _, _ => pure false

private def inferSurfaceValueType (e : Expr) : TermElabM SurfaceValueTypeInfo := do
  let ty ← inferType e
  let ty ← whnf <| ← instantiateMVars ty
  let fn := ty.getAppFn
  let args := ty.getAppArgs
  unless fn.isConstOf ``Tyr.GPU.Codegen.TileIR.ct.Val do
    throwError "expected a TileIR value"
  unless args.size == 1 do
    throwError "internal TileIR value type error"
  let typeExpr := args[0]!
  let reduced ← reduceExprFully typeExpr
  let rFn := reduced.getAppFn
  let rArgs := reduced.getAppArgs
  if rFn.isConstOf ``Tyr.GPU.Codegen.TileIR.ct.Scalar then
    let some elem := ← decodeScalarTypeExpr? rArgs[0]!
      | throwError "unable to decode TileIR surface scalar type"
    return { typeExpr, elem? := some elem, shapeExpr? := none, isScalar := true }
  else if rFn.isConstOf ``Tyr.GPU.Codegen.TileIR.scalarTileTy then
    let some elem := ← decodeScalarTypeExpr? rArgs[0]!
      | throwError "unable to decode TileIR surface scalar type"
    return { typeExpr, elem? := some elem, shapeExpr? := none, isScalar := true }
  else if rFn.isConstOf ``Tyr.GPU.Codegen.TileIR.ct.Tile then
    let some elem := ← decodeScalarTypeExpr? rArgs[0]!
      | throwError "unable to decode TileIR surface tile element type"
    return { typeExpr, elem? := some elem, shapeExpr? := some rArgs[1]!, isScalar := false }
  else if rFn.isConstOf ``Tyr.GPU.Codegen.TileIR.tileTy then
    let some elem := ← decodeScalarTypeExpr? rArgs[0]!
      | throwError "unable to decode TileIR surface tile element type"
    return { typeExpr, elem? := some elem, shapeExpr? := some rArgs[1]!, isScalar := false }
  else if rFn.isConstOf ``Tyr.GPU.Codegen.TileIR.TileType.tile then
    let some elemType := ← decodeElemTypeExpr? rArgs[1]!
      | throwError "unable to decode TileIR surface tile element type"
    let elem? :=
      match elemType with
      | .scalar elem => some elem
      | .ptr _ => none
    let isScalar ← shapeExprIsEmpty rArgs[0]!
    return { typeExpr, elem?, shapeExpr? := some rArgs[0]!, isScalar }
  else
    throwError "unable to decode TileIR surface type"

private def elabCompareSurface
    (hint : String)
    (pred : ComparisonPredicate)
    (lhs rhs : TSyntax `term)
    : TermElabM (TSyntax `term) := do
  let lhsE ← elabTerm lhs none
  let rhsE ← elabTerm rhs none
  let lhsTy ← inferSurfaceValueType lhsE
  let rhsTy ← inferSurfaceValueType rhsE
  let some lhsDtype := lhsTy.elem?
    | throwErrorAt lhs "TileIR comparisons only support scalar tile values"
  let some rhsDtype := rhsTy.elem?
    | throwErrorAt rhs "TileIR comparisons only support scalar tile values"
  unless lhsDtype == rhsDtype do
    let lhsDesc ← renderSurfaceTypeInfo lhsTy
    let rhsDesc ← renderSurfaceTypeInfo rhsTy
    throwErrorAt rhs s!"TileIR comparisons require matching element types, but got {lhsDesc} and {rhsDesc}"
  if ← sameSurfaceShape lhsTy rhsTy then
    `(Tyr.GPU.Codegen.TileIR.ct.compareTileValues $(strLit hint) $(comparisonPredCtor pred) $lhs $rhs)
  else if rhsTy.isScalar then
    `(Tyr.GPU.Codegen.TileIR.ct.compareSurface $(strLit hint) $(comparisonPredCtor pred) $lhs $rhs)
  else if lhsTy.isScalar then
    `(Tyr.GPU.Codegen.TileIR.ct.compareSurfaceLeft $(strLit hint) $(comparisonPredCtor pred) $lhs $rhs)
  else
    let lhsDesc ← renderSurfaceTypeInfo lhsTy
    let rhsDesc ← renderSurfaceTypeInfo rhsTy
    throwErrorAt rhs
      s!"TileIR comparisons require matching shapes or scalar broadcasting, but got {lhsDesc} and {rhsDesc}"

private def elabWhereSurface
    (hint : String)
    (cond trueBranch falseBranch : TSyntax `term)
    : TermElabM (TSyntax `term) := do
  let condE ← elabTerm cond none
  let trueE ← elabTerm trueBranch none
  let falseE ← elabTerm falseBranch none
  let condTy ← inferSurfaceValueType condE
  let trueTy ← inferSurfaceValueType trueE
  let falseTy ← inferSurfaceValueType falseE
  let some condDtype := condTy.elem?
    | throwErrorAt cond "TileIR `ct.where` requires an i1 condition tile"
  unless condDtype == .i1 do
    let condDesc ← renderSurfaceTypeInfo condTy
    throwErrorAt cond s!"TileIR `ct.where` requires an i1 condition tile, but got {condDesc}"
  let some trueDtype := trueTy.elem?
    | throwErrorAt trueBranch "TileIR `ct.where` requires scalar-tile branches"
  let some falseDtype := falseTy.elem?
    | throwErrorAt falseBranch "TileIR `ct.where` requires scalar-tile branches"
  unless trueDtype == falseDtype do
    let trueDesc ← renderSurfaceTypeInfo trueTy
    let falseDesc ← renderSurfaceTypeInfo falseTy
    throwErrorAt falseBranch s!"TileIR `ct.where` branches must have matching element types, but got {trueDesc} and {falseDesc}"
  if (← sameSurfaceShape trueTy falseTy) && (← sameSurfaceShape trueTy condTy) then
    `(Tyr.GPU.Codegen.TileIR.ct.selectSameShape $(strLit hint) $cond $trueBranch $falseBranch)
  else if trueTy.isScalar && falseTy.isScalar then
    `(Tyr.GPU.Codegen.TileIR.ct.selectWithScalarBranches $(strLit hint) $cond $trueBranch $falseBranch)
  else if trueTy.isScalar && (← sameSurfaceShape falseTy condTy) then
    `(Tyr.GPU.Codegen.TileIR.ct.selectWithTrueScalar $(strLit hint) $cond $trueBranch $falseBranch)
  else if (← sameSurfaceShape trueTy condTy) && falseTy.isScalar then
    `(Tyr.GPU.Codegen.TileIR.ct.selectWithFalseScalar $(strLit hint) $cond $trueBranch $falseBranch)
  else if condTy.isScalar && (← sameSurfaceShape trueTy falseTy) then
    `(Tyr.GPU.Codegen.TileIR.ct.selectWithScalarCond $(strLit hint) $cond $trueBranch $falseBranch)
  else
    let condDesc ← renderSurfaceTypeInfo condTy
    let trueDesc ← renderSurfaceTypeInfo trueTy
    let falseDesc ← renderSurfaceTypeInfo falseTy
    throwErrorAt trueBranch
      s!"TileIR `ct.where` requires matching shapes or scalar broadcasting, but got cond={condDesc}, true={trueDesc}, false={falseDesc}"

private def expandSurfaceTerm (stx : TSyntax `term) : MacroM (TSyntax `term) := do
  let (expanded?, _) ← (expandTileExpr? stx "_").run 0
  match expanded? with
  | some expanded => pure expanded
  | none => Macro.throwUnsupported

private def expandEntryTerm
    (entryName : TSyntax `str)
    (binders : Array (TSyntax `tileirBinder))
    (body : TSyntax ``doSeq)
    : MacroM (TSyntax `term) := withRef body do
  let mut params : Array (TSyntax `term) := #[]
  let mut elems : Array (TSyntax `doElem) := #[]
  for binder in binders do
    match binder with
    | `(tileirBinder| ($id:ident : $ty)) =>
        let param ← `(Tyr.GPU.Codegen.TileIR.arg $(strLit id.getId.toString) $ty)
        let wrapped ← `(Tyr.GPU.Codegen.TileIR.ct.param $(strLit id.getId.toString) $ty)
        params := params.push param
        elems := elems.push (← `(doElem| let $id:ident := $wrapped))
    | _ =>
        Macro.throwErrorAt binder "invalid TileIR entry binder"
  elems := elems ++ Lean.Parser.Term.getDoElems body
  let bodySeq ← mkDoSeq elems
  `(Tyr.GPU.Codegen.TileIR.entry
      $entryName
      #[$params,*]
      (set_option backward.do.legacy false in do $bodySeq))

private def rewriteTileIRDoLet? (stx : TSyntax `doElem) : MacroM (Option (TSyntax `doElem)) := do
  match stx with
  | `(doLet| let $[mut%$_]? $decl:letDecl) =>
      match decl with
      | `(letDecl| $declId:letIdDecl) =>
          let { id, binders, value, .. } := Lean.Elab.Term.mkLetIdDeclView declId
          if !binders.isEmpty then
            pure none
          else
            let lhs : TSyntax `term := ⟨id⟩
            let rhs : TSyntax `term := ⟨value⟩
            let (rhs?, _) ← (expandTileExpr? rhs (hintFromPattern lhs)).run 0
            match rhs? with
            | some rhs =>
                some <$> mkBindDoElem lhs rhs
            | none =>
                pure none
      | _ =>
          pure none
  | _ =>
      pure none

private def rewriteTileIRDoLetArrow? (stx : TSyntax `doElem) : MacroM (Option (TSyntax `doElem)) := do
  let rewrite
      (id : Ident)
      (ty? : Option (TSyntax `term))
      (rhs : TSyntax `doElem)
      : MacroM (Option (TSyntax `doElem)) := do
    let lhs : TSyntax `term := mkIdent id.getId
    let rhsTerm? : Option (TSyntax `term) ←
      match rhs with
      | `(doExpr| $rhsTerm:term) =>
          pure (some rhsTerm)
      | `(doElem| if $cond then $thenSeq else $elseSeq) => do
          some <$> `(term| if $cond then do $thenSeq else do $elseSeq)
      | _ =>
          pure none
    match rhsTerm? with
    | some rhsTerm =>
        let (expanded?, _) ← (expandTileExpr? rhsTerm (hintFromPattern lhs)).run 0
        match expanded? with
        | some expanded =>
            let rhsDoElem : TSyntax `doElem ← `(doElem| $expanded:term)
            match ty? with
            | some ty =>
                some <$> `(doElem| let $id:ident : $ty ← $rhsDoElem:doElem)
            | none =>
                some <$> `(doElem| let $id:ident ← $rhsDoElem:doElem)
        | none =>
            pure none
    | _ =>
        pure none
  match stx with
  | `(doElem| let $id:ident ← $rhs:doElem) =>
      rewrite id none rhs
  | `(doElem| let $id:ident : $ty ← $rhs:doElem) =>
      rewrite id (some ty) rhs
  | _ =>
      pure none

private def isTileIREntryMonad : DoElabM Bool := do
  let m := (← read).monadInfo.m.consumeMData
  pure <|
    m.isConstOf ``Tyr.GPU.Codegen.TileIR.EntryM ||
    m.getAppFn.isConstOf ``Tyr.GPU.Codegen.TileIR.EntryM

private def isConstBinderTy : Syntax → Bool
  | `(term| ct.Const $_) => true
  | `(term| Tyr.GPU.Codegen.TileIR.ct.Const $_) => true
  | _ => false

private def constBinderInnerTy? : Syntax → Option Syntax
  | `(term| ct.Const $ty) => some ty
  | `(term| Tyr.GPU.Codegen.TileIR.ct.Const $ty) => some ty
  | _ => none

private def isSupportedConstTy : Syntax → Bool
  | `(term| Nat) => true
  | `(term| _root_.Nat) => true
  | `(term| Int) => true
  | `(term| _root_.Int) => true
  | `(term| Bool) => true
  | `(term| _root_.Bool) => true
  | _ => false

private structure KernelBinder where
  binder : TSyntax `Lean.Parser.Term.bracketedBinder
  id : Ident
  ty : TSyntax `term
  isConst : Bool

private def extractKernelBinders (binders : Array Syntax) :
    CommandElabM (Array KernelBinder) := do
  let mut out : Array KernelBinder := #[]
  for binder in binders do
    match binder with
    | `(bracketedBinderF|($ids* : $ty)) =>
        let isConst := isConstBinderTy ty
        if isConst then
          let some innerTy := constBinderInnerTy? ty
            | throwErrorAt ty "internal TileIR const-binder parsing error"
          unless isSupportedConstTy innerTy do
            throwErrorAt ty "TileIR `ct.Const` kernel parameters currently support only `Nat`, `Int`, and `Bool`"
        for id in ids do
          let id : Ident := ⟨id⟩
          let binder : TSyntax `Lean.Parser.Term.bracketedBinder ←
            `(bracketedBinder| ($id : $ty))
          out := out.push { binder := binder, id := id, ty := ty, isConst := isConst }
    | `(bracketedBinderF|($_ids*)) =>
        throwErrorAt binder "TileIR kernel parameters must have explicit types"
    | _ =>
        throwErrorAt binder "Unsupported TileIR kernel binder; use explicit `(x : ty)` parameters"
  pure out

private def mlirCompanionName (declName : Name) : Name :=
  declName ++ `mlir

private def generateMlirCompanion
    (declName : Name)
    (constBinders : Array (TSyntax `Lean.Parser.Term.bracketedBinder))
    (constArgs : Array Ident)
    : CommandElabM Unit := do
  let companionName := mlirCompanionName declName
  let kernelTerm : TSyntax `term := mkIdent declName
  let cmd : TSyntax `command ←
    if constBinders.isEmpty then
      `(command|
        abbrev $(mkIdent companionName) : String :=
          Tyr.GPU.Codegen.TileIR.renderOptimizedModule $kernelTerm
      )
    else
      `(command|
        abbrev $(mkIdent companionName) $constBinders* : String :=
          Tyr.GPU.Codegen.TileIR.renderOptimizedModule ($kernelTerm $constArgs*)
      )
  elabCommand cmd

@[command_elab tileirKernelDefCmd]
def elabTileIRKernelDef : CommandElab
| `(tileirKernelDefCmd| @[tileir_kernel] def $declId:declId $sig:optDeclSig := do $body:doSeq) => do
    let (binderStx, _type?) := Lean.Elab.expandOptDeclSig sig
    let binders ← extractKernelBinders binderStx.getArgs
    let declIdIdent : Ident := ⟨declId.raw[0]!⟩
    let kernelName := declIdIdent.getId.toString
    let tokId := mkIdentFrom declIdIdent `_tileir_tok0
    let runtimeBinders := binders.filter (fun binder => !binder.isConst)
    let constBinders := binders.filter KernelBinder.isConst
    let paramIds := runtimeBinders.map KernelBinder.id
    let paramTys := runtimeBinders.map KernelBinder.ty
    let constBinderStxs := constBinders.map KernelBinder.binder
    let constParamIds := constBinders.map KernelBinder.id
    let moduleTerm : TSyntax `term ←
      `(tileir_module $(strSyntax kernelName) do
          entry $(strSyntax kernelName)
            $[($paramIds : $paramTys)]*
            ($tokId : .token) do
            $body)
    let generatedDef : TSyntax `command ←
      `(def $declId:declId $[$constBinderStxs]* : Tyr.GPU.Codegen.TileIR.Module := $moduleTerm:term)
    elabCommand generatedDef
    let declName := (← getCurrNamespace) ++ declIdIdent.getId
    generateMlirCompanion declName constBinderStxs constParamIds
    liftTermElabM <|
      Lean.Elab.Term.applyAttributesAt declName #[{ name := `tileirKernelAttr }] .afterTypeChecking
| _ => throwUnsupportedSyntax

@[doElem_elab Lean.Parser.Term.doLet]
def elabTileIRDoLet : DoElab := fun stx dec => do
  unless ← isTileIREntryMonad do
    throwUnsupportedSyntax
  let some rewritten ← liftMacroM <| rewriteTileIRDoLet? stx
    | throwUnsupportedSyntax
  elabDoElem rewritten dec

@[doElem_elab Lean.Parser.Term.doLetArrow]
def elabTileIRDoLetArrow : DoElab := fun stx dec => do
  unless ← isTileIREntryMonad do
    throwUnsupportedSyntax
  let some rewritten ← liftMacroM <| rewriteTileIRDoLetArrow? stx
    | throwUnsupportedSyntax
  elabDoElem rewritten dec

private def rewriteTileIRDoFor
    (stx : TSyntax `doElem)
    (iv : Ident)
    (range : TSyntax `term)
    (body : TSyntax ``doSeq)
    : FrontendM (TSyntax `doElem) := do
  let ((lower, upper, stepVal), prep) ← expandCtRange range
  let initId ← freshTempIdent stx `tileir_loop_init
  let carryId ← freshTempIdent stx `tileir_loop_carry
  let initBind ← liftM <| mkI32ConstBind initId "loop_state" 0
  let continueElem : TSyntax `doElem ← `(doElem| pure (Tyr.GPU.Codegen.TileIR.continue1 $carryId))
  let bodySeq ← liftM <| mkDoSeq (Lean.Parser.Term.getDoElems body |>.push continueElem)
  let loopBody ← wrapModernDoSeq bodySeq
  let loopTerm : TSyntax `term ←
    `(Tyr.GPU.Codegen.TileIR.ct.for1
        $lower
        $upper
        $stepVal
        $initId
        (fun $iv:ident => fun $carryId:ident => $loopBody)
        $(strLit iv.getId.toString)
        $(strLit s!"{iv.getId.toString}_loop"))
  let loopBind ← liftM <| mkBindDoElem (← `(term| _)) loopTerm
  let doneElem : TSyntax `doElem ← `(doElem| pure ())
  let seq ← liftM <| mkDoSeq (prep ++ #[initBind, loopBind, doneElem])
  `(doElem| do $seq)

@[doElem_elab Lean.Parser.Term.doFor]
def elabTileIRDoFor : DoElab := fun stx dec => do
  unless ← isTileIREntryMonad do
    throwUnsupportedSyntax
  let `(doElem| for $iv:ident in $range:term do $body:doSeq) := stx
    | throwUnsupportedSyntax
  unless isCtRuntimeRangeSyntax range do
    throwUnsupportedSyntax
  let info ← inferControlInfoSeq body
  if info.breaks then
    throwErrorAt stx "TileIR `for ... in ct.range(...) do` does not yet support `break`; use `for_tile` for structured loop control"
  if info.continues then
    throwErrorAt stx "TileIR `for ... in ct.range(...) do` does not yet support `continue`; use `for_tile` for structured loop control"
  if info.returnsEarly then
    throwErrorAt stx "TileIR `for ... in ct.range(...) do` does not yet support `return`"
  unless info.reassigns.isEmpty do
    throwErrorAt stx "TileIR `for ... in ct.range(...) do` does not yet support reassigning loop-carried locals; use `for_tile` for now"
  let (rewritten, _) ← liftMacroM <| (rewriteTileIRDoFor stx iv range body).run 0
  elabDoElem rewritten dec

macro_rules
  | `(tileir_module $name do $body) =>
      `(set_option backward.do.legacy false in
          Tyr.GPU.Codegen.TileIR.module_ $name do $body)

macro_rules
  | `(global $name:ident : $ty := $value) =>
      `(Tyr.GPU.Codegen.TileIR.global $(strLit name.getId.toString) $ty $value)

macro_rules
  | `(entry $entryName:str $[$binders:tileirBinder]* do $body:doSeq) => do
      expandEntryTerm entryName binders body

macro_rules
  | `(const_int $value:term : $ty:term) => do
      expandSurfaceTerm (← `(const_int $value:term : $ty:term))
  | `(const_float $value:term : $ty:term) => do
      expandSurfaceTerm (← `(const_float $value:term : $ty:term))
  | `(const_bool $value:term : $ty:term) => do
      expandSurfaceTerm (← `(const_bool $value:term : $ty:term))
  | `(load_ptr_tko $ptr:term, $tok:term : $ty:term) => do
      expandSurfaceTerm (← `(load_ptr_tko $ptr:term, $tok:term : $ty:term))
  | `(get_global $name:ident : $ty:term) => do
      expandSurfaceTerm (← `(get_global $name:ident : $ty:term))
  | `(broadcast $src:term : $ty:term) => do
      expandSurfaceTerm (← `(broadcast $src:term : $ty:term))
  | `(reshape $src:term : $ty:term) => do
      expandSurfaceTerm (← `(reshape $src:term : $ty:term))
  | `(offset $ptr:term, $idx:term : $ty:term) => do
      expandSurfaceTerm (← `(offset $ptr:term, $idx:term : $ty:term))
  | `(make_tensor_view $base:term : $desc:term) => do
      expandSurfaceTerm (← `(make_tensor_view $base:term : $desc:term))
  | `(make_partition_view $src:term : $desc:term) => do
      expandSurfaceTerm (← `(make_partition_view $src:term : $desc:term))
  | `(mmaf $a:term, $b:term, $c:term) => do
      expandSurfaceTerm (← `(mmaf $a:term, $b:term, $c:term))
  | `(mmai $a:term, $b:term, $c:term) => do
      expandSurfaceTerm (← `(mmai $a:term, $b:term, $c:term))
  | `(maxf $lhs:term, $rhs:term) => do
      expandSurfaceTerm (← `(maxf $lhs:term, $rhs:term))
  | `(minf $lhs:term, $rhs:term) => do
      expandSurfaceTerm (← `(minf $lhs:term, $rhs:term))
  | `(for_tile $iv:ident in $range:tileirLoopRange carrying $carrySpec:tileirLoopCarry do $body:doSeq) => do
      expandSurfaceTerm
        (← `(for_tile $iv:ident in $range:tileirLoopRange carrying $carrySpec:tileirLoopCarry do $body:doSeq))
  | `(store_ptr_tko $ptr:term, $value:term, $tok:term) => do
      expandSurfaceTerm (← `(store_ptr_tko $ptr:term, $value:term, $tok:term))
  | `(print_tko $msg:str) => do
      expandSurfaceTerm (← `(print_tko $msg:str))
  | `(continue_tile $value:term) => do
      expandSurfaceTerm (← `(continue_tile $value:term))
  | `(break_tile $value:term) => do
      expandSurfaceTerm (← `(break_tile $value:term))

macro_rules
  | `(ct.comment $msg:str) => do
      expandSurfaceTerm (← `(ct.comment $msg:str))
  | `(ct.bid $axis:term) => do
      expandSurfaceTerm (← `(ct.bid $axis:term))
  | `(ct.num_blocks $axis:term) => do
      expandSurfaceTerm (← `(ct.num_blocks $axis:term))
  | `(ct.cdiv($lhs:term, $rhs:term)) =>
      `(Tyr.GPU.Codegen.TileIR.ct.cdiv $lhs $rhs)
  | `(ct.static_range($stop:term)) =>
      `(Tyr.GPU.Codegen.TileIR.ct.staticRange1 $stop)
  | `(ct.static_range($start:term, $stop:term)) =>
      `(Tyr.GPU.Codegen.TileIR.ct.staticRange2 $start $stop)
  | `(ct.static_range($start:term, $stop:term, $stepVal:term)) =>
      `(Tyr.GPU.Codegen.TileIR.ct.staticRange3 $start $stop $stepVal)
  | `(ct.static_iter($xs:term)) =>
      `(Tyr.GPU.Codegen.TileIR.ct.staticIter $xs)
  | `(ct.iota : $ty:term) => do
      expandSurfaceTerm (← `(ct.iota : $ty:term))
  | `(ct.arange($size:term)) => do
      expandSurfaceTerm (← `(ct.arange($size:term)))
  | `(ct.arange($size:term, $argName:ident := $dtype:term)) => do
      expandSurfaceTerm (← `(ct.arange($size:term, $argName:ident := $dtype:term)))
  | `(ct.load $ptr:term : $ty:term) => do
      expandSurfaceTerm (← `(ct.load $ptr:term : $ty:term))
  | `(ct.load $ptr:term, $name1:ident := $index:term, $name2:ident := $shape:tileirTermTuple) => do
      expandSurfaceTerm (← `(ct.load $ptr:term, $name1:ident := $index:term, $name2:ident := $shape:tileirTermTuple))
  | `(ct.load($ptr:term, $indices:tileirTermTuple, $shape:tileirTermTuple)) => do
      expandSurfaceTerm (← `(ct.load($ptr:term, $indices:tileirTermTuple, $shape:tileirTermTuple)))
  | `(ct.gather($base:term, $idx:term)) => do
      expandSurfaceTerm (← `(ct.gather($base:term, $idx:term)))
  | `(ct.load_view $view:term, $indices:tileirIndexList : $ty:term) => do
      expandSurfaceTerm (← `(ct.load_view $view:term, $indices:tileirIndexList : $ty:term))
  | `(ct.load_global $name:ident : $ty:term) => do
      expandSurfaceTerm (← `(ct.load_global $name:ident : $ty:term))
  | `(ct.full($shape:tileirTermTuple, $value:term, $argName:ident := $dtype:term)) => do
      expandSurfaceTerm (← `(ct.full($shape:tileirTermTuple, $value:term, $argName:ident := $dtype:term)))
  | `(ct.full($shape:tileirTermTuple, $value:term, $dtype:term)) => do
      expandSurfaceTerm (← `(ct.full($shape:tileirTermTuple, $value:term, $dtype:term)))
  | `(ct.full($shape:term, $value:term, $argName:ident := $dtype:term)) => do
      expandSurfaceTerm (← `(ct.full($shape:term, $value:term, $argName:ident := $dtype:term)))
  | `(ct.full($shape:term, $value:term, $dtype:term)) => do
      expandSurfaceTerm (← `(ct.full($shape:term, $value:term, $dtype:term)))
  | `(ct.full_like($like:term, $value:term, $argName:ident := $dtype:term)) => do
      expandSurfaceTerm (← `(ct.full_like($like:term, $value:term, $argName:ident := $dtype:term)))
  | `(ct.full_like($like:term, $value:term, $dtype:term)) => do
      expandSurfaceTerm (← `(ct.full_like($like:term, $value:term, $dtype:term)))
  | `(ct.zeros($shape:tileirTermTuple, $argName:ident := $dtype:term)) => do
      expandSurfaceTerm (← `(ct.zeros($shape:tileirTermTuple, $argName:ident := $dtype:term)))
  | `(ct.zeros($shape:tileirTermTuple, $dtype:term)) => do
      expandSurfaceTerm (← `(ct.zeros($shape:tileirTermTuple, $dtype:term)))
  | `(ct.zeros($shape:term, $argName:ident := $dtype:term)) => do
      expandSurfaceTerm (← `(ct.zeros($shape:term, $argName:ident := $dtype:term)))
  | `(ct.zeros($shape:term, $dtype:term)) => do
      expandSurfaceTerm (← `(ct.zeros($shape:term, $dtype:term)))
  | `(ct.zeros_like($like:term)) => do
      expandSurfaceTerm (← `(ct.zeros_like($like:term)))
  | `(ct.ones($shape:tileirTermTuple, $argName:ident := $dtype:term)) => do
      expandSurfaceTerm (← `(ct.ones($shape:tileirTermTuple, $argName:ident := $dtype:term)))
  | `(ct.ones($shape:tileirTermTuple, $dtype:term)) => do
      expandSurfaceTerm (← `(ct.ones($shape:tileirTermTuple, $dtype:term)))
  | `(ct.ones($shape:term, $argName:ident := $dtype:term)) => do
      expandSurfaceTerm (← `(ct.ones($shape:term, $argName:ident := $dtype:term)))
  | `(ct.ones($shape:term, $dtype:term)) => do
      expandSurfaceTerm (← `(ct.ones($shape:term, $dtype:term)))
  | `(ct.ones_like($like:term)) => do
      expandSurfaceTerm (← `(ct.ones_like($like:term)))
  | `(ct.astype($src:term, $dtype:term)) => do
      expandSurfaceTerm (← `(ct.astype($src:term, $dtype:term)))
  | `(ct.cat($tiles:tileirTermTuple, $dim:term)) => do
      expandSurfaceTerm (← `(ct.cat($tiles:tileirTermTuple, $dim:term)))
  | `(ct.broadcast $src:term : $ty:term) => do
      expandSurfaceTerm (← `(ct.broadcast $src:term : $ty:term))
  | `(ct.broadcast_to($src:term, $shape:tileirTermTuple)) => do
      expandSurfaceTerm (← `(ct.broadcast_to($src:term, $shape:tileirTermTuple)))
  | `(ct.broadcast_to($src:term, $shape:term)) => do
      expandSurfaceTerm (← `(ct.broadcast_to($src:term, $shape:term)))
  | `(ct.reshape $src:term : $ty:term) => do
      expandSurfaceTerm (← `(ct.reshape $src:term : $ty:term))
  | `(ct.reshape($src:term, $shape:tileirTermTuple)) => do
      expandSurfaceTerm (← `(ct.reshape($src:term, $shape:tileirTermTuple)))
  | `(ct.permute($src:term, $perm:tileirTermTuple)) => do
      expandSurfaceTerm (← `(ct.permute($src:term, $perm:tileirTermTuple)))
  | `(ct.transpose($src:term)) => do
      expandSurfaceTerm (← `(ct.transpose($src:term)))
  | `(ct.extract $src:term, $arg1, $arg2) => do
      expandSurfaceTerm (← `(ct.extract $src:term, $arg1, $arg2))
  | `(ct.extract($src:term, $indices:tileirTermTuple, $shape:tileirTermTuple)) => do
      expandSurfaceTerm (← `(ct.extract($src:term, $indices:tileirTermTuple, $shape:tileirTermTuple)))
  | `(ct.offset $ptr:term, $idx:term : $ty:term) => do
      expandSurfaceTerm (← `(ct.offset $ptr:term, $idx:term : $ty:term))
  | `(ct.tensor_view $base:term : $desc:term) => do
      expandSurfaceTerm (← `(ct.tensor_view $base:term : $desc:term))
  | `(ct.partition_view $src:term : $desc:term) => do
      expandSurfaceTerm (← `(ct.partition_view $src:term : $desc:term))
  | `(ct.mma $a:term, $b:term, $c:term) => do
      expandSurfaceTerm (← `(ct.mma $a:term, $b:term, $c:term))
  | `(ct.matmul $a:term, $b:term, $c:term) => do
      expandSurfaceTerm (← `(ct.matmul $a:term, $b:term, $c:term))
  | `(ct.mmai $a:term, $b:term, $c:term) => do
      expandSurfaceTerm (← `(ct.mmai $a:term, $b:term, $c:term))
  | `(ct.max $lhs:term, $rhs:term) => do
      expandSurfaceTerm (← `(ct.max $lhs:term, $rhs:term))
  | `(ct.min $lhs:term, $rhs:term) => do
      expandSurfaceTerm (← `(ct.min $lhs:term, $rhs:term))
  | `(ct.assert $cond:term, $msg:str) => do
      expandSurfaceTerm (← `(ct.assert $cond:term, $msg:str))
  | `(ct.store $ptr:term, $value:term) => do
      expandSurfaceTerm (← `(ct.store $ptr:term, $value:term))
  | `(ct.store $ptr:term, $name1:ident := $index:term, $name2:ident := $value:term) => do
      expandSurfaceTerm (← `(ct.store $ptr:term, $name1:ident := $index:term, $name2:ident := $value:term))
  | `(ct.store($ptr:term, $indices:tileirTermTuple, $value:term)) => do
      expandSurfaceTerm (← `(ct.store($ptr:term, $indices:tileirTermTuple, $value:term)))
  | `(ct.scatter($base:term, $idx:term, $value:term)) => do
      expandSurfaceTerm (← `(ct.scatter($base:term, $idx:term, $value:term)))
  | `(ct.store_view $view:term, $indices:tileirIndexList, $value:term) => do
      expandSurfaceTerm (← `(ct.store_view $view:term, $indices:tileirIndexList, $value:term))
  | `(ct.print $msg:str) => do
      expandSurfaceTerm (← `(ct.print $msg:str))
  | `(ct.static_eval $e:term) =>
      pure e
  | `(ct.static_assert $cond:term) => do
      expandSurfaceTerm (← `(ct.static_assert $cond:term))
  | `(ct.static_assert $cond:term, $msg:term) => do
      expandSurfaceTerm (← `(ct.static_assert $cond:term, $msg:term))
  | `(ct.exp $src:term) => do
      expandSurfaceTerm (← `(ct.exp $src:term))
  | `(ct.exp2 $src:term) => do
      expandSurfaceTerm (← `(ct.exp2 $src:term))
  | `(ct.log $src:term) => do
      expandSurfaceTerm (← `(ct.log $src:term))
  | `(ct.sqrt $src:term) => do
      expandSurfaceTerm (← `(ct.sqrt $src:term))
  | `(ct.rsqrt $src:term) => do
      expandSurfaceTerm (← `(ct.rsqrt $src:term))
  | `(ct.abs $src:term) => do
      expandSurfaceTerm (← `(ct.abs $src:term))
  | `(ct.neg $src:term) => do
      expandSurfaceTerm (← `(ct.neg $src:term))

@[term_elab ctLoadIndexedTerm]
def elabCtLoadIndexed : TermElab := fun stx expectedType => do
  let expanded ← liftMacroM <| expandSurfaceTerm ⟨stx⟩
  elabTerm expanded expectedType

@[term_elab ctLoadIndexedScalarIndexTerm]
def elabCtLoadIndexedScalarIndex : TermElab := fun stx expectedType => do
  let expanded ← liftMacroM <| expandSurfaceTerm ⟨stx⟩
  elabTerm expanded expectedType

@[term_elab ctLoadPositionalTerm]
def elabCtLoadPositional : TermElab := fun stx expectedType => do
  let expanded ← liftMacroM <| expandSurfaceTerm ⟨stx⟩
  elabTerm expanded expectedType

@[term_elab ctExtractIndexedTerm]
def elabCtExtractIndexed : TermElab := fun stx expectedType => do
  let expanded ← liftMacroM <| expandSurfaceTerm ⟨stx⟩
  elabTerm expanded expectedType

@[term_elab ctExtractPositionalTerm]
def elabCtExtractPositional : TermElab := fun stx expectedType => do
  let expanded ← liftMacroM <| expandSurfaceTerm ⟨stx⟩
  elabTerm expanded expectedType

@[term_elab ctStoreIndexedTerm]
def elabCtStoreIndexed : TermElab := fun stx expectedType => do
  let expanded ← liftMacroM <| expandSurfaceTerm ⟨stx⟩
  elabTerm expanded expectedType

@[term_elab ctStoreIndexedScalarIndexTerm]
def elabCtStoreIndexedScalarIndex : TermElab := fun stx expectedType => do
  let expanded ← liftMacroM <| expandSurfaceTerm ⟨stx⟩
  elabTerm expanded expectedType

@[term_elab ctStorePositionalTerm]
def elabCtStorePositional : TermElab := fun stx expectedType => do
  let expanded ← liftMacroM <| expandSurfaceTerm ⟨stx⟩
  elabTerm expanded expectedType

@[term_elab ctEqualTerm]
unsafe def elabCtEqual : TermElab := fun stx expectedType => do
  let `(ctEqualTerm| ct.equal $lhs:term, $rhs:term) := stx
    | throwUnsupportedSyntax
  let expanded ← elabCompareSurface "equal" .equal lhs rhs
  elabTerm expanded expectedType

@[term_elab ctNotEqualTerm]
unsafe def elabCtNotEqual : TermElab := fun stx expectedType => do
  let `(ctNotEqualTerm| ct.not_equal $lhs:term, $rhs:term) := stx
    | throwUnsupportedSyntax
  let expanded ← elabCompareSurface "not_equal" .notEqual lhs rhs
  elabTerm expanded expectedType

@[term_elab ctLessTerm]
unsafe def elabCtLess : TermElab := fun stx expectedType => do
  let `(ctLessTerm| ct.less $lhs:term, $rhs:term) := stx
    | throwUnsupportedSyntax
  let expanded ← elabCompareSurface "less" .lessThan lhs rhs
  elabTerm expanded expectedType

@[term_elab ctLessEqualTerm]
unsafe def elabCtLessEqual : TermElab := fun stx expectedType => do
  let `(ctLessEqualTerm| ct.less_equal $lhs:term, $rhs:term) := stx
    | throwUnsupportedSyntax
  let expanded ← elabCompareSurface "less_equal" .lessThanOrEqual lhs rhs
  elabTerm expanded expectedType

@[term_elab ctGreaterTerm]
unsafe def elabCtGreater : TermElab := fun stx expectedType => do
  let `(ctGreaterTerm| ct.greater $lhs:term, $rhs:term) := stx
    | throwUnsupportedSyntax
  let expanded ← elabCompareSurface "greater" .greaterThan lhs rhs
  elabTerm expanded expectedType

@[term_elab ctGreaterEqualTerm]
unsafe def elabCtGreaterEqual : TermElab := fun stx expectedType => do
  let `(ctGreaterEqualTerm| ct.greater_equal $lhs:term, $rhs:term) := stx
    | throwUnsupportedSyntax
  let expanded ← elabCompareSurface "greater_equal" .greaterThanOrEqual lhs rhs
  elabTerm expanded expectedType

@[term_elab ctWhereTerm]
unsafe def elabCtWhere : TermElab := fun stx expectedType => do
  let `(ctWhereTerm| ct.where($cond:term, $valIfTrue:term, $valIfFalse:term)) := stx
    | throwUnsupportedSyntax
  let expanded ← elabWhereSurface "where" cond valIfTrue valIfFalse
  elabTerm expanded expectedType

end Tyr.GPU.Codegen.TileIR
