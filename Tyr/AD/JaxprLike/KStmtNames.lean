import Tyr.AD.JaxprLike.Core
import Tyr.GPU.Codegen.AST

/-!
# Tyr.AD.JaxprLike.KStmtNames

Canonical op-name helpers for `KStmt -> LeanJaxpr` lowering and rule-pack registration.
-/

namespace Tyr.AD.JaxprLike

open Tyr.GPU.Codegen

/-- Build a dotted `Name` from a string path. -/
private def mkDottedName (dotted : String) : OpName :=
  (dotted.splitOn ".").foldl (init := Lean.Name.anonymous) fun acc part =>
    Lean.Name.str acc part

/-- String tag used in op names and equation params for `ReduceOp`. -/
def kstmtReduceOpTag : ReduceOp → String
  | .Max => "Max"
  | .Min => "Min"
  | .Sum => "Sum"
  | .Prod => "Prod"

/-- String tag used in op names and equation params for `UnaryOp`. -/
def kstmtUnaryOpTag : UnaryOp → String
  | .Exp => "Exp"
  | .Exp2 => "Exp2"
  | .Log => "Log"
  | .Log2 => "Log2"
  | .Abs => "Abs"
  | .Relu => "Relu"
  | .Copy => "Copy"
  | .Neg => "Neg"
  | .Sqrt => "Sqrt"
  | .Rsqrt => "Rsqrt"
  | .Tanh => "Tanh"
  | .FastTanh => "FastTanh"
  | .Sigmoid => "Sigmoid"
  | .Gelu => "Gelu"
  | .Silu => "Silu"
  | .Swish => "Swish"
  | .Sin => "Sin"
  | .Cos => "Cos"
  | .Recip => "Recip"
  | .Square => "Square"
  | .Zero => "Zero"
  | .One => "One"
  | .PosInfty => "PosInfty"
  | .NegInfty => "NegInfty"

/-- String tag used in op names and equation params for `BinaryOp`. -/
def kstmtBinaryOpTag : BinaryOp → String
  | .Add => "Add"
  | .Sub => "Sub"
  | .Mul => "Mul"
  | .Div => "Div"
  | .Max => "Max"
  | .Min => "Min"

/-- String tag used in op names and equation params for `ReduceAxis`. -/
def kstmtReduceAxisTag : ReduceAxis → String
  | .Row => "Row"
  | .Col => "Col"
  | .Full => "Full"

/-- String tag used in op names and equation params for `BroadcastAxis`. -/
def kstmtBroadcastAxisTag : BroadcastAxis → String
  | .Row => "Row"
  | .Col => "Col"

/-- Canonical op name used by `FromKStmt` for unary equations. -/
def kstmtUnaryOpName (op : UnaryOp) : OpName :=
  mkDottedName s!"Tyr.GPU.KStmt.unary.{kstmtUnaryOpTag op}"

/-- Canonical op name used by `FromKStmt` for binary equations. -/
def kstmtBinaryOpName (op : BinaryOp) : OpName :=
  mkDottedName s!"Tyr.GPU.KStmt.binary.{kstmtBinaryOpTag op}"

/-- Canonical op name used by `FromKStmt` for reduce equations. -/
def kstmtReduceOpName (op : ReduceOp) (axis : ReduceAxis) : OpName :=
  mkDottedName s!"Tyr.GPU.KStmt.reduce.{kstmtReduceOpTag op}.{kstmtReduceAxisTag axis}"

/-- Canonical op name used by `FromKStmt` for reduce-accum equations. -/
def kstmtReduceAccumOpName (op : ReduceOp) (axis : ReduceAxis) : OpName :=
  mkDottedName s!"Tyr.GPU.KStmt.reduceAccum.{kstmtReduceOpTag op}.{kstmtReduceAxisTag axis}"

/-- Canonical op name used by `FromKStmt` for broadcast equations. -/
def kstmtBroadcastOpName (axis : BroadcastAxis) : OpName :=
  mkDottedName s!"Tyr.GPU.KStmt.broadcast.{kstmtBroadcastAxisTag axis}"

/-- Canonical op name used by `FromKStmt` for binary-broadcast equations. -/
def kstmtBinaryBroadcastOpName (op : BinaryOp) (axis : BroadcastAxis) : OpName :=
  mkDottedName s!"Tyr.GPU.KStmt.binaryBroadcast.{kstmtBinaryOpTag op}.{kstmtBroadcastAxisTag axis}"

/-- Canonical op name used by `FromKStmt` for `transpose`. -/
def kstmtTransposeOpName : OpName :=
  mkDottedName "Tyr.GPU.KStmt.transpose"

/-- Canonical op name used by `FromKStmt` for `swapLayout`. -/
def kstmtSwapLayoutOpName : OpName :=
  mkDottedName "Tyr.GPU.KStmt.swapLayout"

/-- Canonical op name used by `FromKStmt` for `convert`. -/
def kstmtConvertOpName : OpName :=
  mkDottedName "Tyr.GPU.KStmt.convert"

/-- Canonical op name used by `FromKStmt` for `sliceRows`. -/
def kstmtSliceRowsOpName : OpName :=
  mkDottedName "Tyr.GPU.KStmt.sliceRows"

/-- Canonical op name used by `FromKStmt` for `sliceCols`. -/
def kstmtSliceColsOpName : OpName :=
  mkDottedName "Tyr.GPU.KStmt.sliceCols"

/-- Canonical op name used by `FromKStmt` for `concatCols`. -/
def kstmtConcatColsOpName : OpName :=
  mkDottedName "Tyr.GPU.KStmt.concatCols"

/-- Canonical op name used by `FromKStmt` for `outer`. -/
def kstmtOuterOpName : OpName :=
  mkDottedName "Tyr.GPU.KStmt.outer"

/--
Canonical op name used by LeanJaxpr paths for dot-general style contraction.
This is intentionally independent of the current `KStmt` constructor set.
-/
def kstmtDotGeneralOpName : OpName :=
  mkDottedName "Tyr.GPU.KStmt.dotGeneral"

/-- Canonical no-grad primitive op name (`stop_gradient`) for LeanJaxpr paths. -/
def stopGradientOpName : OpName :=
  mkDottedName "jax.lax.stop_gradient"

/-- Canonical control/value primitive op name (`iota`) for LeanJaxpr paths. -/
def iotaOpName : OpName :=
  mkDottedName "jax.lax.iota"

/-- Common aliases accepted for stop-gradient lowering/rule registration. -/
def allStopGradientOpNames : Array OpName := #[
  stopGradientOpName,
  mkDottedName "Tyr.GPU.KStmt.stopGradient",
  mkDottedName "Graphax.stop_gradient"
]

/-- Common aliases accepted for iota lowering/rule registration. -/
def allIotaOpNames : Array OpName := #[
  iotaOpName,
  mkDottedName "Tyr.GPU.KStmt.iota",
  mkDottedName "Graphax.iota"
]

/-- Common aliases accepted for dot-general lowering/rule registration. -/
def allDotGeneralOpNames : Array OpName := #[
  kstmtDotGeneralOpName,
  mkDottedName "jax.lax.dot_general",
  mkDottedName "Graphax.dot_general"
]

def isStopGradientOpName (op : OpName) : Bool :=
  allStopGradientOpNames.any (· == op)

def isIotaOpName (op : OpName) : Bool :=
  allIotaOpNames.any (· == op)

def isDotGeneralOpName (op : OpName) : Bool :=
  allDotGeneralOpNames.any (· == op)

/-- Canonical op name used by `FromKStmt` for `cumsum`. -/
def kstmtCumsumOpName (axis : ReduceAxis) : OpName :=
  mkDottedName s!"Tyr.GPU.KStmt.cumsum.{kstmtReduceAxisTag axis}"

/-- Canonical op name used by `FromKStmt` for `cumprod`. -/
def kstmtCumprodOpName (axis : ReduceAxis) : OpName :=
  mkDottedName s!"Tyr.GPU.KStmt.cumprod.{kstmtReduceAxisTag axis}"

/-- KStmt unary op universe currently lowered into LeanJaxpr equations. -/
def allKStmtUnaryOps : Array UnaryOp := #[
  .Exp, .Exp2, .Log, .Log2, .Abs, .Relu, .Copy,
  .Neg, .Sqrt, .Rsqrt, .Tanh, .FastTanh, .Sigmoid, .Gelu,
  .Silu, .Swish, .Sin, .Cos, .Recip, .Square,
  .Zero, .One, .PosInfty, .NegInfty
]

/-- KStmt binary op universe currently lowered into LeanJaxpr equations. -/
def allKStmtBinaryOps : Array BinaryOp := #[
  .Add, .Sub, .Mul, .Div, .Max, .Min
]

/-- KStmt reduce op universe currently lowered into LeanJaxpr equations. -/
def allKStmtReduceOps : Array ReduceOp := #[
  .Max, .Min, .Sum, .Prod
]

/-- KStmt axis universe shared by reduce/scan lowering. -/
def allKStmtReduceAxes : Array ReduceAxis := #[
  .Row, .Col, .Full
]

/-- KStmt broadcast axis universe lowered into LeanJaxpr equations. -/
def allKStmtBroadcastAxes : Array BroadcastAxis := #[
  .Row, .Col
]

/-- Unary/binary op-name universe supported by the first KStmt rule packs. -/
def allKStmtUnaryBinaryOpNames : Array OpName :=
  (allKStmtUnaryOps.map kstmtUnaryOpName) ++
  (allKStmtBinaryOps.map kstmtBinaryOpName)

/-- Non-unary/binary op-name universe currently lowered from KStmt. -/
def allKStmtExtendedOpNames : Array OpName := Id.run do
  let mut out : Array OpName := #[
    kstmtTransposeOpName,
    kstmtSwapLayoutOpName,
    kstmtConvertOpName,
    kstmtSliceRowsOpName,
    kstmtSliceColsOpName,
    kstmtConcatColsOpName,
    kstmtOuterOpName,
    kstmtDotGeneralOpName
  ]
  for axis in allKStmtBroadcastAxes do
    out := out.push (kstmtBroadcastOpName axis)
  for op in allKStmtBinaryOps do
    for axis in allKStmtBroadcastAxes do
      out := out.push (kstmtBinaryBroadcastOpName op axis)
  for op in allKStmtReduceOps do
    for axis in allKStmtReduceAxes do
      out := out.push (kstmtReduceOpName op axis)
      out := out.push (kstmtReduceAccumOpName op axis)
  for axis in allKStmtReduceAxes do
    out := out.push (kstmtCumsumOpName axis)
    out := out.push (kstmtCumprodOpName axis)
  return out

/-- Full op-name universe currently lowered from KStmt into LeanJaxpr equations. -/
def allKStmtSupportedOpNames : Array OpName :=
  allKStmtUnaryBinaryOpNames ++ allKStmtExtendedOpNames

end Tyr.AD.JaxprLike
