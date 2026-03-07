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

private def kstmtMMATransposeTag : MMATranspose → String
  | .AB => "AB"
  | .ABt => "ABt"
  | .AtB => "AtB"
  | .AtBt => "AtBt"

/-- Canonical op name used by `FromKStmt` for fused MMA (`dst = a*b + c`). -/
def kstmtMmaOpName (trans : MMATranspose) : OpName :=
  mkDottedName s!"Tyr.GPU.KStmt.mma.{kstmtMMATransposeTag trans}"

/-- Canonical no-grad primitive op name (`stop_gradient`) for LeanJaxpr paths. -/
def stopGradientOpName : OpName :=
  mkDottedName "jax.lax.stop_gradient"

/-- Canonical control/value primitive op name (`iota`) for LeanJaxpr paths. -/
def iotaOpName : OpName :=
  mkDottedName "jax.lax.iota"

/-- Canonical communication/control primitive op name (`device_put`). -/
def devicePutOpName : OpName :=
  mkDottedName "jax.lax.device_put"

/-- Canonical sharding/control primitive op name (`pjit`). -/
def pjitOpName : OpName :=
  mkDottedName "jax.pjit"

/-- Canonical communication primitive op name (`all_gather`). -/
def allGatherOpName : OpName :=
  mkDottedName "jax.lax.all_gather"

/-- Canonical communication primitive op name (`all_to_all`). -/
def allToAllOpName : OpName :=
  mkDottedName "jax.lax.all_to_all"

/-- Canonical communication primitive op name (`reduce_scatter`). -/
def reduceScatterOpName : OpName :=
  mkDottedName "jax.lax.reduce_scatter"

/-- Canonical communication primitive op name (`collective_permute`). -/
def collectivePermuteOpName : OpName :=
  mkDottedName "jax.lax.collective_permute"

/-- Canonical communication primitive op name (`psum`). -/
def psumOpName : OpName :=
  mkDottedName "jax.lax.psum"

/-- Canonical communication primitive op name (`pmean`). -/
def pmeanOpName : OpName :=
  mkDottedName "jax.lax.pmean"

/-- Canonical communication primitive op name (`pmax`). -/
def pmaxOpName : OpName :=
  mkDottedName "jax.lax.pmax"

/-- Canonical communication primitive op name (`pmin`). -/
def pminOpName : OpName :=
  mkDottedName "jax.lax.pmin"

/-- Canonical communication primitive op name (`pswapaxes`). -/
def pswapaxesOpName : OpName :=
  mkDottedName "jax.lax.pswapaxes"

/-- Canonical structural primitive op name (`transpose`). -/
def transposeAliasOpName : OpName :=
  mkDottedName "jax.lax.transpose"

/-- Canonical structural primitive op name (`reshape`). -/
def reshapeAliasOpName : OpName :=
  mkDottedName "jax.lax.reshape"

/-- Canonical structural primitive op name (`squeeze`). -/
def squeezeAliasOpName : OpName :=
  mkDottedName "jax.lax.squeeze"

/-- Canonical structural primitive op name (`broadcast_in_dim`). -/
def broadcastInDimAliasOpName : OpName :=
  mkDottedName "jax.lax.broadcast_in_dim"

/-- Canonical structural primitive op name (`slice`). -/
def sliceAliasOpName : OpName :=
  mkDottedName "jax.lax.slice"

/-- Canonical structural primitive op name (`convert_element_type`). -/
def convertElementTypeAliasOpName : OpName :=
  mkDottedName "jax.lax.convert_element_type"

/-- Canonical structural primitive op name (`concatenate`). -/
def concatenateAliasOpName : OpName :=
  mkDottedName "jax.lax.concatenate"

/-- Canonical reduction primitive op name (`reduce_sum`). -/
def reduceSumAliasOpName : OpName :=
  mkDottedName "jax.lax.reduce_sum"

/-- Canonical reduction primitive op name (`reduce_max`). -/
def reduceMaxAliasOpName : OpName :=
  mkDottedName "jax.lax.reduce_max"

/-- Canonical reduction primitive op name (`reduce_min`). -/
def reduceMinAliasOpName : OpName :=
  mkDottedName "jax.lax.reduce_min"

/-- Canonical control primitive op name (`select_n`). -/
def selectNAliasOpName : OpName :=
  mkDottedName "jax.lax.select_n"

/-- Canonical control primitive op name (`scan`). -/
def scanAliasOpName : OpName :=
  mkDottedName "jax.lax.scan"

/-- Canonical control primitive op name (`scan_p`). -/
def scanPrimAliasOpName : OpName :=
  mkDottedName "jax.lax.scan_p"

/-- Canonical control primitive op name (`cond`). -/
def condAliasOpName : OpName :=
  mkDottedName "jax.lax.cond"

/-- Canonical control primitive op name (`cond_p`). -/
def condPrimAliasOpName : OpName :=
  mkDottedName "jax.lax.cond_p"

/-- Canonical control primitive op name (`select`). -/
def selectAliasOpName : OpName :=
  mkDottedName "jax.lax.select"

/-- Canonical structural primitive op name (`slice_in_dim`). -/
def sliceInDimAliasOpName : OpName :=
  mkDottedName "jax.lax.slice_in_dim"

/-- Canonical structural primitive op name (`pad`). -/
def padAliasOpName : OpName :=
  mkDottedName "jax.lax.pad"

/-- Canonical dynamic-structural primitive op name (`dynamic_slice`). -/
def dynamicSliceAliasOpName : OpName :=
  mkDottedName "jax.lax.dynamic_slice"

/-- Canonical dynamic-structural primitive op name (`dynamic_update_slice`). -/
def dynamicUpdateSliceAliasOpName : OpName :=
  mkDottedName "jax.lax.dynamic_update_slice"

/-- Canonical dynamic-structural primitive op name (`dynamic_update_index_in_dim`). -/
def dynamicUpdateIndexInDimAliasOpName : OpName :=
  mkDottedName "jax.lax.dynamic_update_index_in_dim"

/-- Canonical dynamic-structural primitive op name (`gather`). -/
def gatherAliasOpName : OpName :=
  mkDottedName "jax.lax.gather"

/-- Canonical dynamic-structural primitive op name (`scatter`). -/
def scatterAliasOpName : OpName :=
  mkDottedName "jax.lax.scatter"

/-- Canonical dynamic-structural primitive op name (`scatter_add`). -/
def scatterAddAliasOpName : OpName :=
  mkDottedName "jax.lax.scatter_add"

/-- Canonical dynamic-structural primitive op name (`scatter_min`). -/
def scatterMinAliasOpName : OpName :=
  mkDottedName "jax.lax.scatter_min"

/-- Canonical dynamic-structural primitive op name (`scatter_max`). -/
def scatterMaxAliasOpName : OpName :=
  mkDottedName "jax.lax.scatter_max"

/-- Common aliases accepted for stop-gradient lowering/rule registration. -/
def allStopGradientOpNames : Array OpName := #[
  stopGradientOpName,
  mkDottedName "jax.lax.stop_gradient_p",
  mkDottedName "jax.lax.stop_grad",
  mkDottedName "Tyr.GPU.KStmt.stopGradient",
  mkDottedName "Graphax.stop_gradient"
]

/-- Common aliases accepted for iota lowering/rule registration. -/
def allIotaOpNames : Array OpName := #[
  iotaOpName,
  mkDottedName "jax.lax.iota_p",
  mkDottedName "Tyr.GPU.KStmt.iota",
  mkDottedName "Graphax.iota"
]

/-- Common aliases accepted for `device_put` lowering/rule registration. -/
def allDevicePutOpNames : Array OpName := #[
  devicePutOpName,
  mkDottedName "jax.lax.device_put_p",
  mkDottedName "jax.device_put",
  mkDottedName "Tyr.GPU.KStmt.devicePut",
  mkDottedName "Graphax.device_put"
]

/-- Common aliases accepted for `pjit` lowering/rule registration. -/
def allPjitOpNames : Array OpName := #[
  pjitOpName,
  mkDottedName "jax.lax.pjit_p",
  mkDottedName "jax.lax.pjit",
  mkDottedName "pjit",
  mkDottedName "Tyr.GPU.KStmt.pjit",
  mkDottedName "Graphax.pjit"
]

/-- Common aliases accepted for collective communication primitives. -/
def allCommunicationUnaryOpNames : Array OpName := #[
  allGatherOpName,
  mkDottedName "jax.lax.all_gather_p",
  allToAllOpName,
  mkDottedName "jax.lax.all_to_all_p",
  reduceScatterOpName,
  mkDottedName "jax.lax.reduce_scatter_p",
  collectivePermuteOpName,
  mkDottedName "jax.lax.collective_permute_p",
  psumOpName,
  mkDottedName "jax.lax.psum_p",
  pmeanOpName,
  mkDottedName "jax.lax.pmean_p",
  pmaxOpName,
  mkDottedName "jax.lax.pmax_p",
  pminOpName,
  mkDottedName "jax.lax.pmin_p",
  pswapaxesOpName,
  mkDottedName "jax.lax.pswapaxes_p",
  mkDottedName "jax.lax.all_reduce",
  mkDottedName "jax.lax.all_reduce_p",
  mkDottedName "jax.lax.allgather",
  mkDottedName "jax.lax.allgather_p",
  mkDottedName "Graphax.all_gather",
  mkDottedName "Graphax.all_to_all",
  mkDottedName "Graphax.reduce_scatter",
  mkDottedName "Graphax.collective_permute",
  mkDottedName "Graphax.psum",
  mkDottedName "Graphax.pmean",
  mkDottedName "Graphax.pmax",
  mkDottedName "Graphax.pmin",
  mkDottedName "Graphax.pswapaxes",
  mkDottedName "Tyr.Distributed.allGather",
  mkDottedName "Tyr.Distributed.allReduce"
]

/-- Common aliases accepted for structural unary primitives in LeanJaxpr paths. -/
def allStructuralUnaryAliasOpNames : Array OpName := #[
  transposeAliasOpName,
  mkDottedName "jax.lax.transpose_p",
  reshapeAliasOpName,
  mkDottedName "jax.lax.reshape_p",
  squeezeAliasOpName,
  mkDottedName "jax.lax.squeeze_p",
  broadcastInDimAliasOpName,
  mkDottedName "jax.lax.broadcast_in_dim_p",
  sliceAliasOpName,
  mkDottedName "jax.lax.slice_p",
  sliceInDimAliasOpName,
  mkDottedName "jax.lax.slice_in_dim_p",
  convertElementTypeAliasOpName,
  mkDottedName "jax.lax.convert_element_type_p",
  mkDottedName "Graphax.transpose",
  mkDottedName "Graphax.reshape",
  mkDottedName "Graphax.squeeze",
  mkDottedName "Graphax.broadcast_in_dim",
  mkDottedName "Graphax.slice",
  mkDottedName "Graphax.slice_in_dim",
  mkDottedName "Graphax.convert_element_type"
]

/-- Common aliases accepted for `pad`-style structural binary primitives. -/
def allPadAliasOpNames : Array OpName := #[
  padAliasOpName,
  mkDottedName "jax.lax.pad_p",
  mkDottedName "Graphax.pad"
]

/-- Common aliases accepted for concat-like variadic structural primitives. -/
def allConcatLikeAliasOpNames : Array OpName := #[
  concatenateAliasOpName,
  mkDottedName "jax.lax.concatenate_p",
  mkDottedName "Graphax.concatenate"
]

/-- Common aliases accepted for reduction unary primitives in LeanJaxpr paths. -/
def allReductionUnaryAliasOpNames : Array OpName := #[
  reduceSumAliasOpName,
  mkDottedName "jax.lax.reduce_sum_p",
  reduceMaxAliasOpName,
  mkDottedName "jax.lax.reduce_max_p",
  reduceMinAliasOpName,
  mkDottedName "jax.lax.reduce_min_p",
  mkDottedName "Graphax.reduce_sum",
  mkDottedName "Graphax.reduce_max",
  mkDottedName "Graphax.reduce_min"
]

/-- Common aliases accepted for `select_n` style control primitives. -/
def allSelectNAliasOpNames : Array OpName := #[
  selectAliasOpName,
  mkDottedName "jax.lax.select_p",
  selectNAliasOpName,
  mkDottedName "jax.lax.select_n_p",
  mkDottedName "Graphax.select",
  mkDottedName "Graphax.select_n"
]

/-- `select`-style aliases with fixed ternary arity (pred/true/false). -/
def allSelectFixedArityAliasOpNames : Array OpName := #[
  selectAliasOpName,
  mkDottedName "jax.lax.select_p",
  mkDottedName "Graphax.select"
]

/--
Common aliases for dynamic projection-like primitives.
Derivative edges are emitted from the first input (indices are non-diff here).
-/
def allDynamicProjectionAliasOpNames : Array OpName := #[
  dynamicSliceAliasOpName,
  mkDottedName "jax.lax.dynamic_slice_p",
  gatherAliasOpName,
  mkDottedName "jax.lax.gather_p",
  mkDottedName "Graphax.dynamic_slice",
  mkDottedName "Graphax.gather"
]

/--
Common aliases for dynamic update-like primitives.
Derivative edges are emitted for base/update inputs (indices are non-diff here).
-/
def allDynamicUpdateAliasOpNames : Array OpName := #[
  dynamicUpdateSliceAliasOpName,
  mkDottedName "jax.lax.dynamic_update_slice_p",
  dynamicUpdateIndexInDimAliasOpName,
  mkDottedName "jax.lax.dynamic_update_index_in_dim_p",
  scatterAliasOpName,
  mkDottedName "jax.lax.scatter_p",
  scatterAddAliasOpName,
  mkDottedName "jax.lax.scatter_add_p",
  scatterMinAliasOpName,
  mkDottedName "jax.lax.scatter_min_p",
  scatterMaxAliasOpName,
  mkDottedName "jax.lax.scatter_max_p",
  mkDottedName "Graphax.dynamic_update_slice",
  mkDottedName "Graphax.dynamic_update_index_in_dim",
  mkDottedName "Graphax.scatter",
  mkDottedName "Graphax.scatter_add",
  mkDottedName "Graphax.scatter_min",
  mkDottedName "Graphax.scatter_max"
]

/-- `dynamic_update_index_in_dim` aliases with fixed ternary arity. -/
def allDynamicUpdateIndexInDimAliasOpNames : Array OpName := #[
  dynamicUpdateIndexInDimAliasOpName,
  mkDottedName "jax.lax.dynamic_update_index_in_dim_p",
  mkDottedName "Graphax.dynamic_update_index_in_dim"
]

/-- Common aliases accepted for dot-general lowering/rule registration. -/
def allDotGeneralOpNames : Array OpName := #[
  kstmtDotGeneralOpName,
  mkDottedName "jax.lax.dot_general",
  mkDottedName "jax.lax.dot_general_p",
  mkDottedName "Graphax.dot_general"
]

/--
Common aliases for Graphax/JAX extra unary elemental primitives that are not
part of the current `UnaryOp` KStmt vocabulary.
-/
def allGraphaxExtraUnaryAliasOpNames : Array OpName := #[
  mkDottedName "jax.lax.log1p",
  mkDottedName "jax.lax.log1p_p",
  mkDottedName "Graphax.log1p",
  mkDottedName "jax.lax.asin",
  mkDottedName "jax.lax.asin_p",
  mkDottedName "Graphax.asin",
  mkDottedName "jax.lax.acos",
  mkDottedName "jax.lax.acos_p",
  mkDottedName "Graphax.acos",
  mkDottedName "jax.lax.atan",
  mkDottedName "jax.lax.atan_p",
  mkDottedName "Graphax.atan",
  mkDottedName "jax.lax.tan",
  mkDottedName "jax.lax.tan_p",
  mkDottedName "Graphax.tan",
  mkDottedName "jax.lax.sinh",
  mkDottedName "jax.lax.sinh_p",
  mkDottedName "Graphax.sinh",
  mkDottedName "jax.lax.cosh",
  mkDottedName "jax.lax.cosh_p",
  mkDottedName "Graphax.cosh",
  mkDottedName "jax.lax.asinh",
  mkDottedName "jax.lax.asinh_p",
  mkDottedName "Graphax.asinh",
  mkDottedName "jax.lax.acosh",
  mkDottedName "jax.lax.acosh_p",
  mkDottedName "Graphax.acosh",
  mkDottedName "jax.lax.atanh",
  mkDottedName "jax.lax.atanh_p",
  mkDottedName "Graphax.atanh",
  mkDottedName "jax.lax.erf",
  mkDottedName "jax.lax.erf_p",
  mkDottedName "Graphax.erf",
  mkDottedName "jax.lax.integer_pow",
  mkDottedName "jax.lax.integer_pow_p",
  mkDottedName "Graphax.integer_pow"
]

/--
Common aliases for Graphax/JAX extra binary primitives that are not part of the
current `BinaryOp` KStmt vocabulary.
-/
def allGraphaxExtraBinaryAliasOpNames : Array OpName := #[
  mkDottedName "jax.lax.pow",
  mkDottedName "jax.lax.pow_p",
  mkDottedName "Graphax.pow",
  mkDottedName "jax.lax.atan2",
  mkDottedName "jax.lax.atan2_p",
  mkDottedName "Graphax.atan2"
]

/--
Common aliases for Graphax/JAX comparison primitives that intentionally emit
zero local-Jacobian edges.
-/
def allGraphaxZeroBinaryAliasOpNames : Array OpName := #[
  mkDottedName "jax.lax.eq",
  mkDottedName "jax.lax.eq_p",
  mkDottedName "Graphax.eq",
  mkDottedName "jax.lax.gt",
  mkDottedName "jax.lax.gt_p",
  mkDottedName "Graphax.gt",
  mkDottedName "jax.lax.lt",
  mkDottedName "jax.lax.lt_p",
  mkDottedName "Graphax.lt"
]

/-- Common aliases accepted for `scan` style higher-order control primitives. -/
def allScanAliasOpNames : Array OpName := #[
  scanAliasOpName,
  scanPrimAliasOpName,
  mkDottedName "Graphax.scan"
]

/-- Common aliases accepted for `cond` style higher-order control primitives. -/
def allCondAliasOpNames : Array OpName := #[
  condAliasOpName,
  condPrimAliasOpName,
  mkDottedName "Graphax.cond"
]

/-- Common aliases accepted for higher-order control primitives. -/
def allHigherOrderControlAliasOpNames : Array OpName :=
  allScanAliasOpNames ++ allCondAliasOpNames

/--
Backward-compatible alias retained for existing callers that previously
tracked these names as unsupported.
-/
def allUnsupportedHigherOrderControlAliasOpNames : Array OpName :=
  allHigherOrderControlAliasOpNames

def isStopGradientOpName (op : OpName) : Bool :=
  allStopGradientOpNames.any (· == op)

def isIotaOpName (op : OpName) : Bool :=
  allIotaOpNames.any (· == op)

def isDevicePutOpName (op : OpName) : Bool :=
  allDevicePutOpNames.any (· == op)

def isPjitOpName (op : OpName) : Bool :=
  allPjitOpNames.any (· == op)

def isCommunicationUnaryOpName (op : OpName) : Bool :=
  allCommunicationUnaryOpNames.any (· == op)

def isStructuralUnaryAliasOpName (op : OpName) : Bool :=
  allStructuralUnaryAliasOpNames.any (· == op)

def isPadAliasOpName (op : OpName) : Bool :=
  allPadAliasOpNames.any (· == op)

def isConcatLikeAliasOpName (op : OpName) : Bool :=
  allConcatLikeAliasOpNames.any (· == op)

def isReductionUnaryAliasOpName (op : OpName) : Bool :=
  allReductionUnaryAliasOpNames.any (· == op)

def isSelectNAliasOpName (op : OpName) : Bool :=
  allSelectNAliasOpNames.any (· == op)

def isSelectFixedArityAliasOpName (op : OpName) : Bool :=
  allSelectFixedArityAliasOpNames.any (· == op)

def isScanAliasOpName (op : OpName) : Bool :=
  allScanAliasOpNames.any (· == op)

def isCondAliasOpName (op : OpName) : Bool :=
  allCondAliasOpNames.any (· == op)

def isHigherOrderControlAliasOpName (op : OpName) : Bool :=
  allHigherOrderControlAliasOpNames.any (· == op)

def isDynamicProjectionAliasOpName (op : OpName) : Bool :=
  allDynamicProjectionAliasOpNames.any (· == op)

def isDynamicUpdateAliasOpName (op : OpName) : Bool :=
  allDynamicUpdateAliasOpNames.any (· == op)

def isDynamicUpdateIndexInDimAliasOpName (op : OpName) : Bool :=
  allDynamicUpdateIndexInDimAliasOpNames.any (· == op)

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

/-- KStmt MMA transpose universe lowered into LeanJaxpr equations. -/
def allKStmtMMATransposes : Array MMATranspose := #[
  .AB, .ABt, .AtB, .AtBt
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
  for trans in allKStmtMMATransposes do
    out := out.push (kstmtMmaOpName trans)
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
