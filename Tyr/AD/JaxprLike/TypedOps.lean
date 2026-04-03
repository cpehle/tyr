import Tyr.AD.JaxprLike.Core
import Tyr.AD.JaxprLike.KStmtNames

/-!
# Tyr.AD.JaxprLike.TypedOps

Helpers for deriving typed normalized ops from canonical op names and params.
-/

namespace Tyr.AD.JaxprLike

private def atomName (value : String) : Lean.Name :=
  Lean.Name.str Lean.Name.anonymous value

private def unknownAxisName : Lean.Name :=
  atomName "unknownAxis"

private def findUnaryKStmt? (opName : OpName) : Option Tyr.GPU.Codegen.UnaryOp :=
  allKStmtUnaryOps.find? (fun op => kstmtUnaryOpName op == opName)

private def findBinaryKStmt? (opName : OpName) : Option Tyr.GPU.Codegen.BinaryOp :=
  allKStmtBinaryOps.find? (fun op => kstmtBinaryOpName op == opName)

private def findBroadcastAxis? (opName : OpName) : Option Tyr.GPU.Codegen.BroadcastAxis :=
  allKStmtBroadcastAxes.find? (fun axis => kstmtBroadcastOpName axis == opName)

private def findBinaryBroadcast? (opName : OpName) :
    Option (Tyr.GPU.Codegen.BinaryOp × Tyr.GPU.Codegen.BroadcastAxis) := Id.run do
  for op in allKStmtBinaryOps do
    for axis in allKStmtBroadcastAxes do
      if kstmtBinaryBroadcastOpName op axis == opName then
        return some (op, axis)
  return none

private def findReduce? (opName : OpName) :
    Option (Tyr.GPU.Codegen.ReduceOp × Tyr.GPU.Codegen.ReduceAxis) := Id.run do
  for op in allKStmtReduceOps do
    for axis in allKStmtReduceAxes do
      if kstmtReduceOpName op axis == opName then
        return some (op, axis)
  return none

private def findReduceAccum? (opName : OpName) :
    Option (Tyr.GPU.Codegen.ReduceOp × Tyr.GPU.Codegen.ReduceAxis) := Id.run do
  for op in allKStmtReduceOps do
    for axis in allKStmtReduceAxes do
      if kstmtReduceAccumOpName op axis == opName then
        return some (op, axis)
  return none

private def findMmaTranspose? (opName : OpName) : Option Tyr.GPU.Codegen.MMATranspose :=
  allKStmtMMATransposes.find? (fun trans => kstmtMmaOpName trans == opName)

private def findCumsumAxis? (opName : OpName) : Option Tyr.GPU.Codegen.ReduceAxis :=
  allKStmtReduceAxes.find? (fun axis => kstmtCumsumOpName axis == opName)

private def findCumprodAxis? (opName : OpName) : Option Tyr.GPU.Codegen.ReduceAxis :=
  allKStmtReduceAxes.find? (fun axis => kstmtCumprodOpName axis == opName)

private def controlFlowInfo?
    (opName : OpName)
    (params : OpParams)
    (inputCount outputCount : Nat) :
    Option ControlFlowInfo :=
  if isCondAliasOpName opName || (params.findNat? .condPredicateCount).isSome then
    let predicateDefault := if inputCount = 0 then 0 else 1
    let predicateCount := (params.findNat? .condPredicateCount).getD predicateDefault
    let maxData := inputCount - predicateCount
    some {
      variant := `cond
      staticArgCount := (params.findNat? .controlStaticArgCount).getD 0
      predicateCount := predicateCount
      dataInputCount := (params.findNat? .condDataInputCount).getD maxData
    }
  else if isScanAliasOpName opName || (params.findNat? .scanCarryInputCount).isSome then
    let carryDefault := if inputCount = 0 then 0 else 1
    let carryInputCount := (params.findNat? .scanCarryInputCount).getD carryDefault
    let maxData := inputCount - carryInputCount
    let carryOutputDefault := min carryInputCount outputCount
    some {
      variant := `scan
      staticArgCount := (params.findNat? .controlStaticArgCount).getD 0
      dataInputCount := (params.findNat? .scanDataInputCount).getD maxData
      carryInputCount := carryInputCount
      carryOutputCount := (params.findNat? .scanCarryOutputCount).getD carryOutputDefault
    }
  else
    none

/--
Recover a typed normalized op when the canonical op name is one of the known
KStmt/native normalized primitives or control-flow aliases. Returns `none` when
the op should stay on the generic arity-based path.
-/
def typedOpForNormalizedOp?
    (opName : OpName)
    (params : OpParams)
    (inputCount outputCount : Nat) :
    Option TypedOp :=
  match controlFlowInfo? opName params inputCount outputCount with
  | some info =>
      some (TypedOp.controlFlow info)
  | none =>
      if isDotGeneralOpName opName then
        some <|
          TypedOp.dotGeneral
            ((params.findName? .variant).getD `generic)
            ((params.findNats? .lhsContract).getD #[])
            ((params.findNats? .rhsContract).getD #[])
            ((params.findNats? .lhsBatch).getD #[])
            ((params.findNats? .rhsBatch).getD #[])
      else if opName == kstmtTransposeOpName || opName == transposeAliasOpName then
        some TypedOp.transpose
      else if opName == kstmtSwapLayoutOpName then
        some TypedOp.swapLayout
      else if opName == kstmtConvertOpName || opName == convertElementTypeAliasOpName then
        some TypedOp.convert
      else
        match findUnaryKStmt? opName with
        | some _ => some (TypedOp.unary opName)
        | none =>
            match findBinaryKStmt? opName with
            | some _ => some (TypedOp.binary opName)
            | none =>
                match findReduce? opName with
                | some (op, axis) =>
                    some (TypedOp.reduce
                      (atomName (kstmtReduceOpTag op))
                      (atomName (kstmtReduceAxisTag axis)))
                | none =>
                    match findReduceAccum? opName with
                    | some (op, axis) =>
                        some (TypedOp.reduceAccum
                          (atomName (kstmtReduceOpTag op))
                          (atomName (kstmtReduceAxisTag axis)))
                    | none =>
                        match findBroadcastAxis? opName with
                        | some axis =>
                            some (TypedOp.broadcast (atomName (kstmtBroadcastAxisTag axis)))
                        | none =>
                            match findBinaryBroadcast? opName with
                            | some (op, axis) =>
                                some (TypedOp.binaryBroadcast
                                  (atomName (kstmtBinaryOpTag op))
                                  (atomName (kstmtBroadcastAxisTag axis)))
                            | none =>
                                if opName == kstmtSliceRowsOpName then
                                  some (TypedOp.sliceRows
                                    ((params.findNat? .startRow).getD 0)
                                    ((params.findNat? .numRows).getD 0))
                                else if opName == kstmtSliceColsOpName then
                                  some (TypedOp.sliceCols
                                    ((params.findNat? .startCol).getD 0)
                                    ((params.findNat? .numCols).getD 0))
                                else if opName == kstmtConcatColsOpName then
                                  some TypedOp.concatCols
                                else if opName == kstmtOuterOpName then
                                  some TypedOp.outer
                                else
                                  match findMmaTranspose? opName with
                                  | some trans =>
                                      let variant :=
                                        (params.findName? .variant).getD (atomName s!"mma.{toString trans}")
                                      some (TypedOp.mma variant)
                                  | none =>
                                      match findCumsumAxis? opName with
                                      | some axis =>
                                          some (TypedOp.cumsum (atomName (kstmtReduceAxisTag axis)))
                                      | none =>
                                          match findCumprodAxis? opName with
                                          | some axis =>
                                              some (TypedOp.cumprod (atomName (kstmtReduceAxisTag axis)))
                                          | none =>
                                              if isReductionUnaryAliasOpName opName then
                                                some (TypedOp.reduce opName unknownAxisName)
                                              else
                                                none

/--
Total typed-op derivation used by lowering paths. Unknown ops fall back to the
generic arity-based families so manual/test lowering stays deterministic.
-/
def typedOpForNormalizedOp
    (opName : OpName)
    (params : OpParams)
    (inputCount outputCount : Nat) :
    TypedOp :=
  match typedOpForNormalizedOp? opName params inputCount outputCount with
  | some typed => typed
  | none =>
      match inputCount, outputCount with
      | 0, 1 => TypedOp.nullary opName
      | 1, 1 => TypedOp.unary opName
      | 2, 1 => TypedOp.binary opName
      | 3, 1 => TypedOp.ternary opName
      | arity, 1 => TypedOp.nary opName arity
      | arity, _ => TypedOp.nary opName arity

namespace JEqn

/--
Normalized manual equation helper for tests/fixtures. This uses the shared
typed-op classifier rather than a separate heuristic path.
-/
def ofNormalizedOp
    (id : OpId)
    (op : OpName)
    (invars outvars : Array JVar)
    (params : OpParams := #[])
    (source : SourceRef := {}) :
    JEqn :=
  {
    id := id
    op := op
    invars := invars
    outvars := outvars
    params := params
    typed := typedOpForNormalizedOp op params invars.size outvars.size
    source := source
  }

end JEqn

end Tyr.AD.JaxprLike
