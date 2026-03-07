import Tyr.AD.JaxprLike.KStmtNames
import Tyr.GPU.Codegen.IR

/-!
# Tyr.AD.Elim.LowerKStmt

Conservative lowering from normalized `LeanJaxpr` equations back into GPU
`KStmt` operations for the KStmt-encodable op-name subset.
-/

namespace Tyr.AD.Elim

open Tyr.AD.JaxprLike
open Tyr.GPU.Codegen

private def mkVarId (v : JVar) : VarId :=
  { idx := v.id }

private def malformedEqnError (idx0 : Nat) (eqn : JEqn) (message : String) : String :=
  s!"lowerToKStmts: eqn #{idx0} (`{eqn.op}`): {message}"

private def requireSingleOutVar (idx0 : Nat) (eqn : JEqn) : Except String VarId := do
  if eqn.outvars.size != 1 then
    throw <|
      malformedEqnError idx0 eqn s!"expected exactly one output variable, got {eqn.outvars.size}"
  match eqn.outvars[0]? with
  | none =>
    throw <|
      malformedEqnError idx0 eqn "expected one output variable, got none"
  | some outv =>
    return mkVarId outv

private def requireInVarArity
    (idx0 : Nat)
    (eqn : JEqn)
    (expected : Nat) :
    Except String (Array VarId) := do
  if eqn.invars.size != expected then
    throw <|
      malformedEqnError idx0 eqn s!"expected {expected} input variables, got {eqn.invars.size}"
  return eqn.invars.map mkVarId

private def requireNatParam
    (idx0 : Nat)
    (eqn : JEqn)
    (key : OpParamKey)
    (name : String) :
    Except String Nat :=
  match eqn.params.findNat? key with
  | some value => .ok value
  | none =>
    .error <|
      malformedEqnError idx0 eqn s!"missing Nat parameter `{name}`"

private def decodeUnaryOp? (opName : OpName) : Option UnaryOp :=
  allKStmtUnaryOps.find? (fun op => kstmtUnaryOpName op == opName)

private def decodeBinaryOp? (opName : OpName) : Option BinaryOp :=
  allKStmtBinaryOps.find? (fun op => kstmtBinaryOpName op == opName)

private def decodeReduceOpAxis? (opName : OpName) : Option (ReduceOp × ReduceAxis) := Id.run do
  for op in allKStmtReduceOps do
    for axis in allKStmtReduceAxes do
      if kstmtReduceOpName op axis == opName then
        return some (op, axis)
  return none

private def decodeReduceAccumOpAxis? (opName : OpName) : Option (ReduceOp × ReduceAxis) := Id.run do
  for op in allKStmtReduceOps do
    for axis in allKStmtReduceAxes do
      if kstmtReduceAccumOpName op axis == opName then
        return some (op, axis)
  return none

private def decodeBroadcastAxis? (opName : OpName) : Option BroadcastAxis :=
  allKStmtBroadcastAxes.find? (fun axis => kstmtBroadcastOpName axis == opName)

private def decodeBinaryBroadcast? (opName : OpName) : Option (BinaryOp × BroadcastAxis) := Id.run do
  for op in allKStmtBinaryOps do
    for axis in allKStmtBroadcastAxes do
      if kstmtBinaryBroadcastOpName op axis == opName then
        return some (op, axis)
  return none

private def decodeCumsumAxis? (opName : OpName) : Option ReduceAxis :=
  allKStmtReduceAxes.find? (fun axis => kstmtCumsumOpName axis == opName)

private def decodeCumprodAxis? (opName : OpName) : Option ReduceAxis :=
  allKStmtReduceAxes.find? (fun axis => kstmtCumprodOpName axis == opName)

/-- True iff `opName` can be lowered into a concrete `KStmt` constructor. -/
def isKStmtLowerableOpName (opName : OpName) : Bool :=
  (decodeUnaryOp? opName).isSome ||
  (decodeBinaryOp? opName).isSome ||
  (decodeReduceOpAxis? opName).isSome ||
  (decodeReduceAccumOpAxis? opName).isSome ||
  (decodeBroadcastAxis? opName).isSome ||
  (decodeBinaryBroadcast? opName).isSome ||
  (decodeCumsumAxis? opName).isSome ||
  (decodeCumprodAxis? opName).isSome ||
  opName == kstmtTransposeOpName ||
  opName == kstmtSwapLayoutOpName ||
  opName == kstmtConvertOpName ||
  opName == kstmtSliceRowsOpName ||
  opName == kstmtSliceColsOpName ||
  opName == kstmtConcatColsOpName ||
  opName == kstmtOuterOpName

private def lowerEqnToKStmt (idx0 : Nat) (eqn : JEqn) : Except String KStmt := do
  let dst ← requireSingleOutVar idx0 eqn

  if let some op := decodeUnaryOp? eqn.op then
    let invars ← requireInVarArity idx0 eqn 1
    return .unary op dst invars[0]!

  if let some op := decodeBinaryOp? eqn.op then
    let invars ← requireInVarArity idx0 eqn 2
    return .binary op dst invars[0]! invars[1]!

  if let some (op, axis) := decodeReduceOpAxis? eqn.op then
    let invars ← requireInVarArity idx0 eqn 1
    return .reduce op axis dst invars[0]!

  if let some (op, axis) := decodeReduceAccumOpAxis? eqn.op then
    let invars ← requireInVarArity idx0 eqn 2
    return .reduceAccum op axis dst invars[0]! invars[1]!

  if let some axis := decodeBroadcastAxis? eqn.op then
    let invars ← requireInVarArity idx0 eqn 1
    return .broadcast axis dst invars[0]!

  if let some (op, axis) := decodeBinaryBroadcast? eqn.op then
    let invars ← requireInVarArity idx0 eqn 2
    return .binaryBroadcast op axis dst invars[0]! invars[1]!

  if eqn.op == kstmtTransposeOpName then
    let invars ← requireInVarArity idx0 eqn 1
    return .transpose dst invars[0]!

  if eqn.op == kstmtSwapLayoutOpName then
    let invars ← requireInVarArity idx0 eqn 1
    return .swapLayout dst invars[0]!

  if eqn.op == kstmtConvertOpName then
    let invars ← requireInVarArity idx0 eqn 1
    return .convert dst invars[0]!

  if eqn.op == kstmtSliceRowsOpName then
    let invars ← requireInVarArity idx0 eqn 1
    let startRow ← requireNatParam idx0 eqn .startRow "startRow"
    let numRows ← requireNatParam idx0 eqn .numRows "numRows"
    return .sliceRows dst invars[0]! startRow numRows

  if eqn.op == kstmtSliceColsOpName then
    let invars ← requireInVarArity idx0 eqn 1
    let startCol ← requireNatParam idx0 eqn .startCol "startCol"
    let numCols ← requireNatParam idx0 eqn .numCols "numCols"
    return .sliceCols dst invars[0]! startCol numCols

  if eqn.op == kstmtConcatColsOpName then
    let invars ← requireInVarArity idx0 eqn 2
    return .concatCols dst invars[0]! invars[1]!

  if eqn.op == kstmtOuterOpName then
    let invars ← requireInVarArity idx0 eqn 2
    return .outer dst invars[0]! invars[1]!

  if let some axis := decodeCumsumAxis? eqn.op then
    let invars ← requireInVarArity idx0 eqn 1
    return .cumsum axis dst invars[0]!

  if let some axis := decodeCumprodAxis? eqn.op then
    let invars ← requireInVarArity idx0 eqn 1
    return .cumprod axis dst invars[0]!

  throw <|
    malformedEqnError idx0 eqn "unsupported op name for KStmt lowering"

/--
Lower normalized equations in a `LeanJaxpr` back to `KStmt` statements for the
KStmt-encodable op-name subset.
-/
def lowerToKStmts (jaxpr : LeanJaxpr) : Except (Array String) (Array KStmt) := Id.run do
  let mut stmts : Array KStmt := #[]
  let mut errors : Array String := #[]

  if !jaxpr.constvars.isEmpty then
    errors := errors.push
      s!"lowerToKStmts: constvars are not lowerable to KStmt ({jaxpr.constvars.size} constvars found)"

  for h : idx0 in [:jaxpr.eqns.size] do
    let eqn := jaxpr.eqns[idx0]
    match lowerEqnToKStmt idx0 eqn with
    | .ok stmt =>
      stmts := stmts.push stmt
    | .error err =>
      errors := errors.push err

  if errors.isEmpty then
    return .ok stmts
  else
    return .error errors

end Tyr.AD.Elim
