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

private def decodeMmaTranspose? (opName : OpName) : Option MMATranspose :=
  allKStmtMMATransposes.find? (fun trans => kstmtMmaOpName trans == opName)

private def decodeMMTransposeFromContractAxes?
    (lhsContract rhsContract lhsBatch rhsBatch : Array Nat) : Option MMATranspose :=
  if !lhsBatch.isEmpty || !rhsBatch.isEmpty then
    none
  else if lhsContract == #[1] && rhsContract == #[0] then
    some .AB
  else if lhsContract == #[1] && rhsContract == #[1] then
    some .ABt
  else if lhsContract == #[0] && rhsContract == #[0] then
    some .AtB
  else if lhsContract == #[0] && rhsContract == #[1] then
    some .AtBt
  else
    none

private def shapeDim? (v : JVar) (idx : Nat) : Option Nat := do
  let shape ← v.metaInfo.shape
  shape[idx]?

private def isLeadingBatchAxes (axes : Array Nat) : Bool :=
  axes == (Array.range axes.size)

private def allUnitBatchAxes (v : JVar) (axes : Array Nat) : Bool :=
  axes.all (fun axis => shapeDim? v axis == some 1)

private def axesWithout (rank : Nat) (removed : Array Nat) : Array Nat :=
  (Array.range rank).filter fun axis => !removed.contains axis

private def axisIndex? (axes : Array Nat) (target : Nat) : Option Nat := Id.run do
  for i in [:axes.size] do
    if axes[i]! == target then
      return some i
  return none

private def reindexAxesAfterRemoving?
    (rank : Nat)
    (axes removed : Array Nat) : Option (Array Nat) := do
  let kept := axesWithout rank removed
  let mut out : Array Nat := #[]
  for axis in axes do
    if axis >= rank || removed.contains axis then
      none
    let idx ← axisIndex? kept axis
    out := out.push idx
  return out

private def expectedBatchPrefixShape (rank : Nat) : Array Nat :=
  Array.replicate rank 1

private def shapeAtAxes (shape axes : Array Nat) : Array Nat :=
  axes.map fun axis => shape.getD axis 1

private def compressShapeAfterRemovingAxes
    (shape removed : Array Nat) : Array Nat :=
  shapeAtAxes shape (axesWithout shape.size removed)

private def compressDotGeneralSingletonAxes?
    (lhsContract rhsContract lhsBatch rhsBatch : Array Nat)
    (lhs rhs out : JVar) :
    Option (Array Nat × Array Nat × Array Nat × Array Nat) := do
  if lhsBatch.size != rhsBatch.size then
    none
  else if
      !(allUnitBatchAxes lhs lhsBatch &&
        allUnitBatchAxes rhs rhsBatch) then
    none
  else
    match lhs.metaInfo.shape, rhs.metaInfo.shape, out.metaInfo.shape with
    | some lhsShape, some rhsShape, some outShape =>
      let batchRank := lhsBatch.size
      let lhsNonBatchAxes := axesWithout lhsShape.size lhsBatch
      let rhsNonBatchAxes := axesWithout rhsShape.size rhsBatch
      let lhsNonBatchShape := shapeAtAxes lhsShape lhsNonBatchAxes
      let rhsNonBatchShape := shapeAtAxes rhsShape rhsNonBatchAxes
      let lhsContract' := reindexAxesAfterRemoving? lhsShape.size lhsContract lhsBatch
      let rhsContract' := reindexAxesAfterRemoving? rhsShape.size rhsContract rhsBatch
      match lhsContract', rhsContract' with
      | some lhsContract'', some rhsContract'' =>
        if lhsContract''.size != rhsContract''.size then
          none
        else
          let lhsFreeOrig := (Array.range lhsNonBatchShape.size).filter fun axis => !lhsContract''.contains axis
          let rhsFreeOrig := (Array.range rhsNonBatchShape.size).filter fun axis => !rhsContract''.contains axis
          let lhsFreeExtentsOrig := shapeAtAxes lhsNonBatchShape lhsFreeOrig
          let rhsFreeExtentsOrig := shapeAtAxes rhsNonBatchShape rhsFreeOrig
          let expectedOut :=
            expectedBatchPrefixShape batchRank ++ lhsFreeExtentsOrig ++ rhsFreeExtentsOrig
          if outShape != expectedOut then
            none
          else
            let mut lhsRemoved : Array Nat := #[]
            let mut rhsRemoved : Array Nat := #[]
            let mut lhsContractsKept : Array Nat := #[]
            let mut rhsContractsKept : Array Nat := #[]
            for i in [:lhsContract''.size] do
              let lhsAxis := lhsContract''[i]!
              let rhsAxis := rhsContract''[i]!
              let lhsExtent := lhsNonBatchShape.getD lhsAxis 1
              let rhsExtent := rhsNonBatchShape.getD rhsAxis 1
              if lhsExtent = 1 && rhsExtent = 1 then
                lhsRemoved := lhsRemoved.push lhsAxis
                rhsRemoved := rhsRemoved.push rhsAxis
              else
                lhsContractsKept := lhsContractsKept.push lhsAxis
                rhsContractsKept := rhsContractsKept.push rhsAxis
            for axis in lhsFreeOrig do
              if lhsNonBatchShape.getD axis 1 = 1 then
                lhsRemoved := lhsRemoved.push axis
            for axis in rhsFreeOrig do
              if rhsNonBatchShape.getD axis 1 = 1 then
                rhsRemoved := rhsRemoved.push axis
            let lhsContractCompressed := reindexAxesAfterRemoving? lhsNonBatchShape.size lhsContractsKept lhsRemoved
            let rhsContractCompressed := reindexAxesAfterRemoving? rhsNonBatchShape.size rhsContractsKept rhsRemoved
            match lhsContractCompressed, rhsContractCompressed with
            | some lhsCompressedContracts, some rhsCompressedContracts =>
              some
                ( compressShapeAfterRemovingAxes lhsNonBatchShape lhsRemoved
                , compressShapeAfterRemovingAxes rhsNonBatchShape rhsRemoved
                , lhsCompressedContracts
                , rhsCompressedContracts )
            | _, _ => none
      | _, _ => none
    | _, _, _ => none

private def decodeMMTransposeFromUnitBatchAxes?
    (lhsContract rhsContract lhsBatch rhsBatch : Array Nat)
    (lhs rhs out : JVar) :
    Option MMATranspose :=
  match compressDotGeneralSingletonAxes? lhsContract rhsContract lhsBatch rhsBatch lhs rhs out with
  | some (lhsShape, rhsShape, lhsContract', rhsContract') =>
      if lhsShape.size = 2 && rhsShape.size = 2 then
        decodeMMTransposeFromContractAxes? lhsContract' rhsContract' #[] #[]
      else
        none
  | none => none

private def decodeOuterFromUnitBatchAxes
    (lhsContract rhsContract lhsBatch rhsBatch : Array Nat)
    (lhs rhs out : JVar) :
    Bool :=
  match compressDotGeneralSingletonAxes? lhsContract rhsContract lhsBatch rhsBatch lhs rhs out with
  | some (lhsShape, rhsShape, lhsContract', rhsContract') =>
      lhsContract'.isEmpty && rhsContract'.isEmpty &&
        lhsShape.size = 1 && rhsShape.size = 1
  | none => false

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
  (decodeMmaTranspose? opName).isSome ||
  isDotGeneralOpName opName ||
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

  if isDotGeneralOpName eqn.op then
    let invars ← requireInVarArity idx0 eqn 2
    let lhsContract := (eqn.params.findNats? .lhsContract).getD #[]
    let rhsContract := (eqn.params.findNats? .rhsContract).getD #[]
    let lhsBatch := (eqn.params.findNats? .lhsBatch).getD #[]
    let rhsBatch := (eqn.params.findNats? .rhsBatch).getD #[]
    let lhsVar := eqn.invars[0]!
    let rhsVar := eqn.invars[1]!
    let outVar := eqn.outvars[0]!
    let directTrans? :=
      match decodeMMTransposeFromContractAxes? lhsContract rhsContract lhsBatch rhsBatch with
      | some trans =>
        match lhsVar.metaInfo.shape, rhsVar.metaInfo.shape, outVar.metaInfo.shape with
        | some lhsShape, some rhsShape, some outShape =>
          if lhsBatch.isEmpty && rhsBatch.isEmpty &&
              lhsShape.size = 2 && rhsShape.size = 2 && outShape.size = 2 then
            some trans
          else
            none
        | _, _, _ => some trans
      | none => none
    let trans? :=
      match directTrans? with
      | some trans => some trans
      | none =>
        decodeMMTransposeFromUnitBatchAxes?
          lhsContract rhsContract lhsBatch rhsBatch lhsVar rhsVar outVar
    match trans? with
    | some trans =>
      return .mm trans dst invars[0]! invars[1]!
    | none =>
      if
          (lhsContract.isEmpty && rhsContract.isEmpty && lhsBatch.isEmpty && rhsBatch.isEmpty) ||
          decodeOuterFromUnitBatchAxes lhsContract rhsContract lhsBatch rhsBatch lhsVar rhsVar outVar then
        return .outer dst invars[0]! invars[1]!
      else
        throw <|
          malformedEqnError idx0 eqn
            s!"dot-general parameters are not representable as KStmt mm/outer: lhsContract={lhsContract}, rhsContract={rhsContract}, lhsBatch={lhsBatch}, rhsBatch={rhsBatch}"

  if let some trans := decodeMmaTranspose? eqn.op then
    let invars ← requireInVarArity idx0 eqn 3
    return .mma trans dst invars[0]! invars[1]! invars[2]!

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
