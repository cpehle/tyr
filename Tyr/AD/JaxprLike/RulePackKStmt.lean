import Tyr.AD.JaxprLike.RuleRegistry
import Tyr.AD.JaxprLike.KStmtNames

/-!
# Tyr.AD.JaxprLike.RulePackKStmt

Built-in local-Jacobian rule-pack registration for `FromKStmt` op names.
-/

namespace Tyr.AD.JaxprLike

open Tyr.GPU.Codegen

private def atomName (value : String) : Lean.Name :=
  Lean.Name.str Lean.Name.anonymous value

private def reduceOpTagName (op : ReduceOp) : Lean.Name :=
  atomName (kstmtReduceOpTag op)

private def reduceAxisTagName (axis : ReduceAxis) : Lean.Name :=
  atomName (kstmtReduceAxisTag axis)

private def broadcastAxisTagName (axis : BroadcastAxis) : Lean.Name :=
  atomName (kstmtBroadcastAxisTag axis)

private def unaryJacTag (op : UnaryOp) (mode : Tyr.AD.Sparse.JacMode := .none) :
    Tyr.AD.Sparse.SparseMapTag :=
  .semantic (.unary (kstmtUnaryOpName op) .x mode)

private def binaryJacTag (op : BinaryOp) (arg : Tyr.AD.Sparse.JacArgRole)
    (mode : Tyr.AD.Sparse.JacMode := .none) :
    Tyr.AD.Sparse.SparseMapTag :=
  .semantic (.binary (kstmtBinaryOpName op) arg mode)

/-- 1x1 sparse map used for scalar local Jacobians in early KStmt rule packs. -/
private def scalarJacMap (tag : Tyr.AD.Sparse.SparseMapTag) (weight : Float) : SparseLinearMap :=
  {
    repr := tag
    inDim? := some 1
    outDim? := some 1
    entries := #[({ src := 0, dst := 0, weight := weight } : Tyr.AD.Sparse.SparseEntry)]
  }

private def requireUnaryShape (op : OpName) (eqn : JEqn) : Except RuleError (JVar × JVar) := do
  let outv ←
    match eqn.outvars[0]? with
    | some v => .ok v
    | none => .error (.malformedEqn s!"Unary rule `{op}` requires one output variable.")
  if eqn.outvars.size != 1 then
    .error (.malformedEqn s!"Unary rule `{op}` expects exactly one output variable, got {eqn.outvars.size}.")
  else
    let inv ←
      match eqn.invars[0]? with
      | some v => .ok v
      | none => .error (.malformedEqn s!"Unary rule `{op}` requires one input variable.")
    if eqn.invars.size != 1 then
      .error (.malformedEqn s!"Unary rule `{op}` expects exactly one input variable, got {eqn.invars.size}.")
    else
      .ok (inv, outv)

private def requireBinaryShape (op : OpName) (eqn : JEqn) : Except RuleError (JVar × JVar × JVar) := do
  let outv ←
    match eqn.outvars[0]? with
    | some v => .ok v
    | none => .error (.malformedEqn s!"Binary rule `{op}` requires one output variable.")
  if eqn.outvars.size != 1 then
    .error (.malformedEqn s!"Binary rule `{op}` expects exactly one output variable, got {eqn.outvars.size}.")
  else
    let a ←
      match eqn.invars[0]? with
      | some v => .ok v
      | none => .error (.malformedEqn s!"Binary rule `{op}` requires two input variables.")
    let b ←
      match eqn.invars[1]? with
      | some v => .ok v
      | none => .error (.malformedEqn s!"Binary rule `{op}` requires two input variables.")
    if eqn.invars.size != 2 then
      .error (.malformedEqn s!"Binary rule `{op}` expects exactly two input variables, got {eqn.invars.size}.")
    else
      .ok (a, b, outv)

private def requireNullaryShape (op : OpName) (eqn : JEqn) : Except RuleError JVar := do
  let outv ←
    match eqn.outvars[0]? with
    | some v => .ok v
    | none => .error (.malformedEqn s!"Nullary rule `{op}` requires one output variable.")
  if eqn.outvars.size != 1 then
    .error (.malformedEqn s!"Nullary rule `{op}` expects exactly one output variable, got {eqn.outvars.size}.")
  else if eqn.invars.size != 0 then
    .error (.malformedEqn s!"Nullary rule `{op}` expects zero input variables, got {eqn.invars.size}.")
  else
    .ok outv

private def unaryConstJac? (op : UnaryOp) : Option (Tyr.AD.Sparse.SparseMapTag × Float) :=
  match op with
  | .Copy => some (unaryJacTag .Copy, 1.0)
  | .Neg => some (unaryJacTag .Neg, -1.0)
  | .Zero => some (unaryJacTag .Zero, 0.0)
  | .One => some (unaryJacTag .One, 0.0)
  | .PosInfty => some (unaryJacTag .PosInfty, 0.0)
  | .NegInfty => some (unaryJacTag .NegInfty, 0.0)
  | _ => none

private def unarySymbolicJac? (op : UnaryOp) : Option (Tyr.AD.Sparse.SparseMapTag × Float) :=
  match op with
  | .Exp => some (unaryJacTag .Exp, 1.0)
  | .Exp2 => some (unaryJacTag .Exp2, 1.0)
  | .Log => some (unaryJacTag .Log, 1.0)
  | .Log2 => some (unaryJacTag .Log2, 1.0)
  | .Abs => some (unaryJacTag .Abs, 1.0)
  | .Relu => some (unaryJacTag .Relu, 1.0)
  | .Sqrt => some (unaryJacTag .Sqrt, 1.0)
  | .Rsqrt => some (unaryJacTag .Rsqrt, 1.0)
  | .Tanh => some (unaryJacTag .Tanh, 1.0)
  | .FastTanh => some (unaryJacTag .FastTanh, 1.0)
  | .Sigmoid => some (unaryJacTag .Sigmoid, 1.0)
  | .Gelu => some (unaryJacTag .Gelu, 1.0)
  | .Silu => some (unaryJacTag .Silu, 1.0)
  | .Swish => some (unaryJacTag .Swish, 1.0)
  | .Sin => some (unaryJacTag .Sin, 1.0)
  | .Cos => some (unaryJacTag .Cos, 1.0)
  | .Recip => some (unaryJacTag .Recip, 1.0)
  | .Square => some (unaryJacTag .Square, 1.0)
  | _ => none

private def binaryConstJac? (op : BinaryOp) :
    Option ((Tyr.AD.Sparse.SparseMapTag × Float) × (Tyr.AD.Sparse.SparseMapTag × Float)) :=
  match op with
  | .Add => some ((binaryJacTag .Add .a, 1.0), (binaryJacTag .Add .b, 1.0))
  | .Sub => some ((binaryJacTag .Sub .a, 1.0), (binaryJacTag .Sub .b, -1.0))
  | _ => none

private def binarySymbolicJac? (op : BinaryOp) :
    Option ((Tyr.AD.Sparse.SparseMapTag × Float) × (Tyr.AD.Sparse.SparseMapTag × Float)) :=
  match op with
  | .Mul =>
    some
      ((binaryJacTag .Mul .a .rhsValue, 1.0), (binaryJacTag .Mul .b .lhsValue, 1.0))
  | .Div =>
    some
      ( (binaryJacTag .Div .a .reciprocalRhs, 1.0),
        (binaryJacTag .Div .b .negLhsOverRhsSq, -1.0) )
  | .Max =>
    some
      ((binaryJacTag .Max .a .mask, 1.0), (binaryJacTag .Max .b .complementMask, 1.0))
  | .Min =>
    some
      ((binaryJacTag .Min .a .mask, 1.0), (binaryJacTag .Min .b .complementMask, 1.0))
  | _ => none

private def unaryConstRule
    (op : UnaryOp)
    (jacTag : Tyr.AD.Sparse.SparseMapTag)
    (jacWeight : Float) :
    LocalJacRule :=
  fun eqn _ctx => do
    let (inv, outv) ← requireUnaryShape (kstmtUnaryOpName op) eqn
    .ok #[{ src := inv.id, dst := outv.id, map := scalarJacMap jacTag jacWeight }]

private def binaryConstRule
    (op : BinaryOp)
    (left : Tyr.AD.Sparse.SparseMapTag × Float)
    (right : Tyr.AD.Sparse.SparseMapTag × Float) :
    LocalJacRule :=
  fun eqn _ctx => do
    let (a, b, outv) ← requireBinaryShape (kstmtBinaryOpName op) eqn
    .ok #[
      { src := a.id, dst := outv.id, map := scalarJacMap left.1 left.2 },
      { src := b.id, dst := outv.id, map := scalarJacMap right.1 right.2 }
    ]

private def unarySemanticsRule? (op : UnaryOp) : Option LocalJacRule :=
  match unaryConstJac? op with
  | some (repr, weight) => some (unaryConstRule op repr weight)
  | none =>
    match unarySymbolicJac? op with
    | some (repr, weight) => some (unaryConstRule op repr weight)
    | none => none

private def binarySemanticsRule? (op : BinaryOp) : Option LocalJacRule :=
  match binaryConstJac? op with
  | some (left, right) => some (binaryConstRule op left right)
  | none =>
    match binarySymbolicJac? op with
    | some (left, right) => some (binaryConstRule op left right)
    | none => none

private def unarySymbolicRule
    (opName : OpName)
    (jacTag : Tyr.AD.Sparse.SparseMapTag)
    (jacWeight : Float := 1.0) :
    LocalJacRule :=
  fun eqn _ctx => do
    let (inv, outv) ← requireUnaryShape opName eqn
    .ok #[{ src := inv.id, dst := outv.id, map := scalarJacMap jacTag jacWeight }]

private def binarySymbolicRule
    (opName : OpName)
    (left : Tyr.AD.Sparse.SparseMapTag × Float)
    (right : Tyr.AD.Sparse.SparseMapTag × Float) :
    LocalJacRule :=
  fun eqn _ctx => do
    let (a, b, outv) ← requireBinaryShape opName eqn
    .ok #[
      { src := a.id, dst := outv.id, map := scalarJacMap left.1 left.2 },
      { src := b.id, dst := outv.id, map := scalarJacMap right.1 right.2 }
    ]

private def reduceJacTag (op : ReduceOp) (axis : ReduceAxis) : Tyr.AD.Sparse.SparseMapTag :=
  .semantic (.reduce (reduceOpTagName op) (reduceAxisTagName axis) .x .contract)

private def reduceAccumSrcJacTag (op : ReduceOp) (axis : ReduceAxis) : Tyr.AD.Sparse.SparseMapTag :=
  .semantic (.reduceAccum (reduceOpTagName op) (reduceAxisTagName axis) .src .contract)

private def reduceAccumAccumJacTag (op : ReduceOp) (axis : ReduceAxis) : Tyr.AD.Sparse.SparseMapTag :=
  .semantic (.reduceAccum (reduceOpTagName op) (reduceAxisTagName axis) .accum .carry)

private def broadcastJacTag (axis : BroadcastAxis) : Tyr.AD.Sparse.SparseMapTag :=
  .semantic (.broadcast (broadcastAxisTagName axis) .x .expand)

private def binaryBroadcastJacPair
    (op : BinaryOp)
    (axis : BroadcastAxis) :
    ((Tyr.AD.Sparse.SparseMapTag × Float) × (Tyr.AD.Sparse.SparseMapTag × Float)) :=
  let axisTagName := broadcastAxisTagName axis
  let defaultPair :=
    ( (.semantic (.binaryBroadcast (kstmtBinaryOpName op) axisTagName .tile .tileBroadcast), 1.0),
      (.semantic (.binaryBroadcast (kstmtBinaryOpName op) axisTagName .vec .vecBroadcast), 1.0) )
  match binaryConstJac? op with
  | some (left, right) =>
    ( (.semantic (.binaryBroadcast (kstmtBinaryOpName op) axisTagName .tile .tileBroadcast), left.2),
      (.semantic (.binaryBroadcast (kstmtBinaryOpName op) axisTagName .vec .vecBroadcast), right.2) )
  | none =>
    match binarySymbolicJac? op with
    | some (left, right) =>
      ( (.semantic (.binaryBroadcast (kstmtBinaryOpName op) axisTagName .tile .tileBroadcast), left.2),
        (.semantic (.binaryBroadcast (kstmtBinaryOpName op) axisTagName .vec .vecBroadcast), right.2) )
    | none =>
      defaultPair

private def reduceStructuralRule (op : ReduceOp) (axis : ReduceAxis) : LocalJacRule :=
  unarySymbolicRule
    (kstmtReduceOpName op axis)
    (reduceJacTag op axis)

private def reduceAccumStructuralRule (op : ReduceOp) (axis : ReduceAxis) : LocalJacRule :=
  binarySymbolicRule
    (kstmtReduceAccumOpName op axis)
    (reduceAccumSrcJacTag op axis, 1.0)
    (reduceAccumAccumJacTag op axis, 1.0)

private def broadcastStructuralRule (axis : BroadcastAxis) : LocalJacRule :=
  unarySymbolicRule
    (kstmtBroadcastOpName axis)
    (broadcastJacTag axis)

private def binaryBroadcastStructuralRule (op : BinaryOp) (axis : BroadcastAxis) : LocalJacRule :=
  let pair := binaryBroadcastJacPair op axis
  binarySymbolicRule
    (kstmtBinaryBroadcastOpName op axis)
    pair.1
    pair.2

private def transposeStructuralRule : LocalJacRule :=
  unarySymbolicRule kstmtTransposeOpName (.semantic (.transpose .x .permute))

private def swapLayoutStructuralRule : LocalJacRule :=
  unarySymbolicRule kstmtSwapLayoutOpName (.semantic (.swapLayout .x .layoutPermute))

private def convertStructuralRule : LocalJacRule :=
  unarySymbolicRule kstmtConvertOpName (.semantic (.convert .x .cast))

private def sliceRowsStructuralRule : LocalJacRule :=
  unarySymbolicRule kstmtSliceRowsOpName (.semantic (.sliceRows .x .projection))

private def sliceColsStructuralRule : LocalJacRule :=
  unarySymbolicRule kstmtSliceColsOpName (.semantic (.sliceCols .x .projection))

private def concatColsStructuralRule : LocalJacRule :=
  binarySymbolicRule
    kstmtConcatColsOpName
    (.semantic (.concatCols .lhs .inject), 1.0)
    (.semantic (.concatCols .rhs .inject), 1.0)

private def outerStructuralRule : LocalJacRule :=
  binarySymbolicRule
    kstmtOuterOpName
    (.semantic (.outer .a .kronOther), 1.0)
    (.semantic (.outer .b .kronOther), 1.0)

private def dotGeneralRule (opName : OpName) : LocalJacRule :=
  fun eqn _ctx => do
    let (a, b, outv) ← requireBinaryShape opName eqn
    let lhsContract := (eqn.params.findNats? .lhsContract).getD #[]
    let rhsContract := (eqn.params.findNats? .rhsContract).getD #[]
    let lhsBatch := (eqn.params.findNats? .lhsBatch).getD #[]
    let rhsBatch := (eqn.params.findNats? .rhsBatch).getD #[]
    let variant := (eqn.params.findName? .variant).getD (atomName "generic")
    .ok #[
      { src := a.id, dst := outv.id, map := scalarJacMap (.semantic (.dotGeneral {
          variant := variant
          arg := .lhs
          lhsContract := lhsContract
          rhsContract := rhsContract
          lhsBatch := lhsBatch
          rhsBatch := rhsBatch
        })) 1.0 },
      { src := b.id, dst := outv.id, map := scalarJacMap (.semantic (.dotGeneral {
          variant := variant
          arg := .rhs
          lhsContract := lhsContract
          rhsContract := rhsContract
          lhsBatch := lhsBatch
          rhsBatch := rhsBatch
        })) 1.0 }
    ]

private def stopGradientNoGradRule (opName : OpName) : LocalJacRule :=
  fun eqn _ctx => do
    let (_inv, _outv) ← requireUnaryShape opName eqn
    .ok #[]

private def iotaNoGradRule (opName : OpName) : LocalJacRule :=
  fun eqn _ctx => do
    let _outv ← requireNullaryShape opName eqn
    .ok #[]

private def cumsumStructuralRule (axis : ReduceAxis) : LocalJacRule :=
  unarySymbolicRule
    (kstmtCumsumOpName axis)
    (.semantic (.cumsum (reduceAxisTagName axis) .x .prefix))

private def cumprodStructuralRule (axis : ReduceAxis) : LocalJacRule :=
  unarySymbolicRule
    (kstmtCumprodOpName axis)
    (.semantic (.cumprod (reduceAxisTagName axis) .x .prefixProduct))

/-- Register placeholder local-Jac rules for every KStmt unary and binary op. -/
def registerKStmtUnaryBinaryPlaceholderRules : Lean.CoreM Unit := do
  for op in allKStmtUnaryOps do
    registerLocalJacRule (kstmtUnaryOpName op) defaultPlaceholderRule
  for op in allKStmtBinaryOps do
    registerLocalJacRule (kstmtBinaryOpName op) defaultPlaceholderRule

/-- Register placeholder local-Jac rules for non-unary/binary KStmt op names. -/
def registerKStmtExtendedPlaceholderRules : Lean.CoreM Unit := do
  for op in allKStmtExtendedOpNames do
    registerLocalJacRule op defaultPlaceholderRule

/-- Register placeholder local-Jac rules for all currently lowered KStmt op names. -/
def registerKStmtAllSupportedPlaceholderRules : Lean.CoreM Unit := do
  registerKStmtUnaryBinaryPlaceholderRules
  registerKStmtExtendedPlaceholderRules

/--
Register constant-Jacobian local-Jac rules for linear/constant unary and binary ops:
- unary: `Copy`, `Neg`, `Zero`, `One`, `PosInfty`, `NegInfty`
- binary: `Add`, `Sub`
-/
def registerKStmtLinearSemanticsRules : Lean.CoreM Unit := do
  for op in allKStmtUnaryOps do
    match unaryConstJac? op with
    | some (repr, weight) =>
      registerLocalJacRule (kstmtUnaryOpName op) (unaryConstRule op repr weight)
    | none => pure ()
  for op in allKStmtBinaryOps do
    match binaryConstJac? op with
    | some (left, right) =>
      registerLocalJacRule (kstmtBinaryOpName op) (binaryConstRule op left right)
    | none => pure ()

/--
Register hybrid local-Jac rules for KStmt unary/binary ops:
- use Graphax/AlphaGrad-aligned semantics where available
- use `defaultPlaceholderRule` for remaining ops
-/
def registerKStmtUnaryBinaryHybridRules : Lean.CoreM Unit := do
  for op in allKStmtUnaryOps do
    registerLocalJacRule
      (kstmtUnaryOpName op)
      ((unarySemanticsRule? op).getD defaultPlaceholderRule)
  for op in allKStmtBinaryOps do
    registerLocalJacRule
      (kstmtBinaryOpName op)
      ((binarySemanticsRule? op).getD defaultPlaceholderRule)

/--
Register semantic local-Jac rules for all KStmt unary/binary ops that currently
have Graphax/AlphaGrad-overlap formulas in this registry.
-/
def registerKStmtGraphaxAlphaGradSemanticsRules : Lean.CoreM Unit := do
  for op in allKStmtUnaryOps do
    match unarySemanticsRule? op with
    | some rule => registerLocalJacRule (kstmtUnaryOpName op) rule
    | none => pure ()
  for op in allKStmtBinaryOps do
    match binarySemanticsRule? op with
    | some rule => registerLocalJacRule (kstmtBinaryOpName op) rule
    | none => pure ()

/--
Register structural local-Jacobian rules for non-unary/binary KStmt ops lowered
by `FromKStmt`.
-/
def registerKStmtStructuralSemanticsRules : Lean.CoreM Unit := do
  registerLocalJacRule kstmtTransposeOpName transposeStructuralRule
  registerLocalJacRule kstmtSwapLayoutOpName swapLayoutStructuralRule
  registerLocalJacRule kstmtConvertOpName convertStructuralRule
  registerLocalJacRule kstmtSliceRowsOpName sliceRowsStructuralRule
  registerLocalJacRule kstmtSliceColsOpName sliceColsStructuralRule
  registerLocalJacRule kstmtConcatColsOpName concatColsStructuralRule
  registerLocalJacRule kstmtOuterOpName outerStructuralRule
  registerLocalJacRule kstmtDotGeneralOpName (dotGeneralRule kstmtDotGeneralOpName)

  for axis in allKStmtBroadcastAxes do
    registerLocalJacRule (kstmtBroadcastOpName axis) (broadcastStructuralRule axis)

  for op in allKStmtBinaryOps do
    for axis in allKStmtBroadcastAxes do
      registerLocalJacRule
        (kstmtBinaryBroadcastOpName op axis)
        (binaryBroadcastStructuralRule op axis)

  for op in allKStmtReduceOps do
    for axis in allKStmtReduceAxes do
      registerLocalJacRule
        (kstmtReduceOpName op axis)
        (reduceStructuralRule op axis)
      registerLocalJacRule
        (kstmtReduceAccumOpName op axis)
        (reduceAccumStructuralRule op axis)

  for axis in allKStmtReduceAxes do
    registerLocalJacRule (kstmtCumsumOpName axis) (cumsumStructuralRule axis)
    registerLocalJacRule (kstmtCumprodOpName axis) (cumprodStructuralRule axis)

/--
Register local-Jac rules for all currently lowered KStmt ops:
- unary/binary use semantic-or-placeholder hybrid behavior
- structural/transform ops use explicit structural semantics
-/
def registerKStmtAllSupportedHybridRules : Lean.CoreM Unit := do
  registerKStmtUnaryBinaryHybridRules
  registerKStmtStructuralSemanticsRules

/--
Register no-grad/control primitive semantics used in Graphax/AlphaGrad-style
LeanJaxpr paths:
- `stop_gradient` emits no local-Jacobian edges.
- `iota` emits no local-Jacobian edges.
-/
def registerGraphaxAlphaGradNoGradControlRules : Lean.CoreM Unit := do
  for op in allStopGradientOpNames do
    registerLocalJacRule op (stopGradientNoGradRule op)
  for op in allIotaOpNames do
    registerLocalJacRule op (iotaNoGradRule op)

/-- Register dedicated dot-general local-Jacobian semantics (not outer-only approximation). -/
def registerGraphaxAlphaGradDotGeneralRules : Lean.CoreM Unit := do
  for op in allDotGeneralOpNames do
    registerLocalJacRule op (dotGeneralRule op)

/--
Register a parity-oriented semantic pack aligned with Graphax/AlphaGrad overlap:
- unary/binary overlap semantics
- structural semantics
- no-grad/control semantics
- dot-general semantics
-/
def registerGraphaxAlphaGradParityRules : Lean.CoreM Unit := do
  registerKStmtGraphaxAlphaGradSemanticsRules
  registerKStmtStructuralSemanticsRules
  registerGraphaxAlphaGradNoGradControlRules
  registerGraphaxAlphaGradDotGeneralRules

/-- Register placeholder local-Jac rules for an explicit op-name list. -/
def registerPlaceholderRulesForOps (ops : Array OpName) : Lean.CoreM Unit := do
  for op in ops do
    registerLocalJacRule op defaultPlaceholderRule

end Tyr.AD.JaxprLike
