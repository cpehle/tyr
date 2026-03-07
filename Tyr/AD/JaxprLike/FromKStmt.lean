import Tyr.AD.JaxprLike.Core
import Tyr.AD.JaxprLike.KStmtNames
import Tyr.GPU.Codegen.IR

/-!
# Tyr.AD.JaxprLike.FromKStmt

Minimal conversion from GPU `KStmt` programs into the `LeanJaxpr` scaffold.
This converter supports element-wise unary/binary ops and a structural subset
used by Graphax/AlphaGrad-like elimination backends. Remaining statements are
rejected with explicit diagnostics.
-/

namespace Tyr.AD.JaxprLike

open Tyr.GPU.Codegen

/-- Convert a GPU `VarId` to a `JVar` using the same numeric identifier. -/
private def mkJVar (v : VarId) : JVar :=
  { id := v.idx }

/-- Keep insertion order while deduplicating variable IDs. -/
private def rememberId
    (order : Array Nat)
    (seen : Std.HashSet Nat)
    (id : Nat) :
    Array Nat × Std.HashSet Nat :=
  if seen.contains id then
    (order, seen)
  else
    (order.push id, seen.insert id)

/-- Diagnostic for unsupported `KStmt` constructors. -/
private def unsupportedStmtError (idx0 : Nat) (stmt : KStmt) : String :=
  s!"fromKStmts: unsupported KStmt at index {idx0}: {reprStr stmt}"

/-- Deterministic source marker for statement-level lowering diagnostics. -/
private def stmtSourceRef (idx0 : Nat) : SourceRef :=
  { decl := `Tyr.GPU.Codegen.KStmt
    line? := some (idx0 + 1) }

private def withStmtIdx (idx0 : Nat) (base : OpParams) : OpParams :=
  base ++ #[
    OpParam.mkNat .stmtIdx0 idx0,
    OpParam.mkNat .stmtIdx1 (idx0 + 1)
  ]

private def atomName (value : String) : OpName :=
  Lean.Name.str Lean.Name.anonymous value

/--
Convert GPU `KStmt` IR to a minimal `LeanJaxpr`.

Supported subset:
- `.unary`
- `.binary`
- `.reduce`
- `.reduceAccum`
- `.broadcast`
- `.binaryBroadcast`
- `.transpose`
- `.swapLayout`
- `.convert`
- `.sliceRows`
- `.sliceCols`
- `.concatCols`
- `.outer`
- `.cumsum`
- `.cumprod`

Unsupported statements are accumulated and returned as `.error`.
-/
def fromKStmts (stmts : Array KStmt) : Except (Array String) LeanJaxpr := Id.run do
  let mut eqns : Array JEqn := #[]
  let mut errors : Array String := #[]

  let mut usedOrder : Array Nat := #[]
  let mut usedSet : Std.HashSet Nat := {}
  let mut definedOrder : Array Nat := #[]
  let mut definedSet : Std.HashSet Nat := {}

  for h : idx0 in [:stmts.size] do
    let stmt := stmts[idx0]
    match stmt with
    | .unary op dst src =>
        let (usedOrder', usedSet') := rememberId usedOrder usedSet src.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'

        eqns := eqns.push {
          op := kstmtUnaryOpName op
          invars := #[mkJVar src]
          outvars := #[mkJVar dst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "unary"),
            OpParam.mkName .opTag (atomName (kstmtUnaryOpTag op))
          ]
          source := stmtSourceRef idx0
        }
    | .binary op dst a b =>
        let (usedOrder1, usedSet1) := rememberId usedOrder usedSet a.idx
        let (usedOrder2, usedSet2) := rememberId usedOrder1 usedSet1 b.idx
        usedOrder := usedOrder2
        usedSet := usedSet2
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'

        eqns := eqns.push {
          op := kstmtBinaryOpName op
          invars := #[mkJVar a, mkJVar b]
          outvars := #[mkJVar dst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "binary"),
            OpParam.mkName .opTag (atomName (kstmtBinaryOpTag op))
          ]
          source := stmtSourceRef idx0
        }
    | .reduce op axis dst src =>
        let (usedOrder', usedSet') := rememberId usedOrder usedSet src.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'

        eqns := eqns.push {
          op := kstmtReduceOpName op axis
          invars := #[mkJVar src]
          outvars := #[mkJVar dst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "reduce"),
            OpParam.mkName .opTag (atomName (kstmtReduceOpTag op)),
            OpParam.mkName .axis (atomName (kstmtReduceAxisTag axis))
          ]
          source := stmtSourceRef idx0
        }
    | .reduceAccum op axis dst src accum =>
        let (usedOrder1, usedSet1) := rememberId usedOrder usedSet src.idx
        let (usedOrder2, usedSet2) := rememberId usedOrder1 usedSet1 accum.idx
        usedOrder := usedOrder2
        usedSet := usedSet2
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'

        eqns := eqns.push {
          op := kstmtReduceAccumOpName op axis
          invars := #[mkJVar src, mkJVar accum]
          outvars := #[mkJVar dst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "reduceAccum"),
            OpParam.mkName .opTag (atomName (kstmtReduceOpTag op)),
            OpParam.mkName .axis (atomName (kstmtReduceAxisTag axis))
          ]
          source := stmtSourceRef idx0
        }
    | .broadcast axis dst vec =>
        let (usedOrder', usedSet') := rememberId usedOrder usedSet vec.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'

        eqns := eqns.push {
          op := kstmtBroadcastOpName axis
          invars := #[mkJVar vec]
          outvars := #[mkJVar dst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "broadcast"),
            OpParam.mkName .axis (atomName (kstmtBroadcastAxisTag axis))
          ]
          source := stmtSourceRef idx0
        }
    | .binaryBroadcast op axis dst tile vec =>
        let (usedOrder1, usedSet1) := rememberId usedOrder usedSet tile.idx
        let (usedOrder2, usedSet2) := rememberId usedOrder1 usedSet1 vec.idx
        usedOrder := usedOrder2
        usedSet := usedSet2
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'

        eqns := eqns.push {
          op := kstmtBinaryBroadcastOpName op axis
          invars := #[mkJVar tile, mkJVar vec]
          outvars := #[mkJVar dst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "binaryBroadcast"),
            OpParam.mkName .opTag (atomName (kstmtBinaryOpTag op)),
            OpParam.mkName .axis (atomName (kstmtBroadcastAxisTag axis))
          ]
          source := stmtSourceRef idx0
        }
    | .transpose dst src =>
        let (usedOrder', usedSet') := rememberId usedOrder usedSet src.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'

        eqns := eqns.push {
          op := kstmtTransposeOpName
          invars := #[mkJVar src]
          outvars := #[mkJVar dst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "transpose")
          ]
          source := stmtSourceRef idx0
        }
    | .swapLayout dst src =>
        let (usedOrder', usedSet') := rememberId usedOrder usedSet src.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'

        eqns := eqns.push {
          op := kstmtSwapLayoutOpName
          invars := #[mkJVar src]
          outvars := #[mkJVar dst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "swapLayout")
          ]
          source := stmtSourceRef idx0
        }
    | .convert dst src =>
        let (usedOrder', usedSet') := rememberId usedOrder usedSet src.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'

        eqns := eqns.push {
          op := kstmtConvertOpName
          invars := #[mkJVar src]
          outvars := #[mkJVar dst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "convert")
          ]
          source := stmtSourceRef idx0
        }
    | .sliceRows dst src startRow numRows =>
        let (usedOrder', usedSet') := rememberId usedOrder usedSet src.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'

        eqns := eqns.push {
          op := kstmtSliceRowsOpName
          invars := #[mkJVar src]
          outvars := #[mkJVar dst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "sliceRows"),
            OpParam.mkNat .startRow startRow,
            OpParam.mkNat .numRows numRows
          ]
          source := stmtSourceRef idx0
        }
    | .sliceCols dst src startCol numCols =>
        let (usedOrder', usedSet') := rememberId usedOrder usedSet src.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'

        eqns := eqns.push {
          op := kstmtSliceColsOpName
          invars := #[mkJVar src]
          outvars := #[mkJVar dst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "sliceCols"),
            OpParam.mkNat .startCol startCol,
            OpParam.mkNat .numCols numCols
          ]
          source := stmtSourceRef idx0
        }
    | .concatCols dst left right =>
        let (usedOrder1, usedSet1) := rememberId usedOrder usedSet left.idx
        let (usedOrder2, usedSet2) := rememberId usedOrder1 usedSet1 right.idx
        usedOrder := usedOrder2
        usedSet := usedSet2
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'

        eqns := eqns.push {
          op := kstmtConcatColsOpName
          invars := #[mkJVar left, mkJVar right]
          outvars := #[mkJVar dst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "concatCols")
          ]
          source := stmtSourceRef idx0
        }
    | .outer dst a b =>
        let (usedOrder1, usedSet1) := rememberId usedOrder usedSet a.idx
        let (usedOrder2, usedSet2) := rememberId usedOrder1 usedSet1 b.idx
        usedOrder := usedOrder2
        usedSet := usedSet2
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'

        eqns := eqns.push {
          op := kstmtOuterOpName
          invars := #[mkJVar a, mkJVar b]
          outvars := #[mkJVar dst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "outer")
          ]
          source := stmtSourceRef idx0
        }
    | .cumsum axis dst src =>
        let (usedOrder', usedSet') := rememberId usedOrder usedSet src.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'

        eqns := eqns.push {
          op := kstmtCumsumOpName axis
          invars := #[mkJVar src]
          outvars := #[mkJVar dst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "cumsum"),
            OpParam.mkName .axis (atomName (kstmtReduceAxisTag axis))
          ]
          source := stmtSourceRef idx0
        }
    | .cumprod axis dst src =>
        let (usedOrder', usedSet') := rememberId usedOrder usedSet src.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'

        eqns := eqns.push {
          op := kstmtCumprodOpName axis
          invars := #[mkJVar src]
          outvars := #[mkJVar dst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "cumprod"),
            OpParam.mkName .axis (atomName (kstmtReduceAxisTag axis))
          ]
          source := stmtSourceRef idx0
        }
    | _ =>
        errors := errors.push (unsupportedStmtError idx0 stmt)

  if !errors.isEmpty then
    return .error errors

  let invars := usedOrder.foldl (init := (#[] : Array JVar)) fun acc id =>
    if definedSet.contains id then acc else acc.push { id := id }

  let outvars := definedOrder.foldl (init := (#[] : Array JVar)) fun acc id =>
    if usedSet.contains id then acc else acc.push { id := id }

  return .ok {
    constvars := #[]
    invars := invars
    eqns := eqns
    outvars := outvars
  }

end Tyr.AD.JaxprLike
