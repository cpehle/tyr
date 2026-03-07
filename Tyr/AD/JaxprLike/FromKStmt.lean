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

/-- Convert a GPU `VarId` to a `JVar` using known lowering metadata when available. -/
private def mkJVarFromMeta
    (varMeta : Std.HashMap Nat VarMeta)
    (v : VarId) : JVar :=
  { id := v.idx
    metaInfo := varMeta.getD v.idx {} }

/-- Convert raw numeric ID to `JVar` while preserving propagated metadata. -/
private def mkJVarById
    (varMeta : Std.HashMap Nat VarMeta)
    (id : Nat) : JVar :=
  { id := id
    metaInfo := varMeta.getD id {} }

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

private def rank1Shape (n : Nat) : Array Nat := #[n]

private def rank2Shape (r c : Nat) : Array Nat := #[r, c]

private def mergeVarMeta (existing incoming : VarMeta) : VarMeta :=
  { participation := existing.participation
    shape := existing.shape.orElse (fun _ => incoming.shape)
    dtype := existing.dtype.orElse (fun _ => incoming.dtype)
    sharding := existing.sharding.orElse (fun _ => incoming.sharding)
    aliasGroup? := existing.aliasGroup?.orElse (fun _ => incoming.aliasGroup?) }

private def upsertVarMeta
    (varMeta : Std.HashMap Nat VarMeta)
    (v : VarId)
    (incoming : VarMeta) : Std.HashMap Nat VarMeta :=
  let existing := varMeta.getD v.idx {}
  varMeta.insert v.idx (mergeVarMeta existing incoming)

private def registerDeclaredVarMeta
    (varMeta : Std.HashMap Nat VarMeta)
    (v : VarId)
    (declMeta : VarMeta) : Std.HashMap Nat VarMeta :=
  let existing := varMeta.getD v.idx {}
  -- Declarations should dominate inferred metadata when both are present.
  varMeta.insert v.idx (mergeVarMeta declMeta existing)

private def inferBinaryShape?
    (lhsShape? rhsShape? : Option (Array Nat)) : Option (Array Nat) :=
  match lhsShape?, rhsShape? with
  | some lhs, some rhs =>
    if lhs == rhs then
      some lhs
    else if lhs.size >= rhs.size then
      some lhs
    else
      some rhs
  | some lhs, none => some lhs
  | none, some rhs => some rhs
  | none, none => none

private def inferReduceShape?
    (axis : ReduceAxis)
    (srcShape? : Option (Array Nat)) : Option (Array Nat) :=
  match srcShape? with
  | none => none
  | some shape =>
    if shape.size = 0 then
      some (rank1Shape 1)
    else if shape.size = 1 then
      match axis with
      | .Full => some (rank1Shape 1)
      | .Row => some (rank1Shape 1)
      | .Col => some (rank1Shape 1)
    else
      let r := shape.getD 0 1
      let c := shape.getD 1 1
      match axis with
      | .Row => some (rank1Shape c)
      | .Col => some (rank1Shape r)
      | .Full => some (rank1Shape 1)

private def inferTransposeShape? (srcShape? : Option (Array Nat)) : Option (Array Nat) :=
  match srcShape? with
  | some shape =>
    if shape.size < 2 then
      some shape
    else
      some (rank2Shape (shape.getD 1 1) (shape.getD 0 1))
  | none => none

private def inferSliceRowsShape?
    (srcShape? : Option (Array Nat))
    (numRows : Nat) : Option (Array Nat) :=
  match srcShape? with
  | some shape =>
    if shape.size < 2 then
      some (rank1Shape numRows)
    else
      some (rank2Shape numRows (shape.getD 1 1))
  | none => none

private def inferSliceColsShape?
    (srcShape? : Option (Array Nat))
    (numCols : Nat) : Option (Array Nat) :=
  match srcShape? with
  | some shape =>
    if shape.size < 2 then
      some (rank1Shape numCols)
    else
      some (rank2Shape (shape.getD 0 1) numCols)
  | none => none

private def inferConcatColsShape?
    (lhsShape? rhsShape? : Option (Array Nat)) : Option (Array Nat) :=
  match lhsShape?, rhsShape? with
  | some lhs, some rhs =>
    if lhs.size < 2 || rhs.size < 2 then
      inferBinaryShape? lhsShape? rhsShape?
    else
      some (rank2Shape (lhs.getD 0 1) (lhs.getD 1 1 + rhs.getD 1 1))
  | _, _ => inferBinaryShape? lhsShape? rhsShape?

private def inferOuterShape?
    (lhsShape? rhsShape? : Option (Array Nat)) : Option (Array Nat) := do
  let lhsShape ← lhsShape?
  let rhsShape ← rhsShape?
  let lhsDim := lhsShape.getD 0 1
  let rhsDim := rhsShape.getD 0 1
  some (rank2Shape lhsDim rhsDim)

private def mmContractAxes (trans : MMATranspose) : Array Nat × Array Nat :=
  match trans with
  | .AB => (#[1], #[0])
  | .ABt => (#[1], #[1])
  | .AtB => (#[0], #[0])
  | .AtBt => (#[0], #[1])

private def inferMMShape?
    (trans : MMATranspose)
    (aShape? bShape? : Option (Array Nat)) : Option (Array Nat) := do
  let aShape ← aShape?
  let bShape ← bShape?
  if aShape.size < 2 || bShape.size < 2 then
    none
  else
    let a0 := aShape.getD 0 1
    let a1 := aShape.getD 1 1
    let b0 := bShape.getD 0 1
    let b1 := bShape.getD 1 1
    let outRows :=
      match trans with
      | .AB => a0
      | .ABt => a0
      | .AtB => a1
      | .AtBt => a1
    let outCols :=
      match trans with
      | .AB => b1
      | .ABt => b0
      | .AtB => b1
      | .AtBt => b0
    some (rank2Shape outRows outCols)

/--
Convert GPU `KStmt` IR to a minimal `LeanJaxpr`.

Supported subset:
- declaration statements (`decl*`) are consumed as variable-metadata seeds
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
- `.mm` (lowered as dot-general)
- `.mma` / `.tcgen05Mma`
- `.cumsum`
- `.cumprod`

Unsupported statements are accumulated and returned as `.error`.
-/
def fromKStmts (stmts : Array KStmt) : Except (Array String) LeanJaxpr := Id.run do
  let mut eqns : Array JEqn := #[]
  let mut errors : Array String := #[]
  let mut varMeta : Std.HashMap Nat VarMeta := {}

  let mut usedOrder : Array Nat := #[]
  let mut usedSet : Std.HashSet Nat := {}
  let mut definedOrder : Array Nat := #[]
  let mut definedSet : Std.HashSet Nat := {}

  for h : idx0 in [:stmts.size] do
    let stmt := stmts[idx0]
    match stmt with
    | .declRT v dtype rows cols layout =>
        varMeta := registerDeclaredVarMeta varMeta v {
          shape := some (rank2Shape rows cols)
          dtype := some (toString dtype)
          sharding := some s!"register.{layout}"
        }
    | .declST v dtype rows cols layout =>
        varMeta := registerDeclaredVarMeta varMeta v {
          shape := some (rank2Shape rows cols)
          dtype := some (toString dtype)
          sharding := some s!"shared.{layout}"
        }
    | .declRV v dtype len =>
        varMeta := registerDeclaredVarMeta varMeta v {
          shape := some (rank1Shape len)
          dtype := some (toString dtype)
          sharding := some "register.vec"
        }
    | .declSV v dtype len =>
        varMeta := registerDeclaredVarMeta varMeta v {
          shape := some (rank1Shape len)
          dtype := some (toString dtype)
          sharding := some "shared.vec"
        }
    | .declGPtr v dtype _name =>
        varMeta := registerDeclaredVarMeta varMeta v {
          dtype := some (toString dtype)
          sharding := some "global.ptr"
        }
    | .declKVal v dtype _name =>
        varMeta := registerDeclaredVarMeta varMeta v {
          dtype := some (toString dtype)
          sharding := some "scalar.param"
        }
    | .declSemaphore v =>
        varMeta := registerDeclaredVarMeta varMeta v { participation := .static }
    | .comment _ =>
        pure ()
    | .unary op dst src =>
        let srcMeta := varMeta.getD src.idx {}
        varMeta := upsertVarMeta varMeta dst {
          shape := srcMeta.shape
          dtype := srcMeta.dtype
        }
        let (usedOrder', usedSet') := rememberId usedOrder usedSet src.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'
        let jsrc := mkJVarFromMeta varMeta src
        let jdst := mkJVarFromMeta varMeta dst

        eqns := eqns.push {
          op := kstmtUnaryOpName op
          invars := #[jsrc]
          outvars := #[jdst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "unary"),
            OpParam.mkName .opTag (atomName (kstmtUnaryOpTag op))
          ]
          source := stmtSourceRef idx0
        }
    | .binary op dst a b =>
        let aMeta := varMeta.getD a.idx {}
        let bMeta := varMeta.getD b.idx {}
        varMeta := upsertVarMeta varMeta dst {
          shape := inferBinaryShape? aMeta.shape bMeta.shape
          dtype := aMeta.dtype.orElse (fun _ => bMeta.dtype)
        }
        let (usedOrder1, usedSet1) := rememberId usedOrder usedSet a.idx
        let (usedOrder2, usedSet2) := rememberId usedOrder1 usedSet1 b.idx
        usedOrder := usedOrder2
        usedSet := usedSet2
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'
        let ja := mkJVarFromMeta varMeta a
        let jb := mkJVarFromMeta varMeta b
        let jdst := mkJVarFromMeta varMeta dst

        eqns := eqns.push {
          op := kstmtBinaryOpName op
          invars := #[ja, jb]
          outvars := #[jdst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "binary"),
            OpParam.mkName .opTag (atomName (kstmtBinaryOpTag op))
          ]
          source := stmtSourceRef idx0
        }
    | .reduce op axis dst src =>
        let srcMeta := varMeta.getD src.idx {}
        varMeta := upsertVarMeta varMeta dst {
          shape := inferReduceShape? axis srcMeta.shape
          dtype := srcMeta.dtype
        }
        let (usedOrder', usedSet') := rememberId usedOrder usedSet src.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'
        let jsrc := mkJVarFromMeta varMeta src
        let jdst := mkJVarFromMeta varMeta dst

        eqns := eqns.push {
          op := kstmtReduceOpName op axis
          invars := #[jsrc]
          outvars := #[jdst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "reduce"),
            OpParam.mkName .opTag (atomName (kstmtReduceOpTag op)),
            OpParam.mkName .axis (atomName (kstmtReduceAxisTag axis))
          ]
          source := stmtSourceRef idx0
        }
    | .reduceAccum op axis dst src accum =>
        let srcMeta := varMeta.getD src.idx {}
        let accumMeta := varMeta.getD accum.idx {}
        varMeta := upsertVarMeta varMeta dst {
          shape := accumMeta.shape.orElse (fun _ => inferReduceShape? axis srcMeta.shape)
          dtype := accumMeta.dtype.orElse (fun _ => srcMeta.dtype)
        }
        let (usedOrder1, usedSet1) := rememberId usedOrder usedSet src.idx
        let (usedOrder2, usedSet2) := rememberId usedOrder1 usedSet1 accum.idx
        usedOrder := usedOrder2
        usedSet := usedSet2
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'
        let jsrc := mkJVarFromMeta varMeta src
        let jaccum := mkJVarFromMeta varMeta accum
        let jdst := mkJVarFromMeta varMeta dst

        eqns := eqns.push {
          op := kstmtReduceAccumOpName op axis
          invars := #[jsrc, jaccum]
          outvars := #[jdst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "reduceAccum"),
            OpParam.mkName .opTag (atomName (kstmtReduceOpTag op)),
            OpParam.mkName .axis (atomName (kstmtReduceAxisTag axis))
          ]
          source := stmtSourceRef idx0
        }
    | .broadcast axis dst vec =>
        let vecMeta := varMeta.getD vec.idx {}
        varMeta := upsertVarMeta varMeta dst {
          shape := vecMeta.shape
          dtype := vecMeta.dtype
        }
        let (usedOrder', usedSet') := rememberId usedOrder usedSet vec.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'
        let jvec := mkJVarFromMeta varMeta vec
        let jdst := mkJVarFromMeta varMeta dst

        eqns := eqns.push {
          op := kstmtBroadcastOpName axis
          invars := #[jvec]
          outvars := #[jdst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "broadcast"),
            OpParam.mkName .axis (atomName (kstmtBroadcastAxisTag axis))
          ]
          source := stmtSourceRef idx0
        }
    | .binaryBroadcast op axis dst tile vec =>
        let tileMeta := varMeta.getD tile.idx {}
        let vecMeta := varMeta.getD vec.idx {}
        varMeta := upsertVarMeta varMeta dst {
          shape := tileMeta.shape.orElse (fun _ => vecMeta.shape)
          dtype := tileMeta.dtype.orElse (fun _ => vecMeta.dtype)
        }
        let (usedOrder1, usedSet1) := rememberId usedOrder usedSet tile.idx
        let (usedOrder2, usedSet2) := rememberId usedOrder1 usedSet1 vec.idx
        usedOrder := usedOrder2
        usedSet := usedSet2
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'
        let jtile := mkJVarFromMeta varMeta tile
        let jvec := mkJVarFromMeta varMeta vec
        let jdst := mkJVarFromMeta varMeta dst

        eqns := eqns.push {
          op := kstmtBinaryBroadcastOpName op axis
          invars := #[jtile, jvec]
          outvars := #[jdst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "binaryBroadcast"),
            OpParam.mkName .opTag (atomName (kstmtBinaryOpTag op)),
            OpParam.mkName .axis (atomName (kstmtBroadcastAxisTag axis))
          ]
          source := stmtSourceRef idx0
        }
    | .transpose dst src =>
        let srcMeta := varMeta.getD src.idx {}
        varMeta := upsertVarMeta varMeta dst {
          shape := inferTransposeShape? srcMeta.shape
          dtype := srcMeta.dtype
        }
        let (usedOrder', usedSet') := rememberId usedOrder usedSet src.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'
        let jsrc := mkJVarFromMeta varMeta src
        let jdst := mkJVarFromMeta varMeta dst

        eqns := eqns.push {
          op := kstmtTransposeOpName
          invars := #[jsrc]
          outvars := #[jdst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "transpose")
          ]
          source := stmtSourceRef idx0
        }
    | .swapLayout dst src =>
        let srcMeta := varMeta.getD src.idx {}
        varMeta := upsertVarMeta varMeta dst {
          shape := srcMeta.shape
          dtype := srcMeta.dtype
        }
        let (usedOrder', usedSet') := rememberId usedOrder usedSet src.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'
        let jsrc := mkJVarFromMeta varMeta src
        let jdst := mkJVarFromMeta varMeta dst

        eqns := eqns.push {
          op := kstmtSwapLayoutOpName
          invars := #[jsrc]
          outvars := #[jdst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "swapLayout")
          ]
          source := stmtSourceRef idx0
        }
    | .convert dst src =>
        let srcMeta := varMeta.getD src.idx {}
        varMeta := upsertVarMeta varMeta dst {
          shape := srcMeta.shape
          dtype := srcMeta.dtype
        }
        let (usedOrder', usedSet') := rememberId usedOrder usedSet src.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'
        let jsrc := mkJVarFromMeta varMeta src
        let jdst := mkJVarFromMeta varMeta dst

        eqns := eqns.push {
          op := kstmtConvertOpName
          invars := #[jsrc]
          outvars := #[jdst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "convert")
          ]
          source := stmtSourceRef idx0
        }
    | .sliceRows dst src startRow numRows =>
        let srcMeta := varMeta.getD src.idx {}
        varMeta := upsertVarMeta varMeta dst {
          shape := inferSliceRowsShape? srcMeta.shape numRows
          dtype := srcMeta.dtype
        }
        let (usedOrder', usedSet') := rememberId usedOrder usedSet src.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'
        let jsrc := mkJVarFromMeta varMeta src
        let jdst := mkJVarFromMeta varMeta dst

        eqns := eqns.push {
          op := kstmtSliceRowsOpName
          invars := #[jsrc]
          outvars := #[jdst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "sliceRows"),
            OpParam.mkNat .startRow startRow,
            OpParam.mkNat .numRows numRows
          ]
          source := stmtSourceRef idx0
        }
    | .sliceCols dst src startCol numCols =>
        let srcMeta := varMeta.getD src.idx {}
        varMeta := upsertVarMeta varMeta dst {
          shape := inferSliceColsShape? srcMeta.shape numCols
          dtype := srcMeta.dtype
        }
        let (usedOrder', usedSet') := rememberId usedOrder usedSet src.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'
        let jsrc := mkJVarFromMeta varMeta src
        let jdst := mkJVarFromMeta varMeta dst

        eqns := eqns.push {
          op := kstmtSliceColsOpName
          invars := #[jsrc]
          outvars := #[jdst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "sliceCols"),
            OpParam.mkNat .startCol startCol,
            OpParam.mkNat .numCols numCols
          ]
          source := stmtSourceRef idx0
        }
    | .concatCols dst left right =>
        let leftMeta := varMeta.getD left.idx {}
        let rightMeta := varMeta.getD right.idx {}
        varMeta := upsertVarMeta varMeta dst {
          shape := inferConcatColsShape? leftMeta.shape rightMeta.shape
          dtype := leftMeta.dtype.orElse (fun _ => rightMeta.dtype)
        }
        let (usedOrder1, usedSet1) := rememberId usedOrder usedSet left.idx
        let (usedOrder2, usedSet2) := rememberId usedOrder1 usedSet1 right.idx
        usedOrder := usedOrder2
        usedSet := usedSet2
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'
        let jleft := mkJVarFromMeta varMeta left
        let jright := mkJVarFromMeta varMeta right
        let jdst := mkJVarFromMeta varMeta dst

        eqns := eqns.push {
          op := kstmtConcatColsOpName
          invars := #[jleft, jright]
          outvars := #[jdst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "concatCols")
          ]
          source := stmtSourceRef idx0
        }
    | .outer dst a b =>
        let aMeta := varMeta.getD a.idx {}
        let bMeta := varMeta.getD b.idx {}
        varMeta := upsertVarMeta varMeta dst {
          shape := inferOuterShape? aMeta.shape bMeta.shape
          dtype := aMeta.dtype.orElse (fun _ => bMeta.dtype)
        }
        let (usedOrder1, usedSet1) := rememberId usedOrder usedSet a.idx
        let (usedOrder2, usedSet2) := rememberId usedOrder1 usedSet1 b.idx
        usedOrder := usedOrder2
        usedSet := usedSet2
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'
        let ja := mkJVarFromMeta varMeta a
        let jb := mkJVarFromMeta varMeta b
        let jdst := mkJVarFromMeta varMeta dst

        eqns := eqns.push {
          op := kstmtOuterOpName
          invars := #[ja, jb]
          outvars := #[jdst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "outer")
          ]
          source := stmtSourceRef idx0
        }
    | .mm trans dst a b =>
        let aMeta := varMeta.getD a.idx {}
        let bMeta := varMeta.getD b.idx {}
        let (lhsContract, rhsContract) := mmContractAxes trans
        varMeta := upsertVarMeta varMeta dst {
          shape := inferMMShape? trans aMeta.shape bMeta.shape
          dtype := aMeta.dtype.orElse (fun _ => bMeta.dtype)
        }
        let (usedOrder1, usedSet1) := rememberId usedOrder usedSet a.idx
        let (usedOrder2, usedSet2) := rememberId usedOrder1 usedSet1 b.idx
        usedOrder := usedOrder2
        usedSet := usedSet2
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'
        let ja := mkJVarFromMeta varMeta a
        let jb := mkJVarFromMeta varMeta b
        let jdst := mkJVarFromMeta varMeta dst

        eqns := eqns.push {
          op := kstmtDotGeneralOpName
          invars := #[ja, jb]
          outvars := #[jdst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "mm"),
            OpParam.mkName .opTag (atomName (toString trans)),
            OpParam.mkName .variant (atomName s!"mm.{toString trans}"),
            OpParam.mkNats .lhsContract lhsContract,
            OpParam.mkNats .rhsContract rhsContract,
            OpParam.mkNats .lhsBatch #[],
            OpParam.mkNats .rhsBatch #[]
          ]
          source := stmtSourceRef idx0
        }
    | .mma trans dst a b c =>
        let aMeta := varMeta.getD a.idx {}
        let bMeta := varMeta.getD b.idx {}
        let cMeta := varMeta.getD c.idx {}
        varMeta := upsertVarMeta varMeta dst {
          shape := cMeta.shape.orElse (fun _ => inferMMShape? trans aMeta.shape bMeta.shape)
          dtype := cMeta.dtype.orElse (fun _ => aMeta.dtype.orElse (fun _ => bMeta.dtype))
        }
        let (usedOrder1, usedSet1) := rememberId usedOrder usedSet a.idx
        let (usedOrder2, usedSet2) := rememberId usedOrder1 usedSet1 b.idx
        let (usedOrder3, usedSet3) := rememberId usedOrder2 usedSet2 c.idx
        usedOrder := usedOrder3
        usedSet := usedSet3
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'
        let ja := mkJVarFromMeta varMeta a
        let jb := mkJVarFromMeta varMeta b
        let jc := mkJVarFromMeta varMeta c
        let jdst := mkJVarFromMeta varMeta dst

        eqns := eqns.push {
          op := kstmtMmaOpName trans
          invars := #[ja, jb, jc]
          outvars := #[jdst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "mma"),
            OpParam.mkName .opTag (atomName (toString trans)),
            OpParam.mkName .variant (atomName s!"mma.{toString trans}")
          ]
          source := stmtSourceRef idx0
        }
    | .tcgen05Mma trans dst a b c =>
        let aMeta := varMeta.getD a.idx {}
        let bMeta := varMeta.getD b.idx {}
        let cMeta := varMeta.getD c.idx {}
        varMeta := upsertVarMeta varMeta dst {
          shape := cMeta.shape.orElse (fun _ => inferMMShape? trans aMeta.shape bMeta.shape)
          dtype := cMeta.dtype.orElse (fun _ => aMeta.dtype.orElse (fun _ => bMeta.dtype))
        }
        let (usedOrder1, usedSet1) := rememberId usedOrder usedSet a.idx
        let (usedOrder2, usedSet2) := rememberId usedOrder1 usedSet1 b.idx
        let (usedOrder3, usedSet3) := rememberId usedOrder2 usedSet2 c.idx
        usedOrder := usedOrder3
        usedSet := usedSet3
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'
        let ja := mkJVarFromMeta varMeta a
        let jb := mkJVarFromMeta varMeta b
        let jc := mkJVarFromMeta varMeta c
        let jdst := mkJVarFromMeta varMeta dst

        eqns := eqns.push {
          op := kstmtMmaOpName trans
          invars := #[ja, jb, jc]
          outvars := #[jdst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "tcgen05Mma"),
            OpParam.mkName .opTag (atomName (toString trans)),
            OpParam.mkName .variant (atomName s!"tcgen05Mma.{toString trans}")
          ]
          source := stmtSourceRef idx0
        }
    | .cumsum axis dst src =>
        let srcMeta := varMeta.getD src.idx {}
        varMeta := upsertVarMeta varMeta dst {
          shape := srcMeta.shape
          dtype := srcMeta.dtype
        }
        let (usedOrder', usedSet') := rememberId usedOrder usedSet src.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'
        let jsrc := mkJVarFromMeta varMeta src
        let jdst := mkJVarFromMeta varMeta dst

        eqns := eqns.push {
          op := kstmtCumsumOpName axis
          invars := #[jsrc]
          outvars := #[jdst]
          params := withStmtIdx idx0 #[
            OpParam.mkName .kind (atomName "cumsum"),
            OpParam.mkName .axis (atomName (kstmtReduceAxisTag axis))
          ]
          source := stmtSourceRef idx0
        }
    | .cumprod axis dst src =>
        let srcMeta := varMeta.getD src.idx {}
        varMeta := upsertVarMeta varMeta dst {
          shape := srcMeta.shape
          dtype := srcMeta.dtype
        }
        let (usedOrder', usedSet') := rememberId usedOrder usedSet src.idx
        usedOrder := usedOrder'
        usedSet := usedSet'
        let (definedOrder', definedSet') := rememberId definedOrder definedSet dst.idx
        definedOrder := definedOrder'
        definedSet := definedSet'
        let jsrc := mkJVarFromMeta varMeta src
        let jdst := mkJVarFromMeta varMeta dst

        eqns := eqns.push {
          op := kstmtCumprodOpName axis
          invars := #[jsrc]
          outvars := #[jdst]
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
    if definedSet.contains id then acc else acc.push (mkJVarById varMeta id)

  let outvars := definedOrder.foldl (init := (#[] : Array JVar)) fun acc id =>
    if usedSet.contains id then acc else acc.push (mkJVarById varMeta id)

  return .ok {
    constvars := #[]
    invars := invars
    eqns := eqns
    outvars := outvars
  }

end Tyr.AD.JaxprLike
