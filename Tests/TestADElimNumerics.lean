import Lean.CoreM
import LeanTest
import Tyr.AD.Elim
import Tyr.AD.JaxprLike
import Tyr.AD.Sparse

namespace Tests.ADElimNumerics

open Lean
open LeanTest
open Tyr.AD.Elim
open Tyr.AD.JaxprLike
open Tyr.AD.Sparse

private abbrev SLMap := Tyr.AD.Sparse.SparseLinearMap

def runCoreMResult (x : CoreM α) : IO (Except String α) := do
  let env ← mkEmptyEnvironment
  let ctx : Core.Context := { fileName := "<test>", fileMap := default }
  let state : Core.State := { env := env }
  let eio := x.run ctx state
  let res ← EIO.toBaseIO eio
  match res with
  | .ok (value, _) => pure (.ok value)
  | .error err =>
    let msg ← err.toMessageData.toString
    pure (.error msg)

def runCoreM (x : CoreM α) : IO α := do
  match (← runCoreMResult x) with
  | .ok value => pure value
  | .error msg => throw (IO.userError msg)

private def approx (a b : Float) (tol : Float := 1e-5) : Bool :=
  Float.abs (a - b) <= tol

private def approxVec (lhs rhs : Array Float) (tol : Float := 1e-5) : Bool :=
  if lhs.size != rhs.size then
    false
  else
    (List.range lhs.size).all fun i => approx lhs[i]! rhs[i]! tol

private def assertApproxVec
    (label : String)
    (lhs rhs : Array Float)
    (tol : Float := 1e-5) :
    IO Unit := do
  LeanTest.assertEqual lhs.size rhs.size
    s!"{label} should have matching vector widths."
  LeanTest.assertTrue (approxVec lhs rhs tol)
    s!"{label} mismatch: lhs={reprStr lhs}, rhs={reprStr rhs}"

private def addVecs (lhs rhs : Array Float) : Array Float :=
  let n := max lhs.size rhs.size
  (Array.range n).map fun i => lhs.getD i 0.0 + rhs.getD i 0.0

private def finiteDifferenceGradient
    (f : Array Float → Float)
    (x : Array Float)
    (eps : Float := 1e-4) :
    Array Float :=
  (Array.range x.size).map fun i =>
    let xi := x[i]!
    let xp := x.set! i (xi + eps)
    let xm := x.set! i (xi - eps)
    (f xp - f xm) / (2.0 * eps)

private def mkVar (id : Nat) (shape : Array Nat) : JVar :=
  { id := id, metaInfo := { shape := some shape } }

private def findLocalEdge?
    (edges : Array LocalJacEdge)
    (src dst : Nat) :
    Option LocalJacEdge :=
  edges.find? (fun edge => edge.src = src && edge.dst = dst)

private def edgeMapOrFail
    (label : String)
    (edges : Array LocalJacEdge)
    (src dst : Nat) :
    IO SLMap :=
  match findLocalEdge? edges src dst with
  | some edge => pure edge.map
  | none => LeanTest.fail s!"Missing local-Jac edge {label}: {src} -> {dst}"

private def graphEdgeMapOrFail
    (label : String)
    (g : ElimGraph)
    (src dst : Nat) :
    IO SLMap :=
  match findEdge? g src dst with
  | some map => pure map
  | none => LeanTest.fail s!"Missing elimination-graph edge {label}: {src} -> {dst}"

private def pullbackOrFail
    (label : String)
    (map : SLMap)
    (cotangent : Array Float) :
    IO (Array Float) :=
  match map.pullback cotangent with
  | Except.ok out => pure out
  | Except.error err => LeanTest.fail s!"{label} pullback failed: {err}"

private def denseRowsOrFail
    (label : String)
    (map : SLMap) :
    IO (Array (Array Float)) :=
  match map.toDenseRows with
  | Except.ok rows => pure rows
  | Except.error err => LeanTest.fail s!"{label} dense materialization failed: {err}"

private def assertSameDenseMap
    (label : String)
    (lhs rhs : SLMap) :
    IO Unit := do
  let lhsRows ← denseRowsOrFail label lhs
  let rhsRows ← denseRowsOrFail label rhs
  LeanTest.assertEqual lhsRows.size rhsRows.size
    s!"{label} should agree on row count."
  for i in [:lhsRows.size] do
    assertApproxVec s!"{label} row {i}" lhsRows[i]! rhsRows[i]!

private def extractOrThrow (jaxpr : LeanJaxpr) : CoreM (Array LocalJacEdge) := do
  match (← extractLocalJacEdges jaxpr) with
  | .ok edges => pure edges
  | .error errs =>
      throwError m!"{String.intercalate "\n" (errs.toList.map ruleExecutionErrorToString)}"

private def runForwardOrThrow (jaxpr : LeanJaxpr) : CoreM ElimRunResult := do
  match (← runForwardEliminationOnJaxpr jaxpr) with
  | .ok out => pure out
  | .error err => throwError m!"{err}"

private def runReverseOrThrow (jaxpr : LeanJaxpr) : CoreM ElimRunResult := do
  match (← runReverseEliminationOnJaxpr jaxpr) with
  | .ok out => pure out
  | .error err => throwError m!"{err}"

private def linearAddNegJaxpr : LeanJaxpr :=
  let x := mkVar 1 #[]
  let y := mkVar 2 #[]
  let sum := mkVar 3 #[]
  let loss := mkVar 4 #[]
  LeanJaxpr.mkNormalized
    #[]
    #[x, y]
    #[
      JEqn.ofNormalizedOp 1 (kstmtBinaryOpName .Add) #[x, y] #[sum] #[]
        { decl := `test.linear_add_neg, line? := some 10 },
      JEqn.ofNormalizedOp 2 (kstmtUnaryOpName .Neg) #[sum] #[loss] #[]
        { decl := `test.linear_add_neg, line? := some 11 }
    ]
    #[loss]

private def padReduceSumJaxpr : LeanJaxpr :=
  let base := mkVar 10 #[2, 1]
  let padv := mkVar 11 #[1]
  let padded := mkVar 12 #[6, 1]
  let loss := mkVar 13 #[]
  LeanJaxpr.mkNormalized
    #[]
    #[base, padv]
    #[
      JEqn.ofNormalizedOp 1 padAliasOpName #[base, padv] #[padded]
        #[
          OpParam.mkNats .padLow #[1, 0],
          OpParam.mkNats .padHigh #[2, 0],
          OpParam.mkNats .padInterior #[1, 0],
          OpParam.mkName .sourceOp `Graphax.pad_p
        ]
        { decl := `test.pad_reduce_sum, line? := some 20 },
      JEqn.ofNormalizedOp 2 (kstmtReduceOpName .Sum .Full) #[padded] #[loss] #[]
        { decl := `test.pad_reduce_sum, line? := some 21 }
    ]
    #[loss]

private def scanRoutingJaxpr : LeanJaxpr :=
  let carryIn := mkVar 20 #[2]
  let dataIn := mkVar 21 #[2]
  let carryOut := mkVar 22 #[2]
  let dataOut := mkVar 23 #[2]
  LeanJaxpr.mkNormalized
    #[]
    #[carryIn, dataIn]
    #[
      JEqn.ofNormalizedOp 1 scanAliasOpName #[carryIn, dataIn] #[carryOut, dataOut]
        #[
          OpParam.mkNat .controlStaticArgCount 0,
          OpParam.mkNat .scanCarryInputCount 1,
          OpParam.mkNat .scanDataInputCount 1,
          OpParam.mkNat .scanCarryOutputCount 1,
          OpParam.mkName .sourceOp `Graphax.scan
        ]
        { decl := `test.scan_routing, line? := some 30 }
    ]
    #[carryOut, dataOut]

private def condRoutingJaxpr : LeanJaxpr :=
  let pred := mkVar 30 #[]
  let trueIn := mkVar 31 #[2]
  let falseIn := mkVar 32 #[2]
  let out := mkVar 33 #[2]
  LeanJaxpr.mkNormalized
    #[]
    #[pred, trueIn, falseIn]
    #[
      JEqn.ofNormalizedOp 1 condAliasOpName #[pred, trueIn, falseIn] #[out]
        #[
          OpParam.mkNat .controlStaticArgCount 0,
          OpParam.mkNat .condPredicateCount 1,
          OpParam.mkNat .condDataInputCount 2,
          OpParam.mkName .sourceOp `Graphax.cond
        ]
        { decl := `test.cond_routing, line? := some 40 }
    ]
    #[out]

@[test]
def testLinearScalarLossGradientMatchesFiniteDifferenceAndElimination : IO Unit := do
  let (edges, fwd, rev) ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    let edges ← extractOrThrow linearAddNegJaxpr
    let fwd ← runForwardOrThrow linearAddNegJaxpr
    let rev ← runReverseOrThrow linearAddNegJaxpr
    pure (edges, fwd.graph, rev.graph)
  )

  let xToSum ← edgeMapOrFail "x->sum" edges 1 3
  let yToSum ← edgeMapOrFail "y->sum" edges 2 3
  let sumToLoss ← edgeMapOrFail "sum->loss" edges 3 4

  let xDirect ←
    match Tyr.AD.Sparse.compose xToSum sumToLoss with
    | .ok map => pure map
    | .error err => LeanTest.fail s!"Direct x composition should succeed, got: {err}"
  let yDirect ←
    match Tyr.AD.Sparse.compose yToSum sumToLoss with
    | .ok map => pure map
    | .error err => LeanTest.fail s!"Direct y composition should succeed, got: {err}"

  let xFwd ← graphEdgeMapOrFail "x->loss forward" fwd 1 4
  let yFwd ← graphEdgeMapOrFail "y->loss forward" fwd 2 4
  let xRev ← graphEdgeMapOrFail "x->loss reverse" rev 1 4
  let yRev ← graphEdgeMapOrFail "y->loss reverse" rev 2 4

  assertSameDenseMap "x direct vs forward" xDirect xFwd
  assertSameDenseMap "x forward vs reverse" xFwd xRev
  assertSameDenseMap "y direct vs forward" yDirect yFwd
  assertSameDenseMap "y forward vs reverse" yFwd yRev

  let dx ← pullbackOrFail "x final gradient" xFwd #[1.0]
  let dy ← pullbackOrFail "y final gradient" yFwd #[1.0]
  assertApproxVec "x gradient" dx #[-1.0]
  assertApproxVec "y gradient" dy #[-1.0]

  let fd := finiteDifferenceGradient
    (fun xs => -(xs[0]! + xs[1]!))
    #[2.25, -0.75]
  assertApproxVec "linear finite-difference gradient"
    #[dx[0]!, dy[0]!]
    fd

@[test]
def testPadReduceSumGradientMatchesFiniteDifferenceAndComposedEdges : IO Unit := do
  let (edges, fwd, rev) ← runCoreM (do
    registerGraphaxAlphaGradParityRules
    let edges ← extractOrThrow padReduceSumJaxpr
    let fwd ← runForwardOrThrow padReduceSumJaxpr
    let rev ← runReverseOrThrow padReduceSumJaxpr
    pure (edges, fwd.graph, rev.graph)
  )

  let baseToPadded ← edgeMapOrFail "base->padded" edges 10 12
  let padvToPadded ← edgeMapOrFail "padv->padded" edges 11 12
  let paddedToLoss ← edgeMapOrFail "padded->loss" edges 12 13

  let baseDirect ←
    match Tyr.AD.Sparse.compose baseToPadded paddedToLoss with
    | .ok map => pure map
    | .error err => LeanTest.fail s!"Direct base composition should succeed, got: {err}"
  let padvDirect ←
    match Tyr.AD.Sparse.compose padvToPadded paddedToLoss with
    | .ok map => pure map
    | .error err => LeanTest.fail s!"Direct pad value composition should succeed, got: {err}"

  let baseFwd ← graphEdgeMapOrFail "base->loss forward" fwd 10 13
  let padvFwd ← graphEdgeMapOrFail "padv->loss forward" fwd 11 13
  let baseRev ← graphEdgeMapOrFail "base->loss reverse" rev 10 13
  let padvRev ← graphEdgeMapOrFail "padv->loss reverse" rev 11 13

  assertSameDenseMap "base direct vs forward" baseDirect baseFwd
  assertSameDenseMap "base forward vs reverse" baseFwd baseRev
  assertSameDenseMap "pad value direct vs forward" padvDirect padvFwd
  assertSameDenseMap "pad value forward vs reverse" padvFwd padvRev

  let dBase ← pullbackOrFail "base gradient" baseFwd #[1.0]
  let dPadv ← pullbackOrFail "pad value gradient" padvFwd #[1.0]
  assertApproxVec "base gradient" dBase #[1.0, 1.0]
  assertApproxVec "pad value gradient" dPadv #[4.0]

  let fd := finiteDifferenceGradient
    (fun xs => xs[2]! + xs[0]! + xs[2]! + xs[1]! + xs[2]! + xs[2]!)
    #[2.0, -1.0, 0.5]
  assertApproxVec "pad+reduce finite-difference gradient"
    #[dBase[0]!, dBase[1]!, dPadv[0]!]
    fd

@[test]
def testScanRoutingPullbackNumerics : IO Unit := do
  let edges ← runCoreM (do
    registerGraphaxAlphaGradParityRules
    extractOrThrow scanRoutingJaxpr
  )

  let carryCarry ← edgeMapOrFail "carry->carryOut" edges 20 22
  let carryData ← edgeMapOrFail "carry->dataOut" edges 20 23
  let dataCarry ← edgeMapOrFail "data->carryOut" edges 21 22
  let dataData ← edgeMapOrFail "data->dataOut" edges 21 23

  let carryOutCot : Array Float := #[1.0, -2.0]
  let dataOutCot : Array Float := #[0.25, 3.0]
  let expected := addVecs carryOutCot dataOutCot

  let carryInputCot :=
    addVecs
      (← pullbackOrFail "scan carry->carryOut" carryCarry carryOutCot)
      (← pullbackOrFail "scan carry->dataOut" carryData dataOutCot)
  let dataInputCot :=
    addVecs
      (← pullbackOrFail "scan data->carryOut" dataCarry carryOutCot)
      (← pullbackOrFail "scan data->dataOut" dataData dataOutCot)

  assertApproxVec "scan carry routing" carryInputCot expected
  assertApproxVec "scan data routing" dataInputCot expected

@[test]
def testCondRoutingPullbackNumerics : IO Unit := do
  let edges ← runCoreM (do
    registerGraphaxAlphaGradParityRules
    extractOrThrow condRoutingJaxpr
  )

  LeanTest.assertTrue (!(edges.any (fun edge => edge.src = 30)))
    "Predicate input should not contribute to cond pullback edges."

  let trueBranch ← edgeMapOrFail "true->out" edges 31 33
  let falseBranch ← edgeMapOrFail "false->out" edges 32 33
  let outCot : Array Float := #[2.0, -0.5]

  let trueCot ← pullbackOrFail "cond true branch" trueBranch outCot
  let falseCot ← pullbackOrFail "cond false branch" falseBranch outCot

  assertApproxVec "cond true-branch routing" trueCot outCot
  assertApproxVec "cond false-branch routing" falseCot outCot

end Tests.ADElimNumerics
