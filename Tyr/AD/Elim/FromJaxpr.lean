import Tyr.AD.Elim.Eliminate
import Tyr.AD.Elim.OrderPolicy
import Tyr.AD.JaxprLike.Pipeline

/-!
# Tyr.AD.Elim.FromJaxpr

End-to-end execution adapters:
- normalized `LeanJaxpr` -> local-Jac edges -> elimination
- `KStmt` lowering/build -> local-Jac edges -> elimination
-/

namespace Tyr.AD.Elim

open Tyr.AD.JaxprLike

private def renderRuleExecutionErrors (errs : Array RuleExecutionError) : String :=
  String.intercalate "\n" (errs.toList.map ruleExecutionErrorToString)

private def buildPartitionedGraph?
    (jaxpr : LeanJaxpr)
    (edges : Array LocalJacEdge) :
    Except String ElimGraph := do
  let parts := jaxpr.vertexPartitions
  ofLocalJacEdgesWithPartitions edges parts.inputs parts.outputs parts.eliminable

private def runGraphWithPolicy
    (graph : ElimGraph)
    (policy : OrderPolicy) :
    Except String ElimRunResult := do
  let normalized ← normalizeOrderPolicyAgainstGraph graph policy
  match normalized.baseOrder1? with
  | some order1 =>
    runCompleteElimination graph order1
  | none =>
    .error s!"Order policy `{normalized.source}` does not resolve to a concrete elimination order on this graph."

/-- Build a partitioned elimination graph from an already-normalized `LeanJaxpr`. -/
def buildElimGraphFromJaxpr
    (jaxpr : LeanJaxpr) :
    Lean.CoreM (Except String ElimGraph) := do
  match (← extractLocalJacEdges jaxpr) with
  | .ok edges =>
    return buildPartitionedGraph? jaxpr edges
  | .error errs =>
    return .error <|
      "Local-Jacobian extraction failed:\n" ++ renderRuleExecutionErrors errs

/-- Execute elimination from an already-normalized `LeanJaxpr`. -/
def runEliminationOnJaxpr
    (jaxpr : LeanJaxpr)
    (order : Array JVarId) :
    Lean.CoreM (Except String ElimRunResult) := do
  match (← buildElimGraphFromJaxpr jaxpr) with
  | .ok graph =>
    return runElimination graph order
  | .error err =>
    return .error err

/-- Execute elimination from `LeanJaxpr` using a higher-level order policy. -/
def runEliminationOnJaxprWithPolicy
    (jaxpr : LeanJaxpr)
    (policy : OrderPolicy) :
    Lean.CoreM (Except String ElimRunResult) := do
  match (← buildElimGraphFromJaxpr jaxpr) with
  | .ok graph =>
    return runGraphWithPolicy graph policy
  | .error err =>
    return .error err

/--
Execute complete Graphax-style elimination from `LeanJaxpr` using the graph's
explicit forward eliminable order.
-/
def runForwardEliminationOnJaxpr
    (jaxpr : LeanJaxpr) :
    Lean.CoreM (Except String ElimRunResult) := do
  match (← buildElimGraphFromJaxpr jaxpr) with
  | .ok graph =>
    return runForwardElimination graph
  | .error err =>
    return .error err

/--
Execute complete Graphax-style elimination from `LeanJaxpr` using the graph's
explicit reverse eliminable order.
-/
def runReverseEliminationOnJaxpr
    (jaxpr : LeanJaxpr) :
    Lean.CoreM (Except String ElimRunResult) := do
  match (← buildElimGraphFromJaxpr jaxpr) with
  | .ok graph =>
    return runReverseElimination graph
  | .error err =>
    return .error err

/--
Lower + validate + rule-check + extract local-Jac edges from `KStmt` IR,
then execute elimination in the provided vertex order.
-/
def runEliminationOnKStmts
    (cfg : BuildConfig := {})
    (stmts : Array Tyr.GPU.Codegen.KStmt)
    (order : Array JVarId) :
    Lean.CoreM (Except String ElimRunResult) := do
  match (← buildAndExtractFromKStmts cfg stmts) with
  | .error err =>
    return .error (buildExtractErrorToString err)
  | .ok (jaxpr, edges) =>
    match buildPartitionedGraph? jaxpr edges with
    | .ok graph =>
      return runElimination graph order
    | .error err =>
      return .error err

/-- Execute elimination from `KStmt` IR using a higher-level order policy. -/
def runEliminationOnKStmtsWithPolicy
    (cfg : BuildConfig := {})
    (stmts : Array Tyr.GPU.Codegen.KStmt)
    (policy : OrderPolicy) :
    Lean.CoreM (Except String ElimRunResult) := do
  match (← buildAndExtractFromKStmts cfg stmts) with
  | .error err =>
    return .error (buildExtractErrorToString err)
  | .ok (jaxpr, edges) =>
    match buildPartitionedGraph? jaxpr edges with
    | .ok graph =>
      return runGraphWithPolicy graph policy
    | .error err =>
      return .error err

/--
Lower + validate + extract a partitioned elimination graph from `KStmt` IR,
then execute complete elimination in deterministic forward order.
-/
def runForwardEliminationOnKStmts
    (cfg : BuildConfig := {})
    (stmts : Array Tyr.GPU.Codegen.KStmt) :
    Lean.CoreM (Except String ElimRunResult) := do
  match (← buildAndExtractFromKStmts cfg stmts) with
  | .error err =>
    return .error (buildExtractErrorToString err)
  | .ok (jaxpr, edges) =>
    match buildPartitionedGraph? jaxpr edges with
    | .ok graph =>
      return runForwardElimination graph
    | .error err =>
      return .error err

/--
Lower + validate + extract a partitioned elimination graph from `KStmt` IR,
then execute complete elimination in deterministic reverse order.
-/
def runReverseEliminationOnKStmts
    (cfg : BuildConfig := {})
    (stmts : Array Tyr.GPU.Codegen.KStmt) :
    Lean.CoreM (Except String ElimRunResult) := do
  match (← buildAndExtractFromKStmts cfg stmts) with
  | .error err =>
    return .error (buildExtractErrorToString err)
  | .ok (jaxpr, edges) =>
    match buildPartitionedGraph? jaxpr edges with
    | .ok graph =>
      return runReverseElimination graph
    | .error err =>
      return .error err

end Tyr.AD.Elim
