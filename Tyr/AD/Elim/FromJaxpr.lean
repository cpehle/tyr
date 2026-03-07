import Tyr.AD.Elim.Eliminate
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

/-- Execute elimination from an already-normalized `LeanJaxpr`. -/
def runEliminationOnJaxpr
    (jaxpr : LeanJaxpr)
    (order : Array JVarId) :
    Lean.CoreM (Except String ElimRunResult) := do
  match (← extractLocalJacEdges jaxpr) with
  | .ok edges =>
    return runEliminationOnEdges edges order
  | .error errs =>
    return .error <|
      "Local-Jacobian extraction failed:\n" ++ renderRuleExecutionErrors errs

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
  | .ok (_jaxpr, edges) =>
    return runEliminationOnEdges edges order

end Tyr.AD.Elim
