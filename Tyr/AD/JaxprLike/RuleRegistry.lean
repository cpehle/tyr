import Tyr.AD.JaxprLike.Rules

/-!
# Tyr.AD.JaxprLike.RuleRegistry

Registry for LeanJaxpr local-Jacobian extraction rules.
-/

namespace Tyr.AD.JaxprLike

structure LocalJacRegistry where
  rules : Std.HashMap RuleKey LocalJacRule := {}
  deriving Inhabited

initialize localJacRegistry : Lean.EnvExtension LocalJacRegistry ←
  Lean.registerEnvExtension (pure {})

def registerLocalJacRule (op : OpName) (rule : LocalJacRule) : Lean.CoreM Unit := do
  Lean.modifyEnv fun env =>
    localJacRegistry.modifyState env fun s =>
      { s with rules := s.rules.insert (ruleKeyOfRegisteredOp op) rule }

def getLocalJacRule? (op : OpName) : Lean.CoreM (Option LocalJacRule) := do
  return (localJacRegistry.getState (← Lean.getEnv)).rules.get? (ruleKeyOfRegisteredOp op)

def getLocalJacRuleForEqn? (eqn : JEqn) : Lean.CoreM (Option LocalJacRule) := do
  return (localJacRegistry.getState (← Lean.getEnv)).rules.get? eqn.ruleKey

def runLocalJacRule
    (eqn : JEqn)
    (ctx : RuleContext) :
    Lean.CoreM (Except RuleError (Array LocalJacEdge)) := do
  match (← getLocalJacRuleForEqn? eqn) with
  | some rule => pure <| rule eqn ctx
  | none => pure <| .error (.unsupportedOp eqn.op)

end Tyr.AD.JaxprLike
