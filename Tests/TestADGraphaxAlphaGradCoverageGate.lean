import Lean.CoreM
import LeanTest
import Tyr.AD.JaxprLike

namespace Tests.ADGraphaxAlphaGradCoverageGate

open Lean
open LeanTest
open Tyr.AD.JaxprLike

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

private def dedupOpNames (ops : Array OpName) : Array OpName := Id.run do
  let mut seen : Std.HashSet OpName := {}
  let mut out : Array OpName := #[]
  for op in ops do
    if seen.contains op then
      pure ()
    else
      seen := seen.insert op
      out := out.push op
  return out

private def collectMissingRules (ops : Array OpName) : CoreM (Array OpName) := do
  let mut missing : Array OpName := #[]
  for op in ops do
    if (← getLocalJacRule? op).isNone then
      missing := missing.push op
  return missing

private def declaredParityOpNames : Array OpName :=
  dedupOpNames <|
    allKStmtSupportedOpNames ++
    allStopGradientOpNames ++
    allIotaOpNames ++
    allDevicePutOpNames ++
    allPjitOpNames ++
    allCommunicationUnaryOpNames ++
    allGraphaxExtraUnaryAliasOpNames ++
    allGraphaxExtraBinaryAliasOpNames ++
    allGraphaxZeroBinaryAliasOpNames ++
    allStructuralUnaryAliasOpNames ++
    allPadAliasOpNames ++
    allConcatLikeAliasOpNames ++
    allReductionUnaryAliasOpNames ++
    allSelectNAliasOpNames ++
    allHigherOrderControlAliasOpNames ++
    allDynamicProjectionAliasOpNames ++
    allDynamicUpdateAliasOpNames ++
    allDotGeneralOpNames

@[test]
def testKStmtAllSupportedSemanticsPackCoversDeclaredKStmtUniverse : IO Unit := do
  let missing ← runCoreM (do
    registerKStmtAllSupportedSemanticsRules
    collectMissingRules allKStmtSupportedOpNames
  )
  LeanTest.assertTrue missing.isEmpty
    s!"KStmt all-semantics pack is missing declared KStmt op rules: {reprStr missing}"

@[test]
def testGraphaxAlphaGradParityPackCoversDeclaredSupportMatrixSet : IO Unit := do
  let missing ← runCoreM (do
    registerGraphaxAlphaGradParityRules
    collectMissingRules declaredParityOpNames
  )
  LeanTest.assertTrue missing.isEmpty
    s!"Graphax/AlphaGrad parity pack is missing declared support-matrix rules: {reprStr missing}"

@[test]
def testGraphaxParityPackCoversHigherOrderControlAliases : IO Unit := do
  let missing ← runCoreM (do
    registerGraphaxAlphaGradParityRules
    collectMissingRules allHigherOrderControlAliasOpNames
  )
  LeanTest.assertTrue missing.isEmpty
    s!"scan/cond aliases should be registered with explicit local-Jac semantics: {reprStr missing}"

end Tests.ADGraphaxAlphaGradCoverageGate
