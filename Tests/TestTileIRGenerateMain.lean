import Lean
import LeanTest
import Tyr.GPU.Codegen.TileIR.Specialization

namespace Tests.TileIRGenerateMain

open Lean
open LeanTest
open Tyr.GPU.Codegen.TileIR

private def runCoreMResult (x : CoreM α) : IO (Except String α) := do
  let env ← Lean.importModules #[{ module := `Tyr.GPU.Codegen.TileIR.Specialization }] {}
  let ctx : Core.Context := { fileName := "<tileir-generate-test>", fileMap := default }
  let state : Core.State := { env := env }
  let eio := x.run ctx state
  let res ← EIO.toBaseIO eio
  match res with
  | .ok (value, _) => pure (.ok value)
  | .error err =>
      pure (.error (← err.toMessageData.toString))

private def runCoreM (x : CoreM α) : IO α := do
  match (← runCoreMResult x) with
  | .ok value => pure value
  | .error msg => throw (IO.userError msg)

private def specializationOrderType : Expr :=
  mkForall `rows .default (mkConst ``Nat) <|
    mkForall `cols .default (mkConst ``Int) <|
      mkForall `enabled .default (mkConst ``Bool) <|
        mkConst ``Tyr.GPU.Codegen.TileIR.Module

@[test]
def testRecoverConstParamKindsOrdering : IO Unit := do
  let some kinds ← runCoreM <| Lean.Meta.MetaM.run' do
    Tyr.GPU.Codegen.TileIR.recoverConstParamKinds? specializationOrderType
    | fail "Expected to recover ct.Const parameter kinds from the demo kernel type"
  let expected : Array Tyr.GPU.Codegen.TileIR.ConstParamKind :=
    #[.nat, .int, .bool]
  assertTrue (kinds == expected)
    s!"ct.Const specialization kinds should be recovered in source binder order, got {repr kinds}"

end Tests.TileIRGenerateMain
