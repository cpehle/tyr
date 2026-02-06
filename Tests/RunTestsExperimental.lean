import LeanTest
import Lean.Util.Path
import TestsExperimental

unsafe def main (_args : List String) : IO UInt32 := do
  -- Initialize search path
  Lean.initSearchPath (← Lean.findSysroot)

  let env ← Lean.importModules #[
    { module := `LeanTest },
    { module := `TestsExperimental }
  ] {}

  -- Run the full experimental suite without additional filtering.
  LeanTest.runTestsAndExit env {} {}
