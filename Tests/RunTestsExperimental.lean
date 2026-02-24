import LeanTest
import Lean.Util.Path
import TestsExperimental

/-!
# `Tests.RunTestsExperimental`

Experimental test runner that imports TestsExperimental and executes the experimental suite.

## Overview
- Regression and behavior checks run by the LeanTest-based test suite.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

unsafe def main (_args : List String) : IO UInt32 := do
  -- Initialize search path
  Lean.initSearchPath (← Lean.findSysroot)

  let env ← Lean.importModules #[
    { module := `LeanTest },
    { module := `TestsExperimental }
  ] {}

  -- Run the full experimental suite without additional filtering.
  LeanTest.runTestsAndExit env {} {}
