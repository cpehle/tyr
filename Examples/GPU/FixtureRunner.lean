/- Shared fixture/runner helpers for GPU kernel end-to-end examples. -/

/-!
# `Examples.GPU.FixtureRunner`

GPU example utility module for Fixture Runner.

## Overview
- Example module intended for runnable workflows and reference usage patterns.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace Examples.GPU

structure FixtureSpec where
  dir : System.FilePath
  names : Array String
  deriving Inhabited

def fixturePath (spec : FixtureSpec) (name : String) : System.FilePath :=
  spec.dir / s!"{name}.pt"

def fixturesPresent (spec : FixtureSpec) : IO Bool := do
  let mut ok := true
  for name in spec.names do
    ok := ok && (← (fixturePath spec name).pathExists)
  pure ok

private def parseCommonArgs (args : List String) : Bool × Bool :=
  (args.contains "--regen", args.contains "--gen-only")

def runWithFixtures
    (args : List String)
    (spec : FixtureSpec)
    (generateFixtures : IO Unit)
    (runOnce : IO Bool)
    : IO UInt32 := do
  let (regen, genOnly) := parseCommonArgs args

  if regen then
    generateFixtures
  else if !(← fixturesPresent spec) then
    generateFixtures

  if genOnly then
    return 0

  let ok ← runOnce
  pure (if ok then 0 else 1)

end Examples.GPU
