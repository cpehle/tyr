import Tyr.AutoGrad
import Tests.Test
import LeanTest

namespace Tests.AutoGrad

open Tyr.AD
open LeanTest

-- 1. Define Primal Functions
def square (x : Float) : Float := x * x
def mul (x y : Float) : Float := x * y

-- 2. Define and Register Rules

@[vjp square]
def square_bwd (x dy : Float) : Float := 
  2.0 * x * dy

@[jvp square]
def square_fwd (x dx : Float) : Float × Float :=
  (x * x, 2.0 * x * dx)

@[vjp mul]
def mul_bwd (x y dz : Float) : Float × Float :=
  (dz * y, dz * x)

@[jvp mul]
def mul_fwd (x y dx dy : Float) : Float × Float :=
  (x * y, x * dy + y * dx)

@[test]
def testAttributes : IO Unit := do
  -- This test passes if the file compiles and attributes are processed
  -- We could inspect the environment to see if rules are registered,
  -- but access to `getEnv` requires `CoreM` or similar, not `IO`.
  -- For now, compilation is the main check.
  return

@[test]
def testLinearize : IO Unit := do
  -- Placeholder for functional testing of linearize.
  -- To properly test, we'd need to run `linearize` on a definition
  -- and check the output IR or execute it.
  -- Currently `linearize` is a transformation, execution requires integration.
  return

end Tests.AutoGrad
