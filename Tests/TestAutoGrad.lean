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

/-! ## JVP Execution Tests -/

@[test]
def testSquareJVPExecution : IO Unit := do
  -- Test that JVP rules produce correct tangents
  -- square: f(x) = x², f'(x) = 2x
  -- At x=3, dx=1: f(3)=9, f'(3)*1=6
  let (y, dy) := square_fwd 3.0 1.0
  LeanTest.assertTrue (y == 9.0) s!"square(3) should be 9, got {y}"
  LeanTest.assertTrue (dy == 6.0) s!"d/dx[x²] at x=3 should be 6, got {dy}"

@[test]
def testSquareJVPAtZero : IO Unit := do
  -- At x=0: f(0)=0, f'(0)*1=0
  let (y, dy) := square_fwd 0.0 1.0
  LeanTest.assertTrue (y == 0.0) s!"square(0) should be 0"
  LeanTest.assertTrue (dy == 0.0) s!"d/dx[x²] at x=0 should be 0"

@[test]
def testSquareJVPNegative : IO Unit := do
  -- At x=-2: f(-2)=4, f'(-2)*1=-4
  let (y, dy) := square_fwd (-2.0) 1.0
  LeanTest.assertTrue (y == 4.0) s!"square(-2) should be 4"
  LeanTest.assertTrue (dy == -4.0) s!"d/dx[x²] at x=-2 should be -4, got {dy}"

/-! ## VJP Execution Tests -/

@[test]
def testSquareVJPExecution : IO Unit := do
  -- Test that VJP rules produce correct cotangents
  -- square backward: dx = 2*x*dy
  -- At x=3, dy=1: dx = 6
  let dx := square_bwd 3.0 1.0
  LeanTest.assertTrue (dx == 6.0) s!"grad of x² at x=3 should be 6, got {dx}"

@[test]
def testSquareVJPScaled : IO Unit := do
  -- At x=3, dy=2: dx = 2*3*2 = 12
  let dx := square_bwd 3.0 2.0
  LeanTest.assertTrue (dx == 12.0) s!"grad of x² at x=3 with dy=2 should be 12, got {dx}"

/-! ## Multiplication Tests -/

@[test]
def testMulJVPExecution : IO Unit := do
  -- f(x,y) = x*y, df = x*dy + y*dx
  -- At (2,3), (dx=1,dy=0): f=6, df=3 (only contribution from dx)
  let (f, df) := mul_fwd 2.0 3.0 1.0 0.0
  LeanTest.assertTrue (f == 6.0) s!"2*3 should be 6"
  LeanTest.assertTrue (df == 3.0) s!"d/dx[x*y] at (2,3) should be 3, got {df}"

@[test]
def testMulJVPBothTangents : IO Unit := do
  -- At (2,3), (dx=1,dy=1): f=6, df = 2*1 + 3*1 = 5
  let (f, df) := mul_fwd 2.0 3.0 1.0 1.0
  LeanTest.assertTrue (f == 6.0) s!"2*3 should be 6"
  LeanTest.assertTrue (df == 5.0) s!"df should be x*dy + y*dx = 2+3 = 5, got {df}"

@[test]
def testMulVJPExecution : IO Unit := do
  -- Backward: dx = dz*y, dy = dz*x
  -- At (2,3), dz=1: dx=3, dy=2
  let (dx, dy) := mul_bwd 2.0 3.0 1.0
  LeanTest.assertTrue (dx == 3.0) s!"dx should be y=3, got {dx}"
  LeanTest.assertTrue (dy == 2.0) s!"dy should be x=2, got {dy}"

@[test]
def testMulVJPScaled : IO Unit := do
  -- At (2,3), dz=5: dx=15, dy=10
  let (dx, dy) := mul_bwd 2.0 3.0 5.0
  LeanTest.assertTrue (dx == 15.0) s!"dx should be dz*y=15, got {dx}"
  LeanTest.assertTrue (dy == 10.0) s!"dy should be dz*x=10, got {dy}"

/-! ## Chain Rule Tests -/

-- Manually compute chain rule: d/dx[f(g(x))] = f'(g(x)) * g'(x)
-- For h(x) = square(square(x)) = x^4
-- h'(x) = 4x^3

@[test]
def testChainRuleManual : IO Unit := do
  -- h(x) = (x²)² = x⁴
  -- At x=2: h(2)=16, h'(2)=4*8=32
  let x := 2.0
  let dx := 1.0

  -- Forward pass through first square
  let (y1, dy1) := square_fwd x dx      -- y1=4, dy1=4

  -- Forward pass through second square
  let (y2, dy2) := square_fwd y1 dy1    -- y2=16, dy2=2*4*4=32

  LeanTest.assertTrue (y2 == 16.0) s!"x⁴ at x=2 should be 16, got {y2}"
  LeanTest.assertTrue (dy2 == 32.0) s!"d/dx[x⁴] at x=2 should be 32, got {dy2}"

/-! ## Static Parameter Tests -/

-- A function with a static (non-differentiable) parameter
-- Multiplies x by a constant factor n
def scale (x : Float) (n : Nat) : Float := x * n.toFloat

-- JVP rule: only x has a tangent, n is static
-- Note: n doesn't get a tangent parameter
@[jvp scale, static := [1]]
def scale_fwd (x dx : Float) (n : Nat) : Float × Float :=
  (x * n.toFloat, dx * n.toFloat)

-- VJP rule: only x gets a cotangent, n is static
-- Note: returns only dx, not (dx, dn)
@[vjp scale, static := [1]]
def scale_bwd (x : Float) (n : Nat) (dy : Float) : Float :=
  dy * n.toFloat

@[test]
def testStaticJVP : IO Unit := do
  -- scale(3.0, 5) = 15.0
  -- d/dx[scale(x, 5)] at x=3 with dx=1 should be 5
  let (y, dy) := scale_fwd 3.0 1.0 5
  LeanTest.assertTrue (y == 15.0) s!"scale(3, 5) should be 15, got {y}"
  LeanTest.assertTrue (dy == 5.0) s!"d/dx[scale] at x=3, n=5 should be 5, got {dy}"

@[test]
def testStaticJVPDifferentScale : IO Unit := do
  -- scale(2.0, 10) = 20.0
  -- d/dx[scale(x, 10)] at x=2 with dx=1 should be 10
  let (y, dy) := scale_fwd 2.0 1.0 10
  LeanTest.assertTrue (y == 20.0) s!"scale(2, 10) should be 20, got {y}"
  LeanTest.assertTrue (dy == 10.0) s!"d/dx[scale] with n=10 should be 10, got {dy}"

@[test]
def testStaticVJP : IO Unit := do
  -- Backward: dx = dy * n
  -- At x=3, n=5, dy=1: dx = 5
  let dx := scale_bwd 3.0 5 1.0
  LeanTest.assertTrue (dx == 5.0) s!"grad of scale(x, 5) should be 5, got {dx}"

@[test]
def testStaticVJPScaled : IO Unit := do
  -- At x=3, n=5, dy=2: dx = 10
  let dx := scale_bwd 3.0 5 2.0
  LeanTest.assertTrue (dx == 10.0) s!"grad of scale with dy=2 should be 10, got {dx}"

-- Function with mixed parameters: (diff, static, diff)
def weightedAdd (x : Float) (weight : Nat) (y : Float) : Float :=
  x + (weight.toFloat * y)

-- JVP: x and y get tangents, weight is static
@[jvp weightedAdd, static := [1]]
def weightedAdd_fwd (x dx : Float) (weight : Nat) (y dy : Float) : Float × Float :=
  (x + weight.toFloat * y, dx + weight.toFloat * dy)

-- VJP: returns (dx, dy), no gradient for weight
@[vjp weightedAdd, static := [1]]
def weightedAdd_bwd (x : Float) (weight : Nat) (y : Float) (dz : Float) : Float × Float :=
  (dz, dz * weight.toFloat)

@[test]
def testMixedStaticJVP : IO Unit := do
  -- weightedAdd(2, 3, 4) = 2 + 3*4 = 14
  -- With dx=1, dy=1: dz = 1 + 3*1 = 4
  let (z, dz) := weightedAdd_fwd 2.0 1.0 3 4.0 1.0
  LeanTest.assertTrue (z == 14.0) s!"weightedAdd(2, 3, 4) should be 14, got {z}"
  LeanTest.assertTrue (dz == 4.0) s!"dz should be 1 + 3*1 = 4, got {dz}"

@[test]
def testMixedStaticVJP : IO Unit := do
  -- Backward: dx = dz, dy = dz * weight
  -- At dz=1, weight=3: dx=1, dy=3
  let (dx, dy) := weightedAdd_bwd 2.0 3 4.0 1.0
  LeanTest.assertTrue (dx == 1.0) s!"dx should be 1, got {dx}"
  LeanTest.assertTrue (dy == 3.0) s!"dy should be 3, got {dy}"

@[test]
def testStaticAttributeCompiles : IO Unit := do
  -- This test verifies that the static := [...] attribute syntax compiles
  -- The main check is that the file compiles with the static attribute syntax
  return

/-! ## DifferentiableManifold Typeclass Tests -/

open Tyr.AD.DifferentiableManifold in
@[test]
def testFloatEuclidean : IO Unit := do
  -- Float as EuclideanSpace operations
  let zero : Float := EuclideanSpace.zero
  LeanTest.assertTrue (zero == 0.0) s!"Float zero should be 0.0"

  let sum := EuclideanSpace.add 2.0 3.0
  LeanTest.assertTrue (sum == 5.0) s!"Float add should work, got {sum}"

  let scaled := EuclideanSpace.smul 2.0 3.0
  LeanTest.assertTrue (scaled == 6.0) s!"Float smul should work, got {scaled}"

  let inner := EuclideanSpace.inner 3.0 4.0
  LeanTest.assertTrue (inner == 12.0) s!"Float inner should work, got {inner}"

open Tyr.AD.DifferentiableManifold in
@[test]
def testFloatManifold : IO Unit := do
  -- Float as DifferentiableManifold - test exp map
  let x : Float := 3.0
  let v : Float := 2.0  -- Tangent vector
  let moved := DifferentiableManifold.exp x v
  LeanTest.assertTrue (moved == 5.0) s!"exp(3, 2) should be 5, got {moved}"

  -- Test retract (same as exp for Euclidean)
  let retracted := DifferentiableManifold.retract x v
  LeanTest.assertTrue (retracted == 5.0) s!"retract(3, 2) should be 5, got {retracted}"

open Tyr.AD in
@[test]
def testStaticManifold : IO Unit := do
  -- Static has Unit tangent/cotangent
  let x : torch.Static Float := ⟨42.0⟩
  let zeroTan := DifferentiableManifold.zeroTangent x
  let zeroCot := DifferentiableManifold.zeroCotangent x

  -- Adding units should work
  let unitSum := DifferentiableManifold.addTangent zeroTan zeroTan
  -- Check they are unit
  match zeroTan, zeroCot, unitSum with
  | (), (), () => pure ()

  -- Static exp should not move the point
  let moved := DifferentiableManifold.exp x ()
  LeanTest.assertTrue (moved.val == 42.0) s!"Static exp should not move, got {moved.val}"

open Tyr.AD.DifferentiableManifold in
@[test]
def testGradientStep : IO Unit := do
  -- Test gradient descent step using retraction
  let x : Float := 10.0
  let grad : Float := 2.0  -- Cotangent (gradient)
  let lr := 0.1
  let x' := gradientStep x grad lr
  -- x' = exp(x, -lr * sharp(grad))
  -- For Euclidean: x' = x + (-lr * grad) = 10 - 0.2 = 9.8
  LeanTest.assertTrue (x' == 9.8) s!"Expected 9.8, got {x'}"

open Tyr.AD.DifferentiableManifold in
@[test]
def testMusicalIsomorphisms : IO Unit := do
  -- For Euclidean spaces, sharp and flat are identity
  let x : Float := 5.0
  let cotangent : Float := 3.0
  -- For Float, Tangent x = Float, so sharp/flat are id
  let tangent : Float := sharp (x := x) cotangent
  LeanTest.assertTrue (tangent == 3.0) s!"sharp should be identity for Euclidean, got {tangent}"

  let flatted : Float := flat (x := x) tangent
  LeanTest.assertTrue (flatted == 3.0) s!"flat should be identity for Euclidean, got {flatted}"

end Tests.AutoGrad
