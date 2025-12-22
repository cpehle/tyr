/-
  Tyr/Module/Core.lean

  Equinox-style Module typeclasses for neural networks.
  Provides callable module abstractions with TensorStruct integration.
-/
import Tyr.TensorStruct

namespace torch

/-! ## Module Typeclasses

These typeclasses define the interface for neural network modules:
- `Module`: Pure modules with no side effects in forward pass
- `ModuleIO`: Modules with IO effects (dropout, batch norm in training mode)
- `ModuleCtx`: Modules that take additional context (e.g., training flag, cache)
-/

/-- A pure module: a parameterized function with no side effects.
    The module type `M` must be a TensorStruct for parameter traversal. -/
class Module (M : Type) (In : Type) (Out : Type) where
  [toTensorStruct : TensorStruct M]
  forward : M → In → Out

attribute [instance] Module.toTensorStruct

/-- A module with IO effects (dropout, random sampling, etc.) -/
class ModuleIO (M : Type) (In : Type) (Out : Type) where
  [toTensorStruct : TensorStruct M]
  forward : M → In → IO Out

attribute [instance] ModuleIO.toTensorStruct

/-- A module that takes additional context (training flag, cache, etc.) -/
class ModuleCtx (M : Type) (Ctx : Type) (In : Type) (Out : Type) where
  [toTensorStruct : TensorStruct M]
  forward : M → Ctx → In → IO Out

attribute [instance] ModuleCtx.toTensorStruct

/-! ## Lifting Instances

Automatically lift pure Module to ModuleIO.
-/

/-- Lift a pure Module to ModuleIO -/
instance [inst : Module M In Out] : ModuleIO M In Out where
  toTensorStruct := inst.toTensorStruct
  forward m x := pure (Module.forward m x)

/-! ## Forward Application

Explicit forward pass functions. CoeFun doesn't work well with multi-parameter
typeclasses, so we use explicit syntax instead.
-/

/-- Apply a pure module to input -/
@[inline] def Module.apply [Module M In Out] (m : M) (x : In) : Out :=
  Module.forward m x

/-- Apply an IO module to input -/
@[inline] def ModuleIO.apply [ModuleIO M In Out] (m : M) (x : In) : IO Out :=
  ModuleIO.forward m x

/-- Infix notation for module application: `m |> x` -/
scoped infixl:90 " |> " => Module.apply

/-- Infix notation for IO module application: `m |>! x` -/
scoped infixl:90 " |>! " => ModuleIO.apply

/-! ## Training Context

Common context type for modules that behave differently in training vs inference.
-/

/-- Training context for modules like Dropout, BatchNorm -/
structure TrainingCtx where
  training : Bool := true
  dropout_p : Float := 0.0
  deriving Repr, BEq, Inhabited

namespace TrainingCtx

def train : TrainingCtx := { training := true }
def eval : TrainingCtx := { training := false, dropout_p := 0.0 }

def withDropout (ctx : TrainingCtx) (p : Float) : TrainingCtx :=
  { ctx with dropout_p := if ctx.training then p else 0.0 }

end TrainingCtx

end torch
