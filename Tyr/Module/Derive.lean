/-
  Tyr/Module/Derive.lean

  Macro-based approach for deriving TensorStruct instances.
  Provides `derive_tensor_struct` command for simple structures.
-/
import Lean
import Tyr.TensorStruct

open Lean Elab Command Term Meta

namespace torch

/-! ## TensorStruct Instance Generation

For now, we provide manual instance templates and helper macros.
A full deriving handler requires complex metaprogramming; this simpler
approach works for most cases.

### Field Type Handling

When generating a TensorStruct instance, each field is handled based on its type:

- `T s` → apply function directly: `f field`
- `Static α` → pass through: `field`
- `Frozen s` → apply to tensor: `{ tensor := f field.tensor }`
- `Array α` → recursive: `field.map (TensorStruct.map f)`
- `Option α` → recursive: `field.map (TensorStruct.map f)`
- Nested `TensorStruct` → recursive: `TensorStruct.map f field`

### Example Usage

```lean
-- Define your structure
structure MyLayer where
  weight : T #[n, m]
  bias : T #[n]
  config : Static Config

-- Manually create instance (the macro would generate this)
instance : TensorStruct MyLayer where
  map f l := { weight := f l.weight, bias := f l.bias, config := l.config }
  mapM f l := do pure { weight := ← f l.weight, bias := ← f l.bias, config := l.config }
  zipWith f l1 l2 := { weight := f l1.weight l2.weight, bias := f l1.bias l2.bias, config := l1.config }
  fold f init l := f l.weight (f l.bias init)
```
-/

/-! ## Helper Functions for Manual Instances -/

/-- Helper: map a function over a tensor field -/
@[inline] def mapTensor {s : Shape} (f : ∀ {s}, T s → T s) (t : T s) : T s := f t

/-- Helper: map a function over a frozen field -/
@[inline] def mapFrozen {s : Shape} (f : ∀ {s}, T s → T s) (fr : Frozen s) : Frozen s :=
  { tensor := f fr.tensor }

/-- Helper: skip a static field -/
@[inline] def mapStatic {α : Type} (_ : ∀ {s}, T s → T s) (s : Static α) : Static α := s

/-! ## Instance for Single Tensor

A single tensor is trivially a TensorStruct.
-/

instance {s : Shape} : TensorStruct (T s) where
  map f t := f t
  mapM f t := f t
  zipWith f t1 t2 := f t1 t2
  fold f init t := f t init

/-! ## Tuple Instances

Allow pairs of TensorStructs to be TensorStructs.
-/

instance [TensorStruct α] [TensorStruct β] : TensorStruct (α × β) where
  map f pair := (TensorStruct.map f pair.1, TensorStruct.map f pair.2)
  mapM f pair := do pure (← TensorStruct.mapM f pair.1, ← TensorStruct.mapM f pair.2)
  zipWith f p1 p2 := (TensorStruct.zipWith f p1.1 p2.1, TensorStruct.zipWith f p1.2 p2.2)
  fold f init pair := TensorStruct.fold f (TensorStruct.fold f init pair.1) pair.2

/-! ## Deriving Handler for TensorStruct -/

open Lean Elab Command Term Meta in
/-- Generate TensorStruct instance for a structure.
    All fields are assumed to have TensorStruct instances (T s, Option, Array, Static, etc.) -/
private def mkTensorStructInstanceCmd (typeName : Name) : CommandElabM Unit := do
  let env ← getEnv

  -- Get structure info
  let some structInfo := getStructureInfo? env typeName
    | throwError "{typeName} is not a structure"

  -- Get the inductive info to find parameters
  let some (.inductInfo indInfo) := env.find? typeName
    | throwError "{typeName} is not an inductive type"

  let fieldNames := structInfo.fieldNames

  -- Build field entries for `map`: field := TensorStruct.map f x.field
  let mapFields ← fieldNames.mapM fun fname => do
    let fnameId := mkIdent fname
    `(Lean.Parser.Term.structInstField| $fnameId:ident := TensorStruct.map f x.$fnameId)

  -- Build field entries for `mapM`: let field ← TensorStruct.mapM f x.field
  let mapMBinds ← fieldNames.mapM fun fname => do
    let fnameId := mkIdent fname
    `(doElem| let $fnameId ← TensorStruct.mapM f x.$fnameId)

  -- Build struct literal for mapM result
  let mapMFields ← fieldNames.mapM fun fname => do
    let fnameId := mkIdent fname
    `(Lean.Parser.Term.structInstField| $fnameId:ident := $fnameId)

  -- Build field entries for `zipWith`: field := TensorStruct.zipWith f x.field y.field
  let zipFields ← fieldNames.mapM fun fname => do
    let fnameId := mkIdent fname
    `(Lean.Parser.Term.structInstField| $fnameId:ident := TensorStruct.zipWith f x.$fnameId y.$fnameId)

  -- Build fold expression: TensorStruct.fold f (TensorStruct.fold f ... init x.field1) x.field2
  -- We build this as a right-to-left chain
  let mut foldExpr ← `(init)
  for fname in fieldNames do
    let fnameId := mkIdent fname
    foldExpr ← `(TensorStruct.fold f $foldExpr x.$fnameId)

  -- Get the type parameters
  let numParams := indInfo.numParams

  if numParams == 0 then
    -- No parameters - simple instance
    let typeId := mkIdent typeName
    let instCmd ← `(command|
      instance : TensorStruct $typeId where
        map f x := { $[$mapFields:structInstField],* }
        mapM f x := do
          $[$mapMBinds:doElem]*
          pure { $[$mapMFields:structInstField],* }
        zipWith f x y := { $[$zipFields:structInstField],* }
        fold f init x := $foldExpr
    )
    elabCommand instCmd
  else
    -- Has parameters - need to build binders
    let instCmd ← liftTermElabM do
      Meta.forallTelescopeReducing indInfo.type fun params _ => do
        let paramBinders := params[:numParams]

        -- Build implicit binder syntax for each parameter
        let binderStxs ← paramBinders.toArray.mapM fun param => do
          let paramDecl ← param.fvarId!.getDecl
          let paramName := paramDecl.userName
          let paramType ← Meta.inferType param
          let paramTypeStx ← PrettyPrinter.delab paramType
          `(bracketedBinder| {$(mkIdent paramName) : $paramTypeStx})

        -- Build the applied type
        let paramIdents ← paramBinders.toArray.mapM fun param => do
          let paramDecl ← param.fvarId!.getDecl
          pure (mkIdent paramDecl.userName)

        let appliedType ← `($(mkIdent typeName) $paramIdents*)

        -- Generate instance command with binders
        `(command|
          instance $[$binderStxs]* : TensorStruct $appliedType where
            map f x := { $[$mapFields:structInstField],* }
            mapM f x := do
              $[$mapMBinds:doElem]*
              pure { $[$mapMFields:structInstField],* }
            zipWith f x y := { $[$zipFields:structInstField],* }
            fold f init x := $foldExpr
        )

    elabCommand instCmd

open Lean Elab Deriving in
/-- Deriving handler for TensorStruct -/
def mkTensorStructHandler (typeNames : Array Name) : CommandElabM Bool := do
  if typeNames.isEmpty then return false

  for typeName in typeNames do
    mkTensorStructInstanceCmd typeName

  return true

-- Register the deriving handler
open Lean Elab Deriving in
initialize
  registerDerivingHandler ``TensorStruct mkTensorStructHandler

end torch
