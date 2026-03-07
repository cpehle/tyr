import Tyr.AD.JaxprLike.FromFnBody

/-!
# Tyr.AD.JaxprLike.LowerFnBody

Typed lowering entrypoints for converting Lean IR `FnBody` inputs into
`LeanJaxpr` while preserving existing conversion behavior.
-/

namespace Tyr.AD.JaxprLike

open Lean
open Lean.IR

namespace LowerFnBody

/-- Typed input for lowering a raw Lean IR function body. -/
structure Input where
  declName : Name
  params : Array Param
  body : FnBody
  deriving Inhabited

/-- Typed input for lowering a full Lean IR declaration. -/
structure DeclInput where
  decl : Decl
  deriving Inhabited

/-- Error wrapper preserving `FromFnBody` diagnostics verbatim. -/
structure Error where
  message : String
  deriving Repr, Inhabited

def Error.toString (err : Error) : String :=
  err.message

instance : ToString Error := ⟨Error.toString⟩

/-- Typed entrypoint that wraps `fromFnBody` without fallback behavior. -/
def run (input : Input) : Except Error LeanJaxpr :=
  match fromFnBody input.declName input.params input.body with
  | .ok jaxpr => .ok jaxpr
  | .error message => .error { message := message }

/-- Typed entrypoint that wraps `fromDecl` without fallback behavior. -/
def runDecl (input : DeclInput) : Except Error LeanJaxpr :=
  match fromDecl input.decl with
  | .ok jaxpr => .ok jaxpr
  | .error message => .error { message := message }

end LowerFnBody

end Tyr.AD.JaxprLike
