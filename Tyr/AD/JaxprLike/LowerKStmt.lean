import Tyr.AD.JaxprLike.FromKStmt

/-!
# Tyr.AD.JaxprLike.LowerKStmt

Typed lowering entrypoint for converting GPU `KStmt` inputs into `LeanJaxpr`
while preserving existing conversion behavior.
-/

namespace Tyr.AD.JaxprLike

open Tyr.GPU.Codegen

namespace LowerKStmt

/-- Typed input for lowering a `KStmt` program. -/
structure Input where
  stmts : Array KStmt
  deriving Inhabited

/-- Error wrapper preserving `FromKStmt` diagnostics verbatim. -/
structure Error where
  messages : Array String
  deriving Repr, Inhabited

def Error.toString (err : Error) : String :=
  String.intercalate "\n" err.messages.toList

instance : ToString Error := ⟨Error.toString⟩

/-- Typed entrypoint that wraps `fromKStmts` without fallback behavior. -/
def run (input : Input) : Except Error LeanJaxpr :=
  match fromKStmts input.stmts with
  | .ok jaxpr => .ok jaxpr
  | .error messages => .error { messages := messages }

end LowerKStmt

end Tyr.AD.JaxprLike
