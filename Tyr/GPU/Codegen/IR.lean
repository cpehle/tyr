import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.AST
import Tyr.GPU.Codegen.Stmt
import Tyr.GPU.Codegen.Proof

/-!
# Tyr.GPU.Codegen.IR

`Tyr.GPU.Codegen.IR` is the central intermediate representation for Tyr GPU
kernels. It packages `KStmt` programs with signatures, architecture, resource
metadata, and proof metadata.

Key pieces:

- `KScalarType`: scalar parameter kinds used at kernel boundaries.
- `KStmt`: declarative instruction set imported from `Codegen.Stmt`.
- `KParam`: typed kernel signature parameters.
- `Kernel`: complete lowered kernel record consumed by emitters.

Higher-level modules (`Ops`, `Notation`, `Arch`) construct this IR; emitters
(`EmitNew`) serialize it to target-specific CUDA/C++.
-/

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-- Scalar parameter type for `KVal` kernel parameters. -/
inductive KScalarType where
  | UInt8
  | UInt16
  | UInt32
  | UInt64
  | USize
  | Float
  | Float32
  | Bool
  deriving Repr, Inhabited, BEq

/-- Convert scalar type to C++ type name used in extern/kernel signatures. -/
def KScalarType.toCpp : KScalarType → String
  | .UInt8 => "uint8_t"
  | .UInt16 => "uint16_t"
  | .UInt32 => "uint32_t"
  | .UInt64 => "uint64_t"
  | .USize => "size_t"
  | .Float => "double"
  | .Float32 => "float"
  | .Bool => "uint8_t"

/-- Kernel parameter -/
structure KParam where
  name : String
  dtype : GpuFloat
  isPointer : Bool := false
  scalarPointer : Bool := false
  scalarTy : KScalarType := .UInt64
  deriving Repr, Inhabited, BEq

/-- Complete kernel definition -/
structure Kernel where
  name : String
  arch : GpuArch
  params : Array KParam
  body : Array KStmt
  sharedMemBytes : Nat := 0
  proof : KernelProof := default
  deriving Repr, Inhabited, BEq

end Tyr.GPU.Codegen
