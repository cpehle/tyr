/-
  Tyr/GPU/Codegen/IR.lean

  Kernel IR using VarId for type-safe code generation.
  Replaces string-based KExpr with proper variable tracking.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.AST

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-- GPU kernel statement using VarId -/
inductive KStmt where
  -- Tile declarations
  | declRT (v : VarId) (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout)
  | declST (v : VarId) (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout)
  | declRV (v : VarId) (dtype : GpuFloat) (len : Nat)
  | declSV (v : VarId) (dtype : GpuFloat) (len : Nat)

  -- Memory operations
  | load (dst src : VarId)
  | store (dst src : VarId)
  | loadAsync (dst src : VarId)
  | storeAsync (dst src : VarId)
  | tmaExpect (barrier : VarId) (bytes : Nat)

  -- MMA operations
  | mma (trans : MMATranspose) (dst a b c : VarId)
  | mm (trans : MMATranspose) (dst a b : VarId)
  | mmaFence (dst : VarId)
  | mmaCommitGroup
  | mmaAsyncWait (n : Nat)

  -- Element-wise unary
  | unary (op : UnaryOp) (dst src : VarId)

  -- Element-wise binary
  | binary (op : BinaryOp) (dst a b : VarId)

  -- Scalar operations
  | scalarMul (dst src : VarId) (scalar : Float)

  -- Broadcasting
  | broadcast (axis : BroadcastAxis) (dst vec : VarId)
  | binaryBroadcast (op : BinaryOp) (axis : BroadcastAxis) (dst tile vec : VarId)

  -- Reductions
  | reduce (op : ReduceOp) (axis : ReduceAxis) (dst src : VarId)
  | reduceAccum (op : ReduceOp) (axis : ReduceAxis) (dst src accum : VarId)

  -- Scan/prefix operations
  | cumsum (axis : ReduceAxis) (dst src : VarId)

  -- Outer product
  | outer (dst a b : VarId)

  -- Layout/type conversions
  | swapLayout (dst src : VarId)
  | transpose (dst src : VarId)
  | convert (dst src : VarId)

  -- Masking
  | mask (op : MaskOp) (dst src : VarId) (fillVal : Option Float)

  -- Tile slicing
  | sliceRows (dst src : VarId) (startRow numRows : Nat)
  | sliceCols (dst src : VarId) (startCol numCols : Nat)

  -- Synchronization
  | sync (barrierId : Nat)
  | arrive (barrierId : Nat)

  -- Control flow
  | forLoop (v : VarId) (lo hi : Nat) (body : Array KStmt)
  | comment (text : String)

  deriving Repr, Inhabited

/-- Kernel parameter -/
structure KParam where
  name : String
  dtype : GpuFloat
  isPointer : Bool := false
  deriving Repr, Inhabited

/-- Complete kernel definition -/
structure Kernel where
  name : String
  arch : GpuArch
  params : Array KParam
  body : Array KStmt
  sharedMemBytes : Nat := 0
  deriving Repr, Inhabited

end Tyr.GPU.Codegen
