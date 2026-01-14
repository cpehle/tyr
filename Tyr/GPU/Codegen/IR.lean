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
  | declSemaphore (v : VarId)  -- Semaphore/barrier declaration

  -- Kernel parameter declarations (used by attribute-generated code)
  | declGPtr (v : VarId) (dtype : GpuFloat) (name : String)  -- Global memory pointer param
  | declKVal (v : VarId) (dtype : GpuFloat) (name : String)  -- Scalar value param

  -- Memory operations
  | load (dst src : VarId)
  | store (dst src : VarId)
  | loadAsync (dst src : VarId)
  | storeAsync (dst src : VarId)
  | storeAdd (dst src : VarId)       -- Atomic add for gradient accumulation
  | storeAddAsync (dst src : VarId)  -- Async atomic add (TMA)
  | storeMinAsync (dst src : VarId)  -- Async atomic min (TMA)
  | prefetch (src : VarId)           -- TMA prefetch
  | tmaExpect (barrier : VarId) (bytes : Nat)

  -- TMA operations with global pointers
  | tmaLoad (dst src : VarId) (coord : VarId)      -- TMA load: shared ← global[coord]
  | tmaStore (dst src : VarId) (coord : VarId)     -- TMA store: global[coord] ← shared

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

  -- Element-wise ternary (FMA)
  | ternary (op : TernaryOp) (dst a b c : VarId)

  -- Scalar operations
  | scalarMul (dst src : VarId) (scalar : Float)
  | scalarAdd (dst src : VarId) (scalar : Float)

  -- Broadcasting
  | broadcast (axis : BroadcastAxis) (dst vec : VarId)
  | binaryBroadcast (op : BinaryOp) (axis : BroadcastAxis) (dst tile vec : VarId)

  -- Reductions
  | reduce (op : ReduceOp) (axis : ReduceAxis) (dst src : VarId)
  | reduceAccum (op : ReduceOp) (axis : ReduceAxis) (dst src accum : VarId)

  -- Scan/prefix operations
  | cumsum (axis : ReduceAxis) (dst src : VarId)
  | cumprod (axis : ReduceAxis) (dst src : VarId)  -- Cumulative product (for decay)

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
  | arriveAndWait (barrierId : Nat)

  -- Semaphore operations
  | semaphore (op : SemaphoreOp) (sem : VarId)

  -- Control flow
  | forLoop (v : VarId) (lo hi : Nat) (body : Array KStmt)
  | ifStmt (cond : VarId) (thenBody elseBody : Array KStmt)  -- Conditional
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
