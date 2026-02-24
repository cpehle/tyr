/-
  Tyr/GPU/Codegen/IR.lean

  Kernel IR using VarId for type-safe code generation.
  Replaces string-based KExpr with proper variable tracking.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.AST

/-!
# `Tyr.GPU.Codegen.IR`

GPU code generation component for IR, used to lower high-level tile programs to backend code.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
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

  -- TMA operations with global pointers (legacy single coord)
  | tmaLoad (dst src : VarId) (coord : VarId)      -- TMA load: shared ← global[coord]
  | tmaStore (dst src : VarId) (coord : VarId)     -- TMA store: global[coord] ← shared

  -- Global memory operations with 4D coordinates (ThunderKittens style)
  | loadGlobal (dst src : VarId) (coordB coordD coordR coordC : VarId)
  | storeGlobal (dst src : VarId) (coordB coordD coordR coordC : VarId)
  | loadGlobalAsync (dst src : VarId) (coordB coordD coordR coordC sem : VarId)
  | storeGlobalAsync (dst src : VarId) (coordB coordD coordR coordC : VarId)
  | storeGlobalAdd (dst src : VarId) (coordB coordD coordR coordC : VarId)  -- Atomic add

  -- Vector global memory operations
  | loadVecGlobal (dst src : VarId) (offset : VarId)
  | storeVecGlobal (dst src : VarId) (offset : VarId)
  | storeVecGlobalAdd (dst src : VarId) (offset : VarId)  -- Atomic add for vectors

  -- Distributed / Multimem operations
  | multimemLoadReduce (op : ReduceOp) (dst src : VarId)
  | multimemStore (dst src : VarId)
  | multimemRed (op : ReduceOp) (dst src : VarId)

  -- MMA operations
  | mma (trans : MMATranspose) (dst a b c : VarId)
  | mm (trans : MMATranspose) (dst a b : VarId)
  | mmaFence (dst : VarId)
  | mmaCommitGroup
  | mmaAsyncWait (n : Nat)

  -- Blackwell-specific MMA (tcgen05 / 2-CTA MMA)
  | tcgen05Mma (trans : MMATranspose) (dst a b c : VarId)  -- SM100 2-CTA MMA

  -- Architecture-specific load variants (for explicit control)
  | cpAsyncLoad (dst src : VarId) (coordB coordD coordR coordC sem : VarId)  -- cp.async (SM80)
  | tmaLoadAsync (dst src : VarId) (coordB coordD coordR coordC sem : VarId)  -- TMA (SM90+)

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
  | concatCols (dst left right : VarId)

  -- Synchronization
  | sync (barrierId : Nat)
  | arrive (barrierId : Nat)
  | arriveAndWait (barrierId : Nat)

  -- Named barriers (for warp specialization in FA3)
  | namedBarrierSync (id : Nat) (numThreads : Nat)
  | namedBarrierArrive (id : Nat) (numThreads : Nat)

  -- Warp group operations (for warp specialization)
  | warpGroupIdx (dst : VarId)
  | electOneSync (dst : VarId)

  -- Fence operations (for WGMMA pipelining)
  | fenceViewAsyncShared
  | fenceProxyAsync

  -- Semaphore operations
  | semaphore (op : SemaphoreOp) (sem : VarId)

  -- Control flow
  | forLoop (v : VarId) (lo hi : Nat) (body : Array KStmt)
  | ifStmt (cond : VarId) (thenBody elseBody : Array KStmt)  -- Conditional
  | ifWarpGroup (wgIdx : Nat) (body : Array KStmt)           -- Execute only in specific warp group
  | comment (text : String)

  -- Block/thread index accessors
  | getBlockIdx (dst : VarId) (axis : Nat)   -- axis: 0=x, 1=y, 2=z
  | getThreadIdx (dst : VarId) (axis : Nat)  -- axis: 0=x, 1=y, 2=z

  -- Constants
  | constInt (dst : VarId) (value : Int)     -- Integer constant

  deriving Repr, Inhabited, BEq

/-- Kernel parameter -/
structure KParam where
  name : String
  dtype : GpuFloat
  isPointer : Bool := false
  scalarTy : KScalarType := .UInt64
  deriving Repr, Inhabited, BEq

/-- Complete kernel definition -/
structure Kernel where
  name : String
  arch : GpuArch
  params : Array KParam
  body : Array KStmt
  sharedMemBytes : Nat := 0
  deriving Repr, Inhabited, BEq

end Tyr.GPU.Codegen
