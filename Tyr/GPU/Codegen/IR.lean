import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.AST

/-!
# Tyr.GPU.Codegen.IR

`Tyr.GPU.Codegen.IR` is the central intermediate representation for Tyr GPU kernels.
It models kernels as `KStmt` programs over `VarId`s and operation tags from
`Codegen.AST`.

Key pieces:

- `KScalarType`: scalar parameter kinds used at kernel boundaries.
- `KStmt`: declarative instruction set for allocation, memory, mma, control flow.
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

/-- Concrete ThunderKittens tile descriptor shape required on a global pointer.

Most generated kernels infer these from typed TMA load/store statements. Some
kernel templates need to stage through C++ helper code or raw TK idioms while
still using generated launch wrappers, so they need a declarative way to attach
the same descriptor to the generated `gl<...>` parameter type. -/
inductive GlobalTileDescriptor where
  | st (dtype : GpuFloat) (rows cols : Nat)
  | rowVecSt (dtype : GpuFloat) (rows cols : Nat)
  | colVecSt (dtype : GpuFloat) (rows cols : Nat)
  deriving Repr, Inhabited, BEq, Hashable

/-- GPU kernel statement using VarId -/
inductive KStmt where
  -- Tile declarations
  | declRT (v : VarId) (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout)
  | declST (v : VarId) (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout)
  | declSTArray (v : VarId) (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout) (len : Nat)
  | declSTAlias (v : VarId) (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout)
      (src : VarId) (comment : String)
  | declRV (v : VarId) (dtype : GpuFloat) (len : Nat)
  | declSV (v : VarId) (dtype : GpuFloat) (len : Nat)
  | declSTRowVec (v : VarId) (dtype : GpuFloat) (rows cols : Nat)
  | declSTColVec (v : VarId) (dtype : GpuFloat) (rows cols : Nat)
  | declSTColVecArray (v : VarId) (dtype : GpuFloat) (rows cols len : Nat)
  | declSemaphore (v : VarId)  -- Semaphore/barrier declaration
  | declSemaphoreArray (v : VarId) (len : Nat)

  -- Tensor memory tile declarations (Blackwell SM100)
  | declTT (v : VarId) (dtype : GpuFloat) (rows cols : Nat)  -- TMEM tile (tt<dtype, rows, cols>)

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
  | warpgroupStore (dst src : VarId) -- Warpgroup-wide store into shared memory
  | warpgroupStoreIdx (dst dstIdx src : VarId)
  | tmaStoreCommitGroup              -- Commit TMA store group
  | tmaStoreAsyncWait                -- Wait for outstanding TMA stores
  | prefetch (src : VarId)           -- TMA prefetch
  | tmaExpect (barrier : VarId) (bytes : Nat)
  | blockSync                        -- CTA-wide __syncthreads()
  | groupSync (warps barrierId : Nat) -- ThunderKittens group<N>::sync(barrier)
  | groupSyncVal (warps : Nat) (barrierId : VarId)

  -- TMA operations with global pointers (legacy single coord)
  | tmaLoad (dst src : VarId) (coord : VarId)      -- TMA load: shared ← global[coord]
  | tmaStore (dst src : VarId) (coord : VarId)     -- TMA store: global[coord] ← shared

  -- Global memory operations with 4D coordinates (ThunderKittens style)
  | loadGlobal (dst src : VarId) (coordB coordD coordR coordC : VarId)
  | storeGlobal (dst src : VarId) (coordB coordD coordR coordC : VarId)
  | loadGlobalAsync (dst src : VarId) (coordB coordD coordR coordC sem : VarId)
  | loadGlobalAsyncWarp (dst src : VarId) (coordB coordD coordR coordC sem : VarId)
  | loadGlobalAsyncIdx (dst dstIdx src : VarId) (coordB coordD coordR coordC sem : VarId)
  | loadGlobalAsyncIdxSemIdx (dst dstIdx src : VarId) (coordB coordD coordR coordC sem semIdx : VarId)
  | loadGlobalAsyncWarpIdx (dst dstIdx src : VarId) (coordB coordD coordR coordC sem semIdx : VarId)
  | storeGlobalAsync (dst src : VarId) (coordB coordD coordR coordC : VarId)
  | storeGlobalAsyncIdx (dst src srcIdx : VarId) (coordB coordD coordR coordC : VarId)
  | storeGlobalAdd (dst src : VarId) (coordB coordD coordR coordC : VarId)  -- Atomic add
  | storeGlobalAddWarp (dst src : VarId) (coordB coordD coordR coordC : VarId)  -- Atomic add from current warp
  | requireGlobalTma (ptr : VarId) (descriptor : GlobalTileDescriptor)
  | layoutDim (dst src : VarId) (axis : LayoutDimAxis)

  -- Vector global memory operations
  | loadVecGlobal (dst src : VarId) (offset : VarId)
  | storeVecGlobal (dst src : VarId) (offset : VarId)
  | storeVecGlobalAdd (dst src : VarId) (offset : VarId)  -- Atomic add for vectors
  | loadVecGlobalCoord (dst src : VarId) (coordB coordD coordR coordC : VarId)
  | storeVecGlobalCoord (dst src : VarId) (coordB coordD coordR coordC : VarId)
  | storeVecGlobalAddCoord (dst src : VarId) (coordB coordD coordR coordC : VarId)
  | loadScalarGlobal (dst src offset : VarId)
  | storeScalarGlobal (dst src offset : VarId)

  -- Distributed / Multimem operations
  | multimemLoadReduce (op : ReduceOp) (dst src : VarId)
  | multimemStore (dst src : VarId)
  | multimemRed (op : ReduceOp) (dst src : VarId)

  -- MMA operations
  | mma (trans : MMATranspose) (dst a b c : VarId)
  | mm (trans : MMATranspose) (dst a b : VarId)
  | warpgroupMma (trans : MMATranspose) (dst a b : VarId)
  | warpgroupMm (trans : MMATranspose) (dst a b : VarId)
  | warpgroupMmaIdx (trans : MMATranspose) (dst a aIdx b bIdx : VarId)
  | warpgroupMmIdx (trans : MMATranspose) (dst a aIdx b bIdx : VarId)
  | warpgroupMmaRhsIdx (trans : MMATranspose) (dst a b bIdx : VarId)
  | mmaFence (dst : VarId)
  | mmaCommitGroup
  | mmaAsyncWait (n : Nat)

  -- Blackwell-specific MMA (tcgen05 / 2-CTA MMA)
  | tcgen05Mm (trans : MMATranspose) (dst a b : VarId)     -- SM100 MMA zero-init to TMEM
  | tcgen05Mma (trans : MMATranspose) (dst a b c : VarId)  -- SM100 accumulating MMA to TMEM
  | tcgen05MmaScaled (trans : MMATranspose) (dst a b c scaleA scaleB : VarId)  -- SM100 scaled MMA (MXFP8/NVFP4)
  | tcgen05Commit (sem : VarId) (clusterSize : Nat)        -- Commit tcgen05 results

  -- Tensor memory operations (Blackwell SM100)
  | tmemAllocate (dst pool : VarId) (offset : Nat)         -- Allocate TT from TMEM pool
  | tmemProvision (pool : VarId) (clusterSize : Nat)       -- Provision TMEM across cluster
  | tmemDeprovision (pool : VarId)                         -- Release provisioned TMEM
  | loadScaleTmem (dst src : VarId) (stage : Nat)          -- Load E8M0 scale tile into TMEM
  | tmemSubtile (dst src : VarId) (offset : Nat)           -- Extract subtile from pipelined TMEM

  -- Cluster operations (SM90+ clusters, SM100 2-CTA)
  | clusterIdx (dst : VarId) (axis : Nat)                  -- Cluster CTA rank (axis: 0=x, 1=y, 2=z)
  | clusterTmaLoad (dst src : VarId) (coordB coordD coordR coordC sem : VarId)   -- Cluster-scope TMA load
  | clusterTmaStore (dst src : VarId) (coordB coordD coordR coordC : VarId)      -- Cluster-scope TMA store
  | clusterArrive (sem : VarId)                            -- Cluster barrier arrive
  | clusterWait (sem : VarId)                              -- Cluster barrier wait

  -- Architecture-specific load variants (for explicit control)
  | cpAsyncLoad (dst src : VarId) (coordB coordD coordR coordC sem : VarId)  -- cp.async (SM80)
  | tmaLoadAsync (dst src : VarId) (coordB coordD coordR coordC sem : VarId)  -- TMA (SM90+)

  -- Element-wise unary
  | unary (op : UnaryOp) (dst src : VarId)

  -- Element-wise binary
  | binary (op : BinaryOp) (dst a b : VarId)

  -- Element-wise ternary (FMA)
  | ternary (op : TernaryOp) (dst a b c : VarId)

  -- Element-wise comparison mask
  | eqMask (dst a b : VarId)  -- 1 where a == b, else 0

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
  | warpGroupLaneId (dst : VarId)
  | warpId (dst : VarId)
  | laneId (dst : VarId)
  | electOneSync (dst : VarId)
  | warpgroupDecreaseRegisters (n : Nat)
  | warpgroupIncreaseRegisters (n : Nat)

  -- Fence operations (for WGMMA pipelining)
  | fenceViewAsyncShared
  | fenceProxyAsync

  -- Semaphore operations
  | semaphore (op : SemaphoreOp) (sem : VarId)
  | semaphoreWarp (op : SemaphoreOp) (sem : VarId)
  | semaphoreWaitVal (sem phase : VarId)
  | semaphoreArray (op : SemaphoreOp) (sem idx : VarId)
  | semaphoreArrayWarp (op : SemaphoreOp) (sem idx : VarId)
  | semaphoreArrayWaitVal (sem idx phase : VarId)
  | semaphoreArrayArrive (sem idx : VarId) (count : Nat)

  -- Control flow
  | forLoop (v : VarId) (lo hi : Nat) (body : Array KStmt)
  | forLoopVal (v : VarId) (lo : Nat) (hi : VarId) (body : Array KStmt)
  | forLoopRev (v : VarId) (lo hi : Nat) (body : Array KStmt)
  | forLoopValRev (v : VarId) (lo : Nat) (hi : VarId) (body : Array KStmt)
  | ifStmt (cond : VarId) (thenBody elseBody : Array KStmt)  -- Conditional
  | ifWarpGroup (wgIdx : Nat) (body : Array KStmt)           -- Execute only in specific warp group
  | comment (text : String)

  -- Block/thread index accessors
  | getBlockIdx (dst : VarId) (axis : Nat)   -- axis: 0=x, 1=y, 2=z
  | getThreadIdx (dst : VarId) (axis : Nat)  -- axis: 0=x, 1=y, 2=z

  -- Constants
  | constInt (dst : VarId) (value : Int)     -- Integer constant
  | constFloat (dst : VarId) (value : Float)
  | scalarUnary (op : ScalarUnaryOp) (dst src : VarId)
  | scalarCompare (op : ScalarCompareOp) (dst a b : VarId)
  | scalarBinary (op : ScalarBinaryOp) (dst a b : VarId)
  | scalarSelect (dst cond ifTrue ifFalse : VarId)
  | vecIota (dst : VarId) (start step : Float)
  | vecFillScalar (dst scalar : VarId)
  | raw (code : String)                      -- Exact backend code escape hatch

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
  family : GpuFamily
  params : Array KParam
  body : Array KStmt
  sharedMemBytes : Nat := 0
  launchBounds : Option (Nat × Nat) := none
  deriving Repr, Inhabited, BEq

end Tyr.GPU.Codegen
