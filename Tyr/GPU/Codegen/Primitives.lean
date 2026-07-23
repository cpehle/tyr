import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.AST
import Tyr.GPU.Codegen.GlobalLayout

/-!
# Tyr.GPU.Codegen.Primitives

`Tyr.GPU.Codegen.Primitives` is the core primitive DSL for building kernels.
Functions in this module are strongly typed wrappers around `KStmt` emission.

Highlights:

- allocation helpers (`allocRT`, `allocST`, vectors, semaphores),
- memory operations (shared/global/TMA-style),
- tensor-core math (`mm`, `mma`, async variants),
- reductions, broadcasts, masking, synchronization, and control flow helpers.

Most operations enforce dimension/layout compatibility in types, so invalid
kernels fail during Lean elaboration rather than codegen/runtime.
-/

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-- Emit exact backend code when the typed DSL does not yet expose a primitive.

This is intentionally narrow and should be used to isolate backend-specific
constructs until they are promoted to first-class statements. Most Blackwell
TMEM and cluster operations now have typed wrappers (`allocTT`, `tcgen05Mm`,
`clusterIdx`, etc.). -/
def emitRaw (code : String) : KernelM Unit := do
  emit (.raw code)

/-! ## Tile Allocation -/

/-- Allocate a register tile -/
def allocRT (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row)
    : KernelM (RT dtype rows cols layout) := do
  let v ← freshVar
  emit (.declRT v dtype rows cols layout)
  pure ⟨v⟩

/-- Allocate a shared memory tile -/
def allocST (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row)
    : KernelM (ST dtype rows cols layout) := do
  let v ← freshVar
  emit (.declST v dtype rows cols layout)
  -- Track shared memory usage
  let bytes := rows * cols * dtype.bytes
  modify fun s => { s with sharedMemBytes := s.sharedMemBytes + bytes }
  pure ⟨v⟩

/-- Allocate a ThunderKittens shared tile array in dynamic shared memory. -/
def allocSTArray (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row)
    (len : Nat) : KernelM (STArray dtype rows cols layout len) := do
  let v ← freshVar
  emit (.declSTArray v dtype rows cols layout len)
  modify fun s => { s with sharedMemBytes := s.sharedMemBytes + len * rows * cols * dtype.bytes }
  pure ⟨v⟩

/-- Create a typed shared tile view over an existing shared-memory allocation.

This models the ThunderKittens pattern of reusing a staging buffer for a later
output tile after all users of the original tile have crossed a barrier. The
alias is intentionally not counted as an additional shared-memory allocation. -/
def aliasST {srcDtype : GpuFloat} {srcRows srcCols : Nat} {srcLayout : TileLayout}
    (src : ST srcDtype srcRows srcCols srcLayout)
    (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row)
    (comment : String := "shared memory alias")
    : KernelM (ST dtype rows cols layout) := do
  let v ← freshVar
  emit (.declSTAlias v dtype rows cols layout src.id comment)
  pure ⟨v⟩

/-! ## Global TMA Descriptor Requirements

These helpers attach concrete ThunderKittens tile descriptors to generated
`gl<...>` parameter types without forcing a dummy load/store into the kernel
body. They are useful for kernels that lower a small region through raw TK C++
or through helper templates while still relying on generated launch wrappers. -/

/-- Attach a TK tile descriptor to a global pointer parameter without forcing a
    load/store in the body. The launch wrapper will use this descriptor when
    constructing the `gl<...>` argument. -/
def requireGlobalDescriptor {dtype : GpuFloat}
    (ptr : GPtr dtype) (descriptor : GlobalTileDescriptor) : KernelM Unit := do
  emit (KStmt.requireGlobalTma ptr.id descriptor)

/-- Require an `st<dtype, rows, cols>` tile descriptor on the given global pointer. -/
def requireGlobalST {dtype : GpuFloat}
    (ptr : GPtr dtype) (rows cols : Nat) : KernelM Unit :=
  requireGlobalDescriptor ptr (GlobalTileDescriptor.st dtype rows cols)

/-- Require a `row_vec<st<dtype, rows, cols>>` tile descriptor on the global pointer. -/
def requireGlobalRowVecST {dtype : GpuFloat}
    (ptr : GPtr dtype) (rows cols : Nat) : KernelM Unit :=
  requireGlobalDescriptor ptr (GlobalTileDescriptor.rowVecSt dtype rows cols)

/-- Require a `col_vec<st<dtype, rows, cols>>` tile descriptor on the global pointer. -/
def requireGlobalColVecST {dtype : GpuFloat}
    (ptr : GPtr dtype) (rows cols : Nat) : KernelM Unit :=
  requireGlobalDescriptor ptr (GlobalTileDescriptor.colVecSt dtype rows cols)

/-- Allocate a register vector -/
def allocRV (dtype : GpuFloat) (len : Nat) : KernelM (RV dtype len) := do
  let v ← freshVar
  emit (.declRV v dtype len)
  pure ⟨v⟩

/-- Allocate a shared vector -/
def allocSV (dtype : GpuFloat) (len : Nat) : KernelM (SV dtype len) := do
  let v ← freshVar
  emit (.declSV v dtype len)
  pure ⟨v⟩

/-- Bytes reserved by ThunderKittens for an `sv`, rounded to a 128-byte TMA
    allocation unit on Hopper/Blackwell. -/
private def sharedVecBytes (elements elemBytes : Nat) : Nat :=
  ((elements * elemBytes + 127) / 128) * 128

/-- Allocate a ThunderKittens shared `row_vec<st<T, rows, cols>>`.
    The runtime storage is an `sv<T, cols>`, while the tile descriptor is kept
    in the type so TMA vector IO can infer the matching global descriptor. -/
def allocSTRowVec (dtype : GpuFloat) (rows cols : Nat)
    : KernelM (STRowVec dtype rows cols) := do
  let v ← freshVar
  emit (.declSTRowVec v dtype rows cols)
  modify fun s => { s with sharedMemBytes := s.sharedMemBytes + sharedVecBytes cols dtype.bytes }
  pure ⟨v⟩

/-- Allocate a ThunderKittens shared `col_vec<st<T, rows, cols>>`.
    The runtime storage is an `sv<T, rows>`, while the tile descriptor is kept
    in the type so TMA vector IO can infer the matching global descriptor. -/
def allocSTColVec (dtype : GpuFloat) (rows cols : Nat)
    : KernelM (STColVec dtype rows cols) := do
  let v ← freshVar
  emit (.declSTColVec v dtype rows cols)
  modify fun s => { s with sharedMemBytes := s.sharedMemBytes + sharedVecBytes rows dtype.bytes }
  pure ⟨v⟩

/-- Allocate an array of ThunderKittens shared `col_vec<st<T, rows, cols>>`. -/
def allocSTColVecArray (dtype : GpuFloat) (rows cols len : Nat)
    : KernelM (STColVecArray dtype rows cols len) := do
  let v ← freshVar
  emit (.declSTColVecArray v dtype rows cols len)
  modify fun s => { s with sharedMemBytes := s.sharedMemBytes + len * sharedVecBytes rows dtype.bytes }
  pure ⟨v⟩

/-- Allocate a tensor memory tile (Blackwell TMEM).
    These live in the SM100 tensor memory unit (144 KB per CTA). -/
def allocTT (dtype : GpuFloat) (rows cols : Nat) : KernelM (TT dtype rows cols) := do
  let v ← freshVar
  emit (.declTT v dtype rows cols)
  pure ⟨v⟩

/-- Declare a concrete ThunderKittens tensor-memory allocator. -/
def allocTMEMPool (numCtas : Nat := 1) (clusterSize : Nat := 1)
    : KernelM TMEMPool := do
  let v ← freshVar
  emit (.declTMEMPool v numCtas clusterSize)
  pure ⟨v⟩

/-- Allocate a zero-initialized tensor memory tile -/
def zeroTT (dtype : GpuFloat) (rows cols : Nat) : KernelM (TT dtype rows cols) := do
  let tile ← allocTT dtype rows cols
  emit (.unary .Zero tile.id tile.id)
  pure tile

/-- Allocate a TT from a TMEM pool at a given byte offset. -/
def tmemAllocate (pool : TMEMPool) (dtype : GpuFloat) (rows cols : Nat) (offset : Nat)
    (superlane : Nat := 0) : KernelM (TT dtype rows cols) := do
  let v ← freshVar
  emit (.tmemAllocate v pool.id dtype rows cols superlane offset)
  pure ⟨v⟩

/-- Provision a TMEM pool across the cluster. Must be called inside `electOneSync`. -/
def tmemProvision (pool : TMEMPool) (clusterSize : Nat := 1) : KernelM Unit := do
  emit (.tmemProvision pool.id clusterSize)

/-- Release a provisioned TMEM pool. -/
def tmemDeprovision (pool : TMEMPool) : KernelM Unit := do
  emit (.tmemDeprovision pool.id)

/-- Extract a subtile from a pipelined TMEM allocation at a given offset. -/
def tmemSubtile (src : TT dtype rows cols) (offset : Nat)
    : KernelM (TT dtype rows cols) := do
  let v ← freshVar
  emit (.tmemSubtile v src.id offset)
  pure ⟨v⟩

/-- Semaphore type (barrier) for async operations -/
structure Semaphore where
  /-- Underlying SSA variable id of the shared semaphore allocation. -/
  id : VarId
  deriving Repr

/-! ## Zero/Initialized Allocation -/

/-- Allocate a zero-initialized register tile -/
def zeroRT (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row)
    : KernelM (RT dtype rows cols layout) := do
  let tile ← allocRT dtype rows cols layout
  emit (.unary .Zero tile.id tile.id)
  pure tile

/-- Allocate with negative infinity (for softmax max tracking) -/
def negInftyRV (dtype : GpuFloat) (len : Nat) : KernelM (RV dtype len) := do
  let vec ← allocRV dtype len
  emit (.unary .NegInfty vec.id vec.id)
  pure vec

/-- Allocate a zero-initialized register vector -/
def zeroRV (dtype : GpuFloat) (len : Nat) : KernelM (RV dtype len) := do
  let vec ← allocRV dtype len
  emit (.unary .Zero vec.id vec.id)
  pure vec

/-- Allocate with ones -/
def onesRT (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row)
    : KernelM (RT dtype rows cols layout) := do
  let tile ← allocRT dtype rows cols layout
  emit (.unary .One tile.id tile.id)
  pure tile

/-! ## Memory Operations -/

/-- Load from shared to register (dimensions must match) -/
def load {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : RT dtype rows cols layout)
    (src : ST dtype rows cols layout) : KernelM Unit := do
  emit (.load dst.id src.id)

/-- Load one compile-time fragment of a larger shared tile into registers.
    `rowIdx` and `colIdx` are fragment indices, matching TK's `st.subtile` API. -/
def loadSubtile {dtype : GpuFloat} {rows cols srcRows srcCols : Nat}
    {layout : TileLayout}
    (dst : RT dtype rows cols layout)
    (src : ST dtype srcRows srcCols layout)
    (rowIdx colIdx : Nat) : KernelM Unit := do
  emit (.loadSubtile dst.id src.id rows cols rowIdx colIdx)

/-- Runtime-indexed counterpart of `loadSubtile`; dimensions remain static. -/
def loadSubtileVal {dtype : GpuFloat} {rows cols srcRows srcCols : Nat}
    {layout : TileLayout}
    (dst : RT dtype rows cols layout)
    (src : ST dtype srcRows srcCols layout)
    (rowIdx colIdx : KVal UInt32) : KernelM Unit := do
  emit (.loadSubtileVal dst.id src.id rows cols rowIdx.id colIdx.id)

/-- Load one row fragment per warp from a shared tile using four warps. -/
def warpgroupLoad {dtype : GpuFloat} {dstRows srcRows cols : Nat} {layout : TileLayout}
    (dst : RT dtype dstRows cols layout)
    (src : ST dtype srcRows cols layout)
    (hRows : srcRows = 4 * dstRows := by decide) : KernelM Unit := do
  let _ := hRows
  emit (.warpgroupLoad dst.id src.id)

/-- Store from register to shared -/
def store {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.store dst.id src.id)

/-- Load vector from shared to register -/
def loadVec {dtype : GpuFloat} {len : Nat}
    (dst : RV dtype len)
    (src : SV dtype len) : KernelM Unit := do
  emit (.load dst.id src.id)

/-- Store vector from register to shared -/
def storeVec {dtype : GpuFloat} {len : Nat}
    (dst : SV dtype len)
    (src : RV dtype len) : KernelM Unit := do
  emit (.store dst.id src.id)

/-- Load a `row_vec<st<T, rows, cols>>` from shared memory into a register vector. -/
def loadRowVec {dtype : GpuFloat} {rows cols : Nat}
    (dst : RV dtype cols)
    (src : STRowVec dtype rows cols) : KernelM Unit := do
  emit (.load dst.id src.id)

/-- Store a register vector into a shared `row_vec<st<T, rows, cols>>`. -/
def storeRowVec {dtype : GpuFloat} {rows cols : Nat}
    (dst : STRowVec dtype rows cols)
    (src : RV dtype cols) : KernelM Unit := do
  emit (.store dst.id src.id)

/-- Load a `col_vec<st<T, rows, cols>>` from shared memory into a register vector. -/
def loadColVec {dtype : GpuFloat} {rows cols : Nat}
    (dst : RV dtype rows)
    (src : STColVec dtype rows cols) : KernelM Unit := do
  emit (.load dst.id src.id)

/-- Store a register vector into a shared `col_vec<st<T, rows, cols>>`. -/
def storeColVec {dtype : GpuFloat} {rows cols : Nat}
    (dst : STColVec dtype rows cols)
    (src : RV dtype rows) : KernelM Unit := do
  emit (.store dst.id src.id)

/-! ## Scalar / Vector Runtime Helpers -/

/-- Materialize an integer constant as a runtime scalar. -/
def constIntVal (value : Int) (name : String := "const") : KernelM (KVal UInt32) := do
  let v ← freshVar
  emit (.constInt v value)
  pure ⟨v, name⟩

/-- Materialize an integer constant as a UInt64 runtime scalar (emitted as a
    proper `uint64_t` C literal, so full 64-bit constants are exact). -/
def constUInt64Val (value : Int) (name : String := "const_u64") : KernelM (KVal UInt64) := do
  let v ← freshVar
  emit (.constUInt64 v value.toNat)
  pure ⟨v, name⟩

/-- Materialize a Float32 constant as a runtime scalar. -/
def constFloatVal (value : Float) (name : String := "const_f") : KernelM (KVal Float32) := do
  let v ← freshVar
  emit (.constFloat v value)
  pure ⟨v, name⟩

/-- Allocate a mutable runtime scalar, optionally initialized from another scalar.

This is the scalar analogue of TileLang's `T.alloc_var`: it gives kernels a
named local variable for loop-carried state without leaving the typed DSL. -/
def allocScalar (ty : KScalarType) (name : String := "scalar")
    (init : Option (KVal T) := none) : KernelM (KVal T) := do
  let v ← freshVar
  emit (.declScalar v ty (init.map (·.id)))
  pure ⟨v, name⟩

/-- Allocate a mutable Float32 scalar. -/
def allocFloat32Scalar (init : Option (KVal Float32) := none)
    (name : String := "f32") : KernelM (KVal Float32) :=
  allocScalar .Float32 name init

/-- Assign a runtime scalar variable. -/
def assignScalar {T : Type} (dst src : KVal T) : KernelM Unit := do
  emit (.scalarAssign dst.id src.id)

/-- Cast a runtime scalar to a different scalar type. -/
def castScalar {A B : Type} (dstTy : KScalarType)
    (src : KVal A) (name : String := "cast") : KernelM (KVal B) := do
  let v ← freshVar
  emit (.scalarCast v src.id dstTy)
  pure ⟨v, name⟩

/-- Cast a runtime scalar to UInt64. -/
def castUInt64 {A : Type} (src : KVal A) (name : String := "u64") : KernelM (KVal UInt64) :=
  castScalar .UInt64 src name

/-- Cast a runtime scalar to Float32. -/
def castFloat32 {A : Type} (src : KVal A) (name : String := "f32") : KernelM (KVal Float32) :=
  castScalar .Float32 src name

/-- Load one Float32 scalar from global memory at a runtime offset. -/
def loadFloat32Scalar
    (src : GPtr GpuFloat.Float32)
    (offset : KVal UInt32)
    (name : String := "load_f32")
    : KernelM (KVal Float32) := do
  let v ← freshVar
  emit (.loadScalarGlobal v src.id offset.id)
  pure ⟨v, name⟩

/-- Store one Float32 scalar to global memory at a runtime offset. -/
def storeFloat32Scalar
    (dst : GPtr GpuFloat.Float32)
    (offset : KVal UInt32)
    (src : KVal Float32)
    : KernelM Unit := do
  emit (.storeScalarGlobal dst.id src.id offset.id)

/-- Load one BF16 global element by flat element offset and convert it to Float32. -/
def loadBFloat16FlatAsFloat32
    (src : GPtr GpuFloat.BFloat16)
    (offset : KVal UInt64)
    (name : String := "load_bf16_f32")
    : KernelM (KVal Float32) := do
  let v ← freshVar
  emit (.loadFlatGlobal v src.id offset.id .BFloat16 .Float32)
  pure ⟨v, name⟩

/-- Store one Float32 scalar to a BF16 global element by flat element offset. -/
def storeBFloat16FlatFromFloat32
    (dst : GPtr GpuFloat.BFloat16)
    (offset : KVal UInt64)
    (src : KVal Float32)
    : KernelM Unit := do
  emit (.storeFlatGlobal dst.id offset.id src.id .BFloat16 .Float32)

/-- Allocate a dynamic linear Float32 shared-memory view at an element offset.

The launch wrapper supplies the total dynamic shared-memory byte count. This
mirrors the ergonomic role of TileLang `T.alloc_shared` for runtime extents,
while keeping the offset arithmetic explicit and checkable in Tyr. -/
def sharedFloat32At
    (offsetElems : KVal UInt64)
    (name : String := "shared_f32")
    : KernelM (KShared Float32) := do
  let v ← freshVar
  emit (.declSharedLinear v offsetElems.id .Float32)
  pure ⟨v, name⟩

/-- Load one Float32 from a dynamic linear shared-memory view. -/
def loadSharedFloat32
    (src : KShared Float32)
    (offset : KVal UInt64)
    (name : String := "load_shared_f32")
    : KernelM (KVal Float32) := do
  let v ← freshVar
  emit (.loadSharedLinear v src.id offset.id .Float32)
  pure ⟨v, name⟩

/-- Store one Float32 into a dynamic linear shared-memory view. -/
def storeSharedFloat32
    (dst : KShared Float32)
    (offset : KVal UInt64)
    (src : KVal Float32)
    : KernelM Unit := do
  emit (.storeSharedLinear dst.id offset.id src.id .Float32)

/-- Apply a scalar unary operation to a runtime scalar. -/
def scalarUnary {T : Type} (op : ScalarUnaryOp)
    (src : KVal T) (name : String := "scalar_unary") : KernelM (KVal T) := do
  let v ← freshVar
  emit (.scalarUnary op v src.id)
  pure ⟨v, name⟩

/-- Compare two runtime scalars and return a boolean scalar. -/
def scalarCompare {T : Type} (op : ScalarCompareOp)
    (a b : KVal T) (name : String := "scalar_cmp") : KernelM (KVal Bool) := do
  let v ← freshVar
  emit (.scalarCompare op v a.id b.id)
  pure ⟨v, name⟩

/-- Apply a scalar binary operation to two runtime scalars. -/
def scalarBinary {T : Type} (op : ScalarBinaryOp)
    (a b : KVal T) (name : String := "scalar_binary") : KernelM (KVal T) := do
  let v ← freshVar
  emit (.scalarBinary op v a.id b.id)
  pure ⟨v, name⟩

/-- Select between two runtime scalar values. -/
def scalarSelect {T : Type} (cond : KVal Bool)
    (ifTrue ifFalse : KVal T) (name : String := "scalar_select") : KernelM (KVal T) := do
  let v ← freshVar
  emit (.scalarSelect v cond.id ifTrue.id ifFalse.id)
  pure ⟨v, name⟩

/-- Negate a runtime scalar. -/
def scalarNeg (src : KVal Float32) (name : String := "neg") : KernelM (KVal Float32) :=
  scalarUnary .Neg src name

/-- Exponentiate a runtime scalar. -/
def scalarExp (src : KVal Float32) (name : String := "exp") : KernelM (KVal Float32) :=
  scalarUnary .Exp src name

/-- Base-2 exponentiate a runtime Float32 scalar. -/
def scalarExp2 (src : KVal Float32) (name : String := "exp2") : KernelM (KVal Float32) :=
  scalarUnary .Exp2 src name

/-- Base-2 logarithm of a runtime Float32 scalar. -/
def scalarLog2 (src : KVal Float32) (name : String := "log2") : KernelM (KVal Float32) :=
  scalarUnary .Log2 src name

/-- Reciprocal square root of a runtime Float32 scalar. -/
def scalarRsqrt (src : KVal Float32) (name : String := "rsqrt") : KernelM (KVal Float32) :=
  scalarUnary .Rsqrt src name

/-- Add two runtime scalars. -/
def scalarAddVal {T : Type} (a b : KVal T) (name : String := "add") : KernelM (KVal T) :=
  scalarBinary .Add a b name

/-- Test two runtime scalars for equality. -/
def scalarEq {T : Type} (a b : KVal T) (name : String := "eq") : KernelM (KVal Bool) :=
  scalarCompare .Eq a b name

/-- Test whether one runtime scalar is less than another. -/
def scalarLt {T : Type} (a b : KVal T) (name : String := "lt") : KernelM (KVal Bool) :=
  scalarCompare .Lt a b name

/-- Test whether one runtime scalar is less than or equal to another. -/
def scalarLe {T : Type} (a b : KVal T) (name : String := "le") : KernelM (KVal Bool) :=
  scalarCompare .Le a b name

/-- Test whether one runtime scalar is greater than another. -/
def scalarGt {T : Type} (a b : KVal T) (name : String := "gt") : KernelM (KVal Bool) :=
  scalarCompare .Gt a b name

/-- Test whether one runtime scalar is greater than or equal to another. -/
def scalarGe {T : Type} (a b : KVal T) (name : String := "ge") : KernelM (KVal Bool) :=
  scalarCompare .Ge a b name

/-- Subtract two runtime scalars. -/
def scalarSub {T : Type} (a b : KVal T) (name : String := "sub") : KernelM (KVal T) :=
  scalarBinary .Sub a b name

/-- Multiply two runtime scalars. -/
def scalarMulVal {T : Type} (a b : KVal T) (name : String := "mul") : KernelM (KVal T) :=
  scalarBinary .Mul a b name

/-- Divide two runtime scalars. -/
def scalarDivVal {T : Type} (a b : KVal T) (name : String := "div") : KernelM (KVal T) :=
  scalarBinary .Div a b name

/-- Compute the remainder of two runtime scalars. -/
def scalarMod {T : Type} (a b : KVal T) (name : String := "mod") : KernelM (KVal T) :=
  scalarBinary .Mod a b name

/-- Runtime scalar minimum. -/
def scalarMin {T : Type} (a b : KVal T) (name : String := "min") : KernelM (KVal T) :=
  scalarBinary .Min a b name

/-- Runtime scalar maximum. -/
def scalarMax {T : Type} (a b : KVal T) (name : String := "max") : KernelM (KVal T) :=
  scalarBinary .Max a b name

/-- CTA-wide Float32 sum reduction. All threads receive the same result. -/
def blockReduceSumFloat32
    (src : KVal Float32)
    (name : String := "block_sum")
    : KernelM (KVal Float32) := do
  let v ← freshVar
  emit (.blockReduce .Sum v src.id .Float32)
  pure ⟨v, name⟩

/-- Fill a vector with an arithmetic progression. -/
def iotaVec {dtype : GpuFloat} {len : Nat}
    (dst : RV dtype len) (start : Float := 0.0) (step : Float := 1.0) : KernelM Unit := do
  emit (.vecIota dst.id start step)

/-- Fill a vector with one runtime scalar value. -/
def fillVecScalar {dtype : GpuFloat} {len : Nat} {T : Type}
    (dst : RV dtype len) (scalar : KVal T) : KernelM Unit := do
  emit (.vecFillScalar dst.id scalar.id)

/-! ## Vector Unary/Binary Operations -/

/-- Copy vector -/
def copyVec {dtype : GpuFloat} {len : Nat}
    (dst src : RV dtype len) : KernelM Unit := do
  emit (.unary .Copy dst.id src.id)

/-- Copy a register vector while allowing its physical TK layout to change.
    Unlike `copyVec`, source and destination layout inference is deliberately
    independent; `warp::copy` performs the required warp shuffle. -/
def convertVecLayout {dtype : GpuFloat} {len : Nat}
    (dst src : RV dtype len) : KernelM Unit := do
  emit (.convertVecLayout dst.id src.id)

/-- Zero vector -/
def zeroVec {dtype : GpuFloat} {len : Nat}
    (v : RV dtype len) : KernelM Unit := do
  emit (.unary .Zero v.id v.id)

/-- Vector element-wise add -/
def addVec {dtype : GpuFloat} {len : Nat}
    (dst a b : RV dtype len) : KernelM Unit := do
  emit (.binary .Add dst.id a.id b.id)

/-- Vector element-wise subtract -/
def subVec {dtype : GpuFloat} {len : Nat}
    (dst a b : RV dtype len) : KernelM Unit := do
  emit (.binary .Sub dst.id a.id b.id)

/-- Vector element-wise multiply -/
def mulVec {dtype : GpuFloat} {len : Nat}
    (dst a b : RV dtype len) : KernelM Unit := do
  emit (.binary .Mul dst.id a.id b.id)

/-- Vector element-wise divide -/
def divVec {dtype : GpuFloat} {len : Nat}
    (dst a b : RV dtype len) : KernelM Unit := do
  emit (.binary .Div dst.id a.id b.id)

/-- Vector scalar multiply. -/
def scalarMulVec {dtype : GpuFloat} {len : Nat}
    (dst src : RV dtype len) (scalar : Float) : KernelM Unit := do
  emit (.scalarMul dst.id src.id scalar)

/-- Vector scalar add. -/
def scalarAddVec {dtype : GpuFloat} {len : Nat}
    (dst src : RV dtype len) (scalar : Float) : KernelM Unit := do
  emit (.scalarAdd dst.id src.id scalar)

/-- Vector exp -/
def expVec {dtype : GpuFloat} {len : Nat}
    (dst src : RV dtype len) : KernelM Unit := do
  emit (.unary .Exp dst.id src.id)

/-- Vector base-2 exponent. -/
def exp2Vec {dtype : GpuFloat} {len : Nat}
    (dst src : RV dtype len) : KernelM Unit := do
  emit (.unary .Exp2 dst.id src.id)

/-- Vector log -/
def logVec {dtype : GpuFloat} {len : Nat}
    (dst src : RV dtype len) : KernelM Unit := do
  emit (.unary .Log dst.id src.id)

/-- Vector reciprocal square root. -/
def rsqrtVec {dtype : GpuFloat} {len : Nat}
    (dst src : RV dtype len) : KernelM Unit := do
  emit (.unary .Rsqrt dst.id src.id)

/-- Async load (TMA) -/
def loadAsync {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : ST dtype rows cols layout) : KernelM Unit := do
  emit (.loadAsync dst.id src.id)

/-- Async store (TMA) -/
def storeAsync {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.storeAsync dst.id src.id)

/-- Atomic store-add for gradient accumulation -/
def storeAdd {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.storeAdd dst.id src.id)

/-- Async atomic store-add (TMA) for gradient accumulation -/
def storeAddAsync {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.storeAddAsync dst.id src.id)

/-! ## Matrix Multiply

Tensor-core MMA requires each tile dimension to be a multiple of 16 (WGMMA on Hopper).
The `hM`/`hK`/`hN` auto-params discharge at elaboration for concrete literal dims
via `by decide`; polymorphic callers must supply explicit proofs or propagate
the hypotheses. This surfaces shape errors at kernel-build time rather than
silently emitting broken CUDA.
-/

/-- Matrix multiply-accumulate: D = A @ B + C
    Type system enforces dimensions: A is M×K, B is K×N, C and D are M×N
    Input dtype (bf16/f16) can differ from accumulator dtype (f32) -/
def mma {M K N : Nat} {inDtype accDtype : GpuFloat}
    (dst : RT accDtype M N .Row)
    (a : RT inDtype M K .Row)
    (b : RT inDtype K N .Col)
    (c : RT accDtype M N .Row)
    (hM : M % 16 = 0 := by decide)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hM; let _ := hK; let _ := hN
  emit (.mma .AB dst.id a.id b.id c.id)

/-- Matrix multiply without accumulate: D = A @ B -/
def mm {M K N : Nat} {inDtype accDtype : GpuFloat}
    (dst : RT accDtype M N .Row)
    (a : RT inDtype M K .Row)
    (b : RT inDtype K N .Col)
    (hM : M % 16 = 0 := by decide)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hM; let _ := hK; let _ := hN
  emit (.mm .AB dst.id a.id b.id)

/-- Matrix multiply with B transposed: D = A @ B^T + C
    A is M×K, B is N×K (stored row-major, used transposed) -/
def mmaT {M K N : Nat} {inDtype accDtype : GpuFloat}
    (dst : RT accDtype M N .Row)
    (a : RT inDtype M K .Row)
    (b : RT inDtype N K .Row)
    (c : RT accDtype M N .Row)
    (hM : M % 16 = 0 := by decide)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hM; let _ := hK; let _ := hN
  emit (.mma .ABt dst.id a.id b.id c.id)

/-- Matrix multiply with both transposed: D = A^T @ B^T + C -/
def mmaAtBt {M K N : Nat} {inDtype accDtype : GpuFloat}
    (dst : RT accDtype M N .Row)
    (a : RT inDtype K M .Row)
    (b : RT inDtype N K .Row)
    (c : RT accDtype M N .Row)
    (hM : M % 16 = 0 := by decide)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hM; let _ := hK; let _ := hN
  emit (.mma .AtBt dst.id a.id b.id c.id)

/-- Matrix multiply with A transposed: D = A^T @ B + C -/
def mmaAtB {M K N : Nat} {inDtype accDtype : GpuFloat}
    (dst : RT accDtype M N .Row)
    (a : RT inDtype K M .Col)
    (b : RT inDtype K N .Col)
    (c : RT accDtype M N .Row)
    (hM : M % 16 = 0 := by decide)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hM; let _ := hK; let _ := hN
  emit (.mma .AtB dst.id a.id b.id c.id)

/-- Hopper WGMMA accumulate: `dst += a @ b`, with `b` sourced from shared memory.
    This mirrors ThunderKittens' `warpgroup::mma_AB`. The operation is async;
    call `mmaAsyncWait` before reading `dst`. -/
def warpgroupMma {M K N : Nat} {inDtype accDtype : GpuFloat} {layoutB : TileLayout}
    (dst : RT accDtype M N .Row)
    (a : RT inDtype M K .Row)
    (b : ST inDtype K N layoutB)
    (hM : M % 16 = 0 := by decide)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hM; let _ := hK; let _ := hN
  emit (.warpgroupMma .AB dst.id a.id b.id)

/-- Hopper WGMMA overwrite: `dst = a @ b`, with `b` sourced from shared memory.
    This mirrors ThunderKittens' `warpgroup::mm_AB`. -/
def warpgroupMm {M K N : Nat} {inDtype accDtype : GpuFloat} {layoutB : TileLayout}
    (dst : RT accDtype M N .Row)
    (a : RT inDtype M K .Row)
    (b : ST inDtype K N layoutB)
    (hM : M % 16 = 0 := by decide)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hM; let _ := hK; let _ := hN
  emit (.warpgroupMm .AB dst.id a.id b.id)

/-- Hopper WGMMA accumulate: `dst += a @ b^T`, with `b` sourced from shared memory.
    This mirrors ThunderKittens' `warpgroup::mma_ABt`. -/
def warpgroupMmaT {M K N : Nat} {inDtype accDtype : GpuFloat} {layoutB : TileLayout}
    (dst : RT accDtype M N .Row)
    (a : RT inDtype M K .Row)
    (b : ST inDtype N K layoutB)
    (hM : M % 16 = 0 := by decide)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hM; let _ := hK; let _ := hN
  emit (.warpgroupMma .ABt dst.id a.id b.id)

/-- Hopper WGMMA overwrite: `dst = a @ b^T`, with `b` sourced from shared memory.
    This mirrors ThunderKittens' `warpgroup::mm_ABt`. -/
def warpgroupMmT {M K N : Nat} {inDtype accDtype : GpuFloat} {layoutB : TileLayout}
    (dst : RT accDtype M N .Row)
    (a : RT inDtype M K .Row)
    (b : ST inDtype N K layoutB)
    (hM : M % 16 = 0 := by decide)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hM; let _ := hK; let _ := hN
  emit (.warpgroupMm .ABt dst.id a.id b.id)

/-- Hopper WGMMA accumulate: `dst += a @ b`, with both inputs in shared memory.
    TK's shared/shared overload consumes a 64-row shared tile but writes one
    16-row register subtile per participating warp. -/
def warpgroupMmaShared64x16 {K N : Nat} {inDtype accDtype : GpuFloat}
    {layoutA layoutB : TileLayout}
    (dst : RT accDtype 16 N .Row)
    (a : ST inDtype 64 K layoutA)
    (b : ST inDtype K N layoutB)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hK; let _ := hN
  emit (.warpgroupMma .AB dst.id a.id b.id)

/-- Hopper WGMMA overwrite: `dst = a @ b`, with both inputs in shared memory. -/
def warpgroupMmShared64x16 {K N : Nat} {inDtype accDtype : GpuFloat}
    {layoutA layoutB : TileLayout}
    (dst : RT accDtype 16 N .Row)
    (a : ST inDtype 64 K layoutA)
    (b : ST inDtype K N layoutB)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hK; let _ := hN
  emit (.warpgroupMm .AB dst.id a.id b.id)

/-- Hopper WGMMA accumulate: `dst += a @ b^T`, with both inputs in shared memory.
    This matches TK forward scores: `warpgroup::mma_ABt(scores, q_smem, k_smem)`. -/
def warpgroupMmaSharedT64x16 {K N : Nat} {inDtype accDtype : GpuFloat}
    {layoutA layoutB : TileLayout}
    (dst : RT accDtype 16 N .Row)
    (a : ST inDtype 64 K layoutA)
    (b : ST inDtype N K layoutB)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hK; let _ := hN
  emit (.warpgroupMma .ABt dst.id a.id b.id)

/-- Hopper WGMMA overwrite: `dst = a @ b^T`, with both inputs in shared memory.
    This matches TK forward scores: `warpgroup::mm_ABt(scores, q_smem, k_smem)`. -/
def warpgroupMmSharedT64x16 {K N : Nat} {inDtype accDtype : GpuFloat}
    {layoutA layoutB : TileLayout}
    (dst : RT accDtype 16 N .Row)
    (a : ST inDtype 64 K layoutA)
    (b : ST inDtype N K layoutB)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hK; let _ := hN
  emit (.warpgroupMm .ABt dst.id a.id b.id)

/-- Hopper WGMMA overwrite: `dst = a[idxA] @ b[idxB]^T`, both operands sourced
    from shared tile arrays. This models TK forward's `q_smem[warpgroupid]`
    and `k_smem[kv_idx % stages]` without cloning four stage branches. -/
def warpgroupMmSharedArrayT64x16 {K N lenA lenB : Nat} {inDtype accDtype : GpuFloat}
    {layoutA layoutB : TileLayout}
    (dst : RT accDtype 16 N .Row)
    (a : STArray inDtype 64 K layoutA lenA)
    (aIdx : KVal UInt32)
    (b : STArray inDtype N K layoutB lenB)
    (bIdx : KVal UInt32)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hK; let _ := hN
  emit (.warpgroupMmIdx .ABt dst.id a.id aIdx.id b.id bIdx.id)

/-- Hopper WGMMA accumulate: `dst += a @ b[idxB]`, with B sourced from a
    shared tile array. This models TK forward's `v_smem[kv_idx % stages]`. -/
def warpgroupMmaRhsArray {M K N lenB : Nat} {inDtype accDtype : GpuFloat}
    {layoutB : TileLayout}
    (dst : RT accDtype M N .Row)
    (a : RT inDtype M K .Row)
    (b : STArray inDtype K N layoutB lenB)
    (bIdx : KVal UInt32)
    (hM : M % 16 = 0 := by decide)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hM; let _ := hK; let _ := hN
  emit (.warpgroupMmaRhsIdx .AB dst.id a.id b.id bIdx.id)

/-- Hopper WGMMA accumulate: `dst += a^T @ b`, with both inputs in shared memory.
    This matches TK backward dQ accumulation: `warpgroup::mma_AtB(qg_reg, ds_smem, k_smem)`. -/
def warpgroupMmaSharedAtB64x16 {K N : Nat} {inDtype accDtype : GpuFloat}
    {layoutA layoutB : TileLayout}
    (dst : RT accDtype 16 N .Row)
    (a : ST inDtype K 64 layoutA)
    (b : ST inDtype K N layoutB)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hK; let _ := hN
  emit (.warpgroupMma .AtB dst.id a.id b.id)

/-- Hopper WGMMA overwrite: `dst = a^T @ b`, with both inputs in shared memory.
    This matches TK backward dQ accumulation: `warpgroup::mm_AtB(qg_reg, ds_smem, k_smem)`. -/
def warpgroupMmSharedAtB64x16 {K N : Nat} {inDtype accDtype : GpuFloat}
    {layoutA layoutB : TileLayout}
    (dst : RT accDtype 16 N .Row)
    (a : ST inDtype K 64 layoutA)
    (b : ST inDtype K N layoutB)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hK; let _ := hN
  emit (.warpgroupMm .AtB dst.id a.id b.id)

/-! ## Ternary Operations (FMA) -/

/-- Fused multiply-add: dst = a * b + c -/
def fma {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst a b c : RT dtype rows cols layout) : KernelM Unit := do
  emit (.ternary .FMA dst.id a.id b.id c.id)

/-- FMA pattern for attention: dst = A × B + C (matrix-style) -/
def fmaAxBtC {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst a b c : RT dtype rows cols layout) : KernelM Unit := do
  emit (.ternary .FMAAxBtC dst.id a.id b.id c.id)

/-- FMA pattern: dst = A × C + B -/
def fmaAxCtB {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst a b c : RT dtype rows cols layout) : KernelM Unit := do
  emit (.ternary .FMAAxCtB dst.id a.id b.id c.id)

/-! ## Element-wise Unary Operations -/

/-- Apply unary operation in-place or to different tile -/
def unaryOp {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (op : UnaryOp)
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary op dst.id src.id)

/-- Element-wise exp -/
def exp {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Exp dst.id src.id)

/-- Element-wise exp2 -/
def exp2 {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Exp2 dst.id src.id)

/-- Element-wise log -/
def log {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Log dst.id src.id)

/-- Element-wise sqrt -/
def sqrt {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Sqrt dst.id src.id)

/-- Element-wise rsqrt (1/sqrt) -/
def rsqrt {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Rsqrt dst.id src.id)

/-- Element-wise tanh -/
def tanh {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Tanh dst.id src.id)

/-- Element-wise fast tanh (hardware-accelerated __nv_fast_tanh) -/
def fastTanh {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .FastTanh dst.id src.id)

/-- Element-wise sigmoid -/
def sigmoid {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Sigmoid dst.id src.id)

/-- Element-wise GELU -/
def gelu {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Gelu dst.id src.id)

/-- Element-wise ReLU -/
def relu {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Relu dst.id src.id)

/-- Element-wise negation -/
def neg {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Neg dst.id src.id)

/-- Element-wise absolute value -/
def abs {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Abs dst.id src.id)

/-- Zero out a tile -/
def zero {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (t : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Zero t.id t.id)

/-- Set tile to ones -/
def ones {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (t : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .One t.id t.id)

/-- Copy tile -/
def copy {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Copy dst.id src.id)

/-- Element-wise sin (for rotary embeddings) -/
def sin {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Sin dst.id src.id)

/-- Element-wise cos (for rotary embeddings) -/
def cos {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Cos dst.id src.id)

/-- Element-wise SiLU (x * sigmoid(x)) -/
def silu {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Silu dst.id src.id)

/-- Element-wise Swish (same as SiLU) -/
def swish {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Swish dst.id src.id)

/-- Element-wise reciprocal (1/x) -/
def recip {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Recip dst.id src.id)

/-- Element-wise square (x^2) -/
def square {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.unary .Square dst.id src.id)

/-! ## Element-wise Binary Operations -/

/-- Apply binary operation -/
def binaryOp {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (op : BinaryOp)
    (dst a b : RT dtype rows cols layout) : KernelM Unit := do
  emit (.binary op dst.id a.id b.id)

/-- Element-wise add -/
def add {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst a b : RT dtype rows cols layout) : KernelM Unit := do
  emit (.binary .Add dst.id a.id b.id)

/-- Element-wise subtract -/
def sub {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst a b : RT dtype rows cols layout) : KernelM Unit := do
  emit (.binary .Sub dst.id a.id b.id)

/-- Element-wise multiply -/
def mul {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst a b : RT dtype rows cols layout) : KernelM Unit := do
  emit (.binary .Mul dst.id a.id b.id)

/-- Element-wise divide -/
def div {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst a b : RT dtype rows cols layout) : KernelM Unit := do
  emit (.binary .Div dst.id a.id b.id)

/-- Element-wise max -/
def max {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst a b : RT dtype rows cols layout) : KernelM Unit := do
  emit (.binary .Max dst.id a.id b.id)

/-- Element-wise min -/
def min {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst a b : RT dtype rows cols layout) : KernelM Unit := do
  emit (.binary .Min dst.id a.id b.id)

/-- Scalar multiply -/
def scalarMul {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) (scalar : Float) : KernelM Unit := do
  emit (.scalarMul dst.id src.id scalar)

/-- Scalar add -/
def scalarAdd {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) (scalar : Float) : KernelM Unit := do
  emit (.scalarAdd dst.id src.id scalar)

/-! ## Reduction Operations -/

/-- Row-wise max reduction -/
def rowMax {dtype : GpuFloat} {rows cols : Nat}
    (dst : RV dtype rows)
    (src : RT dtype rows cols .Row) : KernelM Unit := do
  emit (.reduce .Max .Row dst.id src.id)

/-- Row-wise sum reduction -/
def rowSum {dtype : GpuFloat} {rows cols : Nat}
    (dst : RV dtype rows)
    (src : RT dtype rows cols .Row) : KernelM Unit := do
  emit (.reduce .Sum .Row dst.id src.id)

/-- Row-wise min reduction -/
def rowMin {dtype : GpuFloat} {rows cols : Nat}
    (dst : RV dtype rows)
    (src : RT dtype rows cols .Row) : KernelM Unit := do
  emit (.reduce .Min .Row dst.id src.id)

/-- Column-wise max reduction -/
def colMax {dtype : GpuFloat} {rows cols : Nat}
    (dst : RV dtype cols)
    (src : RT dtype rows cols .Row) : KernelM Unit := do
  emit (.reduce .Max .Col dst.id src.id)

/-- Column-wise sum reduction -/
def colSum {dtype : GpuFloat} {rows cols : Nat}
    (dst : RV dtype cols)
    (src : RT dtype rows cols .Row) : KernelM Unit := do
  emit (.reduce .Sum .Col dst.id src.id)

/-- Column-wise min reduction -/
def colMin {dtype : GpuFloat} {rows cols : Nat}
    (dst : RV dtype cols)
    (src : RT dtype rows cols .Row) : KernelM Unit := do
  emit (.reduce .Min .Col dst.id src.id)

/-- Row-wise product reduction -/
def rowProd {dtype : GpuFloat} {rows cols : Nat}
    (dst : RV dtype rows)
    (src : RT dtype rows cols .Row) : KernelM Unit := do
  emit (.reduce .Prod .Row dst.id src.id)

/-- Column-wise product reduction -/
def colProd {dtype : GpuFloat} {rows cols : Nat}
    (dst : RV dtype cols)
    (src : RT dtype rows cols .Row) : KernelM Unit := do
  emit (.reduce .Prod .Col dst.id src.id)

/-- Row-wise max with accumulator -/
def rowMaxAccum {dtype : GpuFloat} {rows cols : Nat}
    (dst : RV dtype rows)
    (src : RT dtype rows cols .Row)
    (accum : RV dtype rows) : KernelM Unit := do
  emit (.reduceAccum .Max .Row dst.id src.id accum.id)

/-- Row-wise sum with accumulator -/
def rowSumAccum {dtype : GpuFloat} {rows cols : Nat}
    (dst : RV dtype rows)
    (src : RT dtype rows cols .Row)
    (accum : RV dtype rows) : KernelM Unit := do
  emit (.reduceAccum .Sum .Row dst.id src.id accum.id)

/-! ## Broadcasting Operations -/

/-- Broadcast row vector to tile -/
def broadcastRow {dtype : GpuFloat} {rows cols : Nat}
    (dst : RT dtype rows cols .Row)
    (vec : RV dtype cols) : KernelM Unit := do
  emit (.broadcast .Row dst.id vec.id)

/-- Broadcast column vector to tile -/
def broadcastCol {dtype : GpuFloat} {rows cols : Nat}
    (dst : RT dtype rows cols .Row)
    (vec : RV dtype rows) : KernelM Unit := do
  emit (.broadcast .Col dst.id vec.id)

/-- Add row vector to each row of tile -/
def addRow {dtype : GpuFloat} {rows cols : Nat}
    (dst tile : RT dtype rows cols .Row)
    (vec : RV dtype cols) : KernelM Unit := do
  emit (.binaryBroadcast .Add .Row dst.id tile.id vec.id)

/-- Subtract row vector from each row of tile -/
def subRow {dtype : GpuFloat} {rows cols : Nat}
    (dst tile : RT dtype rows cols .Row)
    (vec : RV dtype cols) : KernelM Unit := do
  emit (.binaryBroadcast .Sub .Row dst.id tile.id vec.id)

/-- Multiply each row by vector -/
def mulRow {dtype : GpuFloat} {rows cols : Nat}
    (dst tile : RT dtype rows cols .Row)
    (vec : RV dtype cols) : KernelM Unit := do
  emit (.binaryBroadcast .Mul .Row dst.id tile.id vec.id)

/-- Divide each row by vector -/
def divRow {dtype : GpuFloat} {rows cols : Nat}
    (dst tile : RT dtype rows cols .Row)
    (vec : RV dtype cols) : KernelM Unit := do
  emit (.binaryBroadcast .Div .Row dst.id tile.id vec.id)

/-- Add column vector to each column -/
def addCol {dtype : GpuFloat} {rows cols : Nat}
    (dst tile : RT dtype rows cols .Row)
    (vec : RV dtype rows) : KernelM Unit := do
  emit (.binaryBroadcast .Add .Col dst.id tile.id vec.id)

/-- Subtract column vector from each column -/
def subCol {dtype : GpuFloat} {rows cols : Nat}
    (dst tile : RT dtype rows cols .Row)
    (vec : RV dtype rows) : KernelM Unit := do
  emit (.binaryBroadcast .Sub .Col dst.id tile.id vec.id)

/-- Divide each column by vector -/
def divCol {dtype : GpuFloat} {rows cols : Nat}
    (dst tile : RT dtype rows cols .Row)
    (vec : RV dtype rows) : KernelM Unit := do
  emit (.binaryBroadcast .Div .Col dst.id tile.id vec.id)

/-- Multiply each column by vector -/
def mulCol {dtype : GpuFloat} {rows cols : Nat}
    (dst tile : RT dtype rows cols .Row)
    (vec : RV dtype rows) : KernelM Unit := do
  emit (.binaryBroadcast .Mul .Col dst.id tile.id vec.id)

/-! ## Scan Operations -/

/-- Row-wise cumulative sum -/
def cumsumRow {dtype : GpuFloat} {rows cols : Nat}
    (dst src : RT dtype rows cols .Row) : KernelM Unit := do
  emit (.cumsum .Row dst.id src.id)

/-- Column-wise cumulative sum -/
def cumsumCol {dtype : GpuFloat} {rows cols : Nat}
    (dst src : RT dtype rows cols .Row) : KernelM Unit := do
  emit (.cumsum .Col dst.id src.id)

/-! ## Outer Product -/

/-- Outer product: dst[i,j] = a[i] * b[j] -/
def outer {dtype : GpuFloat} {rows cols : Nat}
    (dst : RT dtype rows cols .Row)
    (a : RV dtype rows)
    (b : RV dtype cols) : KernelM Unit := do
  emit (.outer dst.id a.id b.id)

/-! ## Layout/Type Conversions -/

/-- Swap layout (row to col or col to row) -/
def swapLayout {dtype : GpuFloat} {rows cols : Nat} {layout1 layout2 : TileLayout}
    (dst : RT dtype rows cols layout1)
    (src : RT dtype rows cols layout2) : KernelM Unit := do
  emit (.swapLayout dst.id src.id)

/-- Transpose tile -/
def transpose {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : RT dtype cols rows layout)
    (src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.transpose dst.id src.id)

/-- Type conversion (e.g., bf16 to float32) -/
def convert {dtype1 dtype2 : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : RT dtype1 rows cols layout)
    (src : RT dtype2 rows cols layout) : KernelM Unit := do
  emit (.convert dst.id src.id)

/-- Vector type conversion (e.g., bf16 to float32). -/
def convertVec {dtype1 dtype2 : GpuFloat} {len : Nat}
    (dst : RV dtype1 len)
    (src : RV dtype2 len) : KernelM Unit := do
  emit (.convert dst.id src.id)

/-! ## Masking Operations -/

/-- Apply causal mask (zero out upper triangle) -/
def makeCausal {dtype : GpuFloat} {rows cols : Nat}
    (dst src : RT dtype rows cols .Row)
    (fillVal : Option Float := none) : KernelM Unit := do
  emit (.mask .MakeCausal dst.id src.id fillVal)

/-- Lower triangular mask -/
def tril {dtype : GpuFloat} {rows cols : Nat}
    (dst src : RT dtype rows cols .Row)
    (diagonal : Int := 0)
    (fillVal : Option Float := none) : KernelM Unit := do
  emit (.mask (.Tril diagonal) dst.id src.id fillVal)

/-- Upper triangular mask -/
def triu {dtype : GpuFloat} {rows cols : Nat}
    (dst src : RT dtype rows cols .Row)
    (diagonal : Int := 0)
    (fillVal : Option Float := none) : KernelM Unit := do
  emit (.mask (.Triu diagonal) dst.id src.id fillVal)

/-- Transpose causal mask (for backward pass) -/
def makeCausalT {dtype : GpuFloat} {rows cols : Nat}
    (dst src : RT dtype rows cols .Row)
    (fillVal : Option Float := none) : KernelM Unit := do
  emit (.mask .MakeCausalT dst.id src.id fillVal)

/-- Left fill: fill columns 0..colIdx with value -/
def leftFill {dtype : GpuFloat} {rows cols : Nat}
    (dst src : RT dtype rows cols .Row)
    (colIdx : Nat)
    (fillVal : Option Float := none) : KernelM Unit := do
  emit (.mask (.LeftFill colIdx) dst.id src.id fillVal)

/-- Right fill: fill columns colIdx..end with value -/
def rightFill {dtype : GpuFloat} {rows cols : Nat}
    (dst src : RT dtype rows cols .Row)
    (colIdx : Nat)
    (fillVal : Option Float := none) : KernelM Unit := do
  emit (.mask (.RightFill colIdx) dst.id src.id fillVal)

/-- Right fill with a runtime cutoff: fill columns `colVar..end` with `fillVal`.

This is the runtime-parameterized analogue of `rightFill`. ThunderKittens'
`warp::right_fill` already accepts a runtime `int col_idx`; this primitive
threads a Lean `KVal UInt32` through the IR. Used by decode-style attention
where the valid KV length within the last block is determined at runtime
(see `Tyr.GPU.Kernels.tkMhaH100DecodeFwd`). -/
def rightFillVal {dtype : GpuFloat} {rows cols : Nat}
    (dst src : RT dtype rows cols .Row)
    (colVar : KVal UInt32)
    (fillVal : Option Float := none) : KernelM Unit := do
  emit (.maskRightFillVal dst.id src.id colVar.id fillVal)

/-- Upper fill: fill rows 0..rowIdx with value -/
def upperFill {dtype : GpuFloat} {rows cols : Nat}
    (dst src : RT dtype rows cols .Row)
    (rowIdx : Nat)
    (fillVal : Option Float := none) : KernelM Unit := do
  emit (.mask (.UpperFill rowIdx) dst.id src.id fillVal)

/-- Lower fill: fill rows rowIdx..end with value -/
def lowerFill {dtype : GpuFloat} {rows cols : Nat}
    (dst src : RT dtype rows cols .Row)
    (rowIdx : Nat)
    (fillVal : Option Float := none) : KernelM Unit := do
  emit (.mask (.LowerFill rowIdx) dst.id src.id fillVal)

/-- Upper-right fill: fill block at (row, col) to end -/
def upperRightFill {dtype : GpuFloat} {rows cols : Nat}
    (dst src : RT dtype rows cols .Row)
    (rowIdx colIdx : Nat)
    (fillVal : Option Float := none) : KernelM Unit := do
  emit (.mask (.UpperRightFill rowIdx colIdx) dst.id src.id fillVal)

/-! ## Tile Slicing/Concatenation -/

/-- Slice rows from a tile. -/
def sliceRows {dtype : GpuFloat} {dstRows srcRows cols : Nat} {layout : TileLayout}
    (dst : RT dtype dstRows cols layout)
    (src : RT dtype srcRows cols layout)
    (startRow : Nat)
    (numRows : Nat := dstRows) : KernelM Unit := do
  emit (.sliceRows dst.id src.id startRow numRows)

/-- Copy a register subtile into a row slice of a shared tile.
    The emitter lowers this to a `subtile<...>` view on `dst` followed by
    `warp::copy`, which is enough to model TK's per-warp `warpgroup::store`
    into a 64-row shared tile. -/
def storeRowsShared {dtype : GpuFloat} {dstRows srcRows cols : Nat} {layout : TileLayout}
    (dst : ST dtype dstRows cols layout)
    (src : RT dtype srcRows cols layout)
    (startRow : Nat)
    (numRows : Nat := srcRows) : KernelM Unit := do
  emit (.sliceRows dst.id src.id startRow numRows)

/-- ThunderKittens warpgroup store from a warpgroup register fragment into
    shared memory. The source and destination dtypes may differ because TK
    handles the FP32->BF16 conversion used by MHA output stores. -/
def warpgroupStore {dstDtype srcDtype : GpuFloat} {dstRows srcRows cols : Nat}
    {dstLayout srcLayout : TileLayout}
    (dst : ST dstDtype dstRows cols dstLayout)
    (src : RT srcDtype srcRows cols srcLayout) : KernelM Unit := do
  emit (.warpgroupStore dst.id src.id)

/-- ThunderKittens warpgroup store from a per-warp register column vector into a
    shared column-vector tile. This models the `l_smem <- norm_vec` store in
    the H100 forward kernel. -/
def warpgroupStoreColVec {dtype : GpuFloat} {dstRows srcRows cols : Nat}
    (dst : STColVec dtype dstRows cols)
    (src : RV dtype srcRows) : KernelM Unit := do
  emit (.warpgroupStore dst.id src.id)

/-- Warpgroup store into an indexed shared-tile array element. -/
def warpgroupStoreArray {dstDtype srcDtype : GpuFloat} {dstRows srcRows cols len : Nat}
    {dstLayout srcLayout : TileLayout}
    (dst : STArray dstDtype dstRows cols dstLayout len)
    (dstIdx : KVal UInt32)
    (src : RT srcDtype srcRows cols srcLayout) : KernelM Unit := do
  emit (.warpgroupStoreIdx dst.id dstIdx.id src.id)

/-- Warpgroup store into an indexed shared column-vector array element. -/
def warpgroupStoreColVecArray {dtype : GpuFloat} {dstRows srcRows cols len : Nat}
    (dst : STColVecArray dtype dstRows cols len)
    (dstIdx : KVal UInt32)
    (src : RV dtype srcRows) : KernelM Unit := do
  emit (.warpgroupStoreIdx dst.id dstIdx.id src.id)

/-- Slice columns from a tile. -/
def sliceCols {dtype : GpuFloat} {rows dstCols srcCols : Nat} {layout : TileLayout}
    (dst : RT dtype rows dstCols layout)
    (src : RT dtype rows srcCols layout)
    (startCol : Nat)
    (numCols : Nat := dstCols) : KernelM Unit := do
  emit (.sliceCols dst.id src.id startCol numCols)

/-- Concatenate two tiles along columns. -/
def concatCols {dtype : GpuFloat} {rows dstCols leftCols rightCols : Nat} {layout : TileLayout}
    (dst : RT dtype rows dstCols layout)
    (left : RT dtype rows leftCols layout)
    (right : RT dtype rows rightCols layout) : KernelM Unit := do
  emit (.concatCols dst.id left.id right.id)

/-! ## Cumulative/Scan Operations -/

/-- Row-wise cumulative product (for decay in Mamba) -/
def cumprodRow {dtype : GpuFloat} {rows cols : Nat}
    (dst src : RT dtype rows cols .Row) : KernelM Unit := do
  emit (.cumprod .Row dst.id src.id)

/-- Column-wise cumulative product -/
def cumprodCol {dtype : GpuFloat} {rows cols : Nat}
    (dst src : RT dtype rows cols .Row) : KernelM Unit := do
  emit (.cumprod .Col dst.id src.id)

/-! ## TMA Operations -/

/-- TMA prefetch -/
def prefetch {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (src : ST dtype rows cols layout) : KernelM Unit := do
  emit (.prefetch src.id)

/-- Async atomic store-min (TMA) -/
def storeMinAsync {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.storeMinAsync dst.id src.id)

/-- Commit the current async TMA store group. -/
def tmaStoreCommitGroup : KernelM Unit := do
  emit .tmaStoreCommitGroup

/-- Wait for outstanding async TMA stores issued by this warp. -/
def tmaStoreAsyncWait : KernelM Unit := do
  emit .tmaStoreAsyncWait

/-- CTA-wide synchronization. Use sparingly: TK kernels prefer semaphore
    handoffs, but setup/init paths need a full block rendezvous. -/
def blockSync : KernelM Unit := do
  emit .blockSync

/-- ThunderKittens group synchronization (`group<N>::sync(barrierId)`). -/
def groupSync (warps barrierId : Nat) : KernelM Unit := do
  emit (.groupSync warps barrierId)

/-- ThunderKittens group synchronization with a runtime barrier id. -/
def groupSyncVal (warps : Nat) (barrierId : KVal UInt32) : KernelM Unit := do
  emit (.groupSyncVal warps barrierId.id)

/-- TMA load from global pointer to shared tile -/
def tmaLoad {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : GPtr dtype)
    (coord : KVal UInt64) : KernelM Unit := do
  emit (.tmaLoad dst.id src.id coord.id)

/-- TMA store from shared tile to global pointer -/
def tmaStore {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : GPtr dtype)
    (src : ST dtype rows cols layout)
    (coord : KVal UInt64) : KernelM Unit := do
  emit (.tmaStore dst.id src.id coord.id)

/-! ## Global Memory Operations (ThunderKittens Style)

These operations use GlobalLayout and TileCoord/RTileCoord for 4D coordinate-based
memory access, following ThunderKittens' gl and coord patterns.
-/

-- Forward declare GlobalLayout and RTileCoord (will be imported via Codegen.lean)
-- For now, we use VarId directly in the low-level operations

/-- Load tile from global memory to shared memory using 4D coordinates.
    Emits: kittens::load(dst, src, {.b=b, .d=d, .r=r, .c=c}) -/
def loadGlobalCoord {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : GPtr dtype)
    (coordB coordD coordR coordC : VarId)
    : KernelM Unit := do
  emit (.loadGlobal dst.id src.id coordB coordD coordR coordC)

/-- Store tile from shared memory to global memory using 4D coordinates.
    Emits: kittens::store(dst, src, {.b=b, .d=d, .r=r, .c=c}) -/
def storeGlobalCoord {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : GPtr dtype)
    (src : ST dtype rows cols layout)
    (coordB coordD coordR coordC : VarId)
    : KernelM Unit := do
  emit (.storeGlobal dst.id src.id coordB coordD coordR coordC)

/-- Async load from global to shared with semaphore (TMA).
    Emits: kittens::tma::load_async(dst, src, coord, sem) -/
def loadGlobalAsyncCoord {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : GPtr dtype)
    (coordB coordD coordR coordC : VarId)
    (sem : Semaphore)
    : KernelM Unit := do
  emit (.loadGlobalAsync dst.id src.id coordB coordD coordR coordC sem.id)

/-- Async load from global to shared with semaphore (TMA), issued by the
    currently active warp rather than being implicitly restricted to warp 0. -/
def loadGlobalAsyncWarpCoord {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : GPtr dtype)
    (coordB coordD coordR coordC : VarId)
    (sem : Semaphore)
    : KernelM Unit := do
  emit (.loadGlobalAsyncWarp dst.id src.id coordB coordD coordR coordC sem.id)

/-- Async TMA load from global to a shared row-vector tile descriptor, issued by
    the currently active warp. Coordinates are vector-block coordinates, not
    element coordinates; the generated CUDA emits `coord<decltype(dst)>` so TK
    applies the vector length scaling. -/
def loadRowVecGlobalAsyncWarpCoord {dtype : GpuFloat} {rows cols : Nat}
    (dst : STRowVec dtype rows cols)
    (src : GPtr dtype)
    (coordB coordD coordR coordC : VarId)
    (sem : Semaphore)
    : KernelM Unit := do
  emit (.loadGlobalAsyncWarp dst.id src.id coordB coordD coordR coordC sem.id)

/-- Async TMA load from global to a shared column-vector tile descriptor, issued
    by the currently active warp. -/
def loadColVecGlobalAsyncWarpCoord {dtype : GpuFloat} {rows cols : Nat}
    (dst : STColVec dtype rows cols)
    (src : GPtr dtype)
    (coordB coordD coordR coordC : VarId)
    (sem : Semaphore)
    : KernelM Unit := do
  emit (.loadGlobalAsyncWarp dst.id src.id coordB coordD coordR coordC sem.id)

/-- Async store from shared to global (TMA).
    Emits: kittens::tma::store_async(dst, src, coord) -/
def storeGlobalAsyncCoord {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : GPtr dtype)
    (src : ST dtype rows cols layout)
    (coordB coordD coordR coordC : VarId)
    : KernelM Unit := do
  emit (.storeGlobalAsync dst.id src.id coordB coordD coordR coordC)

/-- Async TMA store from a shared row-vector tile descriptor to global memory. -/
def storeRowVecGlobalAsyncCoord {dtype : GpuFloat} {rows cols : Nat}
    (dst : GPtr dtype)
    (src : STRowVec dtype rows cols)
    (coordB coordD coordR coordC : VarId)
    : KernelM Unit := do
  emit (.storeGlobalAsync dst.id src.id coordB coordD coordR coordC)

/-- Async TMA store from a shared column-vector tile descriptor to global memory. -/
def storeColVecGlobalAsyncCoord {dtype : GpuFloat} {rows cols : Nat}
    (dst : GPtr dtype)
    (src : STColVec dtype rows cols)
    (coordB coordD coordR coordC : VarId)
    : KernelM Unit := do
  emit (.storeGlobalAsync dst.id src.id coordB coordD coordR coordC)

/-- Atomic add store from shared to global (for gradient accumulation).
    Emits: kittens::tma::store_add_async(dst, src, coord) -/
def storeGlobalAddCoord {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : GPtr dtype)
    (src : ST dtype rows cols layout)
    (coordB coordD coordR coordC : VarId)
    : KernelM Unit := do
  emit (.storeGlobalAdd dst.id src.id coordB coordD coordR coordC)

/-- Atomic add store from shared to global (for gradient accumulation), issued
    by the currently active warp rather than being implicitly restricted to warp 0. -/
def storeGlobalAddWarpCoord {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : GPtr dtype)
    (src : ST dtype rows cols layout)
    (coordB coordD coordR coordC : VarId)
    : KernelM Unit := do
  emit (.storeGlobalAddWarp dst.id src.id coordB coordD coordR coordC)

/-- Query a dynamic dimension from a global-layout kernel parameter. -/
def layoutDim {dtype : GpuFloat}
    (src : GPtr dtype) (axis : LayoutDimAxis)
    (name : String := "layout_dim") : KernelM (KVal UInt32) := do
  let v ← freshVar
  emit (.layoutDim v src.id axis)
  pure ⟨v, name⟩

/-- Query the batch dimension from a global-layout kernel parameter. -/
def layoutBatch {dtype : GpuFloat} (src : GPtr dtype)
    (name : String := "batch") : KernelM (KVal UInt32) :=
  layoutDim src .Batch name

/-- Query the depth dimension from a global-layout kernel parameter. -/
def layoutDepth {dtype : GpuFloat} (src : GPtr dtype)
    (name : String := "depth") : KernelM (KVal UInt32) :=
  layoutDim src .Depth name

/-- Query the row dimension from a global-layout kernel parameter. -/
def layoutRows {dtype : GpuFloat} (src : GPtr dtype)
    (name : String := "rows") : KernelM (KVal UInt32) :=
  layoutDim src .Rows name

/-- Query the column dimension from a global-layout kernel parameter. -/
def layoutCols {dtype : GpuFloat} (src : GPtr dtype)
    (name : String := "cols") : KernelM (KVal UInt32) :=
  layoutDim src .Cols name

/-- Load vector from global memory.
    Emits: kittens::load(dst, src, offset) -/
def loadVecGlobalCoord {dtype : GpuFloat} {len : Nat}
    (dst : SV dtype len)
    (src : GPtr dtype)
    (offset : VarId)
    : KernelM Unit := do
  emit (.loadVecGlobal dst.id src.id offset)

/-- Store vector to global memory.
    Emits: kittens::store(dst, src, offset) -/
def storeVecGlobalCoord {dtype : GpuFloat} {len : Nat}
    (dst : GPtr dtype)
    (src : SV dtype len)
    (offset : VarId)
    : KernelM Unit := do
  emit (.storeVecGlobal dst.id src.id offset)

/-- Atomic add store vector to global memory.
    Emits: kittens::store_add(dst, src, offset) -/
def storeVecGlobalAddCoord {dtype : GpuFloat} {len : Nat}
    (dst : GPtr dtype)
    (src : SV dtype len)
    (offset : VarId)
    : KernelM Unit := do
  emit (.storeVecGlobalAdd dst.id src.id offset)

/-- Load vector from global memory using a full 4D runtime coordinate. -/
def loadVecGlobalAt {dtype : GpuFloat} {len : Nat}
    (dst : SV dtype len)
    (src : GPtr dtype)
    (coord : RTileCoord)
    : KernelM Unit := do
  emit (.loadVecGlobalCoord dst.id src.id coord.b coord.d coord.r coord.c)

/-- Store vector to global memory using a full 4D runtime coordinate. -/
def storeVecGlobalAt {dtype : GpuFloat} {len : Nat}
    (dst : GPtr dtype)
    (src : SV dtype len)
    (coord : RTileCoord)
    : KernelM Unit := do
  emit (.storeVecGlobalCoord dst.id src.id coord.b coord.d coord.r coord.c)

/-- Atomic-add store vector to global memory using a full 4D runtime coordinate. -/
def storeVecGlobalAddAt {dtype : GpuFloat} {len : Nat}
    (dst : GPtr dtype)
    (src : SV dtype len)
    (coord : RTileCoord)
    : KernelM Unit := do
  emit (.storeVecGlobalAddCoord dst.id src.id coord.b coord.d coord.r coord.c)

/-! ## Distributed / Multimem Operations -/

/-- Multimem load-reduce: dst (reg) = reduce(src (shared) across cluster) -/
def multimemLoadReduce {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : RT dtype rows cols layout)
    (src : ST dtype rows cols layout)
    (op : ReduceOp := .Sum) : KernelM Unit := do
  emit (.multimemLoadReduce op dst.id src.id)

/-- Multimem store: dst (shared across cluster) = src (reg) -/
def multimemStore {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : RT dtype rows cols layout) : KernelM Unit := do
  emit (.multimemStore dst.id src.id)

/-- Multimem reduce: dst (shared across cluster) op= src (reg) -/
def multimemRed {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : RT dtype rows cols layout)
    (op : ReduceOp := .Sum) : KernelM Unit := do
  emit (.multimemRed op dst.id src.id)

/-! ## Semaphore Operations -/

/-- Allocate a semaphore -/
def allocSemaphore : KernelM Semaphore := do
  let v ← freshVar
  emit (.declSemaphore v)
  pure ⟨v⟩

/-- Initialize semaphore with expected arriving threads and transactions. -/
def initSemaphore (sem : Semaphore) (threadCount : Nat := 1) (transactionCount : Nat := 0)
    : KernelM Unit := do
  emit (.semaphore (.Init threadCount transactionCount) sem.id)

/-- Allocate a shared semaphore array. -/
def allocSemaphoreArray (len : Nat) : KernelM (SemaphoreArray len) := do
  let v ← freshVar
  emit (.declSemaphoreArray v len)
  pure ⟨v⟩

/-- Initialize one semaphore array element. -/
def initSemaphoreArray {len : Nat} (sem : SemaphoreArray len) (idx : KVal UInt32)
    (threadCount : Nat := 1) (transactionCount : Nat := 0) : KernelM Unit := do
  emit (.semaphoreArray (.Init threadCount transactionCount) sem.id idx.id)

/-- Invalidate semaphore -/
def invalidateSemaphore (sem : Semaphore) : KernelM Unit := do
  emit (.semaphore .Invalidate sem.id)

/-- Expect bytes on semaphore -/
def expectBytes (sem : Semaphore) (bytes : Nat) : KernelM Unit := do
  emit (.semaphore (.Expect bytes) sem.id)

/-- Expect bytes on one semaphore-array element from warp 0. -/
def expectBytesArray {len : Nat} (sem : SemaphoreArray len) (idx : KVal UInt32)
    (bytes : Nat) : KernelM Unit := do
  emit (.semaphoreArray (.Expect bytes) sem.id idx.id)

/-- Expect bytes on a semaphore from the currently active warp. -/
def expectBytesWarp (sem : Semaphore) (bytes : Nat) : KernelM Unit := do
  emit (.semaphoreWarp (.Expect bytes) sem.id)

/-- Expect bytes on one semaphore-array element from the currently active warp. -/
def expectBytesArrayWarp {len : Nat} (sem : SemaphoreArray len) (idx : KVal UInt32)
    (bytes : Nat) : KernelM Unit := do
  emit (.semaphoreArrayWarp (.Expect bytes) sem.id idx.id)

/-- Wait on semaphore phase bit. -/
def waitSemaphorePhase (sem : Semaphore) (phase : Nat) : KernelM Unit := do
  emit (.semaphore (.Wait phase) sem.id)

/-- Wait on a semaphore using a runtime phase value.
    This matches TK's `wait(sem, phaseVar)` form used in pipelined loops. -/
def waitSemaphorePhaseVal (sem : Semaphore) (phase : KVal UInt32) : KernelM Unit := do
  emit (.semaphoreWaitVal sem.id phase.id)

/-- Wait on one semaphore-array element using a runtime phase value. -/
def waitSemaphoreArrayPhaseVal {len : Nat} (sem : SemaphoreArray len)
    (idx phase : KVal UInt32) : KernelM Unit := do
  emit (.semaphoreArrayWaitVal sem.id idx.id phase.id)

/-- Wait on semaphore phase 0. -/
def waitSemaphore (sem : Semaphore) : KernelM Unit := do
  waitSemaphorePhase sem 0

/-- Arrive at semaphore with transaction count -/
def arriveSemaphore (sem : Semaphore) (count : Nat := 1) : KernelM Unit := do
  emit (.semaphore (.Arrive count) sem.id)

/-- Arrive at a semaphore from the currently active warp. -/
def arriveSemaphoreWarp (sem : Semaphore) (count : Nat := 1) : KernelM Unit := do
  emit (.semaphoreWarp (.Arrive count) sem.id)

/-- Arrive at one semaphore-array element from the currently active warp. -/
def arriveSemaphoreArrayWarp {len : Nat} (sem : SemaphoreArray len)
    (idx : KVal UInt32) (count : Nat := 1) : KernelM Unit := do
  emit (.semaphoreArrayArrive sem.id idx.id count)

/-- Arrive and wait at semaphore -/
def arriveAndWait (barrier : Nat := 0) : KernelM Unit := do
  emit (.arriveAndWait barrier)

/-- Wait on a barrier (for FA3 specialization) -/
def waitBarrier (barrier : Nat) : KernelM Unit := do
  emit (.arriveAndWait barrier)

/-- Signal a barrier (for FA3 specialization) -/
def signalBarrier (barrier : Nat) : KernelM Unit := do
  emit (.arrive barrier)

/-- Reduce the register allocation for the current warpgroup, matching TK's
    explicit producer/loader warpgroup budgeting on Hopper. -/
def warpgroupDecreaseRegisters (n : Nat) : KernelM Unit := do
  emit (.warpgroupDecreaseRegisters n)

/-- Increase the register allocation for the current warpgroup, matching TK's
    explicit consumer warpgroup budgeting on Hopper. -/
def warpgroupIncreaseRegisters (n : Nat) : KernelM Unit := do
  emit (.warpgroupIncreaseRegisters n)

/-- ThunderKittens-style lane streaming of a 64-element shared vector into one
    16x64 register subtile. This matches `stream_tile` from the vendored H100
    attention kernel. -/
def streamTile16x64 (dst : RT GpuFloat.Float32 16 64 .Row) (src : SV GpuFloat.Float32 64)
    : KernelM Unit := do
  let code :=
    "#pragma unroll\n" ++
    "for(int i = 0; i < 4; i++) {\n" ++
    "  int base_col = 16*i + 2*(kittens::laneid()%4);\n" ++
    s!"  {dst.id.toIdent}.tiles[0][i].data[0] = *(float2*)&{src.id.toIdent}[base_col + 0];\n" ++
    s!"  {dst.id.toIdent}.tiles[0][i].data[1] = *(float2*)&{src.id.toIdent}[base_col + 0];\n" ++
    s!"  {dst.id.toIdent}.tiles[0][i].data[2] = *(float2*)&{src.id.toIdent}[base_col + 8];\n" ++
    s!"  {dst.id.toIdent}.tiles[0][i].data[3] = *(float2*)&{src.id.toIdent}[base_col + 8];\n" ++
    "}"
  emit (.raw code)

/-- ThunderKittens-style lane streaming from a shared row-vector tile into one
    16x64 register subtile. This matches the H100 MHA backward helper's
    `stream_tile(reg, row_vec_stage, tic)` after double-buffer indexing has
    already selected the stage. -/
def streamRowVecTile16x64 (dst : RT GpuFloat.Float32 16 64 .Row)
    (src : STRowVec GpuFloat.Float32 64 64) : KernelM Unit := do
  let code :=
    "#pragma unroll\n" ++
    "for(int i = 0; i < 4; i++) {\n" ++
    "  int base_col = 16*i + 2*(kittens::laneid()%4);\n" ++
    s!"  {dst.id.toIdent}.tiles[0][i].data[0] = *(float2*)&{src.id.toIdent}[base_col + 0];\n" ++
    s!"  {dst.id.toIdent}.tiles[0][i].data[1] = *(float2*)&{src.id.toIdent}[base_col + 0];\n" ++
    s!"  {dst.id.toIdent}.tiles[0][i].data[2] = *(float2*)&{src.id.toIdent}[base_col + 8];\n" ++
    s!"  {dst.id.toIdent}.tiles[0][i].data[3] = *(float2*)&{src.id.toIdent}[base_col + 8];\n" ++
    "}"
  emit (.raw code)

/-- ThunderKittens-style subtraction of a streamed 64-element shared vector from
    one 16x64 register subtile. This matches `stream_sub_tile` from the
    vendored H100 attention kernel. -/
def streamSubTile16x64 (dst : RT GpuFloat.Float32 16 64 .Row) (src : SV GpuFloat.Float32 64)
    : KernelM Unit := do
  let code :=
    "#pragma unroll\n" ++
    "for(int i = 0; i < 4; i++) {\n" ++
    "  int base_col = 16*i + 2*(kittens::laneid()%4);\n" ++
    s!"  {dst.id.toIdent}.tiles[0][i].data[0] = base_ops::sub::template op<float2>({dst.id.toIdent}.tiles[0][i].data[0], *(float2*)&{src.id.toIdent}[base_col + 0]);\n" ++
    s!"  {dst.id.toIdent}.tiles[0][i].data[1] = base_ops::sub::template op<float2>({dst.id.toIdent}.tiles[0][i].data[1], *(float2*)&{src.id.toIdent}[base_col + 0]);\n" ++
    s!"  {dst.id.toIdent}.tiles[0][i].data[2] = base_ops::sub::template op<float2>({dst.id.toIdent}.tiles[0][i].data[2], *(float2*)&{src.id.toIdent}[base_col + 8]);\n" ++
    s!"  {dst.id.toIdent}.tiles[0][i].data[3] = base_ops::sub::template op<float2>({dst.id.toIdent}.tiles[0][i].data[3], *(float2*)&{src.id.toIdent}[base_col + 8]);\n" ++
    "}"
  emit (.raw code)

/-- ThunderKittens-style subtraction of a shared row-vector tile from one 16x64
    register subtile. This avoids materializing an intermediate RV just to
    broadcast D across the attention subtile. -/
def streamSubRowVecTile16x64 (dst : RT GpuFloat.Float32 16 64 .Row)
    (src : STRowVec GpuFloat.Float32 64 64) : KernelM Unit := do
  let code :=
    "#pragma unroll\n" ++
    "for(int i = 0; i < 4; i++) {\n" ++
    "  int base_col = 16*i + 2*(kittens::laneid()%4);\n" ++
    s!"  {dst.id.toIdent}.tiles[0][i].data[0] = base_ops::sub::template op<float2>({dst.id.toIdent}.tiles[0][i].data[0], *(float2*)&{src.id.toIdent}[base_col + 0]);\n" ++
    s!"  {dst.id.toIdent}.tiles[0][i].data[1] = base_ops::sub::template op<float2>({dst.id.toIdent}.tiles[0][i].data[1], *(float2*)&{src.id.toIdent}[base_col + 0]);\n" ++
    s!"  {dst.id.toIdent}.tiles[0][i].data[2] = base_ops::sub::template op<float2>({dst.id.toIdent}.tiles[0][i].data[2], *(float2*)&{src.id.toIdent}[base_col + 8]);\n" ++
    s!"  {dst.id.toIdent}.tiles[0][i].data[3] = base_ops::sub::template op<float2>({dst.id.toIdent}.tiles[0][i].data[3], *(float2*)&{src.id.toIdent}[base_col + 8]);\n" ++
    "}"
  emit (.raw code)

/-! ## Synchronization -/

/-- Barrier synchronization -/
def sync (barrier : Nat := 0) : KernelM Unit := do
  emit (.sync barrier)

/-- Signal arrival at barrier -/
def arrive (barrier : Nat := 0) : KernelM Unit := do
  emit (.arrive barrier)

/-- MMA fence for Hopper WGMMA -/
def mmaFence {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : RT dtype rows cols layout) : KernelM Unit := do
  emit (.mmaFence dst.id)

/-- Commit MMA group -/
def mmaCommitGroup : KernelM Unit := do
  emit .mmaCommitGroup

/-- Wait for N MMA groups -/
def mmaAsyncWait (n : Nat := 0) : KernelM Unit := do
  emit (.mmaAsyncWait n)

/-! ## Blackwell tcgen05 MMA Operations -/

/-- Blackwell tcgen05 MMA: zero-init multiply to TMEM destination.
    `mm2_ABt(dst, a, b)` — first MMA that initializes the accumulator.
    Shapes must be multiples of 16 (minimum); B200 2-CTA variants additionally
    require M ≥ 128 (enforce in calling code). -/
def tcgen05Mm {M K N : Nat} {inDtype accDtype : GpuFloat}
    {inLayout : TileLayout} {accLayout : TileLayout}
    (trans : MMATranspose)
    (dst : TT accDtype M N) (a : ST inDtype M K inLayout) (b : ST inDtype N K accLayout)
    (hM : M % 16 = 0 := by decide)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hM; let _ := hK; let _ := hN
  emit (.tcgen05Mm trans dst.id a.id b.id)

/-- Blackwell tcgen05 MMA: accumulating multiply to TMEM destination.
    `mma2_ABt(dst, a, b, c)` — adds to existing accumulator `c`. -/
def tcgen05Mma {M K N : Nat} {inDtype accDtype : GpuFloat}
    {inLayout : TileLayout} {accLayout : TileLayout}
    (trans : MMATranspose)
    (dst : TT accDtype M N) (a : ST inDtype M K inLayout) (b : ST inDtype N K accLayout)
    (c : TT accDtype M N)
    (hM : M % 16 = 0 := by decide)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hM; let _ := hK; let _ := hN
  emit (.tcgen05Mma trans dst.id a.id b.id c.id)

/-- Blackwell tcgen05 scaled MMA with E8M0 scale tiles (MXFP8/NVFP4).
    `mma2_ABt(dst, a, b, c, scaleA, scaleB)`. -/
def tcgen05MmaScaled {M K N : Nat} {inDtype accDtype scaleDtype : GpuFloat}
    {inLayout : TileLayout} {accLayout : TileLayout}
    (trans : MMATranspose)
    (dst : TT accDtype M N) (a : ST inDtype M K inLayout) (b : ST inDtype N K accLayout)
    (c : TT accDtype M N)
    (scaleA : TT scaleDtype M K) (scaleB : TT scaleDtype N K)
    (hM : M % 16 = 0 := by decide)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hM; let _ := hK; let _ := hN
  emit (.tcgen05MmaScaled trans dst.id a.id b.id c.id scaleA.id scaleB.id)

/-- Commit tcgen05 results to the cluster-aware semaphore. -/
def tcgen05Commit (sem : Semaphore) (clusterSize : Nat := 1) : KernelM Unit := do
  emit (.tcgen05Commit sem.id clusterSize)

/-- Load one register fragment per warp from a tensor-memory tile. -/
def tensorLoadAsync {dtype : GpuFloat} {warpRows rows cols : Nat}
    {layout : TileLayout}
    (dst : RT dtype warpRows cols layout) (src : TT dtype rows cols)
    : KernelM Unit := do
  emit (.tensorLoadAsync dst.id src.id)

/-- Wait for all preceding tensor-memory loads. -/
def tensorLoadWait : KernelM Unit := do
  emit .tensorLoadWait

/-! ## Cluster Operations -/

/-- Read cluster CTA index along axis (0=x, 1=y, 2=z).
    Returns which CTA this is within the cluster (e.g. 0 or 1 for 2-CTA). -/
def clusterIdx (axis : Nat := 0) : KernelM (KVal UInt32) := do
  let v ← freshVar
  emit (.clusterIdx v axis)
  pure ⟨v, "cluster_idx"⟩

/-- Cluster-scoped TMA load: visible to all CTAs in the cluster. -/
def clusterTmaLoad {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout) (src : GPtr dtype)
    (coordB coordD coordR coordC : KVal UInt64) (sem : Semaphore)
    : KernelM Unit := do
  emit (.clusterTmaLoad dst.id src.id coordB.id coordD.id coordR.id coordC.id sem.id)

/-- Cluster-scoped TMA store: write from cluster scope. -/
def clusterTmaStore {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : GPtr dtype) (src : ST dtype rows cols layout)
    (coordB coordD coordR coordC : KVal UInt64)
    : KernelM Unit := do
  emit (.clusterTmaStore dst.id src.id coordB.id coordD.id coordR.id coordC.id)

/-- Cluster barrier arrive. -/
def clusterArrive (sem : Semaphore) : KernelM Unit := do
  emit (.clusterArrive sem.id)

/-- Cluster barrier wait. -/
def clusterWait (sem : Semaphore) : KernelM Unit := do
  emit (.clusterWait sem.id)

/-! ## FlashAttention3 Warp Specialization Operations -/

/-- Named barrier synchronization (for warp specialization)
    All threads in the barrier must call this to proceed -/
def namedBarrierSync (barrierId : Nat) (numThreads : Nat) : KernelM Unit := do
  emit (.namedBarrierSync barrierId numThreads)

/-- Named barrier arrive (for warp specialization)
    Signal arrival without waiting -/
def namedBarrierArrive (barrierId : Nat) (numThreads : Nat) : KernelM Unit := do
  emit (.namedBarrierArrive barrierId numThreads)

/-- Get the warp group index (0, 1, 2, ...) within a CTA
    Used to determine producer vs consumer role in FA3 -/
def getWarpGroupIdx : KernelM (KVal UInt32) := do
  let v ← freshVar
  emit (.warpGroupIdx v)
  pure ⟨v, "wg_idx"⟩

/-- Read the lane id within the current warpgroup. ThunderKittens uses this for
    one-thread-per-warpgroup barrier arrivals. -/
def getWarpGroupLaneId (name : String := "wg_lane_id") : KernelM (KVal UInt32) := do
  let v ← freshVar
  emit (.warpGroupLaneId v)
  pure ⟨v, name⟩

/-- Read the current warp id within the CTA. -/
def getWarpId (name : String := "warp_id") : KernelM (KVal UInt32) := do
  let v ← freshVar
  emit (.warpId v)
  pure ⟨v, name⟩

/-- Read the current lane id within the warp. -/
def getLaneId (name : String := "lane_id") : KernelM (KVal UInt32) := do
  let v ← freshVar
  emit (.laneId v)
  pure ⟨v, name⟩

/-- Elect one thread per warp to execute a region
    Returns true for the elected thread -/
def electOneSync : KernelM (KVal Bool) := do
  let v ← freshVar
  emit (.electOneSync v)
  pure ⟨v, "elected"⟩

/-- Fence for view async shared (WGMMA pipelining)
    Ensures shared memory writes are visible to WGMMA -/
def fenceViewAsyncShared : KernelM Unit := do
  emit .fenceViewAsyncShared

/-- Proxy async fence (WGMMA pipelining)
    Ensures async proxy operations complete -/
def fenceProxyAsync : KernelM Unit := do
  emit .fenceProxyAsync

/-- Execute a block of code only in the specified warp group
    Used for producer/consumer specialization in FA3 -/
def ifWarpGroup (wgIdx : Nat) (action : KernelM Unit) : KernelM Unit := do
  emitWarpGroup wgIdx action

/-- Execute a block of code only when the runtime boolean condition is true. -/
def ifThen (cond : KVal Bool) (action : KernelM Unit) : KernelM Unit := do
  emitIf cond.id action

/-- Execute one of two blocks depending on a runtime boolean condition. -/
def ifThenElse (cond : KVal Bool) (thenAction elseAction : KernelM Unit) : KernelM Unit := do
  emitIfElse cond.id thenAction elseAction

/-- Read `blockIdx.{x,y,z}` at runtime. -/
def getBlockIdx (axis : Nat := 0) (name : String := "block_idx") : KernelM (KVal UInt32) := do
  let v ← freshVar
  emit (.getBlockIdx v axis)
  pure ⟨v, name⟩

/-- Read `threadIdx.{x,y,z}` at runtime. -/
def getThreadIdx (axis : Nat := 0) (name : String := "thread_idx") : KernelM (KVal UInt32) := do
  let v ← freshVar
  emit (.getThreadIdx v axis)
  pure ⟨v, name⟩

/-- Read `gridDim.{x,y,z}` at runtime. -/
def getGridDim (axis : Nat := 0) (name : String := "grid_dim") : KernelM (KVal UInt32) := do
  let v ← freshVar
  emit (.getGridDim v axis)
  pure ⟨v, name⟩

/-- Read `blockDim.{x,y,z}` at runtime. -/
def getBlockDim (axis : Nat := 0) (name : String := "block_dim") : KernelM (KVal UInt32) := do
  let v ← freshVar
  emit (.getBlockDim v axis)
  pure ⟨v, name⟩

/-! ## Complex Number Operations -/

/-- Allocate a complex register tile -/
def allocCRT (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row)
    : KernelM (CRT dtype rows cols layout) := do
  let realTile ← allocRT dtype rows cols layout
  let imagTile ← allocRT dtype rows cols layout
  pure ⟨realTile, imagTile⟩

/-- Allocate a zero-initialized complex register tile -/
def zeroCRT (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row)
    : KernelM (CRT dtype rows cols layout) := do
  let realTile ← zeroRT dtype rows cols layout
  let imagTile ← zeroRT dtype rows cols layout
  pure ⟨realTile, imagTile⟩

/-- Allocate a complex shared memory tile -/
def allocCST (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row)
    : KernelM (CST dtype rows cols layout) := do
  let realTile ← allocST dtype rows cols layout
  let imagTile ← allocST dtype rows cols layout
  pure ⟨realTile, imagTile⟩

/-- Allocate a complex register vector -/
def allocCRV (dtype : GpuFloat) (len : Nat) : KernelM (CRV dtype len) := do
  let realVec ← allocRV dtype len
  let imagVec ← allocRV dtype len
  pure ⟨realVec, imagVec⟩

/-- Allocate a complex shared vector -/
def allocCSV (dtype : GpuFloat) (len : Nat) : KernelM (CSV dtype len) := do
  let realVec ← allocSV dtype len
  let imagVec ← allocSV dtype len
  pure ⟨realVec, imagVec⟩

/-- Complex addition: (a+bi) + (c+di) = (a+c) + (b+d)i -/
def complexAdd {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst a b : CRT dtype rows cols layout) : KernelM Unit := do
  add dst.real a.real b.real
  add dst.imag a.imag b.imag

/-- Complex subtraction: (a+bi) - (c+di) = (a-c) + (b-d)i -/
def complexSub {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst a b : CRT dtype rows cols layout) : KernelM Unit := do
  sub dst.real a.real b.real
  sub dst.imag a.imag b.imag

/-- Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i -/
def complexMul {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst a b : CRT dtype rows cols layout) : KernelM Unit := do
  -- Need temp tiles for intermediate results
  let ac ← allocRT dtype rows cols layout
  let bd ← allocRT dtype rows cols layout
  let ad ← allocRT dtype rows cols layout
  let bc ← allocRT dtype rows cols layout
  -- ac = a.real * b.real
  mul ac a.real b.real
  -- bd = a.imag * b.imag
  mul bd a.imag b.imag
  -- ad = a.real * b.imag
  mul ad a.real b.imag
  -- bc = a.imag * b.real
  mul bc a.imag b.real
  -- dst.real = ac - bd
  sub dst.real ac bd
  -- dst.imag = ad + bc
  add dst.imag ad bc

/-- Load complex tile from shared to register -/
def loadComplex {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : CRT dtype rows cols layout)
    (src : CST dtype rows cols layout) : KernelM Unit := do
  load dst.real src.real
  load dst.imag src.imag

/-- Store complex tile from register to shared -/
def storeComplex {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : CST dtype rows cols layout)
    (src : CRT dtype rows cols layout) : KernelM Unit := do
  store dst.real src.real
  store dst.imag src.imag

/-- Complex matrix multiply-accumulate: D = A @ B + C
    Uses: (a+bi)(c+di) = (ac-bd) + (ad+bc)i -/
def complexMma {M K N : Nat} {inDtype accDtype : GpuFloat}
    (dst : CRT accDtype M N .Row)
    (a : CRT inDtype M K .Row)
    (b : CRT inDtype K N .Col)
    (c : CRT accDtype M N .Row)
    (hM : M % 16 = 0 := by decide)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hM; let _ := hK; let _ := hN
  -- real part: ac - bd + c_real
  let ac ← allocRT accDtype M N .Row
  mma ac a.real b.real c.real hM hK hN
  let bd ← allocRT accDtype M N .Row
  mma bd a.imag b.imag (← zeroRT accDtype M N .Row) hM hK hN
  sub dst.real ac bd
  -- imag part: ad + bc + c_imag
  let ad ← allocRT accDtype M N .Row
  mma ad a.real b.imag c.imag hM hK hN
  let bc ← allocRT accDtype M N .Row
  mma bc a.imag b.real (← zeroRT accDtype M N .Row) hM hK hN
  add dst.imag ad bc

/-- Complex matrix multiply with B transposed -/
def complexMmaT {M K N : Nat} {inDtype accDtype : GpuFloat}
    (dst : CRT accDtype M N .Row)
    (a : CRT inDtype M K .Row)
    (b : CRT inDtype N K .Row)
    (c : CRT accDtype M N .Row)
    (hM : M % 16 = 0 := by decide)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hM; let _ := hK; let _ := hN
  -- real part: ac - bd + c_real
  let ac ← allocRT accDtype M N .Row
  mmaT ac a.real b.real c.real hM hK hN
  let bd ← allocRT accDtype M N .Row
  mmaT bd a.imag b.imag (← zeroRT accDtype M N .Row) hM hK hN
  sub dst.real ac bd
  -- imag part: ad + bc + c_imag (note: for transposed B, we use -b.imag)
  let ad ← allocRT accDtype M N .Row
  let negBImag ← allocRT inDtype N K .Row
  neg negBImag b.imag
  mmaT ad a.real negBImag c.imag hM hK hN
  let bc ← allocRT accDtype M N .Row
  mmaT bc a.imag b.real (← zeroRT accDtype M N .Row) hM hK hN
  add dst.imag ad bc

/-! ## Additional Complex Operations (ThunderKittens compatibility) -/

/-- Zero both components of a complex tile -/
def complexZero {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : CRT dtype rows cols layout) : KernelM Unit := do
  zero dst.real
  zero dst.imag

/-- Copy complex tile (with optional type conversion) -/
def complexCopy {dstDtype srcDtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : CRT dstDtype rows cols layout)
    (src : CRT srcDtype rows cols layout) : KernelM Unit := do
  convert dst.real src.real
  convert dst.imag src.imag

/-- Negate complex tile: -(a+bi) = -a - bi -/
def complexNeg {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : CRT dtype rows cols layout) : KernelM Unit := do
  neg dst.real src.real
  neg dst.imag src.imag

/-- Complex conjugate: conj(a+bi) = a - bi -/
def complexConj {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : CRT dtype rows cols layout) : KernelM Unit := do
  copy dst.real src.real
  neg dst.imag src.imag

/-- Multiply complex tile by real scalar: c * (a+bi) = ca + cbi -/
def complexScalarMul {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : CRT dtype rows cols layout) (scalar : Float) : KernelM Unit := do
  scalarMul dst.real src.real scalar
  scalarMul dst.imag src.imag scalar

/-- Add real scalar to complex tile: (a+bi) + c = (a+c) + bi -/
def complexScalarAdd {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : CRT dtype rows cols layout) (scalar : Float) : KernelM Unit := do
  scalarAdd dst.real src.real scalar
  copy dst.imag src.imag

/-- Complex division: (a+bi)/(c+di) = (ac+bd)/(c²+d²) + (bc-ad)/(c²+d²)i -/
def complexDiv {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst a b : CRT dtype rows cols layout) : KernelM Unit := do
  -- Compute denominator: c² + d²
  let cc ← allocRT dtype rows cols layout
  let dd ← allocRT dtype rows cols layout
  let denom ← allocRT dtype rows cols layout
  mul cc b.real b.real
  mul dd b.imag b.imag
  add denom cc dd

  -- Real part: (ac + bd) / denom
  let ac ← allocRT dtype rows cols layout
  let bd ← allocRT dtype rows cols layout
  let realNum ← allocRT dtype rows cols layout
  mul ac a.real b.real
  mul bd a.imag b.imag
  add realNum ac bd
  div dst.real realNum denom

  -- Imag part: (bc - ad) / denom
  let bc ← allocRT dtype rows cols layout
  let ad ← allocRT dtype rows cols layout
  let imagNum ← allocRT dtype rows cols layout
  mul bc a.imag b.real
  mul ad a.real b.imag
  sub imagNum bc ad
  div dst.imag imagNum denom

/-- Complex exponential: e^(a+bi) = e^a * (cos(b) + i*sin(b)) -/
def complexExp {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : CRT dtype rows cols layout) : KernelM Unit := do
  -- Compute e^a (real exponential of real part)
  let expA ← allocRT dtype rows cols layout
  exp expA src.real

  -- Compute cos(b) and sin(b)
  let cosB ← allocRT dtype rows cols layout
  let sinB ← allocRT dtype rows cols layout
  cos cosB src.imag
  sin sinB src.imag

  -- dst.real = e^a * cos(b)
  mul dst.real expA cosB
  -- dst.imag = e^a * sin(b)
  mul dst.imag expA sinB

/-- Swap layout of complex tile -/
def complexSwapLayout {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : CRT dtype rows cols (layout.transpose))
    (src : CRT dtype rows cols layout) : KernelM Unit := do
  swapLayout dst.real src.real
  swapLayout dst.imag src.imag

/-- Transpose complex tile -/
def complexTranspose {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : CRT dtype cols rows layout)
    (src : CRT dtype rows cols layout) : KernelM Unit := do
  transpose dst.real src.real
  transpose dst.imag src.imag

/-- Allocate zero-initialized complex register vector -/
def zeroCRV (dtype : GpuFloat) (len : Nat) : KernelM (CRV dtype len) := do
  let realVec ← zeroRV dtype len
  let imagVec ← zeroRV dtype len
  pure ⟨realVec, imagVec⟩

/-- Zero existing complex register vector -/
def complexZeroVec {dtype : GpuFloat} {len : Nat}
    (dst : CRV dtype len) : KernelM Unit := do
  zeroVec dst.real
  zeroVec dst.imag

/-- Load complex vector from shared to register -/
def loadComplexVec {dtype : GpuFloat} {len : Nat}
    (dst : CRV dtype len)
    (src : CSV dtype len) : KernelM Unit := do
  loadVec dst.real src.real
  loadVec dst.imag src.imag

/-- Store complex vector from register to shared -/
def storeComplexVec {dtype : GpuFloat} {len : Nat}
    (dst : CSV dtype len)
    (src : CRV dtype len) : KernelM Unit := do
  storeVec dst.real src.real
  storeVec dst.imag src.imag

/-- Complex vector addition -/
def complexAddVec {dtype : GpuFloat} {len : Nat}
    (dst a b : CRV dtype len) : KernelM Unit := do
  addVec dst.real a.real b.real
  addVec dst.imag a.imag b.imag

/-- Complex vector multiplication -/
def complexMulVec {dtype : GpuFloat} {len : Nat}
    (dst a b : CRV dtype len) : KernelM Unit := do
  let ac ← allocRV dtype len
  let bd ← allocRV dtype len
  let ad ← allocRV dtype len
  let bc ← allocRV dtype len
  mulVec ac a.real b.real
  mulVec bd a.imag b.imag
  mulVec ad a.real b.imag
  mulVec bc a.imag b.real
  subVec dst.real ac bd
  addVec dst.imag ad bc

end Tyr.GPU.Codegen
