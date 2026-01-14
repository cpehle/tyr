/-
  Tyr/GPU/Codegen/Ops.lean

  Type-safe GPU kernel operations.
  Dimensions are checked at Lean compile time via dependent types.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.AST

namespace Tyr.GPU.Codegen

open Tyr.GPU

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

/-- Allocate a register vector -/
def allocRV (dtype : GpuFloat) (len : Nat) : KernelM (RV dtype len) := do
  let v ← freshVar
  emit (.declRV v dtype len)
  pure ⟨v⟩

/-- Allocate a shared vector -/
def allocSV (dtype : GpuFloat) (len : Nat) : KernelM (SV dtype len) := do
  let v ← freshVar
  emit (.declSV v dtype len)
  let bytes := len * dtype.bytes
  modify fun s => { s with sharedMemBytes := s.sharedMemBytes + bytes }
  pure ⟨v⟩

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

/-! ## Vector Unary/Binary Operations -/

/-- Copy vector -/
def copyVec {dtype : GpuFloat} {len : Nat}
    (dst src : RV dtype len) : KernelM Unit := do
  emit (.unary .Copy dst.id src.id)

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

/-- Vector exp -/
def expVec {dtype : GpuFloat} {len : Nat}
    (dst src : RV dtype len) : KernelM Unit := do
  emit (.unary .Exp dst.id src.id)

/-- Vector log -/
def logVec {dtype : GpuFloat} {len : Nat}
    (dst src : RV dtype len) : KernelM Unit := do
  emit (.unary .Log dst.id src.id)

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

/-! ## Matrix Multiply -/

/-- Matrix multiply-accumulate: D = A @ B + C
    Type system enforces dimensions: A is M×K, B is K×N, C and D are M×N
    Input dtype (bf16/f16) can differ from accumulator dtype (f32) -/
def mma {M K N : Nat} {inDtype accDtype : GpuFloat}
    (dst : RT accDtype M N .Row)
    (a : RT inDtype M K .Row)
    (b : RT inDtype K N .Col)
    (c : RT accDtype M N .Row)
    : KernelM Unit := do
  emit (.mma .AB dst.id a.id b.id c.id)

/-- Matrix multiply without accumulate: D = A @ B -/
def mm {M K N : Nat} {inDtype accDtype : GpuFloat}
    (dst : RT accDtype M N .Row)
    (a : RT inDtype M K .Row)
    (b : RT inDtype K N .Col)
    : KernelM Unit := do
  emit (.mm .AB dst.id a.id b.id)

/-- Matrix multiply with B transposed: D = A @ B^T + C
    A is M×K, B is N×K (stored row-major, used transposed) -/
def mmaT {M K N : Nat} {inDtype accDtype : GpuFloat}
    (dst : RT accDtype M N .Row)
    (a : RT inDtype M K .Row)
    (b : RT inDtype N K .Row)
    (c : RT accDtype M N .Row)
    : KernelM Unit := do
  emit (.mma .ABt dst.id a.id b.id c.id)

/-- Matrix multiply with both transposed: D = A^T @ B^T + C -/
def mmaAtBt {M K N : Nat} {inDtype accDtype : GpuFloat}
    (dst : RT accDtype M N .Row)
    (a : RT inDtype K M .Row)
    (b : RT inDtype N K .Row)
    (c : RT accDtype M N .Row)
    : KernelM Unit := do
  emit (.mma .AtBt dst.id a.id b.id c.id)

/-- Matrix multiply with A transposed: D = A^T @ B + C -/
def mmaAtB {M K N : Nat} {inDtype accDtype : GpuFloat}
    (dst : RT accDtype M N .Row)
    (a : RT inDtype K M .Col)
    (b : RT inDtype K N .Col)
    (c : RT accDtype M N .Row)
    : KernelM Unit := do
  emit (.mma .AtB dst.id a.id b.id c.id)

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

end Tyr.GPU.Codegen
