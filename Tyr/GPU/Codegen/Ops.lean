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

/-- Semaphore type (barrier) for async operations -/
structure Semaphore where
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

/-- Async store from shared to global (TMA).
    Emits: kittens::tma::store_async(dst, src, coord) -/
def storeGlobalAsyncCoord {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : GPtr dtype)
    (src : ST dtype rows cols layout)
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

/-- Initialize semaphore with count -/
def initSemaphore (sem : Semaphore) (count : Nat := 1) : KernelM Unit := do
  emit (.semaphore (.Init count) sem.id)

/-- Invalidate semaphore -/
def invalidateSemaphore (sem : Semaphore) : KernelM Unit := do
  emit (.semaphore .Invalidate sem.id)

/-- Expect bytes on semaphore -/
def expectBytes (sem : Semaphore) (bytes : Nat) : KernelM Unit := do
  emit (.semaphore (.Expect bytes) sem.id)

/-- Wait on semaphore -/
def waitSemaphore (sem : Semaphore) : KernelM Unit := do
  emit (.semaphore .Wait sem.id)

/-- Arrive at semaphore with transaction count -/
def arriveSemaphore (sem : Semaphore) (count : Nat := 1) : KernelM Unit := do
  emit (.semaphore (.Arrive count) sem.id)

/-- Arrive and wait at semaphore -/
def arriveAndWait (barrier : Nat := 0) : KernelM Unit := do
  emit (.arriveAndWait barrier)

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
  -- Capture the body statements
  let startLen := (← get).body.size
  action
  let endLen := (← get).body.size
  let bodyStmts := (← get).body.extract startLen endLen
  -- Replace with ifWarpGroup construct
  modify fun s => { s with body := s.body.extract 0 startLen |>.push (.ifWarpGroup wgIdx bodyStmts) }

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
    : KernelM Unit := do
  -- real part: ac - bd + c_real
  let ac ← allocRT accDtype M N .Row
  mma ac a.real b.real c.real
  let bd ← allocRT accDtype M N .Row
  mma bd a.imag b.imag (← zeroRT accDtype M N .Row)
  sub dst.real ac bd
  -- imag part: ad + bc + c_imag
  let ad ← allocRT accDtype M N .Row
  mma ad a.real b.imag c.imag
  let bc ← allocRT accDtype M N .Row
  mma bc a.imag b.real (← zeroRT accDtype M N .Row)
  add dst.imag ad bc

/-- Complex matrix multiply with B transposed -/
def complexMmaT {M K N : Nat} {inDtype accDtype : GpuFloat}
    (dst : CRT accDtype M N .Row)
    (a : CRT inDtype M K .Row)
    (b : CRT inDtype N K .Row)
    (c : CRT accDtype M N .Row)
    : KernelM Unit := do
  -- real part: ac - bd + c_real
  let ac ← allocRT accDtype M N .Row
  mmaT ac a.real b.real c.real
  let bd ← allocRT accDtype M N .Row
  mmaT bd a.imag b.imag (← zeroRT accDtype M N .Row)
  sub dst.real ac bd
  -- imag part: ad + bc + c_imag (note: for transposed B, we use -b.imag)
  let ad ← allocRT accDtype M N .Row
  let negBImag ← allocRT inDtype N K .Row
  neg negBImag b.imag
  mmaT ad a.real negBImag c.imag
  let bc ← allocRT accDtype M N .Row
  mmaT bc a.imag b.real (← zeroRT accDtype M N .Row)
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
