/-
  Tyr/GPU/Codegen/Notation.lean

  Expression-level DSL notation for GPU kernel operations.
  Provides symbolic operators (@, @ᵀ, +=, etc.) alongside named function equivalents.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.AST
import Tyr.GPU.Codegen.Ops

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-! ## Matrix Multiply Operations (Returning Allocated Result)

These functions allocate the result tile internally and return it,
enabling expression-style syntax: `let c ← a @ b`
-/

/-- Matrix multiply returning newly allocated result: C = A @ B
    Accumulator type defaults to Float32 for bf16/f16 inputs -/
def matmul {M K N : Nat} {inDtype : GpuFloat}
    (a : RT inDtype M K .Row)
    (b : RT inDtype K N .Col)
    (accDtype : GpuFloat := .Float32)
    : KernelM (RT accDtype M N .Row) := do
  let c ← allocRT accDtype M N .Row
  mm c a b
  pure c

/-- Matrix multiply-accumulate returning result: D = A @ B + C -/
def matmulAccum {M K N : Nat} {inDtype accDtype : GpuFloat}
    (a : RT inDtype M K .Row)
    (b : RT inDtype K N .Col)
    (c : RT accDtype M N .Row)
    : KernelM (RT accDtype M N .Row) := do
  let d ← allocRT accDtype M N .Row
  mma d a b c
  pure d

/-- Matrix multiply with B transposed: C = A @ B^T -/
def matmulT {M K N : Nat} {inDtype : GpuFloat}
    (a : RT inDtype M K .Row)
    (b : RT inDtype N K .Row)
    (accDtype : GpuFloat := .Float32)
    : KernelM (RT accDtype M N .Row) := do
  let c ← allocRT accDtype M N .Row
  let zero ← zeroRT accDtype M N .Row
  mmaT c a b zero
  pure c

/-- Matrix multiply-accumulate with B transposed: D = A @ B^T + C -/
def matmulTAccum {M K N : Nat} {inDtype accDtype : GpuFloat}
    (a : RT inDtype M K .Row)
    (b : RT inDtype N K .Row)
    (c : RT accDtype M N .Row)
    : KernelM (RT accDtype M N .Row) := do
  let d ← allocRT accDtype M N .Row
  mmaT d a b c
  pure d

/-- Matrix multiply with A transposed: C = A^T @ B -/
def matmulAt {M K N : Nat} {inDtype : GpuFloat}
    (a : RT inDtype K M .Col)
    (b : RT inDtype K N .Col)
    (accDtype : GpuFloat := .Float32)
    : KernelM (RT accDtype M N .Row) := do
  let c ← allocRT accDtype M N .Row
  let zero ← zeroRT accDtype M N .Row
  mmaAtB c a b zero
  pure c

/-! ## Operator Notation

Symbolic operators for matrix operations.
Use in do-notation: `let c ← a ⬝ b`
-/

-- Matrix multiply: A @ B (using ⬝ to avoid conflict with Lean's @)
-- Note: @ is reserved in Lean, so we use ⬝ (U+2B1D BLACK MEDIUM DIAMOND)

/-- Matrix multiply with default Float32 accumulator -/
def matmulF32 {M K N : Nat} {inDtype : GpuFloat}
    (a : RT inDtype M K .Row)
    (b : RT inDtype K N .Col)
    : KernelM (RT GpuFloat.Float32 M N .Row) :=
  matmul a b .Float32

/-- Matrix multiply transposed with default Float32 accumulator -/
def matmulTF32 {M K N : Nat} {inDtype : GpuFloat}
    (a : RT inDtype M K .Row)
    (b : RT inDtype N K .Row)
    : KernelM (RT GpuFloat.Float32 M N .Row) :=
  matmulT a b .Float32

-- Simple infix notation for matrix multiply
scoped infixl:70 " ⬝ " => matmulF32

-- Matrix multiply with B transposed
scoped infixl:70 " ⬝ᵀ " => matmulTF32

/-! ## In-Place Operations

These modify the destination tile directly.
-/

/-- In-place add: dst += src -/
def addInPlace {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  add dst dst src

/-- In-place subtract: dst -= src -/
def subInPlace {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  sub dst dst src

/-- In-place multiply: dst *= src -/
def mulInPlace {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  mul dst dst src

/-- In-place divide: dst /= src -/
def divInPlace {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : KernelM Unit := do
  div dst dst src

/-! ## Broadcasting Operations with Expression Syntax

These return newly allocated tiles, enabling expression chains.
-/

/-- Subtract column vector from tile (broadcast): result[i,j] = tile[i,j] - vec[i] -/
def subBroadcastCol {dtype : GpuFloat} {rows cols : Nat}
    (tile : RT dtype rows cols .Row)
    (vec : RV dtype rows)
    : KernelM (RT dtype rows cols .Row) := do
  let result ← allocRT dtype rows cols .Row
  subCol result tile vec
  pure result

/-- Divide tile by column vector (broadcast): result[i,j] = tile[i,j] / vec[i] -/
def divBroadcastCol {dtype : GpuFloat} {rows cols : Nat}
    (tile : RT dtype rows cols .Row)
    (vec : RV dtype rows)
    : KernelM (RT dtype rows cols .Row) := do
  let result ← allocRT dtype rows cols .Row
  divCol result tile vec
  pure result

/-- Multiply tile by column vector (broadcast): result[i,j] = tile[i,j] * vec[i] -/
def mulBroadcastCol {dtype : GpuFloat} {rows cols : Nat}
    (tile : RT dtype rows cols .Row)
    (vec : RV dtype rows)
    : KernelM (RT dtype rows cols .Row) := do
  let result ← allocRT dtype rows cols .Row
  mulCol result tile vec
  pure result

/-- Add column vector to tile (broadcast): result[i,j] = tile[i,j] + vec[i] -/
def addBroadcastCol {dtype : GpuFloat} {rows cols : Nat}
    (tile : RT dtype rows cols .Row)
    (vec : RV dtype rows)
    : KernelM (RT dtype rows cols .Row) := do
  let result ← allocRT dtype rows cols .Row
  addCol result tile vec
  pure result

/-- Subtract row vector from tile (broadcast): result[i,j] = tile[i,j] - vec[j] -/
def subBroadcastRow {dtype : GpuFloat} {rows cols : Nat}
    (tile : RT dtype rows cols .Row)
    (vec : RV dtype cols)
    : KernelM (RT dtype rows cols .Row) := do
  let result ← allocRT dtype rows cols .Row
  subRow result tile vec
  pure result

/-- Divide tile by row vector (broadcast): result[i,j] = tile[i,j] / vec[j] -/
def divBroadcastRow {dtype : GpuFloat} {rows cols : Nat}
    (tile : RT dtype rows cols .Row)
    (vec : RV dtype cols)
    : KernelM (RT dtype rows cols .Row) := do
  let result ← allocRT dtype rows cols .Row
  divRow result tile vec
  pure result

/-- Multiply tile by row vector (broadcast): result[i,j] = tile[i,j] * vec[j] -/
def mulBroadcastRow {dtype : GpuFloat} {rows cols : Nat}
    (tile : RT dtype rows cols .Row)
    (vec : RV dtype cols)
    : KernelM (RT dtype rows cols .Row) := do
  let result ← allocRT dtype rows cols .Row
  mulRow result tile vec
  pure result

/-- Add row vector to tile (broadcast): result[i,j] = tile[i,j] + vec[j] -/
def addBroadcastRow {dtype : GpuFloat} {rows cols : Nat}
    (tile : RT dtype rows cols .Row)
    (vec : RV dtype cols)
    : KernelM (RT dtype rows cols .Row) := do
  let result ← allocRT dtype rows cols .Row
  addRow result tile vec
  pure result

/-! ## Reduction Operations with Expression Syntax

These return newly allocated vectors with reduction results.
-/

/-- Row-wise maximum: result[i] = max_j(tile[i,j]) -/
def reduceRowMax {dtype : GpuFloat} {rows cols : Nat}
    (tile : RT dtype rows cols .Row)
    : KernelM (RV dtype rows) := do
  let result ← allocRV dtype rows
  rowMax result tile
  pure result

/-- Row-wise sum: result[i] = sum_j(tile[i,j]) -/
def reduceRowSum {dtype : GpuFloat} {rows cols : Nat}
    (tile : RT dtype rows cols .Row)
    : KernelM (RV dtype rows) := do
  let result ← allocRV dtype rows
  rowSum result tile
  pure result

/-- Row-wise minimum: result[i] = min_j(tile[i,j]) -/
def reduceRowMin {dtype : GpuFloat} {rows cols : Nat}
    (tile : RT dtype rows cols .Row)
    : KernelM (RV dtype rows) := do
  let result ← allocRV dtype rows
  rowMin result tile
  pure result

/-- Column-wise maximum: result[j] = max_i(tile[i,j]) -/
def reduceColMax {dtype : GpuFloat} {rows cols : Nat}
    (tile : RT dtype rows cols .Row)
    : KernelM (RV dtype cols) := do
  let result ← allocRV dtype cols
  colMax result tile
  pure result

/-- Column-wise sum: result[j] = sum_i(tile[i,j]) -/
def reduceColSum {dtype : GpuFloat} {rows cols : Nat}
    (tile : RT dtype rows cols .Row)
    : KernelM (RV dtype cols) := do
  let result ← allocRV dtype cols
  colSum result tile
  pure result

/-- Column-wise minimum: result[j] = min_i(tile[i,j]) -/
def reduceColMin {dtype : GpuFloat} {rows cols : Nat}
    (tile : RT dtype rows cols .Row)
    : KernelM (RV dtype cols) := do
  let result ← allocRV dtype cols
  colMin result tile
  pure result

/-! ## Element-wise Unary Operations (Expression Style)

These return newly allocated tiles with the operation applied.
-/

/-- Element-wise exponential -/
def expTile {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (src : RT dtype rows cols layout)
    : KernelM (RT dtype rows cols layout) := do
  let result ← allocRT dtype rows cols layout
  exp result src
  pure result

/-- Element-wise log -/
def logTile {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (src : RT dtype rows cols layout)
    : KernelM (RT dtype rows cols layout) := do
  let result ← allocRT dtype rows cols layout
  log result src
  pure result

/-- Element-wise sqrt -/
def sqrtTile {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (src : RT dtype rows cols layout)
    : KernelM (RT dtype rows cols layout) := do
  let result ← allocRT dtype rows cols layout
  sqrt result src
  pure result

/-- Element-wise tanh -/
def tanhTile {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (src : RT dtype rows cols layout)
    : KernelM (RT dtype rows cols layout) := do
  let result ← allocRT dtype rows cols layout
  tanh result src
  pure result

/-- Element-wise sigmoid -/
def sigmoidTile {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (src : RT dtype rows cols layout)
    : KernelM (RT dtype rows cols layout) := do
  let result ← allocRT dtype rows cols layout
  sigmoid result src
  pure result

/-- Element-wise ReLU -/
def reluTile {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (src : RT dtype rows cols layout)
    : KernelM (RT dtype rows cols layout) := do
  let result ← allocRT dtype rows cols layout
  relu result src
  pure result

/-- Element-wise GELU -/
def geluTile {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (src : RT dtype rows cols layout)
    : KernelM (RT dtype rows cols layout) := do
  let result ← allocRT dtype rows cols layout
  gelu result src
  pure result

/-- Element-wise negation -/
def negTile {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (src : RT dtype rows cols layout)
    : KernelM (RT dtype rows cols layout) := do
  let result ← allocRT dtype rows cols layout
  neg result src
  pure result

/-- Element-wise reciprocal (1/x) -/
def recipTile {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (src : RT dtype rows cols layout)
    : KernelM (RT dtype rows cols layout) := do
  let result ← allocRT dtype rows cols layout
  recip result src
  pure result

/-! ## Element-wise Binary Operations (Expression Style)

These return newly allocated tiles.
-/

/-- Element-wise add returning new tile -/
def addTiles {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (a b : RT dtype rows cols layout)
    : KernelM (RT dtype rows cols layout) := do
  let result ← allocRT dtype rows cols layout
  add result a b
  pure result

/-- Element-wise subtract returning new tile -/
def subTiles {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (a b : RT dtype rows cols layout)
    : KernelM (RT dtype rows cols layout) := do
  let result ← allocRT dtype rows cols layout
  sub result a b
  pure result

/-- Element-wise multiply returning new tile -/
def mulTiles {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (a b : RT dtype rows cols layout)
    : KernelM (RT dtype rows cols layout) := do
  let result ← allocRT dtype rows cols layout
  mul result a b
  pure result

/-- Element-wise divide returning new tile -/
def divTiles {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (a b : RT dtype rows cols layout)
    : KernelM (RT dtype rows cols layout) := do
  let result ← allocRT dtype rows cols layout
  div result a b
  pure result

/-- Element-wise max returning new tile -/
def maxTiles {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (a b : RT dtype rows cols layout)
    : KernelM (RT dtype rows cols layout) := do
  let result ← allocRT dtype rows cols layout
  max result a b
  pure result

/-- Element-wise min returning new tile -/
def minTiles {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (a b : RT dtype rows cols layout)
    : KernelM (RT dtype rows cols layout) := do
  let result ← allocRT dtype rows cols layout
  min result a b
  pure result

/-! ## Scalar Operations (Expression Style) -/

/-- Scalar multiply returning new tile -/
def scalarMulTile {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (src : RT dtype rows cols layout) (scalar : Float)
    : KernelM (RT dtype rows cols layout) := do
  let result ← allocRT dtype rows cols layout
  scalarMul result src scalar
  pure result

/-- Scalar add returning new tile -/
def scalarAddTile {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (src : RT dtype rows cols layout) (scalar : Float)
    : KernelM (RT dtype rows cols layout) := do
  let result ← allocRT dtype rows cols layout
  scalarAdd result src scalar
  pure result

/-! ## Infix Notation for Tile Operations

These work within do-notation as `let c ← a + b`
-/

-- Element-wise operations
scoped instance {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout} :
    HAdd (RT dtype rows cols layout) (RT dtype rows cols layout)
         (KernelM (RT dtype rows cols layout)) where
  hAdd := addTiles

scoped instance {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout} :
    HSub (RT dtype rows cols layout) (RT dtype rows cols layout)
         (KernelM (RT dtype rows cols layout)) where
  hSub := subTiles

scoped instance {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout} :
    HMul (RT dtype rows cols layout) (RT dtype rows cols layout)
         (KernelM (RT dtype rows cols layout)) where
  hMul := mulTiles

scoped instance {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout} :
    HDiv (RT dtype rows cols layout) (RT dtype rows cols layout)
         (KernelM (RT dtype rows cols layout)) where
  hDiv := divTiles

-- Scalar operations
scoped instance {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout} :
    HMul (RT dtype rows cols layout) Float (KernelM (RT dtype rows cols layout)) where
  hMul := scalarMulTile

scoped instance {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout} :
    HMul Float (RT dtype rows cols layout) (KernelM (RT dtype rows cols layout)) where
  hMul f t := scalarMulTile t f

scoped instance {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout} :
    HAdd (RT dtype rows cols layout) Float (KernelM (RT dtype rows cols layout)) where
  hAdd := scalarAddTile

scoped instance {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout} :
    HAdd Float (RT dtype rows cols layout) (KernelM (RT dtype rows cols layout)) where
  hAdd f t := scalarAddTile t f

/-! ## Masking Operations (Expression Style) -/

/-- Apply causal mask returning new tile -/
def causalMask {dtype : GpuFloat} {rows cols : Nat}
    (src : RT dtype rows cols .Row)
    (fillVal : Float := -1e10)
    : KernelM (RT dtype rows cols .Row) := do
  let result ← allocRT dtype rows cols .Row
  makeCausal result src (some fillVal)
  pure result

/-- Apply transposed causal mask (for backward pass) -/
def causalMaskT {dtype : GpuFloat} {rows cols : Nat}
    (src : RT dtype rows cols .Row)
    (fillVal : Float := -1e10)
    : KernelM (RT dtype rows cols .Row) := do
  let result ← allocRT dtype rows cols .Row
  makeCausalT result src (some fillVal)
  pure result

/-! ## Type Conversion (Expression Style) -/

/-- Convert tile dtype returning new tile -/
def convertTile {srcDtype dstDtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (src : RT srcDtype rows cols layout)
    : KernelM (RT dstDtype rows cols layout) := do
  let result ← allocRT dstDtype rows cols layout
  convert result src
  pure result

/-- Transpose tile returning new tile -/
def transposeTile {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (src : RT dtype rows cols layout)
    : KernelM (RT dtype cols rows layout) := do
  let result ← allocRT dtype cols rows layout
  transpose result src
  pure result

/-! ## Vector Operations (Expression Style) -/

/-- Vector addition returning new vector -/
def addVecs {dtype : GpuFloat} {len : Nat}
    (a b : RV dtype len)
    : KernelM (RV dtype len) := do
  let result ← allocRV dtype len
  addVec result a b
  pure result

/-- Vector subtraction returning new vector -/
def subVecs {dtype : GpuFloat} {len : Nat}
    (a b : RV dtype len)
    : KernelM (RV dtype len) := do
  let result ← allocRV dtype len
  subVec result a b
  pure result

/-- Vector multiplication returning new vector -/
def mulVecs {dtype : GpuFloat} {len : Nat}
    (a b : RV dtype len)
    : KernelM (RV dtype len) := do
  let result ← allocRV dtype len
  mulVec result a b
  pure result

/-- Vector division returning new vector -/
def divVecs {dtype : GpuFloat} {len : Nat}
    (a b : RV dtype len)
    : KernelM (RV dtype len) := do
  let result ← allocRV dtype len
  divVec result a b
  pure result

/-- Vector exp returning new vector -/
def expVector {dtype : GpuFloat} {len : Nat}
    (src : RV dtype len)
    : KernelM (RV dtype len) := do
  let result ← allocRV dtype len
  expVec result src
  pure result

/-- Vector log returning new vector -/
def logVector {dtype : GpuFloat} {len : Nat}
    (src : RV dtype len)
    : KernelM (RV dtype len) := do
  let result ← allocRV dtype len
  logVec result src
  pure result

-- Vector infix operations
scoped instance {dtype : GpuFloat} {len : Nat} :
    HAdd (RV dtype len) (RV dtype len) (KernelM (RV dtype len)) where
  hAdd := addVecs

scoped instance {dtype : GpuFloat} {len : Nat} :
    HSub (RV dtype len) (RV dtype len) (KernelM (RV dtype len)) where
  hSub := subVecs

scoped instance {dtype : GpuFloat} {len : Nat} :
    HMul (RV dtype len) (RV dtype len) (KernelM (RV dtype len)) where
  hMul := mulVecs

scoped instance {dtype : GpuFloat} {len : Nat} :
    HDiv (RV dtype len) (RV dtype len) (KernelM (RV dtype len)) where
  hDiv := divVecs

end Tyr.GPU.Codegen
