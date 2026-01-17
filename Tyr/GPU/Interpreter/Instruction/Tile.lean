/-
  Tyr/GPU/Interpreter/Instruction/Tile.lean

  Tile-level operations (ThunderKittens style).
  Low-level instructions operating on individual tiles.
-/
import Tyr.GPU.Interpreter.Instruction.Base
import Tyr.GPU.Interpreter.Config

namespace Tyr.GPU.Interpreter

open Tyr.GPU

/-- Tile dimensions -/
structure TileDim where
  rows : Nat
  cols : Nat
  deriving Repr, BEq, Hashable, Inhabited

/-- Memory location for tile data -/
inductive MemLoc where
  /-- Global memory with byte offset -/
  | global (offset : Nat)
  /-- Shared memory buffer with id and offset -/
  | shared (bufferId : Nat) (offset : Nat)
  /-- Register file with register id -/
  | register (regId : Nat)
  deriving Repr, BEq, Hashable, Inhabited

/-- Reduction operation types -/
inductive ReduceOp where
  | sum
  | max
  | min
  | prod
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq

/-- Elementwise operation types -/
inductive ElementOp where
  | add
  | sub
  | mul
  | div
  | neg
  | abs
  | exp
  | log
  | sqrt
  | rsqrt
  | tanh
  | sigmoid
  | silu      -- x * sigmoid(x)
  | gelu
  | relu
  | softmax
  | rmsNorm
  | layerNorm
  | rope      -- Rotary Position Embedding
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq

/-- GEMM accumulation mode -/
inductive GemmAccum where
  | overwrite  -- C = A @ B
  | add        -- C += A @ B
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq

/-- Tile operations (ThunderKittens level) -/
inductive TileOp where
  /-- Load tile from src to dst -/
  | load (src : MemLoc) (dst : MemLoc) (dim : TileDim) (dtype : GpuFloat := .BFloat16)
  /-- Store tile from src to dst -/
  | store (src : MemLoc) (dst : MemLoc) (dim : TileDim) (dtype : GpuFloat := .BFloat16)
  /-- Matrix multiply: C = A @ B or C += A @ B -/
  | gemm (a b c : MemLoc) (m n k : Nat) (accum : GemmAccum := .overwrite)
  /-- Reduce along axis -/
  | reduce (src dst : MemLoc) (dim : TileDim) (axis : Nat) (op : ReduceOp)
  /-- Elementwise operation: dst = op(srcs...) -/
  | elementwise (srcs : List MemLoc) (dst : MemLoc) (dim : TileDim) (op : ElementOp)
  /-- Copy tile -/
  | copy (src dst : MemLoc) (dim : TileDim)
  /-- Transpose tile -/
  | transpose (src dst : MemLoc) (dim : TileDim)
  /-- Barrier synchronization -/
  | barrier (scope : BarrierScope)
  /-- No operation -/
  | nop
  deriving Repr, BEq, Inhabited

namespace TileOp

/-- Opcode for each tile operation -/
def opcode : TileOp → Nat
  | .load .. => 1
  | .store .. => 2
  | .gemm .. => 3
  | .reduce .. => 4
  | .elementwise .. => 5
  | .copy .. => 6
  | .transpose .. => 7
  | .barrier .. => 8
  | .nop => 0

/-- Serialize memory location -/
private def serializeMemLoc : MemLoc → Array UInt32
  | .global offset => #[0, natToUInt32 offset]
  | .shared bufferId offset => #[1, natToUInt32 bufferId, natToUInt32 offset]
  | .register regId => #[2, natToUInt32 regId]

/-- Serialize tile dimensions -/
private def serializeTileDim (dim : TileDim) : Array UInt32 :=
  #[natToUInt32 dim.rows, natToUInt32 dim.cols]

/-- Serialize reduce operation -/
private def serializeReduceOp : ReduceOp → UInt32
  | .sum => 0
  | .max => 1
  | .min => 2
  | .prod => 3

/-- Serialize element operation -/
private def serializeElementOp : ElementOp → UInt32
  | .add => 0
  | .sub => 1
  | .mul => 2
  | .div => 3
  | .neg => 4
  | .abs => 5
  | .exp => 6
  | .log => 7
  | .sqrt => 8
  | .rsqrt => 9
  | .tanh => 10
  | .sigmoid => 11
  | .silu => 12
  | .gelu => 13
  | .relu => 14
  | .softmax => 15
  | .rmsNorm => 16
  | .layerNorm => 17
  | .rope => 18

/-- Serialize a tile operation -/
def serialize (op : TileOp) : Array UInt32 :=
  let fields := match op with
    | .load src dst dim _ =>
      serializeMemLoc src ++ serializeMemLoc dst ++ serializeTileDim dim
    | .store src dst dim _ =>
      serializeMemLoc src ++ serializeMemLoc dst ++ serializeTileDim dim
    | .gemm a b c m n k _ =>
      serializeMemLoc a ++ serializeMemLoc b ++ serializeMemLoc c ++
      #[natToUInt32 m, natToUInt32 n, natToUInt32 k]
    | .reduce src dst dim axis reduceOp =>
      serializeMemLoc src ++ serializeMemLoc dst ++ serializeTileDim dim ++
      #[natToUInt32 axis, serializeReduceOp reduceOp]
    | .elementwise srcs dst dim elemOp =>
      let srcArray := srcs.foldl (· ++ serializeMemLoc ·) #[natToUInt32 srcs.length]
      srcArray ++ serializeMemLoc dst ++ serializeTileDim dim ++ #[serializeElementOp elemOp]
    | .copy src dst dim =>
      serializeMemLoc src ++ serializeMemLoc dst ++ serializeTileDim dim
    | .transpose src dst dim =>
      serializeMemLoc src ++ serializeMemLoc dst ++ serializeTileDim dim
    | .barrier scope =>
      #[match scope with | .warp => 0 | .block => 1 | .grid => 2 | .named => 3]
    | .nop => #[]
  serializeWithOpcode op.opcode fields

/-- Estimated cost of tile operation (for scheduling) -/
def cost (op : TileOp) : Float :=
  match op with
  | .load _ _ dim _ => (dim.rows * dim.cols).toFloat * 2.0  -- Memory bound
  | .store _ _ dim _ => (dim.rows * dim.cols).toFloat * 2.0
  | .gemm _ _ _ m n k _ => (m * n * k).toFloat * 2.0  -- FLOPs
  | .reduce _ _ dim _ _ => (dim.rows * dim.cols).toFloat
  | .elementwise srcs _ dim _ => (dim.rows * dim.cols * srcs.length).toFloat
  | .copy _ _ dim => (dim.rows * dim.cols).toFloat
  | .transpose _ _ dim => (dim.rows * dim.cols).toFloat * 1.5
  | .barrier _ => 10.0  -- Sync cost
  | .nop => 0.0

/-- Resource pool for scheduling -/
def pool : TileOp → ResourcePool
  | .load .. => .memory
  | .store .. => .memory
  | .gemm .. => .compute
  | .reduce .. => .compute
  | .elementwise .. => .compute
  | .copy .. => .memory
  | .transpose .. => .mixed
  | .barrier .. => .mixed
  | .nop => .mixed

end TileOp

instance : GpuInstruction TileOp where
  level := .tile
  opcode := 0  -- Varies by operation
  serialize := TileOp.serialize
  cost := TileOp.cost
  tags := { pool := .mixed }

/-- Get the actual opcode for a specific TileOp value -/
def TileOp.getOpcode (op : TileOp) : Nat := op.opcode

end Tyr.GPU.Interpreter
