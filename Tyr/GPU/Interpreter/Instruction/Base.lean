/-
  Tyr/GPU/Interpreter/Instruction/Base.lean

  Base typeclass for GPU interpreter instructions.
  Defines the interface that all instructions must implement.
-/
import Tyr.GPU.Types

namespace Tyr.GPU.Interpreter

open Tyr.GPU

/-- Instruction level: tile ops (low-level) vs fused ops (high-level) -/
inductive InstructionLevel where
  /-- Low-level: single tile operation (ThunderKittens style) -/
  | tile
  /-- High-level: fused multi-step operation (Megakernels style) -/
  | fused
  /-- Control: barriers, synchronization, control flow -/
  | control
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq

/-- Resource pool for scheduling -/
inductive ResourcePool where
  /-- Memory-bound operations (loads, stores) -/
  | memory
  /-- Compute-bound operations (GEMM, reductions) -/
  | compute
  /-- Mixed or undetermined -/
  | mixed
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq

/-- Tags for instruction metadata -/
structure InstructionTags where
  /-- Resource pool for scheduling -/
  pool : ResourcePool := .mixed
  /-- Whether instruction requires synchronization after -/
  needsSync : Bool := false
  /-- Custom tags -/
  custom : List (String × String) := []
  deriving Repr, Inhabited

/-- Globals state - tracks model parameters and buffers.
    This is a minimal interface; concrete implementations will extend it. -/
class Globals (G : Type) where
  /-- Number of SMs on the device -/
  smCount : G → Nat
  /-- Device identifier -/
  device : G → String

/-- Base instruction typeclass.
    All GPU interpreter instructions must implement this. -/
class GpuInstruction (α : Type) where
  /-- Instruction level (tile vs fused) -/
  level : InstructionLevel
  /-- Unique opcode for dispatch -/
  opcode : Nat
  /-- Serialize instruction to array of UInt32 words -/
  serialize : α → Array UInt32
  /-- Estimated cost for scheduling (higher = more expensive) -/
  cost : α → Float
  /-- Previous opcode this instruction depends on (for implicit sequencing) -/
  prevOpcode : Option Nat := none
  /-- Instruction tags for scheduling hints -/
  tags : InstructionTags := {}

/-- Composable instructions can be decomposed into simpler ops -/
class ComposableInstruction (α : Type) extends GpuInstruction α where
  /-- Decompose into a list of sub-instructions -/
  decompose : α → List α

/-- No-op instruction (opcode 0) -/
structure NoOp where
  deriving Repr, BEq, Inhabited

instance : GpuInstruction NoOp where
  level := .control
  opcode := 0
  serialize _ := #[0]
  cost _ := 0.0

/-- Serialize a Nat to UInt32 -/
def natToUInt32 (n : Nat) : UInt32 := n.toUInt32

/-- Serialize a list of Nats to UInt32 array with length prefix -/
def serializeNatList (ns : List Nat) : Array UInt32 :=
  #[natToUInt32 ns.length] ++ (ns.map natToUInt32).toArray

/-- Serialize an optional Nat (0 for none) -/
def serializeOptionNat (opt : Option Nat) : UInt32 :=
  match opt with
  | none => 0
  | some n => natToUInt32 n

/-- Pad serialized instruction to fixed width -/
def padToWidth (words : Array UInt32) (width : Nat) : Array UInt32 :=
  if words.size >= width then words
  else words ++ (List.replicate (width - words.size) (0 : UInt32)).toArray

/-- Standard serialization: opcode followed by fields -/
def serializeWithOpcode (opcode : Nat) (fields : Array UInt32) (width : Nat := 32) : Array UInt32 :=
  padToWidth (#[natToUInt32 opcode] ++ fields) width

end Tyr.GPU.Interpreter
