/-
  Tyr/GPU/Interpreter/DAG/Node.lean

  DAG node types for instruction scheduling.
  Supports both tile-level and fused operations.
-/
import Tyr.GPU.Interpreter.Instruction.Base
import Tyr.GPU.Interpreter.Instruction.Tile
import Tyr.GPU.Interpreter.Instruction.Transformer

/-!
# `Tyr.GPU.Interpreter.DAG.Node`

DAG utilities for the GPU interpreter, covering Node for dependency-aware scheduling.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace Tyr.GPU.Interpreter

/-- Existential wrapper for any instruction type -/
inductive AnyInstruction where
  | tile (op : TileOp)
  | transformer (op : TransformerOp)
  | noop
  deriving Repr, BEq, Inhabited

namespace AnyInstruction

/-- Get the opcode of the wrapped instruction -/
def opcode : AnyInstruction → Nat
  | .tile op => op.opcode
  | .transformer op => op.opcode
  | .noop => 0

/-- Get the instruction level -/
def level : AnyInstruction → InstructionLevel
  | .tile _ => .tile
  | .transformer _ => .fused
  | .noop => .control

/-- Serialize the wrapped instruction -/
def serialize : AnyInstruction → Array UInt32
  | .tile op => op.serialize
  | .transformer op => op.serialize
  | .noop => #[0]

/-- Get the cost of the wrapped instruction -/
def cost : AnyInstruction → Float
  | .tile op => op.cost
  | .transformer op => op.cost
  | .noop => 0.0

/-- Get the resource pool -/
def pool : AnyInstruction → ResourcePool
  | .tile op => op.pool
  | .transformer op => op.pool
  | .noop => .mixed

end AnyInstruction

/-- Unique identifier for a DAG node -/
structure NodeId where
  val : Nat
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq, Ord

instance : ToString NodeId where
  toString n := s!"node_{n.val}"

instance : LT NodeId where
  lt a b := a.val < b.val

/-- DAG node representing a single instruction with dependencies -/
structure DAGNode where
  /-- Unique identifier -/
  id : NodeId
  /-- The instruction at this node -/
  instruction : AnyInstruction
  /-- IDs of nodes this node depends on -/
  dependencies : List NodeId := []
  /-- Cached cost for scheduling -/
  cachedCost : Float := 0.0
  /-- Priority for scheduling (computed from critical path) -/
  priority : Float := 0.0
  /-- Scheduled start time (filled by scheduler) -/
  startTime : Float := 1e30
  /-- Scheduled end time (filled by scheduler) -/
  endTime : Float := 1e30
  /-- Assigned SM index (filled by scheduler) -/
  assignedSM : Option Nat := none
  deriving Repr, Inhabited

namespace DAGNode

/-- Create a new DAG node -/
def mk' (id : Nat) (instr : AnyInstruction) (deps : List NodeId := []) : DAGNode :=
  { id := ⟨id⟩
    instruction := instr
    dependencies := deps
    cachedCost := instr.cost
    startTime := 1e30
    endTime := 1e30 }

/-- Get the opcode -/
def opcode (node : DAGNode) : Nat := node.instruction.opcode

/-- Serialize the instruction -/
def serialize (node : DAGNode) : Array UInt32 := node.instruction.serialize

/-- Check if this node has no dependencies -/
def isRoot (node : DAGNode) : Bool := node.dependencies.isEmpty

/-- Get the cost -/
def cost (node : DAGNode) : Float := node.cachedCost

end DAGNode

/-- Complete DAG structure -/
structure DAG where
  /-- All nodes in topological order -/
  nodes : Array DAGNode
  /-- Root nodes (no dependencies) -/
  roots : List NodeId
  /-- Terminal nodes (no dependents) -/
  terminals : List NodeId
  deriving Repr, Inhabited

namespace DAG

/-- Create an empty DAG -/
def empty : DAG := { nodes := #[], roots := [], terminals := [] }

/-- Number of nodes in the DAG -/
def size (dag : DAG) : Nat := dag.nodes.size

/-- Get a node by ID -/
def getNode? (dag : DAG) (id : NodeId) : Option DAGNode :=
  dag.nodes.find? (·.id == id)

/-- Get a node by ID (with default) -/
def getNode! (dag : DAG) (id : NodeId) : DAGNode :=
  dag.getNode? id |>.getD default

/-- Check if the DAG is empty -/
def isEmpty (dag : DAG) : Bool := dag.nodes.isEmpty

/-- Get all nodes as a list -/
def toList (dag : DAG) : List DAGNode := dag.nodes.toList

/-- Get children of a node (nodes that depend on it) -/
def children (dag : DAG) (nodeId : NodeId) : List NodeId :=
  dag.nodes.toList.filterMap fun node =>
    if node.dependencies.contains nodeId then some node.id else none

/-- Compute the set of all nodes reachable from a given node -/
partial def reachableFrom (dag : DAG) (nodeId : NodeId) : List NodeId :=
  let direct := dag.children nodeId
  direct ++ direct.flatMap (dag.reachableFrom ·)

/-- Validate that the DAG has no cycles (simplified check) -/
def isAcyclic (dag : DAG) : Bool :=
  -- Simple check: ensure no node depends on a later node in the array
  let indexed := dag.nodes.toList.zip (List.range dag.nodes.size)
  indexed.all fun (node, i) =>
    node.dependencies.all fun depId =>
      match dag.nodes.findIdx? (·.id == depId) with
      | some j => j < i
      | none => false

end DAG

/-- Serialized instruction ready for GPU transfer -/
structure SerializedInstruction where
  /-- Raw UInt32 words -/
  words : Array UInt32
  /-- Original node ID for debugging -/
  sourceNode : Option NodeId := none
  deriving Repr, Inhabited

namespace SerializedInstruction

/-- Create from a DAG node -/
def fromNode (node : DAGNode) (width : Nat := 32) : SerializedInstruction :=
  { words := padToWidth node.serialize width
    sourceNode := some node.id }

/-- Create a no-op instruction -/
def noop (width : Nat := 32) : SerializedInstruction :=
  { words := padToWidth #[0] width
    sourceNode := none }

end SerializedInstruction

end Tyr.GPU.Interpreter
