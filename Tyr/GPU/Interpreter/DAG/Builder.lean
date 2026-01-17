/-
  Tyr/GPU/Interpreter/DAG/Builder.lean

  Monadic DSL for building instruction DAGs.
  Supports both explicit dependencies and implicit opcode-based sequencing.
-/
import Tyr.GPU.Interpreter.DAG.Node
import Std.Data.HashMap

namespace Tyr.GPU.Interpreter

open Std (HashMap)

/-- State for the DAG builder -/
structure DAGBuilderState where
  /-- Next available node ID -/
  nextId : Nat := 0
  /-- Accumulated nodes -/
  nodes : Array DAGNode := #[]
  /-- Last node ID for each opcode (for implicit sequencing) -/
  lastByOpcode : HashMap Nat NodeId := {}
  /-- All root node IDs -/
  roots : List NodeId := []
  deriving Inhabited

/-- DAG builder monad -/
abbrev DAGBuilder := StateM DAGBuilderState

namespace DAGBuilder

/-- Generate a fresh node ID -/
def freshId : DAGBuilder NodeId := do
  let s ← get
  let id : NodeId := ⟨s.nextId⟩
  set { s with nextId := s.nextId + 1 }
  return id

/-- Emit a tile operation with explicit dependencies -/
def emitTile (op : TileOp) (deps : List NodeId := []) : DAGBuilder NodeId := do
  let id ← freshId
  let node : DAGNode := DAGNode.mk' id.val (.tile op) deps
  modify fun s =>
    let isRoot := deps.isEmpty
    { s with
      nodes := s.nodes.push node
      lastByOpcode := s.lastByOpcode.insert op.opcode id
      roots := if isRoot then id :: s.roots else s.roots }
  return id

/-- Emit a transformer operation with explicit dependencies -/
def emitTransformer (op : TransformerOp) (deps : List NodeId := []) : DAGBuilder NodeId := do
  let id ← freshId
  let node : DAGNode := DAGNode.mk' id.val (.transformer op) deps
  modify fun s =>
    let isRoot := deps.isEmpty
    { s with
      nodes := s.nodes.push node
      lastByOpcode := s.lastByOpcode.insert op.opcode id
      roots := if isRoot then id :: s.roots else s.roots }
  return id

/-- Emit any instruction with explicit dependencies -/
def emit (instr : AnyInstruction) (deps : List NodeId := []) : DAGBuilder NodeId := do
  let id ← freshId
  let node : DAGNode := DAGNode.mk' id.val instr deps
  modify fun s =>
    let isRoot := deps.isEmpty
    { s with
      nodes := s.nodes.push node
      lastByOpcode := s.lastByOpcode.insert instr.opcode id
      roots := if isRoot then id :: s.roots else s.roots }
  return id

/-- Emit after all of the given nodes complete -/
def emitAfter (deps : List NodeId) (instr : AnyInstruction) : DAGBuilder NodeId :=
  emit instr deps

/-- Emit a tile op after dependencies -/
def emitTileAfter (deps : List NodeId) (op : TileOp) : DAGBuilder NodeId :=
  emitTile op deps

/-- Emit a transformer op after dependencies -/
def emitTransformerAfter (deps : List NodeId) (op : TransformerOp) : DAGBuilder NodeId :=
  emitTransformer op deps

/-- Emit with implicit dependency on last instruction of same opcode -/
def emitSequential (instr : AnyInstruction) : DAGBuilder NodeId := do
  let s ← get
  let deps := match s.lastByOpcode.get? instr.opcode with
    | some id => [id]
    | none => []
  emit instr deps

/-- Emit tile op with implicit sequencing by opcode -/
def emitTileSequential (op : TileOp) : DAGBuilder NodeId := do
  emitSequential (.tile op)

/-- Emit transformer op with implicit sequencing -/
def emitTransformerSequential (op : TransformerOp) : DAGBuilder NodeId := do
  -- Use prevOpcode for implicit dependency
  let s ← get
  let deps := match op.prevOpcode with
    | some prevOp =>
      match s.lastByOpcode.get? prevOp with
      | some id => [id]
      | none => []
    | none => []
  emitTransformer op deps

/-- Get the last emitted node with a specific opcode -/
def getLastByOpcode (opcode : Nat) : DAGBuilder (Option NodeId) := do
  return (← get).lastByOpcode.get? opcode

/-- Get all currently emitted nodes -/
def getNodes : DAGBuilder (Array DAGNode) := do
  return (← get).nodes

/-- Build the final DAG from the accumulated state -/
def build : DAGBuilder DAG := do
  let s ← get
  let nodes := s.nodes

  -- Compute terminals (nodes with no dependents)
  -- Collect all dependency IDs into a list
  let allDepIds := nodes.foldl (fun acc n => acc ++ n.dependencies) []
  let terminals := nodes.toList.filterMap fun n =>
    if !allDepIds.contains n.id then some n.id else none

  return {
    nodes := nodes
    roots := s.roots.reverse
    terminals := terminals
  }

end DAGBuilder

/-- Run the DAG builder and extract the DAG -/
def runDAGBuilder (builder : DAGBuilder Unit) : DAG :=
  let initState : DAGBuilderState := { lastByOpcode := {} }
  let (_, s) := (builder *> DAGBuilder.build).run initState
  let (dag, _) := DAGBuilder.build.run s
  dag

/-- Build a DAG from a builder action -/
def buildDAG (builder : DAGBuilder Unit) : DAG :=
  let initState : DAGBuilderState := { lastByOpcode := {} }
  let (dag, _) := (builder *> DAGBuilder.build).run initState
  dag

end Tyr.GPU.Interpreter
