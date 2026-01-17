/-
  Tyr/GPU/Interpreter/Scheduler/DAGBased.lean

  DAG-based scheduling with dependency tracking.
  Simplified implementation.
-/
import Tyr.GPU.Interpreter.DAG.Node
import Tyr.GPU.Interpreter.DAG.Builder
import Tyr.GPU.Interpreter.Scheduler.Sequential
import Std.Data.HashMap

namespace Tyr.GPU.Interpreter

open Std (HashMap)

/-- Create an array with n copies of a value -/
private def mkArrayN (n : Nat) (a : α) : Array α :=
  (List.replicate n a).toArray

/-- Get element at index with default -/
private def getAt {α : Type} [Inhabited α] (arr : Array α) (i : Nat) : α :=
  arr.getD i default

/-- Simple DAG-based scheduling: respect dependencies, assign to least-loaded SM -/
def dagSchedule (dag : DAG) (smCount : Nat) (_useCost : Bool := true) : Schedule := Id.run do
  -- Initialize SM load tracking
  let mut smLoads : Array Float := mkArrayN smCount 0.0
  let mut queues : Array (Array SerializedInstruction) := mkArrayN smCount #[]

  -- Track which nodes are complete
  let mut completed : HashMap NodeId Bool := {}

  -- Convert to list for easier manipulation
  let mut remainingList := dag.nodes.toList

  -- Simple greedy algorithm: process nodes in order, respecting dependencies
  while remainingList.length > 0 do
    -- Find a ready node (all dependencies completed)
    let readyNode? := remainingList.find? fun node =>
      node.dependencies.all fun depId => completed.get? depId |>.getD false

    match readyNode? with
    | none => break  -- No progress possible (shouldn't happen for valid DAGs)
    | some node =>
      -- Remove this node from remaining
      remainingList := remainingList.filter (·.id != node.id)

      -- Find least-loaded SM
      let mut minIdx := 0
      let mut minLoad := getAt smLoads 0
      for i in [1:smCount] do
        if getAt smLoads i < minLoad then
          minIdx := i
          minLoad := getAt smLoads i

      -- Assign to that SM
      let serialized := SerializedInstruction.fromNode node
      queues := queues.modify minIdx (·.push serialized)
      smLoads := smLoads.set! minIdx (minLoad + node.cost)

      -- Mark as completed
      completed := completed.insert node.id true

  let maxLen := queues.foldl (fun m q => max m q.size) 0
  { smQueues := queues, maxQueueLen := maxLen, smCount := smCount }

/-- Wave-based scheduling: group by opcode, then load-balance within wave -/
def waveSchedule (dag : DAG) (smCount : Nat) : Schedule := Id.run do
  -- Group nodes by opcode
  let mut waves : HashMap Nat (Array DAGNode) := {}
  for node in dag.nodes do
    let opcode := node.opcode
    let existing := waves.get? opcode |>.getD #[]
    waves := waves.insert opcode (existing.push node)

  -- Process each wave, assigning to least-loaded SM
  let mut queues := mkArrayN smCount #[]
  let mut smCosts : Array Float := mkArrayN smCount 0.0

  for (_, waveNodes) in waves.toList do
    -- Sort by cost (biggest first)
    let sortedNodes := waveNodes.toList.mergeSort (·.cost > ·.cost)

    for node in sortedNodes do
      -- Find least-loaded SM
      let mut minIdx := 0
      let mut minCost := getAt smCosts 0
      for i in [1:smCount] do
        if getAt smCosts i < minCost then
          minIdx := i
          minCost := getAt smCosts i

      -- Assign to that SM
      let serialized := SerializedInstruction.fromNode node
      queues := queues.modify minIdx (·.push serialized)
      smCosts := smCosts.set! minIdx (minCost + node.cost)

  let maxLen := queues.foldl (fun m q => max m q.size) 0
  { smQueues := queues, maxQueueLen := maxLen, smCount := smCount }

/-- Main scheduling entry point -/
def schedule (dag : DAG) (smCount : Nat) (strategy : ScheduleStrategy) : Schedule :=
  match strategy with
  | .sequential => roundRobinSchedule dag smCount
  | .zigZag => zigZagSchedule dag smCount
  | .dagBased useCost => dagSchedule dag smCount useCost
  | .wave => waveSchedule dag smCount
  | .pooled frac => poolSchedule dag smCount frac

end Tyr.GPU.Interpreter
