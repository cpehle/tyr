/-
  Tyr/GPU/Interpreter/Scheduler/Sequential.lean

  Sequential scheduling algorithms: round-robin, zig-zag.
  Simple schedulers that don't consider dependencies.
-/
import Tyr.GPU.Interpreter.DAG.Node
import Tyr.GPU.Interpreter.Config

namespace Tyr.GPU.Interpreter

/-- Create an array with n copies of a value -/
private def arrayReplicate (n : Nat) (a : α) : Array α :=
  (List.replicate n a).toArray

/-- Schedule result: per-SM instruction queues -/
structure Schedule where
  /-- Instruction queues for each SM -/
  smQueues : Array (Array SerializedInstruction)
  /-- Maximum queue length (for padding) -/
  maxQueueLen : Nat
  /-- Number of SMs -/
  smCount : Nat
  deriving Repr, Inhabited

namespace Schedule

/-- Create an empty schedule -/
def empty (smCount : Nat) : Schedule :=
  { smQueues := arrayReplicate smCount #[]
    maxQueueLen := 0
    smCount := smCount }

/-- Total number of instructions across all SMs -/
def totalInstructions (s : Schedule) : Nat :=
  s.smQueues.foldl (· + ·.size) 0

/-- Pad all queues to the same length with no-ops -/
def padQueues (s : Schedule) (width : Nat := 32) : Schedule :=
  let noop := SerializedInstruction.noop width
  let paddedQueues := s.smQueues.map fun queue =>
    if queue.size < s.maxQueueLen then
      queue ++ arrayReplicate (s.maxQueueLen - queue.size) noop
    else queue
  { s with smQueues := paddedQueues }

/-- Get flattened instructions (all SMs concatenated) -/
def flatten (s : Schedule) : Array SerializedInstruction :=
  s.smQueues.foldl (· ++ ·) #[]

end Schedule

/-- Round-robin assignment: distribute instructions evenly across SMs -/
def roundRobinSchedule (dag : DAG) (smCount : Nat) : Schedule :=
  let queues := arrayReplicate smCount #[]
  let (queues, _) := dag.nodes.foldl (fun (qs, i) node =>
    let serialized := SerializedInstruction.fromNode node
    let qs' := qs.modify i (·.push serialized)
    (qs', (i + 1) % smCount)
  ) (queues, 0)
  let maxLen := queues.foldl (fun m q => max m q.size) 0
  { smQueues := queues, maxQueueLen := maxLen, smCount := smCount }

/-- Zig-zag assignment: 0,1,2,3,3,2,1,0,0,1,2,3,... for better locality -/
def zigZagSchedule (dag : DAG) (smCount : Nat) : Schedule :=
  let queues := arrayReplicate smCount #[]
  let (queues, _) := dag.nodes.foldl (fun (qs, i) node =>
    let serialized := SerializedInstruction.fromNode node
    let baseId := i % (smCount * 2)
    let smIdx := if baseId < smCount then baseId else smCount - 1 - (baseId - smCount)
    let qs' := qs.modify smIdx (·.push serialized)
    (qs', i + 1)
  ) (queues, 0)
  let maxLen := queues.foldl (fun m q => max m q.size) 0
  { smQueues := queues, maxQueueLen := maxLen, smCount := smCount }

/-- Pool-based assignment: separate memory and compute SMs -/
def poolSchedule (dag : DAG) (smCount : Nat) (memoryFraction : Float) : Schedule :=
  let memSMs := (smCount.toFloat * memoryFraction).toUInt64.toNat
  let computeSMs := smCount - memSMs

  -- Partition instructions by pool
  let (memInstrs, computeInstrs) := dag.nodes.foldl (fun (mem, comp) node =>
    match node.instruction.pool with
    | .memory => (mem.push node, comp)
    | .compute => (mem, comp.push node)
    | .mixed => (mem, comp.push node)  -- Default to compute
  ) (#[], #[])

  -- Round-robin within each pool
  let memQueues := arrayReplicate memSMs #[]
  let (memQueues, _) := memInstrs.foldl (fun (qs, i) node =>
    let serialized := SerializedInstruction.fromNode node
    (qs.modify i (·.push serialized), (i + 1) % memSMs)
  ) (memQueues, 0)

  let computeQueues := arrayReplicate computeSMs #[]
  let (computeQueues, _) := computeInstrs.foldl (fun (qs, i) node =>
    let serialized := SerializedInstruction.fromNode node
    (qs.modify i (·.push serialized), (i + 1) % computeSMs)
  ) (computeQueues, 0)

  let allQueues := memQueues ++ computeQueues
  let maxLen := allQueues.foldl (fun m q => max m q.size) 0
  { smQueues := allQueues, maxQueueLen := maxLen, smCount := smCount }

/-- Schedule a DAG using the specified strategy -/
def scheduleSequential (dag : DAG) (smCount : Nat) (strategy : ScheduleStrategy) : Schedule :=
  match strategy with
  | .sequential => roundRobinSchedule dag smCount
  | .zigZag => zigZagSchedule dag smCount
  | .pooled frac => poolSchedule dag smCount frac
  | _ => roundRobinSchedule dag smCount  -- Fallback

end Tyr.GPU.Interpreter
