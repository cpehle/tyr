/-
  Tyr/Data/TaskClass.lean

  Open Task typeclass for extensible task definitions.

  This replaces the closed AnyTask union with an open typeclass pattern.
  New tasks can be added by defining instances without modifying existing code.

  Key types:
  - Task: Typeclass for task implementations
  - BoxedTask: Existential wrapper for heterogeneous collections
  - TaskMixture: Works with any Task instances
-/
import Tyr.Data.Task

namespace torch.Data.TaskClass

open torch.Data.Task

/-! ## Evaluation Types -/

/-- Evaluation type for tasks -/
inductive EvalType where
  | categorical  -- Multiple choice, model picks a letter
  | generative   -- Free-form generation, extract answer
  deriving Repr, BEq, Inhabited

/-- Result of evaluating a response -/
structure EvalResult where
  /-- Whether the response is correct -/
  correct : Bool
  /-- Confidence score (0.0 to 1.0) -/
  score : Float
  /-- Expected answer -/
  expected : String
  /-- Predicted answer -/
  predicted : String
  deriving Repr

/-! ## Task Typeclass -/

/-- Open typeclass for task implementations.

    Any type can become a task by providing these operations.
    This replaces the closed AnyTask inductive with an open extension point.

    Example:
    ```
    instance : Task MyCustomTask where
      name := "my_custom"
      evalType _ := .generative
      size t := t.examples.size
      getExample t idx := t.examples[idx]?.map toConversation
      evaluate _ conv response := { correct := ..., ... }
    ```
-/
class EvalTask (α : Type) where
  /-- Task identifier -/
  name : String
  /-- Evaluation type (categorical vs generative) -/
  evalType : α → EvalType
  /-- Number of examples in this task -/
  size : α → Nat
  /-- Get example at index as a Conversation -/
  getExample : α → Nat → Option Conversation
  /-- Evaluate a response against expected answer -/
  evaluate : α → Conversation → String → EvalResult
  /-- Compute reward for RL (defaults to evaluation score) -/
  reward : α → Conversation → String → Float :=
    fun task conv response => (evaluate task conv response).score

/-! ## Boxed Task (Existential Wrapper)

For heterogeneous collections of tasks, we need an existential wrapper
that hides the concrete type while preserving Task operations.
-/

/-- Operations on a boxed (type-erased) task -/
structure TaskOps where
  name : String
  evalType : EvalType
  size : Nat
  getExample : Nat → Option Conversation
  evaluate : Conversation → String → EvalResult
  reward : Conversation → String → Float

instance : Inhabited TaskOps where
  default := {
    name := ""
    evalType := .categorical
    size := 0
    getExample := fun _ => none
    evaluate := fun _ _ => { correct := false, score := 0, expected := "", predicted := "" }
    reward := fun _ _ => 0.0
  }

/-- Create TaskOps from a concrete task -/
def TaskOps.ofTask [EvalTask α] (task : α) : TaskOps where
  name := EvalTask.name (α := α)
  evalType := EvalTask.evalType task
  size := EvalTask.size task
  getExample := EvalTask.getExample task
  evaluate := EvalTask.evaluate task
  reward := EvalTask.reward task

/-- A boxed task that hides the concrete type.

    Use this for heterogeneous collections like task mixtures.
    The concrete type is erased but operations are preserved.
-/
structure BoxedTask where
  /-- Operations on the task -/
  ops : TaskOps

instance : Inhabited BoxedTask where
  default := { ops := default }

/-- Box a concrete task into a BoxedTask -/
def boxTask [EvalTask α] (task : α) : BoxedTask :=
  { ops := TaskOps.ofTask task }

-- Forward operations to the boxed task
namespace BoxedTask

def name (t : BoxedTask) : String := t.ops.name
def evalType (t : BoxedTask) : EvalType := t.ops.evalType
def size (t : BoxedTask) : Nat := t.ops.size
def getExample (t : BoxedTask) (idx : Nat) : Option Conversation := t.ops.getExample idx
def evaluate (t : BoxedTask) (conv : Conversation) (response : String) : EvalResult :=
  t.ops.evaluate conv response
def reward (t : BoxedTask) (conv : Conversation) (response : String) : Float :=
  t.ops.reward conv response

end BoxedTask

/-! ## Task Mixture with Typeclass -/

/-- Entry in a task mixture -/
structure MixtureEntry where
  task : BoxedTask
  weight : Nat := 1

instance : Inhabited MixtureEntry where
  default := { task := default, weight := 1 }

/-- LCG random number generator for deterministic shuffling -/
private def lcgNext (state : UInt64) : UInt64 :=
  state * 6364136223846793005 + 1442695040888963407

/-- Fisher-Yates shuffle -/
private def shuffleIndices (arr : Array (Nat × Nat)) (seed : UInt64) : Array (Nat × Nat) := Id.run do
  if arr.size <= 1 then return arr
  let mut result := arr
  let mut state := seed
  for i in [:(arr.size - 1)] do
    state := lcgNext state
    let range := arr.size - i
    let j := i + (state % range.toUInt64).toNat
    let tmp := result[i]!
    result := result.set! i result[j]!
    result := result.set! j tmp
  return result

/-- A mixture of boxed tasks with deterministic interleaving -/
structure GenericTaskMixture where
  entries : Array MixtureEntry
  /-- Shuffled index map: global idx → (task idx, local idx) -/
  indexMap : Array (Nat × Nat)
  seed : UInt64

/-- Create a task mixture from boxed entries -/
def GenericTaskMixture.create (entries : Array MixtureEntry) (seed : UInt64 := 42)
    : GenericTaskMixture := Id.run do
  -- Build index map
  let mut indices : Array (Nat × Nat) := #[]
  for taskIdx in [:entries.size] do
    let entry := entries[taskIdx]!
    let taskSize := entry.task.size
    for _ in [:entry.weight] do
      for localIdx in [:taskSize] do
        indices := indices.push (taskIdx, localIdx)

  let shuffled := shuffleIndices indices seed
  { entries, indexMap := shuffled, seed }

def GenericTaskMixture.size (mix : GenericTaskMixture) : Nat := mix.indexMap.size

def GenericTaskMixture.getExample (mix : GenericTaskMixture) (index : Nat) : Option Conversation := do
  let (taskIdx, localIdx) ← mix.indexMap[index]?
  let entry ← mix.entries[taskIdx]?
  entry.task.getExample localIdx

def GenericTaskMixture.evaluate (mix : GenericTaskMixture) (index : Nat) (response : String)
    : Option EvalResult := do
  let (taskIdx, localIdx) ← mix.indexMap[index]?
  let entry ← mix.entries[taskIdx]?
  let conv ← entry.task.getExample localIdx
  return entry.task.evaluate conv response

def GenericTaskMixture.reward (mix : GenericTaskMixture) (index : Nat) (response : String)
    : Option Float := do
  let (taskIdx, localIdx) ← mix.indexMap[index]?
  let entry ← mix.entries[taskIdx]?
  let conv ← entry.task.getExample localIdx
  return entry.task.reward conv response

/-! ## Convenience Functions -/

/-- Create a mixture entry from any EvalTask instance -/
def entry [EvalTask α] (task : α) (weight : Nat := 1) : MixtureEntry :=
  { task := boxTask task, weight }

/-- Create a mixture from a list of tasks with equal weights -/
def mixtureSame (tasks : Array BoxedTask) (seed : UInt64 := 42) : GenericTaskMixture :=
  GenericTaskMixture.create (tasks.map fun t => { task := t, weight := 1 }) seed

/-- Create a mixture from weighted task entries -/
def mixtureWeighted (entries : Array MixtureEntry) (seed : UInt64 := 42) : GenericTaskMixture :=
  GenericTaskMixture.create entries seed

end torch.Data.TaskClass
