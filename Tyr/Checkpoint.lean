import Tyr.TensorStruct

/-!
# Tyr.Checkpoint

`Tyr.Checkpoint` provides model-agnostic checkpoint persistence based on `TensorStruct`.
It enables saving and restoring parameter trees (and mirrored optimizer trees) without
model-specific serialization code.

## Major Components

- `CheckpointMeta`: iteration/loss metadata persisted with checkpoints.
- Generic tensor-tree save/load via `saveParams` and `loadParams`.
- Full checkpoint helpers (`saveCheckpoint`, `loadCheckpoint`).
- Optimizer-state variants that reuse the same TensorStruct traversal patterns.

## Scope

This module targets straightforward local checkpoint persistence for training workflows.
It prioritizes generic structure traversal and reproducible load/save behavior over
custom binary formats or distributed snapshot orchestration.
-/

namespace torch.checkpoint

open torch

private def parseNatLine (key : String) (line : String) : Except String Nat :=
  let value := (line.drop key.length).trimAscii.toString
  match value.toNat? with
  | some n => Except.ok n
  | none => Except.error s!"Invalid Nat value for {key.dropEnd 1}: {value}"

private def parseFloatLit? (s : String) : Option Float :=
  let trimmed := s.trimAscii.toString
  if trimmed.isEmpty then
    none
  else
    let negative := trimmed.startsWith "-"
    let body := if negative then (trimmed.drop 1).toString else trimmed
    let unsigned? :=
      match body.splitOn "." with
      | [whole] =>
        whole.toNat?.map Nat.toFloat
      | [whole, frac] =>
        match whole.toNat?, frac.toNat? with
        | some w, some f =>
          let denom : Float := (Nat.pow 10 frac.length).toFloat
          some (w.toFloat + f.toFloat / denom)
        | _, _ => none
      | _ => none
    unsigned?.map fun x => if negative then -x else x

private def parseFloatLine (key : String) (line : String) : Except String Float :=
  let value := (line.drop key.length).trimAscii.toString
  match parseFloatLit? value with
  | some x => Except.ok x
  | none => Except.error s!"Invalid Float value for {key.dropEnd 1}: {value}"

/-- Checkpoint metadata -/
structure CheckpointMeta where
  iteration : Nat
  bestValLoss : Float
  trainLoss : Float
  optimCount : Nat := 0
  deriving Repr, Inhabited

/-- Save checkpoint metadata to a file -/
def saveCheckpointMeta (m : CheckpointMeta) (path : String) : IO Unit := do
  let content := s!"iteration={m.iteration}\nbestValLoss={m.bestValLoss}\ntrainLoss={m.trainLoss}\noptimCount={m.optimCount}"
  IO.FS.writeFile path content

/-- Parse checkpoint metadata from a file -/
def loadCheckpointMeta (path : String) : IO CheckpointMeta := do
  let content ← IO.FS.readFile path
  let lines := content.splitOn "\n"
  let mut iteration : Nat := 0
  let mut bestValLoss : Float := 1e10
  let mut trainLoss : Float := 0.0
  let mut optimCount : Nat := 0
  for line in lines do
    if line.startsWith "iteration=" then
      match parseNatLine "iteration=" line with
      | .ok value => iteration := value
      | .error err => throw <| IO.userError err
    else if line.startsWith "bestValLoss=" then
      match parseFloatLine "bestValLoss=" line with
      | .ok value => bestValLoss := value
      | .error err => throw <| IO.userError err
    else if line.startsWith "trainLoss=" then
      match parseFloatLine "trainLoss=" line with
      | .ok value => trainLoss := value
      | .error err => throw <| IO.userError err
    else if line.startsWith "optimCount=" then
      match parseNatLine "optimCount=" line with
      | .ok value => optimCount := value
      | .error err => throw <| IO.userError err
  return { iteration, bestValLoss, trainLoss, optimCount }

/-- Check if a checkpoint exists -/
def checkpointExists (dir : String) : IO Bool := do
  data.fileExists (dir ++ "/meta.txt")

/-! ## TensorStruct-based Save/Load

These functions use the TensorStruct typeclass to generically save and load
any model structure containing tensors. Tensors are saved with auto-generated
sequential names based on traversal order.
-/

/-- State for tracking tensor index during save -/
structure SaveState where
  index : IO.Ref Nat

/-- Save all tensors in a TensorStruct to a directory with a namePrefix -/
def saveParams [TensorStruct α] (params : α) (dir : String) (namePrefix : String := "param") : IO Unit := do
  IO.FS.createDirAll dir
  let indexRef ← IO.mkRef 0
  let _ ← TensorStruct.fold (fun {s} (t : T s) (acc : IO Unit) => do
    acc
    let idx ← indexRef.get
    let path := s!"{dir}/{namePrefix}_{idx}.pt"
    data.saveTensor t path
    indexRef.set (idx + 1)
  ) (pure ()) params
  let finalIdx ← indexRef.get
  IO.println s!"Saved {finalIdx} tensors to {dir}"

/-- Load all tensors in a TensorStruct from a directory with a namePrefix.
    Requires a template structure to know the shapes. -/
def loadParams [TensorStruct α] (template : α) (dir : String) (namePrefix : String := "param") : IO α := do
  let indexRef ← IO.mkRef 0
  let result ← TensorStruct.mapM (fun {s} (_ : T s) => do
    let idx ← indexRef.get
    let path := s!"{dir}/{namePrefix}_{idx}.pt"
    let t ← data.loadTensor s path
    indexRef.set (idx + 1)
    pure t
  ) template
  let finalIdx ← indexRef.get
  IO.println s!"Loaded {finalIdx} tensors from {dir}"
  return TensorStruct.makeLeafParams result

/-- Save full checkpoint (params + metadata) -/
def saveCheckpoint [TensorStruct α]
    (params : α)
    (iteration : Nat)
    (bestValLoss : Float)
    (trainLoss : Float)
    (dir : String)
    (namePrefix : String := "param") : IO Unit := do
  saveParams params dir namePrefix
  saveCheckpointMeta { iteration, bestValLoss, trainLoss } (dir ++ "/meta.txt")
  IO.println s!"Checkpoint saved at iteration {iteration}"

/-- Load checkpoint (params + metadata) -/
def loadCheckpoint [TensorStruct α]
    (template : α)
    (dir : String)
    (namePrefix : String := "param") : IO (α × CheckpointMeta) := do
  let params ← loadParams template dir namePrefix
  let m ← loadCheckpointMeta (dir ++ "/meta.txt")
  IO.println s!"Checkpoint loaded from iteration {m.iteration}"
  return (params, m)

/-! ## Optimizer State Checkpointing

For optimizer states that mirror the model structure (like Adam's mu/nu),
use saveParams/loadParams with different namePrefixes.
-/

/-- Save optimizer state (for optimizers with matching structure like Adam) -/
def saveOptimizerState [TensorStruct α]
    (mu nu : α)
    (count : Nat)
    (dir : String) : IO Unit := do
  saveParams mu dir "optim_mu"
  saveParams nu dir "optim_nu"
  IO.FS.writeFile (dir ++ "/optim_count.txt") (toString count)
  IO.println s!"Optimizer state saved to {dir}"

/-- Load optimizer state -/
def loadOptimizerState [TensorStruct α]
    (template : α)
    (dir : String) : IO (α × α × Nat) := do
  let mu ← loadParams template dir "optim_mu"
  let nu ← loadParams template dir "optim_nu"
  let countStr ← IO.FS.readFile (dir ++ "/optim_count.txt")
  let count := countStr.trimAscii.toString.toNat!
  IO.println s!"Optimizer state loaded from {dir} (count={count})"
  return (mu, nu, count)

/-- Check if optimizer state exists -/
def optimStateExists (dir : String) : IO Bool := do
  data.fileExists (dir ++ "/optim_count.txt")

end torch.checkpoint
