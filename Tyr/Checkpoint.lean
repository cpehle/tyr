/-
  General Checkpoint System

  Provides model-agnostic save/load functionality using TensorStruct.
  Enables training resumption and model export for any model type.
-/
import Tyr.TensorStruct

/-!
# `Tyr.Checkpoint`

Provides metadata parsing plus TensorStruct-based save and load routines for model parameters and optimizer states.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace torch.checkpoint

open torch

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
      iteration := (line.drop 10).toNat!
    else if line.startsWith "bestValLoss=" then
      let valStr := line.drop 12
      bestValLoss := valStr.toNat!.toFloat
    else if line.startsWith "trainLoss=" then
      let valStr := line.drop 10
      trainLoss := valStr.toNat!.toFloat
    else if line.startsWith "optimCount=" then
      optimCount := (line.drop 11).toNat!
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
