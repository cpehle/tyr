/-
  Tyr/GPU/Codegen/Monad.lean

  Kernel builder monad using standard StateM.
  Enables do-notation for natural kernel construction.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.IR

/-!
# `Tyr.GPU.Codegen.Monad`

GPU code generation component for Monad, used to lower high-level tile programs to backend code.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-- Kernel building state -/
structure KernelState where
  /-- Next available variable ID -/
  nextId : Nat := 0
  /-- Accumulated kernel statements -/
  body : Array KStmt := #[]
  /-- Target GPU architecture -/
  arch : GpuArch := .SM90
  /-- Shared memory usage tracking -/
  sharedMemBytes : Nat := 0
  deriving Inhabited, Repr

/-- Kernel builder monad - just standard StateM! -/
abbrev KernelM := StateM KernelState

/-- Generate a fresh variable ID -/
def freshVar : KernelM VarId := do
  let s ← get
  let v : VarId := ⟨s.nextId⟩
  set { s with nextId := s.nextId + 1 }
  pure v

/-- Emit a statement to the kernel body -/
def emit (stmt : KStmt) : KernelM Unit := do
  modify fun s => { s with body := s.body.push stmt }

/-- Add a comment to the kernel -/
def comment (text : String) : KernelM Unit := do
  emit (.comment text)

/-- Set the target architecture -/
def setArch (arch : GpuArch) : KernelM Unit := do
  modify fun s => { s with arch := arch }

/-- Build a Kernel from the accumulated state -/
def buildKernel (name : String) (params : Array KParam := #[]) : KernelM Kernel := do
  let s ← get
  pure {
    name := name
    arch := s.arch
    params := params
    body := s.body
    sharedMemBytes := s.sharedMemBytes
  }

/-- Run the kernel builder and extract the result -/
def runKernelM (arch : GpuArch := .SM90) (m : KernelM α) : α × KernelState :=
  m.run { arch := arch }

/-- Run and extract just the kernel.
    Note: nextId starts at params.size to avoid conflicts with parameter VarIds -/
def buildKernelM (name : String) (arch : GpuArch := .SM90)
    (params : Array KParam := #[]) (m : KernelM Unit) : Kernel :=
  -- Start nextId at params.size so freshVar doesn't conflict with parameter VarIds
  let (_, state) := m.run { arch := arch, nextId := params.size }
  {
    name := name
    arch := state.arch
    params := params
    body := state.body
    sharedMemBytes := state.sharedMemBytes
  }

/-- Capture loop body: saves state, runs body, extracts statements, restores -/
def captureBody (body : KernelM Unit) : KernelM (Array KStmt) := do
  let savedBody := (← get).body
  modify fun s => { s with body := #[] }
  body
  let capturedBody := (← get).body
  modify fun s => { s with body := savedBody }
  pure capturedBody

end Tyr.GPU.Codegen
