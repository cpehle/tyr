/-
  Tyr/GPU/Codegen/Constraints.lean

  Type-level constraints for GPU kernel operations.
  Enables compile-time checking of dimension requirements.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.AST
import Tyr.GPU.Codegen.Ops

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-! ## Dimension Constraints

Typeclasses for expressing and checking dimension requirements at compile time.
MMA operations typically require dimensions to be multiples of 16.
-/

/-- Proof that n is divisible by d -/
class DivisibleBy (n : Nat) (d : Nat) : Prop where
  proof : n % d = 0

/-- Proof that n is a multiple of 16 (common MMA requirement) -/
abbrev Mult16 (n : Nat) := DivisibleBy n 16

/-- Proof that n is a multiple of 8 -/
abbrev Mult8 (n : Nat) := DivisibleBy n 8

/-- Proof that n is a multiple of 32 (warp size) -/
abbrev WarpAligned (n : Nat) := DivisibleBy n 32

/-- Proof that n is a multiple of 128 (warp group size) -/
abbrev WarpGroupAligned (n : Nat) := DivisibleBy n 128

-- Auto-derive instances for common multiples of 16
instance : DivisibleBy 16 16 where proof := by native_decide
instance : DivisibleBy 32 16 where proof := by native_decide
instance : DivisibleBy 48 16 where proof := by native_decide
instance : DivisibleBy 64 16 where proof := by native_decide
instance : DivisibleBy 80 16 where proof := by native_decide
instance : DivisibleBy 96 16 where proof := by native_decide
instance : DivisibleBy 112 16 where proof := by native_decide
instance : DivisibleBy 128 16 where proof := by native_decide
instance : DivisibleBy 144 16 where proof := by native_decide
instance : DivisibleBy 160 16 where proof := by native_decide
instance : DivisibleBy 176 16 where proof := by native_decide
instance : DivisibleBy 192 16 where proof := by native_decide
instance : DivisibleBy 256 16 where proof := by native_decide
instance : DivisibleBy 512 16 where proof := by native_decide
instance : DivisibleBy 1024 16 where proof := by native_decide

-- Common multiples of 8
instance : DivisibleBy 8 8 where proof := by native_decide
instance : DivisibleBy 16 8 where proof := by native_decide
instance : DivisibleBy 24 8 where proof := by native_decide
instance : DivisibleBy 32 8 where proof := by native_decide
instance : DivisibleBy 40 8 where proof := by native_decide
instance : DivisibleBy 48 8 where proof := by native_decide
instance : DivisibleBy 56 8 where proof := by native_decide
instance : DivisibleBy 64 8 where proof := by native_decide
instance : DivisibleBy 72 8 where proof := by native_decide
instance : DivisibleBy 80 8 where proof := by native_decide
instance : DivisibleBy 88 8 where proof := by native_decide
instance : DivisibleBy 96 8 where proof := by native_decide
instance : DivisibleBy 104 8 where proof := by native_decide
instance : DivisibleBy 112 8 where proof := by native_decide
instance : DivisibleBy 120 8 where proof := by native_decide
instance : DivisibleBy 128 8 where proof := by native_decide
instance : DivisibleBy 256 8 where proof := by native_decide
instance : DivisibleBy 512 8 where proof := by native_decide
instance : DivisibleBy 1024 8 where proof := by native_decide

-- Warp-aligned sizes
instance : DivisibleBy 32 32 where proof := by native_decide
instance : DivisibleBy 64 32 where proof := by native_decide
instance : DivisibleBy 96 32 where proof := by native_decide
instance : DivisibleBy 128 32 where proof := by native_decide
instance : DivisibleBy 256 32 where proof := by native_decide
instance : DivisibleBy 512 32 where proof := by native_decide
instance : DivisibleBy 1024 32 where proof := by native_decide

-- Warp group aligned sizes (128 threads = 4 warps)
instance : DivisibleBy 128 128 where proof := by native_decide
instance : DivisibleBy 256 128 where proof := by native_decide
instance : DivisibleBy 384 128 where proof := by native_decide
instance : DivisibleBy 512 128 where proof := by native_decide
instance : DivisibleBy 1024 128 where proof := by native_decide

/-! ## Constrained MMA Operations

These versions of MMA operations include compile-time dimension checks.
-/

/-- Matrix multiply with dimension constraints -/
def mmaConstrained {M K N : Nat} {inDtype accDtype : GpuFloat}
    [Mult16 M] [Mult16 K] [Mult16 N]
    (dst : RT accDtype M N .Row)
    (a : RT inDtype M K .Row)
    (b : RT inDtype K N .Col)
    (c : RT accDtype M N .Row)
    : KernelM Unit := do
  emit (.mma .AB dst.id a.id b.id c.id)

/-- Matrix multiply without accumulate, with constraints -/
def mmConstrained {M K N : Nat} {inDtype accDtype : GpuFloat}
    [Mult16 M] [Mult16 K] [Mult16 N]
    (dst : RT accDtype M N .Row)
    (a : RT inDtype M K .Row)
    (b : RT inDtype K N .Col)
    : KernelM Unit := do
  emit (.mm .AB dst.id a.id b.id)

/-- Matrix multiply with B transposed, with constraints -/
def mmaTConstrained {M K N : Nat} {inDtype accDtype : GpuFloat}
    [Mult16 M] [Mult16 K] [Mult16 N]
    (dst : RT accDtype M N .Row)
    (a : RT inDtype M K .Row)
    (b : RT inDtype N K .Row)
    (c : RT accDtype M N .Row)
    : KernelM Unit := do
  emit (.mma .ABt dst.id a.id b.id c.id)

/-! ## Layout Constraints

Typeclasses for ensuring correct layouts in operations.
-/

/-- Proof that a tile has Row layout -/
class HasRowLayout (T : Type) : Prop

/-- Proof that a tile has Col layout -/
class HasColLayout (T : Type) : Prop

instance {dtype : GpuFloat} {rows cols : Nat} : HasRowLayout (RT dtype rows cols .Row) where
instance {dtype : GpuFloat} {rows cols : Nat} : HasColLayout (RT dtype rows cols .Col) where
instance {dtype : GpuFloat} {rows cols : Nat} : HasRowLayout (ST dtype rows cols .Row) where
instance {dtype : GpuFloat} {rows cols : Nat} : HasColLayout (ST dtype rows cols .Col) where

/-! ## Resource Budget Tracking (Opt-in)

Types for tracking register and shared memory usage.
-/

/-- GPU resource budget configuration -/
structure ResourceBudget where
  /-- Maximum registers per thread (255 for SM90) -/
  maxRegisters : Nat := 255
  /-- Maximum shared memory in bytes (99KB for SM90) -/
  maxSharedMem : Nat := 99000
  /-- Maximum warps per SM -/
  maxWarps : Nat := 16
  deriving Repr, Inhabited

/-- Default budget for SM90 (Hopper) -/
def sm90Budget : ResourceBudget := {
  maxRegisters := 255
  maxSharedMem := 99000
  maxWarps := 16
}

/-- Default budget for SM80 (Ampere) -/
def sm80Budget : ResourceBudget := {
  maxRegisters := 255
  maxSharedMem := 48000
  maxWarps := 16
}

/-- Resource usage tracking state -/
structure ResourceUsage where
  /-- Estimated register usage per thread -/
  registers : Nat := 0
  /-- Shared memory usage in bytes -/
  sharedMem : Nat := 0
  /-- Number of register tiles allocated -/
  regTiles : Nat := 0
  /-- Number of shared tiles allocated -/
  sharedTiles : Nat := 0
  deriving Repr, Inhabited

/-- Extended kernel state with resource tracking -/
structure KernelStateExt where
  /-- Base kernel state -/
  base : KernelState
  /-- Resource usage -/
  usage : ResourceUsage
  /-- Resource budget -/
  budget : ResourceBudget
  deriving Repr, Inhabited

/-- Kernel builder monad with resource tracking -/
abbrev KernelMExt := StateM KernelStateExt

/-- Convert KernelM action to KernelMExt -/
def liftKernelM (m : KernelM α) : KernelMExt α := do
  let s ← get
  let (a, newBase) := m.run s.base
  set { s with base := newBase }
  pure a

/-- Check if budget allows allocation -/
def checkBudget (bytes : Nat) (isShared : Bool) : KernelMExt Bool := do
  let s ← get
  if isShared then
    pure (s.usage.sharedMem + bytes ≤ s.budget.maxSharedMem)
  else
    -- Rough estimate: each tile uses ~(rows * cols) / 32 registers
    pure true

/-- Allocate register tile with budget tracking -/
def allocRTTracked (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row)
    : KernelMExt (RT dtype rows cols layout) := do
  let bytes := rows * cols * dtype.bytes
  let withinBudget ← checkBudget bytes false
  if !withinBudget then
    -- Could emit a warning or error here
    pure ()
  let tile ← liftKernelM (allocRT dtype rows cols layout)
  modify fun s => { s with
    usage := { s.usage with
      regTiles := s.usage.regTiles + 1
      registers := s.usage.registers + (rows * cols / 32)
    }
  }
  pure tile

/-- Allocate shared tile with budget tracking -/
def allocSTTracked (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row)
    : KernelMExt (ST dtype rows cols layout) := do
  let bytes := rows * cols * dtype.bytes
  let withinBudget ← checkBudget bytes true
  if !withinBudget then
    -- Could emit a warning or error here
    pure ()
  let tile ← liftKernelM (allocST dtype rows cols layout)
  modify fun s => { s with
    usage := { s.usage with
      sharedTiles := s.usage.sharedTiles + 1
      sharedMem := s.usage.sharedMem + bytes
    }
  }
  pure tile

/-- Run KernelMExt and extract results with usage report -/
def runKernelMExt (budget : ResourceBudget := sm90Budget) (m : KernelMExt α)
    : α × KernelState × ResourceUsage :=
  let initState : KernelStateExt := {
    base := { arch := .SM90 }
    usage := {}
    budget := budget
  }
  let (a, finalState) := m.run initState
  (a, finalState.base, finalState.usage)

/-! ## Architecture Capabilities

Typeclasses for expressing architecture-specific features.
-/

/-- GPU architecture type -/
inductive Arch where
  | SM80   -- Ampere
  | SM90   -- Hopper
  | SM100  -- Blackwell (future)
  deriving Repr, BEq, Inhabited

/-- Architecture has TMA (Tensor Memory Accelerator) support -/
class HasTMA (arch : Arch) : Prop

/-- Architecture has WGMMA (Warp Group MMA) support -/
class HasWGMMA (arch : Arch) : Prop

/-- Architecture has FP8 support -/
class HasFP8 (arch : Arch) : Prop

/-- Architecture has cluster support -/
class HasCluster (arch : Arch) : Prop

-- SM90 (Hopper) capabilities
instance : HasTMA Arch.SM90 where
instance : HasWGMMA Arch.SM90 where
instance : HasFP8 Arch.SM90 where
instance : HasCluster Arch.SM90 where

-- SM100 (Blackwell) capabilities (superset of SM90)
instance : HasTMA Arch.SM100 where
instance : HasWGMMA Arch.SM100 where
instance : HasFP8 Arch.SM100 where
instance : HasCluster Arch.SM100 where

-- SM80 (Ampere) has none of the Hopper features
-- (No instances for SM80)

/-! ## Constrained TMA Operations

TMA operations that require architecture support.
-/

/-- TMA load requiring HasTMA capability -/
def tmaLoadConstrained {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    {arch : Arch} [HasTMA arch]
    (dst : ST dtype rows cols layout)
    (src : GPtr dtype)
    (coord : KVal UInt64) : KernelM Unit := do
  emit (.tmaLoad dst.id src.id coord.id)

/-- TMA store requiring HasTMA capability -/
def tmaStoreConstrained {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    {arch : Arch} [HasTMA arch]
    (dst : GPtr dtype)
    (src : ST dtype rows cols layout)
    (coord : KVal UInt64) : KernelM Unit := do
  emit (.tmaStore dst.id src.id coord.id)

end Tyr.GPU.Codegen
