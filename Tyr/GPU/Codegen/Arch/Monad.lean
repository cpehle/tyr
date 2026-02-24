/-
  Tyr/GPU/Codegen/Arch/Monad.lean

  Architecture-indexed kernel monad for type-safe multi-architecture kernel generation.
  Operations are tagged with their minimum required architecture level.
-/
import Tyr.GPU.Codegen.Arch.Level
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.IR

/-!
# `Tyr.GPU.Codegen.Arch.Monad`

Architecture-specific GPU code generation support for Monad within the ThunderKittens backend.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace Tyr.GPU.Codegen.Arch

open Tyr.GPU
open Tyr.GPU.Codegen

/-- Kernel monad indexed by minimum required architecture.
    The minArch parameter tracks the minimum architecture capability needed
    for all operations that have been composed in this computation.

    Example: A computation that uses TMA would have minArch = Hopper,
    while a computation using only basic MMA would have minArch = Ampere. -/
structure ArchKernelM (minArch : ArchLevel) (α : Type) where
  /-- The underlying KernelM computation -/
  run : KernelM α
  deriving Inhabited

namespace ArchKernelM

/-- Run an ArchKernelM computation, extracting the underlying KernelM -/
def toKernelM {minArch : ArchLevel} (m : ArchKernelM minArch α) : KernelM α := m.run

/-- Lift a KernelM computation into ArchKernelM with Ampere as the base architecture.
    Since Ampere is supported by all architectures, this is always safe. -/
def liftKernelM (m : KernelM α) : ArchKernelM .Ampere α := ⟨m⟩

instance : Functor (ArchKernelM minArch) where
  map f m := ⟨f <$> m.run⟩

instance : Pure (ArchKernelM minArch) where
  pure a := ⟨pure a⟩

instance : Bind (ArchKernelM minArch) where
  bind m f := ⟨do let a ← m.run; (f a).run⟩

instance : Applicative (ArchKernelM minArch) where
  seq f x := ⟨Seq.seq f.run (fun _ => (x ()).run)⟩

instance : Monad (ArchKernelM minArch) := {}

instance : MonadLift KernelM (ArchKernelM .Ampere) where
  monadLift m := ⟨m⟩

/-- Lift portable (Ampere) operations into any architecture.
    Since Ampere capabilities are available on all architectures,
    Ampere operations can be used anywhere. -/
def liftPortable (m : ArchKernelM .Ampere α) : ArchKernelM arch α := ⟨m.run⟩

/-- Lift from a lower architecture to a higher one.
    Requires proof that minArch ≤ targetArch. -/
def liftArch (m : ArchKernelM minArch α) (h : minArch ≤ targetArch := by decide)
    : ArchKernelM targetArch α := ⟨m.run⟩

/-- Use an arch-specific operation when targeting a compatible architecture.
    The proof h ensures we're targeting an architecture that supports the operation. -/
def requireArch (op : ArchKernelM minArch α)
    (h : minArch ≤ targetArch := by decide) : ArchKernelM targetArch α := ⟨op.run⟩

/-- Combine two computations, taking the max of their architecture requirements -/
def combine {arch1 arch2 : ArchLevel}
    (m1 : ArchKernelM arch1 α) (m2 : ArchKernelM arch2 β)
    : ArchKernelM (if arch1.toNat ≥ arch2.toNat then arch1 else arch2) (α × β) := ⟨do
  let a ← m1.run
  let b ← m2.run
  pure (a, b)⟩

/-- Sequence two computations, ignoring the architecture indexing at runtime -/
def andThen' {arch1 arch2 : ArchLevel} (m1 : ArchKernelM arch1 Unit) (m2 : ArchKernelM arch2 α)
    : KernelM α := do
  m1.run
  m2.run

end ArchKernelM

/-- Generate a fresh variable -/
def archFreshVar {minArch : ArchLevel} : ArchKernelM minArch VarId := ⟨freshVar⟩

/-- Emit a statement -/
def archEmit {minArch : ArchLevel} (stmt : KStmt) : ArchKernelM minArch Unit := ⟨emit stmt⟩

/-- Add a comment -/
def archComment {minArch : ArchLevel} (text : String) : ArchKernelM minArch Unit := ⟨comment text⟩

/-- Get current state -/
def archGet {minArch : ArchLevel} : ArchKernelM minArch KernelState := ⟨get⟩

/-- Modify state -/
def archModify {minArch : ArchLevel} (f : KernelState → KernelState) : ArchKernelM minArch Unit :=
  ⟨modify f⟩

/-- Helper to run an ArchKernelM computation and build a kernel -/
def runArchKernel (arch : ArchLevel) (m : ArchKernelM arch Unit)
    (name : String) (params : Array KParam := #[]) : Kernel :=
  buildKernelM name arch.toGpuArch params m.run

/-- Run an ArchKernelM and extract just the kernel state -/
def evalArchKernel (arch : ArchLevel) (m : ArchKernelM arch Unit) : KernelState :=
  (m.run.run { arch := arch.toGpuArch }).2

/-- Capture the body of an ArchKernelM for use in control flow -/
def archCaptureBody {minArch : ArchLevel} (body : ArchKernelM minArch Unit)
    : ArchKernelM minArch (Array KStmt) := ⟨captureBody body.run⟩

/-- Create a for loop with architecture indexing -/
def archForLoop {minArch : ArchLevel}
    (lo hi : Nat) (body : ArchKernelM minArch Unit) : ArchKernelM minArch Unit := ⟨do
  let v ← freshVar
  let bodyStmts ← captureBody body.run
  emit (.forLoop v lo hi bodyStmts)⟩

/-- Create an if statement with architecture indexing -/
def archIfStmt {minArch : ArchLevel}
    (cond : VarId) (thenBody elseBody : ArchKernelM minArch Unit)
    : ArchKernelM minArch Unit := ⟨do
  let thenStmts ← captureBody thenBody.run
  let elseStmts ← captureBody elseBody.run
  emit (.ifStmt cond thenStmts elseStmts)⟩

/-- Synchronization operations -/
def archSync {minArch : ArchLevel} (barrier : Nat := 0) : ArchKernelM minArch Unit :=
  ⟨emit (.sync barrier)⟩

end Tyr.GPU.Codegen.Arch
