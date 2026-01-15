/-
  Tyr/GPU/Codegen/Loop.lean

  ForIn instance for GPU kernel loops using Lean's standard for syntax.

  Provides two loop styles:
  1. `for i in krange lo hi do ...` - standard Lean4 for syntax with VarId access
  2. `forLoop lo hi do ...` - simple loop without index access (legacy)
-/
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad

namespace Tyr.GPU.Codegen

/-! ## Kernel Loop Index

A runtime loop index that can be used in coordinate calculations.
This wraps a VarId so it can be used with RTileCoord and global memory operations.
-/

/-- Kernel loop index - represents a runtime loop variable.
    Use this with `for i in krange lo hi do ...` to get access to the loop variable. -/
structure KIdx where
  /-- The underlying VarId -/
  id : VarId
  deriving Repr, Inhabited

/-- Get the VarId from a kernel index -/
def KIdx.varId (idx : KIdx) : VarId := idx.id

/-! ## Kernel Range

A range for GPU kernel loops. Unlike Std.Range, this generates GPU IR
rather than iterating at Lean compile time.
-/

/-- Range for GPU loops -/
structure KRange where
  lo : Nat
  hi : Nat
  deriving Repr, Inhabited

/-- Create a kernel range.
    Usage: `for i in krange 0 numBlocks do ...` -/
def krange (lo hi : Nat) : KRange := ⟨lo, hi⟩

/-- Notation for kernel ranges: gpu[lo:hi]
    Alternative syntax using brackets -/
scoped notation "gpu[" lo ":" hi "]" => krange lo hi

/-- Size of range -/
def KRange.size (r : KRange) : Nat := r.hi - r.lo

/-! ## ForIn Instance

Enables standard Lean4 `for` syntax for kernel loops.
The loop body receives a KIdx that can be used in coordinate calculations.

Example:
```lean
for kvIdx in krange 0 numKvBlocks do
  let coord := { b := batchId, d := headId, r := kvIdx.id, c := zeroId : RTileCoord }
  loadGlobalCoord sK K_ptr coord.b coord.d coord.r coord.c
```
-/

/-- ForIn instance for KernelM with KRange, yielding KIdx.
    This enables `for i in krange lo hi do ...` syntax in kernels.

    The loop variable `i` is a KIdx wrapping the VarId of the loop counter.
    Use `i.id` or `i.varId` to access the VarId for coordinate calculations.

    Note: The loop body is executed once to capture the IR statements.
    The actual iteration happens at CUDA runtime, not Lean compile time.
-/
instance : ForIn KernelM KRange KIdx where
  forIn range init f := do
    let loopVar ← freshVar
    let idx : KIdx := ⟨loopVar⟩
    let capturedBody ← captureBody do
      -- Execute body once with the KIdx to capture statements
      -- The return value is discarded - we only care about side effects (emitted statements)
      let _ ← f idx init
      pure ()
    emit (.forLoop loopVar range.lo range.hi capturedBody)
    pure init

/-! ## Legacy Loop Functions

These are kept for backward compatibility. New code should prefer the `for` syntax.
-/

/-- For loop that captures body as IR statements (legacy).
    Prefer: `for _ in krange lo hi do ...` -/
def forLoop (lo hi : Nat) (body : KernelM Unit) : KernelM Unit := do
  let loopVar ← freshVar
  let capturedBody ← captureBody body
  emit (.forLoop loopVar lo hi capturedBody)

/-- For loop with VarId exposed (legacy).
    Prefer: `for i in krange lo hi do ... i.id ...` -/
def forLoopVar (lo hi : Nat) (body : VarId → KernelM Unit) : KernelM Unit := do
  let loopVar ← freshVar
  let capturedBody ← captureBody (body loopVar)
  emit (.forLoop loopVar lo hi capturedBody)

/-- For loop with iteration variable exposed (legacy).
    Note: This passes a Nat, not VarId - use forLoopVar for runtime index access.
    Prefer: `for i in krange lo hi do ...` -/
def forLoopIdx (lo hi : Nat) (body : Nat → KernelM Unit) : KernelM Unit := do
  let loopVar ← freshVar
  -- Execute body with a representative index to capture statements
  let capturedBody ← captureBody (body lo)
  emit (.forLoop loopVar lo hi capturedBody)

/-! ## Loop Utilities -/

/-- Unrolled loop (for small, known iteration counts).
    This fully unrolls at Lean compile time - use for small fixed-size loops. -/
def unroll (n : Nat) (body : Nat → KernelM Unit) : KernelM Unit := do
  for i in List.range n do
    body i

/-- Pipelined loop with prologue/main/epilogue for software pipelining -/
def pipelinedLoop (lo hi stages : Nat) (body : Nat → KernelM Unit) : KernelM Unit := do
  comment s!"Pipelined loop: {stages} stages"
  -- Prologue: fill pipeline
  comment "Prologue"
  for i in List.range (Nat.min stages (hi - lo)) do
    body (lo + i)
  -- Main loop
  if hi - lo > stages then
    comment "Main loop"
    forLoop (lo + stages) hi do
      body lo  -- Representative body
  -- Epilogue would be similar

end Tyr.GPU.Codegen
