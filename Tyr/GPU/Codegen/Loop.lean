/-
  Tyr/GPU/Codegen/Loop.lean

  ForIn instance for GPU kernel loops using Lean's standard for syntax.
-/
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad

namespace Tyr.GPU.Codegen

/-- Range for GPU loops -/
structure KRange where
  lo : Nat
  hi : Nat
  deriving Repr, Inhabited

/-- Create a kernel range -/
def krange (lo hi : Nat) : KRange := ⟨lo, hi⟩

/-- Notation for kernel ranges: [lo:hi] -/
scoped notation "[" lo ":" hi "]" => krange lo hi

/-- Size of range -/
def KRange.size (r : KRange) : Nat := r.hi - r.lo

/-- For loop that captures body as IR statements -/
def forLoop (lo hi : Nat) (body : KernelM Unit) : KernelM Unit := do
  let loopVar ← freshVar
  let capturedBody ← captureBody body
  emit (.forLoop loopVar lo hi capturedBody)

/-- For loop with iteration variable exposed -/
def forLoopIdx (lo hi : Nat) (body : Nat → KernelM Unit) : KernelM Unit := do
  let loopVar ← freshVar
  -- Execute body with a representative index to capture statements
  let capturedBody ← captureBody (body lo)
  emit (.forLoop loopVar lo hi capturedBody)

/-- ForIn instance for KernelM with KRange
    This enables `for _ in [0:n] do ...` syntax in kernels.

    Note: The loop body is executed once to capture the IR statements.
    The actual iteration happens at CUDA runtime, not Lean compile time.
-/
instance : ForIn KernelM KRange Nat where
  forIn range init f := do
    let loopVar ← freshVar
    let capturedBody ← captureBody do
      -- Execute body once with initial index to capture statements
      -- The return value is discarded - we only care about side effects (emitted statements)
      let _ ← f range.lo init
      pure ()
    emit (.forLoop loopVar range.lo range.hi capturedBody)
    pure init

/-- Unrolled loop (for small, known iteration counts) -/
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
