import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Capabilities

/-!
# Tyr.GPU.Codegen.Monad

`Tyr.GPU.Codegen.Monad` provides the stateful kernel-construction API.
`KernelM` accumulates IR statements while tracking fresh variable IDs,
target architecture, and shared-memory usage.

This layer turns imperative-looking `do` notation into immutable IR:

- `freshVar` allocates symbolic names,
- `emit` appends `KStmt`,
- `buildKernelM` seals the final `Kernel`.

Most user-facing DSL functions in `Ops` and `Notation` are thin wrappers over
this monad.
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
  /-- Optional CUDA launch bounds: `(maxThreadsPerBlock, minBlocksPerSM)`. -/
  launchBounds : Option (Nat × Nat) := none
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

/-- Attach CUDA `__launch_bounds__` metadata to the generated kernel. -/
def setLaunchBounds (maxThreadsPerBlock minBlocksPerSM : Nat) : KernelM Unit := do
  modify fun s => { s with launchBounds := some (maxThreadsPerBlock, minBlocksPerSM) }

/-- Build a Kernel from the accumulated state -/
def buildKernel (name : String) (params : Array KParam := #[]) : KernelM Kernel := do
  let s ← get
  pure {
    name := name
    arch := s.arch
    params := params
    body := s.body
    sharedMemBytes := s.sharedMemBytes
    launchBounds := s.launchBounds
  }

/-- Run the kernel builder and extract the result -/
def runKernelM (arch : GpuArch := .SM90) (m : KernelM α) : α × KernelState :=
  m.run { arch := arch }

/-- Maximum shared memory in bytes for a runtime `GpuArch` value.
    Mirrors `GpuCapabilities.maxSharedMem` but is usable without a typeclass
    instance at hand (dynamic dispatch on `arch`). -/
def maxSharedMemForArch (arch : GpuArch) : Nat :=
  match arch with
  | .SM80 => GpuCapabilities.maxSharedMem .SM80
  | .SM90 => GpuCapabilities.maxSharedMem .SM90
  | .SM100 => GpuCapabilities.maxSharedMem .SM100

/-- Compare a finalized kernel's `sharedMemBytes` against the architecture's max.
    Returns `.error` with a descriptive message if the budget is exceeded. -/
def checkSharedMemBudget (k : Kernel) : Except String Unit :=
  let maxBytes := maxSharedMemForArch k.arch
  if k.sharedMemBytes > maxBytes then
    .error s!"Kernel {k.name}: shared memory usage {k.sharedMemBytes} bytes exceeds \
             architecture {repr k.arch} limit of {maxBytes} bytes"
  else
    .ok ()

/-- Run and extract just the kernel.
    Note: nextId starts at params.size to avoid conflicts with parameter VarIds -/
def buildKernelM (name : String) (arch : GpuArch := .SM90)
    (params : Array KParam := #[]) (m : KernelM Unit) : Kernel :=
  -- Start nextId at params.size so freshVar doesn't conflict with parameter VarIds
  let (_, state) := m.run { arch := arch, nextId := params.size }
  let k : Kernel := {
    name := name
    arch := state.arch
    params := params
    body := state.body
    sharedMemBytes := state.sharedMemBytes
    launchBounds := state.launchBounds
  }
  -- Warn (do not fail) if the shared-memory budget is exceeded. Some tests
  -- deliberately over-subscribe shared memory expecting dynamic allocation.
  match checkSharedMemBudget k with
  | .ok _ => k
  | .error msg => dbg_trace s!"[tyr] warning: {msg}"; k

/-- Capture loop body: saves state, runs body, extracts statements, restores -/
def captureBlock (body : KernelM α) : KernelM (α × Array KStmt) := do
  let savedBody := (← get).body
  modify fun s => { s with body := #[] }
  let value ← body
  let capturedBody := (← get).body
  modify fun s => { s with body := savedBody }
  pure (value, capturedBody)

/-- Capture a statement block, discarding the action result. -/
def captureBody (body : KernelM Unit) : KernelM (Array KStmt) := do
  let (_, capturedBody) ← captureBlock body
  pure capturedBody

/-- Emit one structured statement whose body is produced by a scoped builder. -/
def emitScoped (wrap : Array KStmt → KStmt) (body : KernelM Unit) : KernelM Unit := do
  let capturedBody ← captureBody body
  emit (wrap capturedBody)

/-- Emit an `if` statement from scoped builder actions. -/
def emitIf (cond : VarId) (thenAction : KernelM Unit) : KernelM Unit :=
  emitScoped (fun thenBody => .ifStmt cond thenBody #[]) thenAction

/-- Emit an `if/else` statement from scoped builder actions. -/
def emitIfElse (cond : VarId) (thenAction elseAction : KernelM Unit) : KernelM Unit := do
  let thenBody ← captureBody thenAction
  let elseBody ← captureBody elseAction
  emit (.ifStmt cond thenBody elseBody)

/-- Emit a warpgroup-specialized scoped block. -/
def emitWarpGroup (wgIdx : Nat) (action : KernelM Unit) : KernelM Unit :=
  emitScoped (fun body => .ifWarpGroup wgIdx body) action

end Tyr.GPU.Codegen
