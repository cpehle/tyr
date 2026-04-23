/-
  Tyr/GPU/Codegen/KernelTemplate.lean

  Phase composition and fused kernel patterns.
  Provides KernelPhase for sequential phase composition with barriers,
  and FusedGemm for the common GEMM accumulation + epilogue callback pattern.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.AST
import Tyr.GPU.Codegen.Primitives
import Tyr.GPU.Codegen.Loop
import Tyr.GPU.Codegen.Pipeline
import Tyr.GPU.Codegen.PersistentKernel

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-! ## Kernel Phase Composition

A kernel template is a sequence of named phases separated by barriers.
Each phase emits its own IR statements, and the template sums shared memory.
-/

/-- A single kernel phase -/
structure KernelPhase where
  /-- Phase name (emitted as comment) -/
  name : String
  /-- Extra raw shared memory (bytes) beyond what `allocST` inside `emit` accounts for. -/
  extraRawBytes : Nat := 0
  /-- Phase body -/
  emit : KernelM Unit

/-- A kernel template built from ordered phases -/
structure KernelTemplate where
  /-- Kernel name -/
  name : String
  /-- Target architecture -/
  arch : GpuArch := .SM90
  /-- Kernel parameters -/
  params : Array KParam := #[]
  /-- Ordered phases -/
  phases : Array KernelPhase := #[]
  deriving Inhabited

/-- Add a phase to a kernel template -/
def KernelTemplate.addPhase (tmpl : KernelTemplate) (phase : KernelPhase)
    : KernelTemplate :=
  { tmpl with phases := tmpl.phases.push phase }

/-- Build a `Kernel` from a template by emitting all phases with barriers between them. -/
def KernelTemplate.build (tmpl : KernelTemplate) : Kernel :=
  buildKernelM tmpl.name tmpl.arch tmpl.params do
    -- Track extra shared memory from phases (raw bytes beyond allocST).
    let totalExtra := tmpl.phases.foldl (fun acc p => acc + p.extraRawBytes) 0
    if totalExtra > 0 then
      modify fun s => { s with sharedMemBytes := s.sharedMemBytes + totalExtra }
    -- Emit phases with barriers
    for i in List.range tmpl.phases.size do
      match tmpl.phases[i]? with
      | none => pure ()
      | some phase =>
      comment s!"=== Phase: {phase.name} ==="
      KernelPhase.emit phase
      -- Barrier between phases (skip after last)
      if i + 1 < tmpl.phases.size then
        sync

/-! ## Fused GEMM Configuration

The fused GEMM pattern: accumulate tiles in a loop, then call a user-supplied
epilogue with the accumulator. This is the Flux pattern generalized.
-/

/-- Configuration for a fused GEMM -/
structure FusedGemmConfig where
  /-- Tile rows (M dimension) -/
  tileM : Nat := 64
  /-- Tile inner dimension (K dimension) -/
  tileK : Nat := 64
  /-- Tile columns (N dimension) -/
  tileN : Nat := 64
  /-- Number of K-dimension blocks to iterate over -/
  kBlocks : Nat := 8
  /-- Input data type -/
  inDtype : GpuFloat := .BFloat16
  /-- Output/accumulator data type -/
  outDtype : GpuFloat := .Float32
  deriving Repr, Inhabited

/-- Emit a fused GEMM: MMA accumulation loop followed by an epilogue callback.

    The epilogue receives the accumulated register tile and can perform
    arbitrary post-processing (bias add, activation, store, etc.).

    This generates:
    1. Zero-initialized accumulator
    2. K-block loop with MMA
    3. Epilogue callback with the accumulator -/
def fusedGemm (cfg : FusedGemmConfig)
    (lhsShared : ST cfg.inDtype cfg.tileM cfg.tileK .Row)
    (rhsShared : ST cfg.inDtype cfg.tileK cfg.tileN .Col)
    (epilogue : RT cfg.outDtype cfg.tileM cfg.tileN .Row → KernelM Unit)
    (hM : cfg.tileM % 16 = 0 := by decide)
    (hK : cfg.tileK % 16 = 0 := by decide)
    (hN : cfg.tileN % 16 = 0 := by decide)
    : KernelM Unit := do
  let _ := hM; let _ := hK; let _ := hN
  comment s!"Fused GEMM: {cfg.tileM}x{cfg.tileN}x{cfg.tileK}, {cfg.kBlocks} K-blocks"
  -- Allocate accumulator
  let accum ← zeroRT cfg.outDtype cfg.tileM cfg.tileN
  -- Allocate register tiles for MMA operands
  let aReg ← allocRT cfg.inDtype cfg.tileM cfg.tileK
  let bReg ← allocRT cfg.inDtype cfg.tileK cfg.tileN .Col
  -- K-block loop
  forLoop 0 cfg.kBlocks do
    load aReg lhsShared
    load bReg rhsShared
    mma accum aReg bReg accum hM hK hN
    sync
  -- Epilogue
  comment "GEMM epilogue"
  epilogue accum

end Tyr.GPU.Codegen
