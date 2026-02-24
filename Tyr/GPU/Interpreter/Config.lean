/-
  Tyr/GPU/Interpreter/Config.lean

  Configuration types for GPU interpreters.
  Supports both ThunderKittens-style (simple producer/consumer) and
  Megakernels-style (5 warp roles, page-based memory, DAG scheduling).
-/
import Tyr.GPU.Types

/-!
# `Tyr.GPU.Interpreter.Config`

Configuration types for CUDA interpreters, including warp roles, memory models, and scheduling options.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace Tyr.GPU.Interpreter

open Tyr.GPU

/-- Warp role in the interpreter kernel -/
inductive WarpRole where
  /-- ThunderKittens: loads data to shared memory -/
  | producer
  /-- Both: main compute warps -/
  | consumer
  /-- Megakernels: TMA loads from global → shared -/
  | loader
  /-- Megakernels: TMA stores from shared → global -/
  | storer
  /-- Megakernels: triggers operation execution -/
  | launcher
  /-- Megakernels: instruction fetch, page allocation, semaphore setup -/
  | controller
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq

instance : ToString WarpRole where
  toString
    | .producer => "producer"
    | .consumer => "consumer"
    | .loader => "loader"
    | .storer => "storer"
    | .launcher => "launcher"
    | .controller => "controller"

/-- Memory model for the interpreter -/
inductive MemoryModel where
  /-- ThunderKittens: fixed ring buffer stages for pipelining -/
  | ringBuffer (stages : Nat)
  /-- Megakernels: dynamic page allocation -/
  | pageBased (numPages : Nat)
  /-- Hybrid: both ring buffers and pages available -/
  | hybrid (rings : Nat) (pages : Nat)
  deriving Repr, BEq, Hashable, Inhabited

/-- Scheduling strategy for instruction dispatch -/
inductive ScheduleStrategy where
  /-- Execute instructions in order, round-robin across SMs -/
  | sequential
  /-- Respect DAG dependencies, optionally use cost model for load balancing -/
  | dagBased (useCostModel : Bool := true)
  /-- Group by opcode (wave), then load-balance within each wave -/
  | wave
  /-- Zig-zag assignment for better memory locality -/
  | zigZag
  /-- Pool-based: separate memory and compute SMs -/
  | pooled (memoryFraction : Float)
  deriving Repr, BEq, Inhabited

/-- Barrier scope for synchronization -/
inductive BarrierScope where
  | warp       -- __syncwarp()
  | block      -- __syncthreads()
  | grid       -- Cooperative groups grid sync
  | named      -- Named barrier (mbarrier)
  deriving Repr, BEq, Hashable, Inhabited, DecidableEq

/-- Complete interpreter configuration -/
structure InterpreterConfig where
  /-- Warp roles used in this interpreter -/
  warpRoles : List WarpRole
  /-- Memory management model -/
  memoryModel : MemoryModel
  /-- Scheduling strategy -/
  scheduling : ScheduleStrategy
  /-- Number of consumer warps (main compute) -/
  numConsumerWarps : Nat := 8
  /-- Pipeline depth for instruction prefetching -/
  instructionPipelineDepth : Nat := 2
  /-- Words per serialized instruction -/
  wordsPerInstruction : Nat := 32
  /-- Target GPU architecture -/
  arch : GpuArch := .SM90
  deriving Repr, Inhabited

/-- ThunderKittens-style configuration: simple producer/consumer -/
def thunderKittensConfig : InterpreterConfig := {
  warpRoles := [.producer, .consumer]
  memoryModel := .ringBuffer 3
  scheduling := .sequential
  numConsumerWarps := 4
  instructionPipelineDepth := 1
}

/-- Megakernels-style configuration: 5 warp roles, DAG scheduling -/
def megakernelsConfig : InterpreterConfig := {
  warpRoles := [.loader, .storer, .launcher, .controller, .consumer]
  memoryModel := .pageBased 4
  scheduling := .dagBased true
  numConsumerWarps := 8
  instructionPipelineDepth := 2
}

/-- Hybrid configuration: loader + consumer with mixed memory -/
def hybridConfig : InterpreterConfig := {
  warpRoles := [.loader, .consumer]
  memoryModel := .hybrid 2 4
  scheduling := .dagBased false
  numConsumerWarps := 6
}

/-- Calculate total number of warps needed -/
def InterpreterConfig.totalWarps (cfg : InterpreterConfig) : Nat :=
  let specializedWarps := cfg.warpRoles.filter (· != .consumer) |>.length
  specializedWarps + cfg.numConsumerWarps

/-- Calculate threads per block -/
def InterpreterConfig.threadsPerBlock (cfg : InterpreterConfig) : Nat :=
  cfg.totalWarps * 32

/-- Check if configuration uses producer/consumer model -/
def InterpreterConfig.isProducerConsumer (cfg : InterpreterConfig) : Bool :=
  cfg.warpRoles.contains .producer

/-- Check if configuration uses megakernel model -/
def InterpreterConfig.isMegakernel (cfg : InterpreterConfig) : Bool :=
  cfg.warpRoles.contains .controller

end Tyr.GPU.Interpreter
