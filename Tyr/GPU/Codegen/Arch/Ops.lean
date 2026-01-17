/-
  Tyr/GPU/Codegen/Arch/Ops.lean

  Operation dispatch typeclasses for architecture-polymorphic operations.
  Each operation typeclass has instances for different architectures that
  emit the optimal code for that architecture.
-/
import Tyr.GPU.Codegen.Arch.Level
import Tyr.GPU.Codegen.Arch.Monad
import Tyr.GPU.Codegen.Arch.Capabilities
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Ops

namespace Tyr.GPU.Codegen.Arch

open Tyr.GPU
open Tyr.GPU.Codegen

/-! ## MMA Operation Dispatch

MMA uses warp mma on Ampere and WGMMA on Hopper+.
The typeclass instances select the appropriate implementation.
-/

/-- MMA dispatch typeclass: selects optimal MMA implementation for each architecture -/
class HasMMA (arch : ArchLevel) where
  /-- Matrix multiply-accumulate: D = A @ B + C -/
  mma {M K N : Nat} {inDtype accDtype : GpuFloat}
      (dst : RT accDtype M N .Row)
      (a : RT inDtype M K .Row)
      (b : RT inDtype K N .Col)
      (c : RT accDtype M N .Row)
      : ArchKernelM arch Unit
  /-- Matrix multiply with B transposed: D = A @ B^T + C -/
  mmaT {M K N : Nat} {inDtype accDtype : GpuFloat}
      (dst : RT accDtype M N .Row)
      (a : RT inDtype M K .Row)
      (b : RT inDtype N K .Row)
      (c : RT accDtype M N .Row)
      : ArchKernelM arch Unit

/-- Ampere MMA: uses standard warp mma instructions -/
instance : HasMMA .Ampere where
  mma dst a b c := ⟨emit (.mma .AB dst.id a.id b.id c.id)⟩
  mmaT dst a b c := ⟨emit (.mma .ABt dst.id a.id b.id c.id)⟩

/-- Hopper MMA: uses WGMMA with fence/commit semantics -/
instance : HasMMA .Hopper where
  mma dst a b c := ⟨do
    emit (.mmaFence dst.id)
    emit (.mma .AB dst.id a.id b.id c.id)
    emit .mmaCommitGroup⟩
  mmaT dst a b c := ⟨do
    emit (.mmaFence dst.id)
    emit (.mma .ABt dst.id a.id b.id c.id)
    emit .mmaCommitGroup⟩

/-- Blackwell MMA: uses WGMMA with fence/commit (same as Hopper for now) -/
instance : HasMMA .Blackwell where
  mma dst a b c := ⟨do
    emit (.mmaFence dst.id)
    emit (.mma .AB dst.id a.id b.id c.id)
    emit .mmaCommitGroup⟩
  mmaT dst a b c := ⟨do
    emit (.mmaFence dst.id)
    emit (.mma .ABt dst.id a.id b.id c.id)
    emit .mmaCommitGroup⟩

/-! ## Async Load Dispatch

Async loads use cp.async on Ampere and TMA on Hopper+.
-/

/-- Async load dispatch: selects optimal async load for each architecture -/
class HasAsyncLoad (arch : ArchLevel) where
  /-- Async load from global to shared with semaphore -/
  loadAsync {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
      (dst : ST dtype rows cols layout)
      (src : GPtr dtype)
      (coordB coordD coordR coordC : VarId)
      (sem : Semaphore)
      : ArchKernelM arch Unit

/-- Ampere async load: uses cp.async -/
instance : HasAsyncLoad .Ampere where
  loadAsync dst src coordB coordD coordR coordC sem := ⟨
    emit (.loadGlobalAsync dst.id src.id coordB coordD coordR coordC sem.id)⟩

/-- Hopper async load: uses TMA -/
instance : HasAsyncLoad .Hopper where
  loadAsync dst src coordB coordD coordR coordC sem := ⟨
    emit (.loadGlobalAsync dst.id src.id coordB coordD coordR coordC sem.id)⟩

/-- Blackwell async load: uses TMA -/
instance : HasAsyncLoad .Blackwell where
  loadAsync dst src coordB coordD coordR coordC sem := ⟨
    emit (.loadGlobalAsync dst.id src.id coordB coordD coordR coordC sem.id)⟩

/-! ## Async Store Dispatch -/

/-- Async store dispatch: selects optimal async store for each architecture -/
class HasAsyncStore (arch : ArchLevel) where
  /-- Async store from shared to global -/
  storeAsync {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
      (dst : GPtr dtype)
      (src : ST dtype rows cols layout)
      (coordB coordD coordR coordC : VarId)
      : ArchKernelM arch Unit
  /-- Async atomic add store -/
  storeAddAsync {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
      (dst : GPtr dtype)
      (src : ST dtype rows cols layout)
      (coordB coordD coordR coordC : VarId)
      : ArchKernelM arch Unit

instance : HasAsyncStore .Ampere where
  storeAsync dst src coordB coordD coordR coordC := ⟨
    emit (.storeGlobalAsync dst.id src.id coordB coordD coordR coordC)⟩
  storeAddAsync dst src coordB coordD coordR coordC := ⟨
    emit (.storeGlobalAdd dst.id src.id coordB coordD coordR coordC)⟩

instance : HasAsyncStore .Hopper where
  storeAsync dst src coordB coordD coordR coordC := ⟨
    emit (.storeGlobalAsync dst.id src.id coordB coordD coordR coordC)⟩
  storeAddAsync dst src coordB coordD coordR coordC := ⟨
    emit (.storeGlobalAdd dst.id src.id coordB coordD coordR coordC)⟩

instance : HasAsyncStore .Blackwell where
  storeAsync dst src coordB coordD coordR coordC := ⟨
    emit (.storeGlobalAsync dst.id src.id coordB coordD coordR coordC)⟩
  storeAddAsync dst src coordB coordD coordR coordC := ⟨
    emit (.storeGlobalAdd dst.id src.id coordB coordD coordR coordC)⟩

/-! ## Tile Allocation Dispatch

Tile allocation with architecture-optimal sizes.
-/

/-- Tile allocation dispatch: uses optimal tile sizes per architecture -/
class HasTileAlloc (arch : ArchLevel) where
  /-- Allocate register tile with optimal size -/
  allocRT (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row)
      : ArchKernelM arch (RT dtype rows cols layout)
  /-- Allocate shared tile with optimal size -/
  allocST (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row)
      : ArchKernelM arch (ST dtype rows cols layout)
  /-- Allocate zero-initialized register tile -/
  zeroRT (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row)
      : ArchKernelM arch (RT dtype rows cols layout)

private def allocRTImpl (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout)
    : KernelM (RT dtype rows cols layout) := do
  let v ← freshVar
  emit (.declRT v dtype rows cols layout)
  pure ⟨v⟩

private def allocSTImpl (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout)
    : KernelM (ST dtype rows cols layout) := do
  let v ← freshVar
  emit (.declST v dtype rows cols layout)
  let bytes := rows * cols * dtype.bytes
  modify fun s => { s with sharedMemBytes := s.sharedMemBytes + bytes }
  pure ⟨v⟩

private def zeroRTImpl (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout)
    : KernelM (RT dtype rows cols layout) := do
  let tile ← allocRTImpl dtype rows cols layout
  emit (.unary .Zero tile.id tile.id)
  pure tile

instance : HasTileAlloc .Ampere where
  allocRT dtype rows cols layout := ⟨allocRTImpl dtype rows cols layout⟩
  allocST dtype rows cols layout := ⟨allocSTImpl dtype rows cols layout⟩
  zeroRT dtype rows cols layout := ⟨zeroRTImpl dtype rows cols layout⟩

instance : HasTileAlloc .Hopper where
  allocRT dtype rows cols layout := ⟨allocRTImpl dtype rows cols layout⟩
  allocST dtype rows cols layout := ⟨allocSTImpl dtype rows cols layout⟩
  zeroRT dtype rows cols layout := ⟨zeroRTImpl dtype rows cols layout⟩

instance : HasTileAlloc .Blackwell where
  allocRT dtype rows cols layout := ⟨allocRTImpl dtype rows cols layout⟩
  allocST dtype rows cols layout := ⟨allocSTImpl dtype rows cols layout⟩
  zeroRT dtype rows cols layout := ⟨zeroRTImpl dtype rows cols layout⟩

/-! ## Smart Operations (Auto-dispatch)

These operations automatically select the optimal implementation
based on the target architecture.
-/

/-- Smart MMA: dispatches to optimal implementation -/
def smartMMA [HasMMA arch] {M K N : Nat} {inDtype accDtype : GpuFloat}
    (dst : RT accDtype M N .Row)
    (a : RT inDtype M K .Row)
    (b : RT inDtype K N .Col)
    (c : RT accDtype M N .Row) : ArchKernelM arch Unit :=
  HasMMA.mma dst a b c

/-- Smart MMA with B transposed: dispatches to optimal implementation -/
def smartMMAT [HasMMA arch] {M K N : Nat} {inDtype accDtype : GpuFloat}
    (dst : RT accDtype M N .Row)
    (a : RT inDtype M K .Row)
    (b : RT inDtype N K .Row)
    (c : RT accDtype M N .Row) : ArchKernelM arch Unit :=
  HasMMA.mmaT dst a b c

/-- Smart async load: dispatches to optimal implementation -/
def smartLoadAsync [HasAsyncLoad arch] {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : GPtr dtype)
    (coordB coordD coordR coordC : VarId)
    (sem : Semaphore) : ArchKernelM arch Unit :=
  HasAsyncLoad.loadAsync dst src coordB coordD coordR coordC sem

/-- Smart async store: dispatches to optimal implementation -/
def smartStoreAsync [HasAsyncStore arch] {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : GPtr dtype)
    (src : ST dtype rows cols layout)
    (coordB coordD coordR coordC : VarId) : ArchKernelM arch Unit :=
  HasAsyncStore.storeAsync dst src coordB coordD coordR coordC

/-! ## Conditional Execution Based on Architecture -/

/-- Execute Hopper-specific operation if supported, otherwise use fallback -/
def whenHopper (hopperOp : ArchKernelM .Hopper Unit)
               (fallback : ArchKernelM .Ampere Unit)
               (arch : ArchLevel) : KernelM Unit :=
  if arch.supports .Hopper then hopperOp.run else fallback.run

/-- Execute Blackwell-specific operation if supported, otherwise use fallback -/
def whenBlackwell (blackwellOp : ArchKernelM .Blackwell Unit)
                  (fallback : ArchKernelM .Hopper Unit)
                  (arch : ArchLevel) : KernelM Unit :=
  if arch.supports .Blackwell then blackwellOp.run else fallback.run

/-- Execute operation based on TMA capability -/
def whenTMA (tmaOp : ArchKernelM arch Unit)
            (fallback : ArchKernelM arch Unit) : ArchKernelM arch Unit :=
  if arch.capabilities.hasTMA then tmaOp else fallback

/-- Execute operation based on WGMMA capability -/
def whenWGMMA (wgmmaOp : ArchKernelM arch Unit)
              (fallback : ArchKernelM arch Unit) : ArchKernelM arch Unit :=
  if arch.capabilities.hasWGMMA then wgmmaOp else fallback

/-! ## Architecture-Polymorphic Tile Operations

These lift existing Codegen.Ops operations into the architecture-indexed monad.
-/

section ArchOps
variable {arch : ArchLevel}

/-- Load from shared to register -/
def archLoad {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : RT dtype rows cols layout)
    (src : ST dtype rows cols layout) : ArchKernelM arch Unit :=
  ⟨emit (.load dst.id src.id)⟩

/-- Store from register to shared -/
def archStore {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : ST dtype rows cols layout)
    (src : RT dtype rows cols layout) : ArchKernelM arch Unit :=
  ⟨emit (.store dst.id src.id)⟩

/-- Element-wise add -/
def archAdd {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst a b : RT dtype rows cols layout) : ArchKernelM arch Unit :=
  ⟨emit (.binary .Add dst.id a.id b.id)⟩

/-- Element-wise subtract -/
def archSub {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst a b : RT dtype rows cols layout) : ArchKernelM arch Unit :=
  ⟨emit (.binary .Sub dst.id a.id b.id)⟩

/-- Element-wise multiply -/
def archMul {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst a b : RT dtype rows cols layout) : ArchKernelM arch Unit :=
  ⟨emit (.binary .Mul dst.id a.id b.id)⟩

/-- Element-wise divide -/
def archDiv {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst a b : RT dtype rows cols layout) : ArchKernelM arch Unit :=
  ⟨emit (.binary .Div dst.id a.id b.id)⟩

/-- Element-wise exp -/
def archExp {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : ArchKernelM arch Unit :=
  ⟨emit (.unary .Exp dst.id src.id)⟩

/-- Element-wise log -/
def archLog {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : ArchKernelM arch Unit :=
  ⟨emit (.unary .Log dst.id src.id)⟩

/-- Zero a tile -/
def archZero {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (t : RT dtype rows cols layout) : ArchKernelM arch Unit :=
  ⟨emit (.unary .Zero t.id t.id)⟩

/-- Copy tile -/
def archCopy {dtype : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst src : RT dtype rows cols layout) : ArchKernelM arch Unit :=
  ⟨emit (.unary .Copy dst.id src.id)⟩

/-- Type conversion -/
def archConvert {dtype1 dtype2 : GpuFloat} {rows cols : Nat} {layout : TileLayout}
    (dst : RT dtype1 rows cols layout)
    (src : RT dtype2 rows cols layout) : ArchKernelM arch Unit :=
  ⟨emit (.convert dst.id src.id)⟩

/-- Row-wise max reduction -/
def archRowMax {dtype : GpuFloat} {rows cols : Nat}
    (dst : RV dtype rows)
    (src : RT dtype rows cols .Row) : ArchKernelM arch Unit :=
  ⟨emit (.reduce .Max .Row dst.id src.id)⟩

/-- Row-wise sum reduction -/
def archRowSum {dtype : GpuFloat} {rows cols : Nat}
    (dst : RV dtype rows)
    (src : RT dtype rows cols .Row) : ArchKernelM arch Unit :=
  ⟨emit (.reduce .Sum .Row dst.id src.id)⟩

/-- Causal mask -/
def archMakeCausal {dtype : GpuFloat} {rows cols : Nat}
    (dst src : RT dtype rows cols .Row)
    (fillVal : Option Float := none) : ArchKernelM arch Unit :=
  ⟨emit (.mask .MakeCausal dst.id src.id fillVal)⟩

/-- Subtract column vector from each column -/
def archSubCol {dtype : GpuFloat} {rows cols : Nat}
    (dst tile : RT dtype rows cols .Row)
    (vec : RV dtype rows) : ArchKernelM arch Unit :=
  ⟨emit (.binaryBroadcast .Sub .Col dst.id tile.id vec.id)⟩

/-- Divide each column by vector -/
def archDivCol {dtype : GpuFloat} {rows cols : Nat}
    (dst tile : RT dtype rows cols .Row)
    (vec : RV dtype rows) : ArchKernelM arch Unit :=
  ⟨emit (.binaryBroadcast .Div .Col dst.id tile.id vec.id)⟩

/-- Allocate a register vector -/
def archAllocRV (dtype : GpuFloat) (len : Nat) : ArchKernelM arch (RV dtype len) := ⟨do
  let v ← freshVar
  emit (.declRV v dtype len)
  pure ⟨v⟩⟩

/-- Allocate a zero-initialized register vector -/
def archZeroRV (dtype : GpuFloat) (len : Nat) : ArchKernelM arch (RV dtype len) := ⟨do
  let vec ← Codegen.allocRV dtype len
  emit (.unary .Zero vec.id vec.id)
  pure vec⟩

/-- Allocate vector with negative infinity -/
def archNegInftyRV (dtype : GpuFloat) (len : Nat) : ArchKernelM arch (RV dtype len) := ⟨do
  let vec ← Codegen.allocRV dtype len
  emit (.unary .NegInfty vec.id vec.id)
  pure vec⟩

/-- Allocate a semaphore -/
def archAllocSemaphore : ArchKernelM arch Semaphore := ⟨do
  let v ← freshVar
  emit (.declSemaphore v)
  pure ⟨v⟩⟩

/-- Row-wise max with accumulator -/
def archRowMaxAccum {dtype : GpuFloat} {rows cols : Nat}
    (dst : RV dtype rows)
    (src : RT dtype rows cols .Row)
    (accum : RV dtype rows) : ArchKernelM arch Unit :=
  ⟨emit (.reduceAccum .Max .Row dst.id src.id accum.id)⟩

/-- Row-wise sum with accumulator -/
def archRowSumAccum {dtype : GpuFloat} {rows cols : Nat}
    (dst : RV dtype rows)
    (src : RT dtype rows cols .Row)
    (accum : RV dtype rows) : ArchKernelM arch Unit :=
  ⟨emit (.reduceAccum .Sum .Row dst.id src.id accum.id)⟩

end ArchOps

end Tyr.GPU.Codegen.Arch
