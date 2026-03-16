/-
  Tyr/GPU/Codegen/Launch.lean

  Launch parameter computation for GPU kernels.
  Provides standard grid sizing, persistent kernel grid sizing,
  and occupancy-aware launch configuration.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.ArchConfig

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-! ## Launch Configuration -/

/-- Grid dimensions (blocks in x, y, z) -/
structure GridDim where
  x : Nat := 1
  y : Nat := 1
  z : Nat := 1
  deriving Repr, Inhabited, BEq

/-- Block dimensions (threads in x, y, z) -/
structure BlockDim where
  x : Nat := 128
  y : Nat := 1
  z : Nat := 1
  deriving Repr, Inhabited, BEq

/-- Complete launch configuration for a GPU kernel -/
structure LaunchConfig where
  /-- Grid dimensions (number of blocks) -/
  grid : GridDim
  /-- Block dimensions (threads per block) -/
  block : BlockDim
  /-- Dynamic shared memory in bytes -/
  sharedMemBytes : Nat := 0
  /-- Whether this is a cooperative launch -/
  cooperative : Bool := false
  /-- Cluster size (SM90+ only, 1 = no clustering) -/
  clusterSize : Nat := 1
  deriving Repr, Inhabited, BEq

/-- Total number of threads in a block -/
def BlockDim.numThreads (b : BlockDim) : Nat :=
  b.x * b.y * b.z

/-- Total number of blocks in the grid -/
def GridDim.numBlocks (g : GridDim) : Nat :=
  g.x * g.y * g.z

/-! ## Standard Grid Computation -/

/-- Divide rounding up: (a + b - 1) / b -/
def divCeil (a b : Nat) : Nat :=
  (a + b - 1) / b

/-- Compute standard launch config from problem and tile dimensions.
    Grid is sized to cover the problem with one block per tile. -/
def computeLaunchConfig (kernel : Kernel)
    (problemM problemN : Nat) (tileM tileN : Nat)
    (blockDim : BlockDim := { x := 128 }) : LaunchConfig :=
  let gridX := divCeil problemN tileN
  let gridY := divCeil problemM tileM
  { grid := { x := gridX, y := gridY }
    block := blockDim
    sharedMemBytes := kernel.sharedMemBytes }

/-- Compute launch config for batched kernels (3D grid). -/
def computeBatchedLaunchConfig (kernel : Kernel)
    (batch problemM problemN : Nat) (tileM tileN : Nat)
    (blockDim : BlockDim := { x := 128 }) : LaunchConfig :=
  let gridX := divCeil problemN tileN
  let gridY := divCeil problemM tileM
  { grid := { x := gridX, y := gridY, z := batch }
    block := blockDim
    sharedMemBytes := kernel.sharedMemBytes }

/-! ## Persistent Kernel Grid Sizing -/

/-- Compute launch config for persistent kernels.
    Uses `numSMs * wavesPerSM` blocks to keep the GPU saturated. -/
def computePersistentLaunchConfig (kernel : Kernel)
    (numSMs : Nat) (wavesPerSM : Nat := 1)
    (blockDim : BlockDim := { x := 128 }) : LaunchConfig :=
  let numCTAs := numSMs * wavesPerSM
  { grid := { x := numCTAs }
    block := blockDim
    sharedMemBytes := kernel.sharedMemBytes
    cooperative := true }

/-- Compute persistent launch config using the architecture's typical SM count. -/
def computePersistentLaunchConfigForArch (kernel : Kernel)
    (arch : GpuArch) (wavesPerSM : Nat := 1)
    (blockDim : BlockDim := { x := 128 }) : LaunchConfig :=
  computePersistentLaunchConfig kernel arch.capabilities.typicalSMs wavesPerSM blockDim

/-! ## Occupancy-Aware Launch Configuration -/

/-- Estimate maximum blocks per SM based on shared memory and register usage.
    This is a simplified model; actual occupancy depends on the CUDA runtime. -/
def estimateBlocksPerSM (arch : GpuArch) (sharedMemBytes : Nat)
    (blockDim : BlockDim) : Nat :=
  let caps := arch.capabilities
  let maxBySharedMem :=
    if sharedMemBytes > 0 then caps.maxSharedMem / sharedMemBytes
    else 32  -- upper bound
  let maxByThreads := 2048 / blockDim.numThreads  -- typical max 2048 threads/SM
  Nat.min maxBySharedMem maxByThreads |>.max 1

/-- Compute occupancy-aware launch config that tries to maximize SM utilization.
    Falls back to standard grid sizing if occupancy-based count exceeds the
    problem-required block count. -/
def computeOccupancyLaunchConfig (kernel : Kernel)
    (problemM problemN : Nat) (tileM tileN : Nat)
    (arch : GpuArch) (blockDim : BlockDim := { x := 128 }) : LaunchConfig :=
  let caps := arch.capabilities
  let blocksPerSM := estimateBlocksPerSM arch kernel.sharedMemBytes blockDim
  let occupancyBlocks := caps.typicalSMs * blocksPerSM
  let gridX := divCeil problemN tileN
  let gridY := divCeil problemM tileM
  let problemBlocks := gridX * gridY
  -- Use the smaller of occupancy-optimal and problem-required
  if problemBlocks ≤ occupancyBlocks then
    -- Standard tiling is fine
    { grid := { x := gridX, y := gridY }
      block := blockDim
      sharedMemBytes := kernel.sharedMemBytes }
  else
    -- Persistent style with occupancy-optimal blocks
    { grid := { x := occupancyBlocks }
      block := blockDim
      sharedMemBytes := kernel.sharedMemBytes
      cooperative := true }

/-! ## Utilities -/

/-- Generate a C++ launch configuration string for debugging/code generation -/
def LaunchConfig.toCppComment (lc : LaunchConfig) : String :=
  let cluster := if lc.clusterSize > 1 then s!", cluster={lc.clusterSize}" else ""
  let coop := if lc.cooperative then ", cooperative" else ""
  s!"// Launch: grid=({lc.grid.x}, {lc.grid.y}, {lc.grid.z}), " ++
  s!"block=({lc.block.x}, {lc.block.y}, {lc.block.z}), " ++
  s!"smem={lc.sharedMemBytes}{cluster}{coop}"

end Tyr.GPU.Codegen
