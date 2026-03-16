/-
  Tyr/GPU/Codegen/TileDispatch.lean

  Multi-tile runtime dispatch: compile multiple kernel variants with
  different tile sizes and select the best one at runtime based on
  problem dimensions.
-/
import Tyr.GPU.Types
import Tyr.GPU.Codegen.Var
import Tyr.GPU.Codegen.TileTypes
import Tyr.GPU.Codegen.IR
import Tyr.GPU.Codegen.Monad
import Tyr.GPU.Codegen.EmitNew
import Tyr.GPU.Codegen.KernelTemplate
import Tyr.GPU.Codegen.Launch

namespace Tyr.GPU.Codegen

open Tyr.GPU

/-! ## Tile Variant Configuration -/

/-- A single tile variant for multi-tile dispatch -/
structure TileVariant where
  /-- Variant name suffix -/
  name : String
  /-- Tile M dimension -/
  tileM : Nat := 64
  /-- Tile N dimension -/
  tileN : Nat := 64
  /-- Tile K dimension -/
  tileK : Nat := 64
  /-- Minimum problem M dimension for this variant to be selected -/
  minM : Nat := 0
  /-- Minimum problem N dimension for this variant to be selected -/
  minN : Nat := 0
  /-- Block dimension for this variant -/
  blockDim : BlockDim := { x := 128 }
  deriving Repr, Inhabited, BEq

/-- Configuration for multi-tile dispatch -/
structure TileDispatchConfig where
  /-- Base kernel name -/
  name : String
  /-- Target architecture -/
  arch : GpuArch := .SM90
  /-- Kernel parameters (shared across all variants) -/
  params : Array KParam := #[]
  /-- Ordered list of variants (first matching wins) -/
  variants : Array TileVariant := #[]
  deriving Repr, Inhabited

/-! ## Tile Dispatch Result -/

/-- Result of building a tile dispatch: one kernel per variant -/
structure TileDispatchResult where
  /-- Compiled kernel variants -/
  kernels : Array (TileVariant × Kernel)
  /-- The dispatch config used -/
  config : TileDispatchConfig
  deriving Repr

/-! ## Build and Select -/

/-- Build one `Kernel` per tile variant.
    The `body` callback receives the variant and should emit the kernel body. -/
def buildTileDispatch (cfg : TileDispatchConfig)
    (body : TileVariant → KernelM Unit) : TileDispatchResult :=
  let kernels := cfg.variants.map fun variant =>
    let kernelName := s!"{cfg.name}_{variant.name}"
    let kernel := buildKernelM kernelName cfg.arch cfg.params (body variant)
    (variant, kernel)
  { kernels, config := cfg }

/-- Select the best tile variant for the given problem dimensions.
    Returns the first variant whose minM/minN constraints are satisfied.
    Falls back to the last variant (assumed to be the smallest/default). -/
def selectVariant (cfg : TileDispatchConfig) (problemM problemN : Nat)
    : TileVariant :=
  let found := cfg.variants.find? fun v =>
    problemM ≥ v.minM && problemN ≥ v.minN
  found.getD (cfg.variants.back!)

/-- Compute grid dimensions for a tile variant and problem size -/
def computeGridDims (problemM problemN tileM tileN : Nat) : GridDim :=
  { x := divCeil problemN tileN
    y := divCeil problemM tileM }

/-- Select a variant and compute its launch config for the given problem -/
def selectAndComputeLaunch (dispatch : TileDispatchResult)
    (problemM problemN : Nat) : TileVariant × Kernel × LaunchConfig :=
  let variant := selectVariant dispatch.config problemM problemN
  let kernel := dispatch.kernels.find? (fun (v, _) => v == variant)
    |>.map Prod.snd |>.getD dispatch.kernels[0]!.2
  let lc := computeLaunchConfig kernel problemM problemN variant.tileM variant.tileN variant.blockDim
  (variant, kernel, lc)

/-! ## Code Generation -/

/-- Generate C++ source for all dispatch variants -/
def TileDispatchResult.generateAll (dispatch : TileDispatchResult) : String :=
  let sources := dispatch.kernels.map fun (_, kernel) =>
    generateKernel kernel
  -- Combine: use the header from the first, strip duplicates from rest
  if h : sources.size > 0 then
    let first := sources[0]
    let rest := sources.extract 1 sources.size |>.map fun src =>
      -- Strip the common header
      let lines := src.splitOn "\n"
      let bodyLines := lines.filter fun line =>
        !line.startsWith "#include" && !line.startsWith "using namespace"
      String.intercalate "\n" bodyLines
    first ++ "\n" ++ String.intercalate "\n" rest.toList
  else
    ""

end Tyr.GPU.Codegen
