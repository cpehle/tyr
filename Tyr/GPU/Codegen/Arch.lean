import Tyr.GPU.Codegen.Arch.Level
import Tyr.GPU.Codegen.Arch.Monad
import Tyr.GPU.Codegen.Arch.Capabilities
import Tyr.GPU.Codegen.Arch.Ops
import Tyr.GPU.Codegen.Arch.Polymorphic

/-!
# Tyr.GPU.Codegen.Arch

`Tyr.GPU.Codegen.Arch` is the architecture-polymorphic facade for GPU kernel generation.
It re-exports the hierarchy, typed monad, capability records, dispatch operations,
and polymorphic compilation utilities.

## Why This Layer Exists

The base DSL can emit kernels for a chosen `GpuArch`, but many kernels need one
source definition with architecture-specific lowering choices (Ampere vs Hopper vs
Blackwell). This module provides that bridge.

## Architecture Hierarchy

```text
SM80 (Ampere)  <  SM90 (Hopper)  <  SM100 (Blackwell)
   - warp mma      - + WGMMA         - + tcgen05
   - cp.async      - + TMA           - + 2-CTA MMA
   - 164KB smem    - + 228KB smem    - + 256KB smem
```

## Typical Use

1. Build kernels in `ArchKernelM` with architecture-aware helpers.
2. Use `smart*` operations (`smartMMA`, `smartLoadAsync`, etc.).
3. Compile a `PolyKernel` for one or many architecture targets.

This keeps kernel intent centralized while still generating tuned backend code.
-/

namespace Tyr.GPU.Codegen

-- Re-export Arch namespace for convenience
export Arch (
  -- Level.lean exports
  ArchLevel
  ArchLe

  -- Monad.lean exports
  ArchKernelM
  archFreshVar
  archEmit
  archComment
  archGet
  archModify
  archSync
  archForLoop
  archIfStmt
  archCaptureBody
  runArchKernel
  evalArchKernel

  -- Capabilities.lean exports
  ArchConfig
  ArchCapabilitiesRecord
  HasTMACapability
  HasWGMMACapability
  HasFP8Capability
  HasDSMCapability

  -- Ops.lean exports
  HasMMA
  HasAsyncLoad
  HasAsyncStore
  HasTileAlloc
  smartMMA
  smartMMAT
  smartLoadAsync
  smartStoreAsync
  whenHopper
  whenBlackwell
  whenTMA
  whenWGMMA

  -- Polymorphic.lean exports
  PolyKernel
  PolyKernelBody
  CompiledPolyKernel
  RegisteredPolyKernel
  mkPolyKernel
  mkPolyKernelWithConfig
  mkArchPolyKernel
  PolyKernelBuilder
  polyKernel
  generateUnifiedLauncher
  generateHeaderDecls
)

end Tyr.GPU.Codegen
