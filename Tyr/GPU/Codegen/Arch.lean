/-
  Tyr/GPU/Codegen/Arch.lean

  Multi-architecture kernel generation for ThunderKittens.
  This module re-exports all architecture-related functionality.

  Architecture Hierarchy:
    SM80 (Ampere)  ⊂  SM90 (Hopper)  ⊂  SM100 (Blackwell)
       └─ warp mma      └─ + WGMMA        └─ + tcgen05
       └─ cp.async      └─ + TMA          └─ + 2-CTA MMA
       └─ 164KB smem    └─ + 228KB smem   └─ + 256KB smem

  Usage:
    -- Import this module to get all architecture functionality
    import Tyr.GPU.Codegen.Arch

    -- Define a polymorphic kernel
    def myKernel : PolyKernel := mkPolyKernelWithConfig "my_kernel"
      #[{ name := "input", dtype := .BFloat16, isPointer := true }]
      fun arch cfg => do
        comment s!"Using tile size: {cfg.mmaTileSize}"
        -- Kernel body uses smart operations that auto-dispatch
        ...

    -- Compile for all architectures
    #eval IO.println (myKernel.compile.cppSource)
-/

import Tyr.GPU.Codegen.Arch.Level
import Tyr.GPU.Codegen.Arch.Monad
import Tyr.GPU.Codegen.Arch.Capabilities
import Tyr.GPU.Codegen.Arch.Ops
import Tyr.GPU.Codegen.Arch.Polymorphic

/-!
# `Tyr.GPU.Codegen.Arch`

GPU code generation component for Arch, used to lower high-level tile programs to backend code.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
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
