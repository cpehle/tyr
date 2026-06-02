import Tyr.ShapeSync.Graph

/-!
# ShapeSync architecture profiles

The ShapeSync graph is intentionally backend-neutral, but backend integrations
still need a compact way to state which synchronization mechanisms and producer
kinds an architecture exposes.  These profiles are descriptive metadata, not
proofs of backend runtime semantics.
-/

namespace Tyr.ShapeSync

inductive BarrierScope where
| cta
| warpGroup
| cluster
| system
deriving Repr, BEq, DecidableEq

namespace BarrierScope

def render : BarrierScope → String
  | .cta => "cta"
  | .warpGroup => "warp_group"
  | .cluster => "cluster"
  | .system => "system"

end BarrierScope

structure ArchitectureProfile where
  name : String
  defaultThreadCtx : ThreadCtx := {}
  barrierScopes : List BarrierScope := [.cta]
  producerKinds : List ProducerKind := [.simt]
  supportsNamedBarrier : Bool := false
  supportsMBarrier : Bool := false
deriving Repr, BEq, DecidableEq

namespace ArchitectureProfile

def supportsScope (profile : ArchitectureProfile) (scope : BarrierScope) : Bool :=
  profile.barrierScopes.any (fun s => s == scope)

def supportsProducerKind (profile : ArchitectureProfile) (kind : ProducerKind) : Bool :=
  profile.producerKinds.any (fun k => k == kind)

def genericGpu : ArchitectureProfile :=
  {
    name := "generic_gpu"
    barrierScopes := [.cta]
    producerKinds := [.simt]
  }

def cudaSm80 : ArchitectureProfile :=
  {
    name := "cuda_sm80"
    barrierScopes := [.cta]
    producerKinds := [.simt, .cpAsync]
  }

def cudaSm90 : ArchitectureProfile :=
  {
    name := "cuda_sm90"
    barrierScopes := [.cta, .warpGroup, .cluster]
    producerKinds := [.simt, .cpAsync, .tma]
    supportsNamedBarrier := true
    supportsMBarrier := true
  }

def cudaSm100 : ArchitectureProfile :=
  {
    name := "cuda_sm100"
    barrierScopes := [.cta, .warpGroup, .cluster]
    producerKinds := [.simt, .cpAsync, .tma]
    supportsNamedBarrier := true
    supportsMBarrier := true
  }

end ArchitectureProfile

end Tyr.ShapeSync
