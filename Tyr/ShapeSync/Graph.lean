import Tyr.ShapeSync.Thread

/-!
# Generic dependence graphs for producer/consumer lowering

The graph layer is intentionally small: nodes have a role, a participation
guard, and a list of resource accesses.  It is enough to express the core of
warp-specialized producer/consumer analysis without mentioning TVM statements.
-/

namespace Tyr.ShapeSync

abbrev NodeId := Nat
abbrev ResourceId := Nat

inductive AccessKind where
| read
| write
| readWrite
deriving Repr, BEq, DecidableEq

namespace AccessKind

def reads : AccessKind → Bool
  | .read | .readWrite => true
  | .write => false

def writes : AccessKind → Bool
  | .write | .readWrite => true
  | .read => false

end AccessKind

inductive ProducerKind where
| tma
| simt
| cpAsync
deriving Repr, BEq, DecidableEq

namespace ProducerKind

def isAsyncLike : ProducerKind → Bool
  | .simt | .cpAsync => true
  | .tma => false

end ProducerKind

inductive NodeRole where
| producer (kind : ProducerKind)
| consumer
| other
deriving Repr, BEq, DecidableEq

structure Access where
  resource : ResourceId
  kind : AccessKind
deriving Repr, BEq, DecidableEq

namespace Access

def read (resource : ResourceId) : Access :=
  { resource, kind := .read }

def write (resource : ResourceId) : Access :=
  { resource, kind := .write }

def readWrite (resource : ResourceId) : Access :=
  { resource, kind := .readWrite }

end Access

structure GraphNode where
  id : NodeId
  role : NodeRole := .other
  guard : ThreadPred := .top
  accesses : List Access := []
deriving Repr, BEq, DecidableEq

namespace GraphNode

def producerKind? (node : GraphNode) : Option ProducerKind :=
  match node.role with
  | .producer kind => some kind
  | _ => none

def isProducer (node : GraphNode) : Bool :=
  node.producerKind?.isSome

def isAsyncProducer (node : GraphNode) : Bool :=
  match node.producerKind? with
  | some kind => kind.isAsyncLike
  | none => false

def isPureTmaProducer (node : GraphNode) : Bool :=
  node.producerKind? == some .tma

private def accessesResourceWith (node : GraphNode) (resource : ResourceId)
    (p : AccessKind → Bool) : Bool :=
  node.accesses.any fun access => access.resource == resource && p access.kind

def readsResource (node : GraphNode) (resource : ResourceId) : Bool :=
  accessesResourceWith node resource AccessKind.reads

def writesResource (node : GraphNode) (resource : ResourceId) : Bool :=
  accessesResourceWith node resource AccessKind.writes

def touchesResource (node : GraphNode) (resource : ResourceId) : Bool :=
  readsResource node resource || writesResource node resource

def writtenResources (node : GraphNode) : List ResourceId :=
  node.accesses.filterMap fun access =>
    if access.kind.writes then some access.resource else none

end GraphNode

private def concatMap (xs : List α) (f : α → List β) : List β :=
  xs.foldr (fun x acc => f x ++ acc) []

private def enumerateFrom (i : Nat) : List α → List (Nat × α)
  | [] => []
  | x :: xs => (i, x) :: enumerateFrom (i + 1) xs

private def minNat? : List Nat → Option Nat
  | [] => none
  | x :: xs => some (xs.foldl Nat.min x)

structure AccessGraph where
  nodes : List GraphNode
deriving Repr, BEq, DecidableEq

structure DependencyWindow where
  producer : GraphNode
  resource : ResourceId
  consumer? : Option GraphNode
  firstRead? : Option Nat
  waitPos : Nat
  releasePos : Nat
deriving Repr, BEq, DecidableEq

namespace DependencyWindow

def isPureTma (window : DependencyWindow) : Bool :=
  window.producer.isPureTmaProducer

def adjustedWait (window : DependencyWindow) (waitPos : Nat) : DependencyWindow :=
  { window with waitPos }

end DependencyWindow

namespace AccessGraph

def indexedNodes (graph : AccessGraph) : List (Nat × GraphNode) :=
  enumerateFrom 0 graph.nodes

def producerNodes (graph : AccessGraph) : List GraphNode :=
  graph.nodes.filter GraphNode.isProducer

def consumerStream (graph : AccessGraph) : List (Nat × GraphNode) :=
  enumerateFrom 0 (graph.nodes.filter fun node => !node.isProducer)

def consumerCount (graph : AccessGraph) : Nat :=
  graph.consumerStream.length

def hasAsyncProducer (graph : AccessGraph) : Bool :=
  graph.nodes.any GraphNode.isAsyncProducer

def firstConsumerRead? (graph : AccessGraph) (resource : ResourceId) : Option (Nat × GraphNode) :=
  graph.consumerStream.find? fun (_, node) => node.readsResource resource

def lastConsumerAccess? (graph : AccessGraph) (resource : ResourceId) : Option Nat :=
  graph.consumerStream.foldl
    (fun acc (idx, node) => if node.touchesResource resource then some idx else acc)
    none

def dependencyWindows (graph : AccessGraph) : List DependencyWindow :=
  concatMap graph.producerNodes fun producer =>
    producer.writtenResources.filterMap fun resource =>
      match graph.lastConsumerAccess? resource with
      | none => none
      | some lastAccess =>
          let firstRead := graph.firstConsumerRead? resource
          let firstReadIdx? := firstRead.map Prod.fst
          let consumer? := firstRead.map Prod.snd
          let waitPos := firstReadIdx?.getD 0
          some {
            producer
            resource
            consumer?
            firstRead? := firstReadIdx?
            waitPos
            releasePos := lastAccess + 1
          }

def earliestAsyncRead (graph : AccessGraph) : Nat :=
  let resources :=
    concatMap (graph.producerNodes.filter GraphNode.isAsyncProducer)
      GraphNode.writtenResources
  let firstReads := resources.filterMap fun resource =>
    (graph.firstConsumerRead? resource).map Prod.fst
  (minNat? firstReads).getD graph.consumerCount

def adjustedWindowsForAsync (graph : AccessGraph) : List DependencyWindow :=
  let windows := graph.dependencyWindows
  if graph.hasAsyncProducer then
    let earliest := graph.earliestAsyncRead
    windows.map fun window => window.adjustedWait (Nat.min window.waitPos earliest)
  else
    windows

private def sameWaitRelease : List DependencyWindow → Bool
  | [] => true
  | window :: rest =>
      rest.all fun other =>
        other.waitPos == window.waitPos && other.releasePos == window.releasePos

def canMergePureTmaBarriers (windows : List DependencyWindow)
    (hasAsyncProducer : Bool := false) : Bool :=
  !hasAsyncProducer &&
    windows.length > 1 &&
    windows.all DependencyWindow.isPureTma &&
    sameWaitRelease windows

end AccessGraph

end Tyr.ShapeSync
