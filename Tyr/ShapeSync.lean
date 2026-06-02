import Tyr.ShapeSync.Shape
import Tyr.ShapeSync.Thread
import Tyr.ShapeSync.Graph
import Tyr.ShapeSync.Arch
import Tyr.ShapeSync.ProducerConsumer
import Tyr.ShapeSync.Proof

/-!
# Tyr.ShapeSync

Generic Lean-native reasoning for shape and synchronization lowering.

This package extracts the reusable proof core behind TileLang-style analyzer
queries without depending on TVM/TIR:

- shape/range containment and non-unit extent compatibility,
- static divisibility/alignment obligations,
- finite thread-domain predicates,
- synchronization participant-count obligations,
- resource-dependence graphs for producer/consumer barrier placement,
- a low-friction `shape_sync` tactic backed by Lean's own automation.
-/
