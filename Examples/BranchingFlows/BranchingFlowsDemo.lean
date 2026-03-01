/-
  Examples/BranchingFlows/BranchingFlowsDemo.lean

  Minimal, deterministic demo for the Lean BranchingFlows port.
  Focuses on the combinatorial components (forest sampling + bridge output)
  with a toy linear "bridge" on Float states.
-/
import Tyr
import Tyr.Model.BranchingFlows

namespace Examples.BranchingFlows

open torch
open torch.branching

private def clamp01 (x : Float) : Float :=
  max 0.0 (min x 1.0)

private def uniformTimeDist : TimeDist :=
  { cdf := fun t => clamp01 t,
    pdf := fun t => if t < 0.0 || t > 1.0 then 0.0 else 1.0,
    quantile := fun p => clamp01 p }

private def linBridge (_ : Unit) (x0 x1 : Float) (t0 t1 : Float) : Float :=
  let dt := t1 - t0
  x0 + (x1 - x0) * dt

private def meanMerge (a b : Float) (w1 w2 : Nat) : Float :=
  let w := (w1 + w2).toFloat
  if w == 0.0 then a else (a * w1.toFloat + b * w2.toFloat) / w

def runDemo : IO Unit := do
  let x1 : BranchingState Float :=
    BranchingState.mkDefault #[1.0, 2.0, 3.0, 4.0] #[0, 0, 1, 1]
  let x1s := #[x1]
  let times := #[0.5]
  let x0Sampler : FlowNode Float → Float := fun node => node.data
  let (result, _rng) :=
    branchingBridge linBridge () x0Sampler x1s times
      uniformTimeDist uniformTimeDist (sequentialUniformPolicy Float) meanMerge
      (rng := { state := 123 })

  IO.println "=== BranchingFlows Demo (Lean) ==="
  IO.println s!"Batch size: {result.t.size}"
  IO.println s!"Segments in batch[0]: {result.segments[0]!.size}"
  IO.println s!"Xt length: {result.Xt[0]!.state.size}"
  IO.println s!"Used time: {result.t[0]!}"
  IO.println s!"Descendants: {repr result.descendants[0]!}"

def _root_.main (_args : List String) : IO UInt32 := do
  runDemo
  pure 0

end Examples.BranchingFlows
