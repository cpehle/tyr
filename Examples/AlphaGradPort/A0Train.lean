import Examples.AlphaGradPort.Tasks

/-!
  Examples/AlphaGradPort/A0Train.lean

  Shared AlphaGrad-style search loop utilities used by port examples.
-/

namespace Examples.AlphaGradPort

open Tyr.AD.Elim

inductive SearchBackend where
  | gumbel
  | dag
  | dagGumbel
  deriving Repr, Inhabited

instance : ToString SearchBackend where
  toString
    | .gumbel => "gumbel"
    | .dag => "dag"
    | .dagGumbel => "dag-gumbel"

structure RunConfig where
  episodes : Nat := 16
  seed : UInt64 := 250197
  backend : SearchBackend := .dagGumbel
  logEvery : Nat := 4
  deriving Repr

structure RunSummary where
  taskName : String
  episodes : Nat
  backend : SearchBackend
  bestReward : Float
  avgReward : Float
  bestActions0 : Array ActionId0
  bestOrder1 : Array VertexId1
  forwardReward : Float
  reverseReward : Float
  deriving Repr

private def forwardOrder1 (n : Nat) : Array VertexId1 :=
  (Array.range n).map (· + 1)

private def reverseOrder1 (n : Nat) : Array VertexId1 :=
  (Array.range n).reverse.map (· + 1)

private def evalOrderReward? (task : TaskSpec) (order1 : Array VertexId1) : Except String Float := do
  let actions0 ← verticesToActions? task.numVertices order1
  let s0 ← initAlphaGradStateFromEdges? task.edges task.numVertices
  let sf ← replayActions? task.envCfg s0 actions0
  pure sf.cumulativeReward

private def runEpisode?
    (task : TaskSpec)
    (backend : SearchBackend)
    (seed : UInt64) :
    Except String AlphaGradEpisodeResult :=
  match backend with
  | .gumbel =>
    searchEpisodeFromEdges? task.envCfg task.mctsCfg seed task.edges task.numVertices
  | .dag =>
    searchEpisodeDagFromEdges? task.envCfg task.mctsCfg seed task.edges task.numVertices
  | .dagGumbel =>
    searchEpisodeDagGumbelFromEdges? task.envCfg task.mctsCfg seed task.edges task.numVertices

private def avgReward (sum : Float) (episodes : Nat) : Float :=
  if episodes = 0 then 0.0 else sum / Float.ofNat episodes

def runTask (task : TaskSpec) (cfg : RunConfig := {}) : IO (Except String RunSummary) := do
  let forwardReward : Float ←
    match evalOrderReward? task (forwardOrder1 task.numVertices) with
    | .ok r => pure r
    | .error msg =>
      return .error s!"Failed to evaluate forward baseline for {task.name}: {msg}"

  let reverseReward : Float ←
    match evalOrderReward? task (reverseOrder1 task.numVertices) with
    | .ok r => pure r
    | .error msg =>
      return .error s!"Failed to evaluate reverse baseline for {task.name}: {msg}"

  IO.println s!"[AlphaGradPort] task={task.name} backend={cfg.backend} episodes={cfg.episodes} vertices={task.numVertices} edges={task.edges.size}"
  IO.println s!"[AlphaGradPort] baseline forward={forwardReward}, reverse={reverseReward}"

  if cfg.episodes = 0 then
    return .ok {
      taskName := task.name
      episodes := 0
      backend := cfg.backend
      bestReward := 0.0
      avgReward := 0.0
      bestActions0 := #[]
      bestOrder1 := #[]
      forwardReward := forwardReward
      reverseReward := reverseReward
    }

  let mut bestReward := -1.0e30
  let mut bestActions0 : Array ActionId0 := #[]
  let mut bestOrder1 : Array VertexId1 := #[]
  let mut totalReward := 0.0

  for ep in [:cfg.episodes] do
    let epSeed := cfg.seed + UInt64.ofNat (ep + 1)
    match runEpisode? task cfg.backend epSeed with
    | .error msg =>
      return .error s!"Episode {ep + 1} failed for {task.name}: {msg}"
    | .ok out =>
      totalReward := totalReward + out.totalReward
      if out.totalReward > bestReward then
        bestReward := out.totalReward
        bestActions0 := out.actions0
        bestOrder1 := out.order1
      let shouldLog :=
        cfg.logEvery > 0 &&
          (((ep + 1) % cfg.logEvery = 0) || (ep + 1 = cfg.episodes))
      if shouldLog then
        IO.println s!"[AlphaGradPort] ep={ep + 1}/{cfg.episodes} reward={out.totalReward} best={bestReward}"

  let summary : RunSummary := {
    taskName := task.name
    episodes := cfg.episodes
    backend := cfg.backend
    bestReward := bestReward
    avgReward := avgReward totalReward cfg.episodes
    bestActions0 := bestActions0
    bestOrder1 := bestOrder1
    forwardReward := forwardReward
    reverseReward := reverseReward
  }

  IO.println s!"[AlphaGradPort] done task={task.name} best={summary.bestReward} avg={summary.avgReward}"
  IO.println s!"[AlphaGradPort] best_actions0={summary.bestActions0}"
  IO.println s!"[AlphaGradPort] best_order1={summary.bestOrder1}"
  return .ok summary

end Examples.AlphaGradPort
