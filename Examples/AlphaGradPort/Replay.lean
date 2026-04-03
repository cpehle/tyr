import Lean.Data.Json
import Lean.Data.Json.FromToJson

namespace Examples.AlphaGradPort

open Lean

inductive ReplayKind where
  | ppo
  | alphazero
  deriving Repr, Inhabited, BEq, ToJson, FromJson

structure ReplaySample where
  kind : ReplayKind
  features : Array Float
  action : Nat := 0
  oldLogProb : Float := 0.0
  reward : Float := 0.0
  valueTarget : Float := 0.0
  advantage : Float := 0.0
  policyTarget : Array Float := #[]
  done : Bool := false
  deriving Repr, Inhabited, ToJson, FromJson

structure ReplayBuffer where
  capacity : Nat
  cursor : Nat := 0
  isFull : Bool := false
  samples : Array ReplaySample := #[]
  deriving Repr, Inhabited, ToJson, FromJson

def ReplayBuffer.empty (capacity : Nat) : ReplayBuffer :=
  { capacity := if capacity = 0 then 1 else capacity }

def ReplayBuffer.size (buf : ReplayBuffer) : Nat :=
  if buf.isFull then
    buf.capacity
  else
    buf.samples.size

def ReplayBuffer.orderedSamples (buf : ReplayBuffer) : Array ReplaySample :=
  if !buf.isFull then
    buf.samples
  else
    buf.samples.extract buf.cursor buf.samples.size ++
      buf.samples.extract 0 buf.cursor

def ReplayBuffer.filterByKind (buf : ReplayBuffer) (kind : ReplayKind) : Array ReplaySample :=
  buf.orderedSamples.filter (fun sample => sample.kind == kind)

def ReplayBuffer.push (buf : ReplayBuffer) (sample : ReplaySample) : ReplayBuffer :=
  let capacity := if buf.capacity = 0 then 1 else buf.capacity
  if buf.samples.size < capacity then
    let samples := buf.samples.push sample
    let isFull := samples.size >= capacity
    let cursor :=
      if isFull then
        0
      else
        samples.size
    { capacity, cursor, isFull, samples }
  else
    let cursor := buf.cursor % capacity
    let samples := buf.samples.set! cursor sample
    let cursor' := (cursor + 1) % capacity
    { capacity, cursor := cursor', isFull := true, samples }

def ReplayBuffer.pushBatch (buf : ReplayBuffer) (batch : Array ReplaySample) : ReplayBuffer :=
  batch.foldl (init := buf) fun acc sample => acc.push sample

private def lcgA : UInt64 := 6364136223846793005
private def lcgC : UInt64 := 1442695040888963407

private def mix (x : UInt64) : UInt64 :=
  x * lcgA + lcgC

def ReplayBuffer.sampleBatch
    (buf : ReplayBuffer)
    (seed : UInt64)
    (batchSize : Nat)
    (kind? : Option ReplayKind := none) :
    Except String (Array ReplaySample × UInt64) := do
  let pool :=
    match kind? with
    | some kind => buf.filterByKind kind
    | none => buf.orderedSamples
  if pool.isEmpty then
    throw "Replay buffer has no samples for the requested batch."
  let actualBatch := if batchSize = 0 then 1 else batchSize
  let mut out : Array ReplaySample := #[]
  let mut key := seed
  for _ in [:actualBatch] do
    key := mix key
    let idx := (key.toNat) % pool.size
    out := out.push (pool.getD idx default)
  pure (out, key)

def ReplayBuffer.latestBatch
    (buf : ReplayBuffer)
    (batchSize : Nat)
    (kind? : Option ReplayKind := none) :
    Array ReplaySample :=
  let pool :=
    match kind? with
    | some kind => buf.filterByKind kind
    | none => buf.orderedSamples
  let n := Nat.min batchSize pool.size
  pool.extract (pool.size - n) pool.size

def saveReplayBuffer (path : System.FilePath) (buf : ReplayBuffer) : IO Unit := do
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  IO.FS.writeFile path (Lean.toJson buf).compress

def loadReplayBuffer (path : System.FilePath) : IO ReplayBuffer := do
  let content ← IO.FS.readFile path
  match Json.parse content with
  | .error err =>
    throw <| IO.userError s!"Replay buffer JSON parse failed at {path}: {err}"
  | .ok json =>
    match (Lean.fromJson? json : Except String ReplayBuffer) with
    | .error err =>
      throw <| IO.userError s!"Replay buffer decode failed at {path}: {err}"
    | .ok buf =>
      pure buf

end Examples.AlphaGradPort
