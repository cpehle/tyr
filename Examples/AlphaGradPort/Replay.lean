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

private structure PackedFloatArrays where
  offsets : Array Nat
  values : Array Float
  deriving Repr, Inhabited, ToJson, FromJson

private structure ReplayBufferDiskV2 where
  version : Nat := 2
  capacity : Nat
  cursor : Nat
  isFull : Bool
  kinds : Array ReplayKind
  features : PackedFloatArrays
  actions : Array Nat
  oldLogProbs : Array Float
  rewards : Array Float
  valueTargets : Array Float
  advantages : Array Float
  policyTargets : PackedFloatArrays
  dones : Array Bool
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

private def packFloatArrays
    (rows : Array (Array Float)) :
    PackedFloatArrays := Id.run do
  let mut offsets : Array Nat := #[0]
  let mut values : Array Float := #[]
  let mut cursor := 0
  for row in rows do
    cursor := cursor + row.size
    offsets := offsets.push cursor
    for value in row do
      values := values.push value
  return { offsets := offsets, values := values }

private def unpackFloatArrays?
    (packed : PackedFloatArrays) :
    Except String (Array (Array Float)) := do
  if packed.offsets.isEmpty then
    pure #[]
  else if packed.offsets.getD 0 1 != 0 then
    throw "Packed float arrays must start at offset 0."
  else
    let mut rows : Array (Array Float) := #[]
    for i in [1:packed.offsets.size] do
      let start := packed.offsets.getD (i - 1) 0
      let stop := packed.offsets.getD i start
      if start > stop || stop > packed.values.size then
        throw s!"Packed float array offsets are invalid at row {i - 1}: [{start}, {stop})."
      rows := rows.push (packed.values.extract start stop)
    pure rows

private def ReplayBuffer.toDiskV2 (buf : ReplayBuffer) : ReplayBufferDiskV2 :=
  let samples := buf.orderedSamples
  {
    capacity := buf.capacity
    cursor := buf.cursor
    isFull := buf.isFull
    kinds := samples.map (·.kind)
    features := packFloatArrays (samples.map (·.features))
    actions := samples.map (·.action)
    oldLogProbs := samples.map (·.oldLogProb)
    rewards := samples.map (·.reward)
    valueTargets := samples.map (·.valueTarget)
    advantages := samples.map (·.advantage)
    policyTargets := packFloatArrays (samples.map (·.policyTarget))
    dones := samples.map (·.done)
  }

private def ReplayBuffer.ofDiskV2
    (disk : ReplayBufferDiskV2) :
    Except String ReplayBuffer := do
  let features ← unpackFloatArrays? disk.features
  let policyTargets ← unpackFloatArrays? disk.policyTargets
  let sampleCount := disk.kinds.size
  let sameLen (n : Nat) (label : String) : Except String Unit :=
    if n = sampleCount then
      pure ()
    else
      throw s!"Replay disk {label} length mismatch: expected {sampleCount}, got {n}."
  sameLen features.size "features"
  sameLen disk.actions.size "actions"
  sameLen disk.oldLogProbs.size "oldLogProbs"
  sameLen disk.rewards.size "rewards"
  sameLen disk.valueTargets.size "valueTargets"
  sameLen disk.advantages.size "advantages"
  sameLen policyTargets.size "policyTargets"
  sameLen disk.dones.size "dones"
  let mut samples : Array ReplaySample := #[]
  for i in [:sampleCount] do
    samples := samples.push {
      kind := disk.kinds.getD i default
      features := features.getD i #[]
      action := disk.actions.getD i 0
      oldLogProb := disk.oldLogProbs.getD i 0.0
      reward := disk.rewards.getD i 0.0
      valueTarget := disk.valueTargets.getD i 0.0
      advantage := disk.advantages.getD i 0.0
      policyTarget := policyTargets.getD i #[]
      done := disk.dones.getD i false
    }
  pure {
    capacity := if disk.capacity = 0 then 1 else disk.capacity
    cursor := disk.cursor
    isFull := disk.isFull
    samples := samples
  }

def saveReplayBuffer (path : System.FilePath) (buf : ReplayBuffer) : IO Unit := do
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  IO.FS.writeFile path (Lean.toJson buf.toDiskV2).compress

def loadReplayBuffer (path : System.FilePath) : IO ReplayBuffer := do
  let content ← IO.FS.readFile path
  match Json.parse content with
  | .error err =>
    throw <| IO.userError s!"Replay buffer JSON parse failed at {path}: {err}"
  | .ok json =>
    match (Lean.fromJson? json : Except String ReplayBufferDiskV2) with
    | .ok disk =>
      match ReplayBuffer.ofDiskV2 disk with
      | .ok buf => pure buf
      | .error err =>
        throw <| IO.userError s!"Replay buffer V2 decode failed at {path}: {err}"
    | .error _ =>
      match (Lean.fromJson? json : Except String ReplayBuffer) with
      | .error err =>
        throw <| IO.userError s!"Replay buffer decode failed at {path}: {err}"
      | .ok buf =>
        pure buf

end Examples.AlphaGradPort
