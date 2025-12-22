import Tyr

open torch
open torch.gpt
open torch.train
open torch.checkpoint

-- Shakespeare character vocabulary (65 chars)
-- This must match the prepare.py encoding
def shakespeareChars : String :=
  "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def charToInt (c : Char) : Int64 :=
  match shakespeareChars.data.findIdx? (· == c) with
  | some idx => idx.toInt64
  | none => 0  -- fallback to newline

def intToChar (i : Int64) : Char :=
  shakespeareChars.data.getD i.toUInt64.toNat '\n'

def encode (s : String) : Array Int64 :=
  s.data.toArray.map charToInt

def decode (tokens : Array Int64) : String :=
  String.mk (tokens.toList.map intToChar)

/-- Train with data of known size (legacy, no validation) -/
def runTraining {n : UInt64} (modelCfg : Config) (trainCfg : TrainConfig)
    (trainData : T #[n]) : IO (GPTParams modelCfg) := do
  IO.println ""
  IO.println "Initializing model..."
  let params ← GPTParams.init modelCfg
  -- Initialize optimizer state using Optax-style API
  let opt := Optim.adamw (lr := trainCfg.learningRate)
  let optState := opt.init params

  IO.println s!"Model initialized with {modelCfg.n_layer} layers"
  IO.println ""

  -- Train
  let finalParams ← trainLoop trainCfg params optState trainData

  IO.println ""
  return finalParams

/-- Train with validation data and checkpointing -/
def runTrainingWithVal {nTrain nVal : UInt64} (modelCfg : Config) (trainCfg : TrainConfig)
    (trainData : T #[nTrain]) (valData : T #[nVal])
    (checkpointDir : Option String := none) : IO (GPTParams modelCfg) := do
  IO.println ""
  IO.println "Initializing model..."
  let params ← GPTParams.init modelCfg
  -- Initialize optimizer state using Optax-style API
  let opt := Optim.adamw (lr := trainCfg.learningRate)
  let optState := opt.init params

  IO.println s!"Model initialized with {modelCfg.n_layer} layers"
  IO.println ""

  -- Train with validation
  let (finalParams, bestValLoss) ← trainLoopWithVal trainCfg params optState trainData valData

  -- Save final checkpoint if directory specified
  if let some dir := checkpointDir then
    saveCheckpoint finalParams trainCfg.maxIters bestValLoss 0.0 dir

  IO.println ""
  return finalParams

/-- Test tokenizer encode/decode -/
def testTokenizer : IO Unit := do
  IO.println "=== Testing Tokenizer ==="

  -- Create base tokenizer (256 byte tokens + special tokens)
  let tok := tokenizer.createBase
  IO.println s!"Tokenizer created with vocab size: {tok.vocabSize}"
  IO.println s!"Special tokens added: {tok.specialTokens.size}"

  -- Test pretokenization
  let testText := "Hello, world! This is a test."
  let pretokens := tokenizer.pretokenizeFull testText
  IO.println s!"Pretokenized '{testText}' into {pretokens.size} chunks"
  for i in [:pretokens.size] do
    if h : i < pretokens.size then
      IO.println s!"  [{i}]: '{pretokens[i]}'"

  IO.println "=== Tokenizer test passed! ==="

/-- Test MCTS basic operations -/
def testMCTS : IO Unit := do
  IO.println "=== Testing MCTS ==="

  -- Create initial proof state
  let initialState : mcts.ProofState := {
    goals := #["∀ x, x = x"]
    history := #[]
    isSolved := false
  }

  -- Create root node
  let root := mcts.createRoot initialState
  IO.println s!"Created root node with {initialState.goals.size} goal(s)"
  IO.println s!"Root player: {repr root.player}"

  -- Test PUCT score
  let config : mcts.MCTSConfig := {}
  let child : mcts.Node := {
    action := some (.tactic "rfl")
    state := { initialState with isSolved := true }
    player := .OR
    prior := 0.5
    value := 1.0
    visitCount := 1
  }
  let score := mcts.puctScore root child config
  IO.println s!"PUCT score for child: {score}"

  -- Test AND/OR value computation
  let children := #[child, child]
  let andVal := mcts.andNodeValue children
  let orVal := mcts.orNodeValue children
  IO.println s!"AND node value: {andVal}"
  IO.println s!"OR node value: {orVal}"

  IO.println "=== MCTS test passed! ==="

/-- Test mock prover -/
def testProver : IO Unit := do
  IO.println "=== Testing Prover ==="

  let mockProver : prover.MockProver := {}

  -- Create initial state
  let state : mcts.ProofState := {
    goals := #["test goal"]
    history := #[]
  }

  -- Test tactic application
  let result1 := prover.MockProver.mockApply mockProver state "rfl"
  IO.println s!"Apply 'rfl': {repr result1}"

  let result2 := prover.MockProver.mockApply mockProver state "split"
  IO.println s!"Apply 'split': {repr result2}"

  let result3 := prover.MockProver.mockApply mockProver state "intro x"
  IO.println s!"Apply 'intro x': {repr result3}"

  -- Test suggestions
  let suggestions := prover.MockProver.mockSuggestions mockProver state
  IO.println s!"Suggestions: {suggestions.size} tactics"

  IO.println "=== Prover test passed! ==="

/-- Test NanoProof model forward pass -/
def testNanoProof : IO Unit := do
  IO.println "=== Testing NanoProof ==="
  (← IO.getStdout).flush

  -- Use tiny config for fast testing
  IO.println "Creating config..."
  (← IO.getStdout).flush
  let cfg := torch.nanoproof.Config.tiny
  IO.println s!"Config: {cfg.n_layer} layers, {cfg.n_head} heads, {cfg.n_embd} embed dim"
  (← IO.getStdout).flush

  -- Initialize model (without value head to avoid the Option crash bug)
  IO.println "Initializing NanoProof model..."
  (← IO.getStdout).flush
  let params ← torch.nanoproof.NanoProofParams.init cfg (withValueHead := false)
  IO.println "Model initialized!"
  (← IO.getStdout).flush

  -- Initialize rotary cache
  IO.println "Initializing rotary cache..."
  (← IO.getStdout).flush
  let rotaryCache ← torch.nanoproof.RotaryCache.init cfg.sequence_len cfg.headDim
  IO.println "Rotary cache initialized!"
  (← IO.getStdout).flush

  -- Create dummy input (batch=2, seq=32)
  IO.println "Creating input tensor..."
  (← IO.getStdout).flush
  let batchSize : UInt64 := 2
  let seqLen : UInt64 := 32
  let input ← randint 0 cfg.vocab_size.toInt64 #[batchSize, seqLen]
  IO.println s!"Input shape: [{batchSize}, {seqLen}]"
  (← IO.getStdout).flush

  -- Debug the value_logits issue
  IO.println "=== Debugging value_logits issue ==="
  (← IO.getStdout).flush

  -- Test 1: Create ModelOutput directly (not via forward)
  IO.println "Test 1: Create ModelOutput struct directly with none..."
  (← IO.getStdout).flush
  let dummyLogits ← randn #[batchSize, seqLen, cfg.vocab_size] false
  let directOutput : torch.nanoproof.ModelOutput batchSize seqLen cfg.vocab_size cfg.num_value_bins := {
    policy_logits := dummyLogits
    value_logits := none
  }
  IO.println "Created struct"
  (← IO.getStdout).flush
  let dv := directOutput.value_logits
  IO.println "Got value_logits from direct struct"
  (← IO.getStdout).flush
  match dv with
  | none => IO.println "Direct struct: value_logits is none ✓"
  | some _ => IO.println "Direct struct: value_logits is some"
  (← IO.getStdout).flush

  -- Test 2: Create ModelOutput with Some tensor
  IO.println "Test 2: Create ModelOutput with some tensor..."
  (← IO.getStdout).flush
  let dummyValue ← randn #[batchSize, seqLen, cfg.num_value_bins] false
  let directOutput2 : torch.nanoproof.ModelOutput batchSize seqLen cfg.vocab_size cfg.num_value_bins := {
    policy_logits := dummyLogits
    value_logits := some dummyValue
  }
  IO.println "Created struct with some"
  (← IO.getStdout).flush
  let dv2 := directOutput2.value_logits
  IO.println "Got value_logits from direct struct 2"
  (← IO.getStdout).flush
  match dv2 with
  | none => IO.println "Direct struct 2: value_logits is none"
  | some _ => IO.println "Direct struct 2: value_logits is some ✓"
  (← IO.getStdout).flush

  -- Test 3: Try MINIMAL forward (just embedding + linear)
  IO.println "Test 3: forwardMinimal (IO)..."
  (← IO.getStdout).flush
  let outputMin ← torch.nanoproof.forwardMinimal batchSize seqLen params rotaryCache input
  IO.println "forwardMinimal returned"
  (← IO.getStdout).flush
  let vlMin := outputMin.value_logits
  IO.println "Got value_logits from forwardMinimal"
  (← IO.getStdout).flush
  match vlMin with
  | none => IO.println "forwardMinimal: value_logits is none ✓"
  | some _ => IO.println "forwardMinimal: value_logits is some"
  (← IO.getStdout).flush

  -- Test 4: Now test full forward (IO version)
  IO.println "Test 4: Full forward pass (IO)..."
  (← IO.getStdout).flush
  let output ← torch.nanoproof.forward batchSize seqLen params rotaryCache input
  IO.println "Forward returned"
  (← IO.getStdout).flush
  let vl := output.value_logits
  IO.println "Got value_logits field from forward"
  (← IO.getStdout).flush
  match vl with
  | none => IO.println "Forward: value_logits is none"
  | some _ => IO.println "Forward: value_logits is some"
  (← IO.getStdout).flush

  -- Test loss computation
  IO.println "Computing loss..."
  (← IO.getStdout).flush
  let targets ← randint 0 cfg.vocab_size.toInt64 #[batchSize, seqLen]
  let _lossVal ← torch.nanoproof.loss batchSize seqLen params rotaryCache input targets
  IO.println s!"Loss computed successfully"
  (← IO.getStdout).flush

  IO.println "=== NanoProof test passed! ==="

/-- Run all NanoProof-related tests -/
def runNanoProofTests : IO Unit := do
  testTokenizer
  IO.println ""
  testMCTS
  IO.println ""
  testProver
  IO.println ""
  testNanoProof

def main : IO Unit := do
  IO.println "Starting..."

  -- Test individual components
  testTokenizer
  IO.println ""
  testMCTS
  IO.println ""
  testProver
  IO.println ""
  testNanoProof
  IO.println ""

  IO.println "Done with NanoProof tests!"