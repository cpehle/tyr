import Tyr
import Tyr.Checkpoint
import Examples.GPT.GPT
import Examples.GPT.Train
import Examples.NanoProof.Model
import LeanTest

open torch
open torch.gpt
open torch.train
open torch.checkpoint

-- Shakespeare character vocabulary (65 chars)
def shakespeareChars : String :=
  "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def charToInt (c : Char) : Int64 :=
  match shakespeareChars.toList.findIdx? (· == c) with
  | some idx => idx.toInt64
  | none => 0  -- fallback to newline

def intToChar (i : Int64) : Char :=
  shakespeareChars.toList.getD i.toUInt64.toNat '\n'

def encode (s : String) : Array Int64 :=
  s.toList.toArray.map charToInt

def decode (tokens : Array Int64) : String :=
  String.ofList (tokens.toList.map intToChar)

/-- Test tokenizer encode/decode -/
@[test]
def testTokenizer : IO Unit := do
  let tok := tokenizer.createBase
  LeanTest.assertTrue (tok.vocabSize > 0) "Vocab size should be positive"

  let testText := "Hello, world! This is a test."
  let pretokens := tokenizer.pretokenizeFull testText
  LeanTest.assertTrue (pretokens.size > 0) "Should produce pretokens"

/-- Test NanoProof model forward pass -/
@[test]
def testNanoProof : IO Unit := do
  let cfg := torch.nanoproof.Config.tiny
  let params ← torch.nanoproof.NanoProofParams.init cfg (withValueHead := false)
  let rotaryCache ← torch.nanoproof.RotaryCache.init cfg.sequence_len cfg.headDim

  let batchSize : UInt64 := 2
  let seqLen : UInt64 := 32
  let input ← randint 0 cfg.vocab_size.toInt64 #[batchSize, seqLen]

  -- Test 1: Create ModelOutput directly (not via forward)
  let dummyLogits ← randn #[batchSize, seqLen, cfg.vocab_size] false
  let directOutput : torch.nanoproof.ModelOutput batchSize seqLen cfg.vocab_size cfg.num_value_bins := {
    policy_logits := dummyLogits
    value_logits := none
  }
  
  match directOutput.value_logits with
  | none => pure ()
  | some _ => LeanTest.fail "Direct struct: value_logits should be none"

  -- Test 2: Create ModelOutput with Some tensor
  let dummyValue ← randn #[batchSize, seqLen, cfg.num_value_bins] false
  let directOutput2 : torch.nanoproof.ModelOutput batchSize seqLen cfg.vocab_size cfg.num_value_bins := {
    policy_logits := dummyLogits
    value_logits := some dummyValue
  }
  
  match directOutput2.value_logits with
  | none => LeanTest.fail "Direct struct 2: value_logits should be some"
  | some _ => pure ()

  -- Test 3: Try MINIMAL forward (just embedding + linear)
  let outputMin ← torch.nanoproof.forwardMinimal batchSize seqLen params rotaryCache input
  match outputMin.value_logits with
  | none => pure ()
  | some _ => LeanTest.fail "forwardMinimal: value_logits should be none"

  -- Test 4: Now test full forward (IO version)
  let output ← torch.nanoproof.forward batchSize seqLen params rotaryCache input
  match output.value_logits with
  | none => pure ()
  | some _ => pure ()

  -- Test loss computation
  let targets ← randint 0 cfg.vocab_size.toInt64 #[batchSize, seqLen]
  let _lossVal ← torch.nanoproof.loss batchSize seqLen params rotaryCache input targets
  pure ()