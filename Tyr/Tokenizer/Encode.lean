/-
  Token Encoding

  BPE encoding implementation.
-/
import Tyr.Tokenizer.Types
import Tyr.Tokenizer.ByteLevel
import Tyr.Tokenizer.Pretokenize

namespace tokenizer

/-- Convert bytes to base token IDs (one token per byte) -/
def bytesToBaseTokens (tok : BPETokenizer) (bytes : ByteArray) : Array TokenId := Id.run do
  let mut result := Array.mkEmpty bytes.size
  for b in bytes.toList do
    -- Each byte maps to a single-byte token
    let singleByte := ByteArray.mk #[b]
    match tok.bytesToId.get? singleByte with
    | some id => result := result.push id
    | none => pure ()  -- Skip unknown bytes (shouldn't happen with proper vocab)
  return result

/-- Find the highest priority merge pair in a token sequence.
    Returns the index and the merge rule if found. -/
def findBestMerge (tok : BPETokenizer) (tokens : Array TokenId) : Option (Nat × MergeRule) := Id.run do
  if tokens.size < 2 then return none

  let mut bestIdx : Option Nat := none
  let mut bestPriority : Nat := tok.merges.size  -- Lower is better

  -- Check each adjacent pair
  for i in [:tokens.size - 1] do
    let left := tokens[i]!
    let right := tokens[i + 1]!
    -- Look up merge
    match tok.mergeLookup.get? (left, right) with
    | some _ =>
      -- Find priority (index in merge list)
      -- This is O(n) which could be optimized with a priority lookup table
      for priority in [:tok.merges.size] do
        let rule := tok.merges[priority]!
        if rule.left == left && rule.right == right then
          if priority < bestPriority then
            bestPriority := priority
            bestIdx := some i
          break
    | none => pure ()

  match bestIdx with
  | none => return none
  | some idx =>
    let left := tokens[idx]!
    let right := tokens[idx + 1]!
    for rule in tok.merges do
      if rule.left == left && rule.right == right then
        return some (idx, rule)
    return none

/-- Apply a single BPE merge at a given position -/
def applyMerge (tokens : Array TokenId) (idx : Nat) (mergeResult : TokenId) : Array TokenId := Id.run do
  if idx + 1 >= tokens.size then return tokens
  let mut newTokens := Array.mkEmpty (tokens.size - 1)
  for i in [:idx] do
    newTokens := newTokens.push tokens[i]!
  newTokens := newTokens.push mergeResult
  for i in [idx + 2:tokens.size] do
    newTokens := newTokens.push tokens[i]!
  return newTokens

/-- BPE encode a single word (pretokenized chunk) -/
partial def encodeWord (tok : BPETokenizer) (word : String) : Array TokenId := Id.run do
  -- Convert to byte-level then to base tokens
  let bytes := word.toUTF8
  let mut tokens := bytesToBaseTokens tok bytes

  if tokens.isEmpty then return #[]

  -- Iteratively apply merges until no more can be applied
  let mut changed := true
  let mut iterations := 0
  let maxIter := 10000  -- Safety limit

  while changed && iterations < maxIter do
    changed := false
    iterations := iterations + 1
    match findBestMerge tok tokens with
    | some (idx, rule) =>
      tokens := applyMerge tokens idx rule.result
      changed := true
    | none => pure ()

  return tokens

/-- Encode a full text string to tokens -/
def encode (tok : BPETokenizer) (text : String) : Array TokenId := Id.run do
  -- Pretokenize first
  let chunks := pretokenizeFull text
  let mut encResult : Array TokenId := #[]

  for chunk in chunks do
    let tokens := encodeWord tok chunk
    encResult := encResult ++ tokens

  return encResult

/-- Encode with special token handling.
    Special tokens in the input are detected and replaced with their IDs. -/
def encodeWithSpecials (tok : BPETokenizer) (text : String) : Array TokenId := Id.run do
  -- Simple approach: split on special tokens, encode each part
  -- For now, just use regular encode
  -- TODO: implement proper special token detection
  encode tok text

/-- Get the number of tokens for a text (without returning them) -/
def countTokens (tok : BPETokenizer) (text : String) : Nat :=
  (encode tok text).size

end tokenizer
