/-
  Token Decoding

  Convert token IDs back to strings.
-/
import Tyr.Tokenizer.Types
import Tyr.Tokenizer.ByteLevel

namespace tokenizer

/-- Decode a single token ID to bytes -/
def decodeToken (tok : BPETokenizer) (id : TokenId) : ByteArray :=
  -- Check if it's a special token first
  match tok.idToSpecial.get? id with
  | some specialStr => specialStr.toUTF8
  | none =>
    -- Regular token
    match tok.idToBytes[id.toNat]? with
    | some bytes => bytes
    | none => ByteArray.empty

/-- Decode an array of token IDs to a string -/
def decode (tok : BPETokenizer) (ids : Array TokenId) : String := Id.run do
  let mut bytes := ByteArray.empty
  for id in ids do
    bytes := bytes ++ decodeToken tok id
  -- Convert bytes to UTF-8 string, replacing invalid sequences
  String.fromUTF8! bytes

/-- Decode tokens, returning bytes instead of string -/
def decodeToBytes (tok : BPETokenizer) (ids : Array TokenId) : ByteArray := Id.run do
  let mut bytes := ByteArray.empty
  for id in ids do
    bytes := bytes ++ decodeToken tok id
  return bytes

/-- Decode a single token to string (for debugging) -/
def decodeOne (tok : BPETokenizer) (id : TokenId) : String :=
  String.fromUTF8! (decodeToken tok id)

end tokenizer
