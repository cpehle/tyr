/-
  Token Decoding

  Convert token IDs back to strings.
-/
import Tyr.Tokenizer.Types
import Tyr.Tokenizer.ByteLevel

/-!
# `Tyr.Tokenizer.Decode`

Tokenizer submodule for Decode, used in text preprocessing and generation pipelines.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

namespace tokenizer

/-- Decode UTF-8 bytes, falling back to byte-level rendering for invalid streams. -/
private def decodeUtf8Lossy (bytes : ByteArray) : String :=
  match String.fromUTF8? bytes with
  | some s => s
  | none => bytesToByteLevel bytes

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
  return decodeUtf8Lossy bytes

/-- Decode tokens, returning bytes instead of string -/
def decodeToBytes (tok : BPETokenizer) (ids : Array TokenId) : ByteArray := Id.run do
  let mut bytes := ByteArray.empty
  for id in ids do
    bytes := bytes ++ decodeToken tok id
  return bytes

/-- Decode a single token to string (for debugging) -/
def decodeOne (tok : BPETokenizer) (id : TokenId) : String :=
  decodeUtf8Lossy (decodeToken tok id)

end tokenizer
