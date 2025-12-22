/-
  Tokenizer I/O

  Binary file format for loading/saving tokenizers.

  Format:
  - Header: magic (8 bytes "TYR_BPE1"), version (u32), vocab_size (u32),
            num_merges (u32), num_special (u32)
  - Vocabulary: For each token: length (u16), bytes (variable)
  - Merges: For each merge: left (u32), right (u32), result (u32)
  - Special tokens: For each: length (u16), string (UTF-8), id (u32)
-/
import Tyr.Tokenizer.Types
import Tyr.Tokenizer.SpecialTokens

namespace tokenizer

/-- Magic number for the file format -/
def magic : ByteArray := "TYR_BPE1".toUTF8

/-- Current format version -/
def formatVersion : UInt32 := 1

/-- Write a UInt32 to bytes (little-endian) -/
def writeU32 (n : UInt32) : ByteArray :=
  ByteArray.mk #[
    (n &&& 0xFF).toUInt8,
    ((n >>> 8) &&& 0xFF).toUInt8,
    ((n >>> 16) &&& 0xFF).toUInt8,
    ((n >>> 24) &&& 0xFF).toUInt8
  ]

/-- Write a UInt16 to bytes (little-endian) -/
def writeU16 (n : UInt16) : ByteArray :=
  ByteArray.mk #[
    (n &&& 0xFF).toUInt8,
    ((n >>> 8) &&& 0xFF).toUInt8
  ]

/-- Read a UInt32 from bytes at offset (little-endian) -/
def readU32 (bytes : ByteArray) (offset : Nat) : Option UInt32 :=
  if offset + 4 > bytes.size then none
  else
    let b0 := bytes[offset]!
    let b1 := bytes[offset + 1]!
    let b2 := bytes[offset + 2]!
    let b3 := bytes[offset + 3]!
    some (b0.toUInt32 ||| (b1.toUInt32 <<< 8) ||| (b2.toUInt32 <<< 16) ||| (b3.toUInt32 <<< 24))

/-- Read a UInt16 from bytes at offset (little-endian) -/
def readU16 (bytes : ByteArray) (offset : Nat) : Option UInt16 :=
  if offset + 2 > bytes.size then none
  else
    let b0 := bytes[offset]!
    let b1 := bytes[offset + 1]!
    some (b0.toUInt16 ||| (b1.toUInt16 <<< 8))

/-- Serialize a tokenizer to bytes -/
def serialize (tok : BPETokenizer) : ByteArray := Id.run do
  let mut result := magic

  -- Header
  result := result ++ writeU32 formatVersion
  result := result ++ writeU32 tok.vocabSize
  result := result ++ writeU32 tok.merges.size.toUInt32
  result := result ++ writeU32 tok.specialTokens.size.toUInt32

  -- Vocabulary
  for bytes in tok.idToBytes do
    result := result ++ writeU16 bytes.size.toUInt16
    result := result ++ bytes

  -- Merges
  for merge in tok.merges do
    result := result ++ writeU32 merge.left
    result := result ++ writeU32 merge.right
    result := result ++ writeU32 merge.result

  -- Special tokens (iterate through the HashMap)
  for (str, id) in tok.specialTokens.toList do
    let strBytes := str.toUTF8
    result := result ++ writeU16 strBytes.size.toUInt16
    result := result ++ strBytes
    result := result ++ writeU32 id

  return result

/-- Deserialize a tokenizer from bytes -/
def deserialize (bytes : ByteArray) : Option BPETokenizer := do
  -- Check magic
  if bytes.size < 8 then failure
  for i in [:8] do
    if bytes[i]! != magic[i]! then failure

  -- Read header
  let version ← readU32 bytes 8
  if version != formatVersion then failure

  let vocabSize ← readU32 bytes 12
  let numMerges ← readU32 bytes 16
  let numSpecial ← readU32 bytes 20

  let mut offset := 24

  -- Read vocabulary
  let mut idToBytes := Array.mkEmpty vocabSize.toNat
  let mut bytesToId : Std.HashMap ByteArray TokenId := {}
  for i in [:vocabSize.toNat] do
    let len ← readU16 bytes offset
    offset := offset + 2
    if offset + len.toNat > bytes.size then failure
    let tokenBytes := bytes.extract offset (offset + len.toNat)
    idToBytes := idToBytes.push tokenBytes
    bytesToId := bytesToId.insert tokenBytes i.toUInt32
    offset := offset + len.toNat

  -- Read merges
  let mut merges := Array.mkEmpty numMerges.toNat
  let mut mergeLookup : Std.HashMap (TokenId × TokenId) TokenId := {}
  for _ in [:numMerges.toNat] do
    let left ← readU32 bytes offset
    let right ← readU32 bytes (offset + 4)
    let result ← readU32 bytes (offset + 8)
    offset := offset + 12
    let rule := { left := left, right := right, result := result : MergeRule }
    merges := merges.push rule
    mergeLookup := mergeLookup.insert (left, right) result

  -- Read special tokens
  let mut specialTokens : Std.HashMap String TokenId := {}
  let mut idToSpecial : Std.HashMap TokenId String := {}
  for _ in [:numSpecial.toNat] do
    let len ← readU16 bytes offset
    offset := offset + 2
    if offset + len.toNat > bytes.size then failure
    let strBytes := bytes.extract offset (offset + len.toNat)
    let str := String.fromUTF8! strBytes
    offset := offset + len.toNat
    let id ← readU32 bytes offset
    offset := offset + 4
    specialTokens := specialTokens.insert str id
    idToSpecial := idToSpecial.insert id str

  return {
    vocabSize := vocabSize
    idToBytes := idToBytes
    bytesToId := bytesToId
    merges := merges
    mergeLookup := mergeLookup
    specialTokens := specialTokens
    idToSpecial := idToSpecial
  }

/-- Save tokenizer to file -/
def save (tok : BPETokenizer) (path : String) : IO Unit := do
  let bytes := serialize tok
  IO.FS.writeBinFile path bytes

/-- Load tokenizer from file -/
def load (path : String) : IO BPETokenizer := do
  let bytes ← IO.FS.readBinFile path
  match deserialize bytes with
  | some tok => return tok
  | none => throw (IO.userError "Failed to parse tokenizer file")

/-- Create a base tokenizer with 256 byte tokens and special tokens -/
def createBase : BPETokenizer := Id.run do
  let mut idToBytes := Array.mkEmpty 512
  let mut bytesToId : Std.HashMap ByteArray TokenId := {}

  -- Add 256 byte tokens
  for i in [:256] do
    let b := ByteArray.mk #[i.toUInt8]
    idToBytes := idToBytes.push b
    bytesToId := bytesToId.insert b i.toUInt32

  let tok : BPETokenizer := {
    vocabSize := 256
    idToBytes := idToBytes
    bytesToId := bytesToId
    merges := #[]
    mergeLookup := {}
    specialTokens := {}
    idToSpecial := {}
  }

  -- Add special tokens
  addSpecialTokens tok allSpecialTokens 256

end tokenizer
