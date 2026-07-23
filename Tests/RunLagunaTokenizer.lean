/-
  Tests/RunLagunaTokenizer.lean

  Validates the pure-Lean Laguna tokenizer (Tyr/Tokenizer/Laguna.lean) against
  ground truth generated with the HuggingFace `tokenizers` package from
  dev/laguna_reference/tokenizer.json (see Tests/fixtures/laguna/).
-/
import Tyr.Tokenizer.Laguna
import Lean.Data.Json

open tokenizer tokenizer.laguna

private def parseJsonFile (path : String) : IO Lean.Json := do
  let contents ← IO.FS.readFile path
  match Lean.Json.parse contents with
  | .ok json => pure json
  | .error err => throw (IO.userError s!"Failed to parse JSON at {path}: {err}")

private def getObjVal? (j : Lean.Json) (key : String) : Option Lean.Json :=
  match j with
  | .obj kvs => Std.TreeMap.Raw.get? kvs key
  | _ => none

private def getStr (j : Lean.Json) (key : String) : IO String :=
  match getObjVal? j key with
  | some (.str s) => pure s
  | _ => throw (IO.userError s!"fixture entry missing string field {key}")

private def getArr (j : Lean.Json) (key : String) : IO (Array Lean.Json) :=
  match getObjVal? j key with
  | some (.arr a) => pure a
  | _ => throw (IO.userError s!"fixture entry missing array field {key}")

private def jsonToIds (arr : Array Lean.Json) : IO (Array TokenId) := do
  let mut out : Array TokenId := #[]
  for j in arr do
    match Lean.Json.getNat? j with
    | .ok n => out := out.push n.toUInt32
    | .error e => throw (IO.userError s!"invalid token id in fixture: {e}")
  pure out

private def firstDiff (a b : Array TokenId) : Nat := Id.run do
  let n := min a.size b.size
  for i in [:n] do
    if a[i]! != b[i]! then
      return i
  n

structure Stats where
  passed : Nat := 0
  failed : Nat := 0

private def check (st : IO.Ref Stats) (cond : Bool) (msg : String) : IO Unit := do
  if cond then
    st.modify (fun s => { s with passed := s.passed + 1 })
    IO.println s!"PASS: {msg}"
  else
    st.modify (fun s => { s with failed := s.failed + 1 })
    IO.println s!"FAIL: {msg}"

def main : IO Unit := do
  -- Locate fixture + reference tokenizer (run from repo root or Tests/)
  let mut fixturePath? : Option String := none
  for p in #["Tests/fixtures/laguna/tokenizer_cases.json",
             "../Tests/fixtures/laguna/tokenizer_cases.json"] do
    if ← System.FilePath.pathExists p then
      fixturePath? := some p
      break
  let fixturePath ← match fixturePath? with
    | some p => pure p
    | none => throw (IO.userError "fixture tokenizer_cases.json not found (run from repo root)")

  let mut modelDir? : Option String := none
  for d in #["dev/laguna_reference", "../dev/laguna_reference"] do
    if ← System.FilePath.pathExists s!"{d}/tokenizer.json" then
      modelDir? := some d
      break
  let modelDir ← match modelDir? with
    | some d => pure d
    | none => throw (IO.userError "dev/laguna_reference/tokenizer.json not found")

  let st ← IO.mkRef ({} : Stats)

  let tok ← loadTokenizer modelDir
  check st (tok.vocabSize == 100352) s!"vocabSize={tok.vocabSize}"
  check st (tok.addedTokens.size == 70) s!"addedTokens={tok.addedTokens.size}"
  check st (eosTokenId == 2 && bosTokenId == 2 && padTokenId == 9 &&
            clsTokenId == 10 && assistantEndTokenId == 24)
    "special id constants (eos=2, bos=2, pad=9, cls=10, </assistant>=24)"
  check st (tok.padTokenId' == 9) s!"loaded padTokenId={tok.padTokenId'}"

  let fixture ← parseJsonFile fixturePath

  -- Encode cases: exact id equality with HF tokenizers output
  let cases ← getArr fixture "cases"
  let mut totalIds : Nat := 0
  for c in cases do
    let name ← getStr c "name"
    let text ← getStr c "text"
    let expected ← jsonToIds (← getArr c "ids")
    let got := encodeText tok text
    totalIds := totalIds + expected.size
    if got == expected then
      check st true s!"encode[{name}] ({expected.size} ids)"
    else
      let d := firstDiff got expected
      let gotSlice := got.extract d (min (d + 8) got.size)
      let expSlice := expected.extract d (min (d + 8) expected.size)
      check st false s!"encode[{name}] first diff at {d}: got {gotSlice}... vs expected {expSlice}... (lens {got.size}/{expected.size})"
    -- Roundtrip: decode(encode(x)) == 〈|EOS|〉 ++ x (BOS prepend)
    let rt := decodeTokens tok got
    check st (rt == eosToken ++ text) s!"roundtrip[{name}]"

  -- Decode cases from HF
  let decodeCases ← getArr fixture "decode_cases"
  for c in decodeCases do
    let ids ← jsonToIds (← getArr c "ids")
    let expected ← getStr c "text"
    let got := decodeTokens tok ids
    check st (got == expected) s!"decode {ids.extract 0 (min 6 ids.size)}... -> {expected.take 20}..."

  -- Chat template cases (rendered with jinja2 from chat_template.jinja)
  let chatCases ← getArr fixture "chat"
  for c in chatCases do
    let prompt ← getStr c "prompt"
    let thinking ← getStr c "thinking"
    let notThinking ← getStr c "not_thinking"
    check st (chatTemplate prompt true == thinking) s!"chatTemplate thinking for {prompt.take 24}..."
    check st (chatTemplate prompt false == notThinking) s!"chatTemplate not_thinking for {prompt.take 24}..."

  -- Encoding a chat-templated prompt starts with double BOS (template text +
  -- post-processor both contribute 〈|EOS|〉), matching HF behavior.
  let chatIds := encodeText tok (chatTemplate "hi" true)
  check st (chatIds.size >= 2 && chatIds[0]! == 2 && chatIds[1]! == 2)
    "chat-templated encoding starts with [2, 2]"

  -- Seeded random fuzz fixture (exact id equality + roundtrip)
  let mut fuzzPath? : Option String := none
  for p in #["Tests/fixtures/laguna/tokenizer_fuzz.json",
             "../Tests/fixtures/laguna/tokenizer_fuzz.json"] do
    if ← System.FilePath.pathExists p then
      fuzzPath? := some p
      break
  let mut fuzzCount : Nat := 0
  match fuzzPath? with
  | some fp =>
    let fuzz ← parseJsonFile fp
    let fuzzCases ← getArr fuzz "cases"
    fuzzCount := fuzzCases.size
    let mut idx : Nat := 0
    for c in fuzzCases do
      let text ← getStr c "text"
      let expected ← jsonToIds (← getArr c "ids")
      let got := encodeText tok text
      if got != expected then
        let d := firstDiff got expected
        check st false s!"fuzz[{idx}] len={text.length} first diff at {d} (lens {got.size}/{expected.size})"
      else
        st.modify (fun s => { s with passed := s.passed + 1 })
      let rt := decodeTokens tok got
      if rt != eosToken ++ text then
        check st false s!"fuzz roundtrip[{idx}]"
      else
        st.modify (fun s => { s with passed := s.passed + 1 })
      idx := idx + 1
  | none => IO.println "fuzz fixture not found, skipping"

  let s ← st.get
  IO.println s!"Encoded {cases.size} fixture strings ({totalIds} expected ids) + {decodeCases.size} decode + {chatCases.size} chat cases + {fuzzCount} fuzz"
  IO.println s!"{s.passed} passed, {s.failed} failed"
  if s.failed > 0 then
    throw (IO.userError s!"{s.failed} Laguna tokenizer test(s) failed")
  IO.println "All Laguna tokenizer tests passed."
