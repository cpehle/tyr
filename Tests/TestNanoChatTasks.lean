import LeanTest
import Lean.Data.Json
import Examples.NanoChat.Tasks.LLM

/-!
# `Tests.TestNanoChatTasks`

NanoChat task parsing tests for JSON task formats and helper behavior.

## Overview
- Regression and behavior checks run by the LeanTest-based test suite.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

open torch.Tasks.LLM

private def parseJsonOrFail (raw : String) : IO Lean.Json := do
  match Lean.Json.parse raw with
  | .ok json => pure json
  | .error e => LeanTest.fail s!"JSON parse failed: {e}"

@[test]
def testMMLUAuxiliaryTrainWrapperParsing : IO Unit := do
  let json ← parseJsonOrFail
    "{\"train\":{\"question\":\"Q?\",\"choices\":[\"A1\",\"A2\",\"A3\",\"A4\"],\"answer\":2,\"subject\":\"test\"}}"
  let some ex := MMLUExample.fromJson? json
    | LeanTest.fail "Expected MMLU example from wrapped auxiliary_train row"
  LeanTest.assertEqual ex.question "Q?"
  LeanTest.assertEqual ex.choices.size 4
  LeanTest.assertEqual ex.answer 2
  LeanTest.assertEqual ex.subject "test"

@[test]
def testSmolTalkParsingWithOptionalSystem : IO Unit := do
  let json ← parseJsonOrFail
    "{\"messages\":[{\"role\":\"system\",\"content\":\"S\"},{\"role\":\"user\",\"content\":\"U1\"},{\"role\":\"assistant\",\"content\":\"A1\"},{\"role\":\"user\",\"content\":\"U2\"},{\"role\":\"assistant\",\"content\":\"A2\"}],\"source\":\"unit\"}"
  let some ex := SmolTalkExample.fromJson? json
    | LeanTest.fail "Expected valid SmolTalk example"
  LeanTest.assertEqual ex.messages.size 5
  match ex.messages.toList with
  | m0 :: m1 :: m2 :: _ =>
    LeanTest.assertEqual m0.content "S"
    LeanTest.assertEqual m1.content "U1"
    LeanTest.assertEqual m2.content "A1"
  | _ =>
    LeanTest.fail "Expected at least 3 messages"

@[test]
def testSmolTalkRejectsBadRoleOrder : IO Unit := do
  let json ← parseJsonOrFail
    "{\"messages\":[{\"role\":\"assistant\",\"content\":\"bad\"},{\"role\":\"user\",\"content\":\"u\"}]}"
  match SmolTalkExample.fromJson? json with
  | some _ => LeanTest.fail "Expected invalid SmolTalk row to be rejected"
  | none => pure ()

@[test]
def testCustomJSONLoader : IO Unit := do
  let tmpPath : System.FilePath := ⟨"/tmp/tyr_customjson_test.jsonl"⟩
  let line1 := "[{\"role\":\"user\",\"content\":\"hello\"},{\"role\":\"assistant\",\"content\":\"world\"}]"
  let line2 := "[{\"role\":\"user\",\"content\":\"u1\"},{\"role\":\"assistant\",\"content\":\"a1\"},{\"role\":\"user\",\"content\":\"u2\"},{\"role\":\"assistant\",\"content\":\"a2\"}]"
  IO.FS.writeFile tmpPath s!"{line1}\n{line2}\n"
  let convs ← loadCustomJSONConversations tmpPath
  LeanTest.assertEqual convs.size 2
  match convs.toList with
  | conv0 :: conv1 :: _ =>
    LeanTest.assertEqual conv0.messages.size 2
    LeanTest.assertEqual conv1.messages.size 4
    match conv0.messages.toList with
    | m0 :: m1 :: _ =>
      LeanTest.assertEqual m0.content "hello"
      LeanTest.assertEqual m1.content "world"
    | _ =>
      LeanTest.fail "Expected at least 2 messages in first conversation"
  | _ =>
    LeanTest.fail "Expected at least 2 conversations"
