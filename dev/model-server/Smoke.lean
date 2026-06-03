import Capnp.Rpc
import TyrModelServer.Protocol

open TyrModelServer
open TyrModelServer.Capnp.model_gateway

private def assertEq [BEq α] [Repr α] (label : String) (got expected : α) : IO Unit := do
  unless got == expected do
    throw <| IO.userError s!"{label}: expected {reprStr expected}, got {reprStr got}"

private def mkUnixTestAddress : IO (String × System.FilePath) := do
  let n ← IO.rand 0 1000000000
  let path : System.FilePath := s!"/tmp/tyr-model-server-smoke-{n}.sock"
  pure (s!"unix:{path}", path)

private def mockServer : ModelGateway.TypedServer :=
  { info := fun _ _ _ =>
      pure <| buildInfoResponse {
        modelId := "mock/qwen36"
        source := "mock-source"
        modelDir := "/tmp/mock"
        device := "cpu"
        maxContextTokens := 4096
        vocabSize := 32000
        supportsThinking := true
        supportsVision := false
      }
    generate := fun _ request _ => do
      let req := ModelGateway.generate_Params.ofReader request
      let text :=
        match req.messages.back? with
        | some msg => s!"echo:{msg.content}"
        | none => "echo:"
      pure <| buildGenerateResponse {
        text := text
        generatedTokens := #[11, 22, 33]
        promptTokenCount := 7
        generatedTokenCount := 3
        stopReason := "max_new_tokens"
      }
    tokenize := fun _ request _ => do
      let req := ModelGateway.tokenize_Params.ofReader request
      pure <| buildTokenizeResponse #[UInt32.ofNat req.text.length]
    decode := fun _ request _ => do
      let req := ModelGateway.decode_Params.ofReader request
      pure <| buildDecodeResponse s!"decoded:{req.tokens.size}" }

def main (_argv : List String) : IO UInt32 := do
  let (address, socketPath) ← mkUnixTestAddress
  let runtime ← Capnp.Rpc.Runtime.init
  try
    let bootstrap ← ModelGateway.registerTypedTarget runtime mockServer
    let server ← runtime.newServer bootstrap
    let listener ← server.listen address
    let client ← runtime.newClient address
    server.accept listener
    let remoteTarget ← client.bootstrap

    let info ← Capnp.Rpc.RuntimeM.run runtime do
      ModelGateway.callInfoTypedM remoteTarget buildInfoPayload
    let infoRes := infoResultsOfReader info.reader
    assertEq "info.modelId" infoRes.modelId "mock/qwen36"
    assertEq "info.device" infoRes.device "cpu"

    let generatePayload := buildGeneratePayload
      #[{ role := "user", content := "hello" }]
      (maxNewTokens := 3)
      (temperature := 0.0)
      (topK := 0)
      (topP := 1.0)
      (enableThinking := false)
    let generated ← Capnp.Rpc.RuntimeM.run runtime do
      ModelGateway.callGenerateTypedM remoteTarget generatePayload
    let genRes := generateResultsOfReader generated.reader
    assertEq "generate.text" genRes.text "echo:hello"
    assertEq "generate.tokens" genRes.generatedTokens #[11, 22, 33]

    let tokenize ← Capnp.Rpc.RuntimeM.run runtime do
      ModelGateway.callTokenizeTypedM remoteTarget (buildTokenizePayload "abcd")
    let tokenizeRes := tokenizeResultsOfReader tokenize.reader
    assertEq "tokenize.tokens" tokenizeRes.tokens #[4]

    let decode ← Capnp.Rpc.RuntimeM.run runtime do
      ModelGateway.callDecodeTypedM remoteTarget (buildDecodePayload #[1, 2, 3])
    let decodeRes := decodeResultsOfReader decode.reader
    assertEq "decode.text" decodeRes.text "decoded:3"

    runtime.releaseTarget remoteTarget
    client.release
    server.release
    runtime.releaseListener listener
    runtime.releaseTarget bootstrap
    pure 0
  finally
    runtime.shutdown
    if ← socketPath.pathExists then
      try
        IO.FS.removeFile socketPath
      catch _ =>
        pure ()
