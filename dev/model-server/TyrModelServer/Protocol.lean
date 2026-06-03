import Capnp.Rpc
import Tyr.Tokenizer.Qwen36
import TyrModelServer.Capnp.model_gateway

namespace TyrModelServer

open Capnp
open TyrModelServer.Capnp.model_gateway

private def buildPayload
    (build : Capnp.BuilderM Unit)
    (capTable : Capnp.CapTable := Capnp.emptyCapTable)
    (segmentWords : Nat := 64) : Capnp.Rpc.Payload := Id.run do
  let (_, builder) := (do build).run (Capnp.initMessageBuilder segmentWords)
  { msg := Capnp.buildMessage builder, capTable := capTable }

private def buildRootPayload
    (initRoot : Capnp.BuilderM α)
    (fill : α → Capnp.BuilderM Unit)
    (segmentWords : Nat := 64) : Capnp.Rpc.Payload :=
  buildPayload (do
    let root ← initRoot
    fill root) (segmentWords := segmentWords)

inductive ChatRole where
  | system
  | user
  | assistant
  deriving BEq, Repr

def ChatRole.ofText? (text : String) : Option ChatRole :=
  match text.trimAscii.toString.toLower with
  | "system" => some .system
  | "user" => some .user
  | "assistant" => some .assistant
  | _ => none

def ChatRole.toText : ChatRole → String
  | .system => "system"
  | .user => "user"
  | .assistant => "assistant"

private def renderAssistantSuffix (enableThinking : Bool) : String :=
  if enableThinking then
    "<|im_start|>assistant\n<think>\n"
  else
    tokenizer.qwen36.assistantGenerationSuffix

def renderQwenConversation
    (messages : Array ChatMessage)
    (enableThinking : Bool := false) : Except String String := do
  if messages.isEmpty then
    throw "model gateway requires at least one chat message"
  let body := messages.foldl (init := "") fun acc msg =>
    match ChatRole.ofText? msg.role with
    | some role =>
        acc ++ s!"<|im_start|>{role.toText}\n{msg.content}<|im_end|>\n"
    | none =>
        acc ++ s!"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
  pure (body ++ renderAssistantSuffix enableThinking)

def buildInfoPayload : Capnp.Rpc.Payload :=
  buildRootPayload ModelGateway.info_Params.initRoot (fun _ => pure ())

def buildGeneratePayload
    (messages : Array ChatMessage)
    (maxNewTokens : UInt32 := 256)
    (temperature : Float := 0.0)
    (topK : UInt32 := 0)
    (topP : Float := 1.0)
    (enableThinking : Bool := false) : Capnp.Rpc.Payload :=
  buildRootPayload ModelGateway.generate_Params.initRoot
    (fun root =>
      ModelGateway.generate_Params.Builder.setFromValue root {
        messages := messages
        maxNewTokens := maxNewTokens
        temperature := temperature
        topK := topK
        topP := topP
        enableThinking := enableThinking
      })

def buildTokenizePayload (text : String) : Capnp.Rpc.Payload :=
  buildRootPayload ModelGateway.tokenize_Params.initRoot
    (fun root => ModelGateway.tokenize_Params.Builder.setFromValue root { text := text })

def buildDecodePayload (tokens : Array UInt32) : Capnp.Rpc.Payload :=
  buildRootPayload ModelGateway.decode_Params.initRoot
    (fun root => ModelGateway.decode_Params.Builder.setFromValue root { tokens := tokens })

def buildInfoResponse (value : ModelGateway.info_Results) : Capnp.Rpc.Payload :=
  buildRootPayload ModelGateway.info_Results.initRoot
    (fun root => ModelGateway.info_Results.Builder.setFromValue root value)

def buildGenerateResponse (value : ModelGateway.generate_Results) : Capnp.Rpc.Payload :=
  buildRootPayload ModelGateway.generate_Results.initRoot
    (fun root => ModelGateway.generate_Results.Builder.setFromValue root value)

def buildTokenizeResponse (tokens : Array UInt32) : Capnp.Rpc.Payload :=
  buildRootPayload ModelGateway.tokenize_Results.initRoot
    (fun root => ModelGateway.tokenize_Results.Builder.setFromValue root { tokens := tokens })

def buildDecodeResponse (text : String) : Capnp.Rpc.Payload :=
  buildRootPayload ModelGateway.decode_Results.initRoot
    (fun root => ModelGateway.decode_Results.Builder.setFromValue root { text := text })

def infoResultsOfReader (reader : ModelGateway.info_Results.Reader) : ModelGateway.info_Results :=
  ModelGateway.info_Results.ofReader reader

def generateResultsOfReader (reader : ModelGateway.generate_Results.Reader) : ModelGateway.generate_Results :=
  ModelGateway.generate_Results.ofReader reader

def tokenizeResultsOfReader (reader : ModelGateway.tokenize_Results.Reader) : ModelGateway.tokenize_Results :=
  ModelGateway.tokenize_Results.ofReader reader

def decodeResultsOfReader (reader : ModelGateway.decode_Results.Reader) : ModelGateway.decode_Results :=
  ModelGateway.decode_Results.ofReader reader

end TyrModelServer
