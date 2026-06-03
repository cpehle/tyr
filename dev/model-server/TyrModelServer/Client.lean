import Capnp.Rpc
import TyrModelServer.Protocol

namespace TyrModelServer

open TyrModelServer.Capnp.model_gateway

structure ModelGatewayConnection where
  runtime : Capnp.Rpc.Runtime
  client : Capnp.Rpc.RuntimeClientRef
  gateway : ModelGateway
  address : String

def ModelGatewayConnection.connect (address : String) : IO ModelGatewayConnection := do
  let runtime ← Capnp.Rpc.Runtime.init
  try
    let client ← runtime.newClient address
    let gateway ← client.bootstrap
    pure { runtime := runtime, client := client, gateway := gateway, address := address }
  catch err =>
    runtime.shutdown
    throw err

def ModelGatewayConnection.close (conn : ModelGatewayConnection) : IO Unit := do
  conn.runtime.releaseTarget conn.gateway
  conn.client.release
  conn.runtime.shutdown

def withModelGatewayConnection (address : String) (action : ModelGatewayConnection → IO α) : IO α := do
  let conn ← ModelGatewayConnection.connect address
  try
    action conn
  finally
    conn.close

private def withResponse
    (runtime : Capnp.Rpc.Runtime)
    (response : Capnp.Rpc.TypedPayload α)
    (k : α → IO β) : IO β := do
  try
    k response.reader
  finally
    runtime.releaseCapTable response.capTable

def ModelGatewayConnection.info
    (conn : ModelGatewayConnection) : IO ModelGateway.info_Results := do
  let response ← Capnp.Rpc.RuntimeM.run conn.runtime do
    ModelGateway.callInfoTypedM conn.gateway buildInfoPayload
  withResponse conn.runtime response fun reader =>
    pure (infoResultsOfReader reader)

def ModelGatewayConnection.generate
    (conn : ModelGatewayConnection)
    (messages : Array ChatMessage)
    (maxNewTokens : UInt32 := 256)
    (temperature : Float := 0.0)
    (topK : UInt32 := 0)
    (topP : Float := 1.0)
    (enableThinking : Bool := false) : IO ModelGateway.generate_Results := do
  let payload := buildGeneratePayload messages maxNewTokens temperature topK topP enableThinking
  let response ← Capnp.Rpc.RuntimeM.run conn.runtime do
    ModelGateway.callGenerateTypedM conn.gateway payload
  withResponse conn.runtime response fun reader =>
    pure (generateResultsOfReader reader)

def ModelGatewayConnection.generateUserText
    (conn : ModelGatewayConnection)
    (prompt : String)
    (systemPrompt? : Option String := none)
    (maxNewTokens : UInt32 := 256)
    (temperature : Float := 0.0)
    (topK : UInt32 := 0)
    (topP : Float := 1.0)
    (enableThinking : Bool := false) : IO ModelGateway.generate_Results := do
  let messages :=
    match systemPrompt? with
    | some systemPrompt =>
        #[
          { role := "system", content := systemPrompt },
          { role := "user", content := prompt }
        ]
    | none =>
        #[{ role := "user", content := prompt }]
  conn.generate messages maxNewTokens temperature topK topP enableThinking

def ModelGatewayConnection.tokenize
    (conn : ModelGatewayConnection)
    (text : String) : IO ModelGateway.tokenize_Results := do
  let response ← Capnp.Rpc.RuntimeM.run conn.runtime do
    ModelGateway.callTokenizeTypedM conn.gateway (buildTokenizePayload text)
  withResponse conn.runtime response fun reader =>
    pure (tokenizeResultsOfReader reader)

def ModelGatewayConnection.decode
    (conn : ModelGatewayConnection)
    (tokens : Array UInt32) : IO ModelGateway.decode_Results := do
  let response ← Capnp.Rpc.RuntimeM.run conn.runtime do
    ModelGateway.callDecodeTypedM conn.gateway (buildDecodePayload tokens)
  withResponse conn.runtime response fun reader =>
    pure (decodeResultsOfReader reader)

end TyrModelServer
