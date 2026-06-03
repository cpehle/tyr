import Std
import Tyr.Model.Qwen36
import Tyr.TensorStruct
import Tyr.Tokenizer.Qwen36
import TyrModelServer.Protocol

namespace TyrModelServer

open torch
open torch.qwen36
open TyrModelServer.Capnp.model_gateway

structure ServerArgs where
  source : String := "Qwen/Qwen3.6-35B-A3B"
  revision : String := "main"
  cacheDir : String := "~/.cache/huggingface/tyr-models"
  device : String := "auto"
  address : String := "unix:/tmp/tyr-qwen36.sock"
  deriving Inhabited

structure LoadedQwen36Model where
  source : String
  revision : String
  modelDir : String
  device : Device
  tokenizer : tokenizer.qwen36.QwenTokenizer
  cfg : Config
  model : Qwen36ForCausalLM cfg

structure RunningServer where
  runtime : Capnp.Rpc.Runtime
  bootstrap : ModelGateway
  server : Capnp.Rpc.RuntimeServerRef
  listener : Capnp.Rpc.Listener
  address : String
  socketPath? : Option System.FilePath

private def parseNatArg (name : String) (v : String) : IO UInt32 := do
  match v.toNat? with
  | some n => pure (UInt32.ofNat n)
  | none => throw <| IO.userError s!"invalid {name}: {v}"

private def printHelp : IO Unit := do
  IO.println "Usage: lake exe tyr_model_server [options]"
  IO.println "  --source <path-or-repo>      Local model dir or HF repo id"
  IO.println "  --revision <rev>             HF revision/branch/tag (default: main)"
  IO.println "  --cache-dir <path>           Local cache for downloaded model files"
  IO.println "  --device <auto|cpu|mps|cuda[:n]>  Execution device (default: auto)"
  IO.println "  --address <addr>             Cap'n Proto RPC address (default: unix:/tmp/tyr-qwen36.sock)"
  IO.println "Examples:"
  IO.println "  lake exe tyr_model_server --device mps"
  IO.println "  lake exe tyr_model_server --source /weights/qwen36 --address unix:/tmp/qwen36.sock"

private partial def parseArgsLoop (xs : List String) (acc : ServerArgs) : IO (Option ServerArgs) := do
  match xs with
  | [] => pure (some acc)
  | "--source" :: v :: rest =>
      parseArgsLoop rest { acc with source := v }
  | "--revision" :: v :: rest =>
      parseArgsLoop rest { acc with revision := v }
  | "--cache-dir" :: v :: rest =>
      parseArgsLoop rest { acc with cacheDir := v }
  | "--device" :: v :: rest =>
      parseArgsLoop rest { acc with device := v }
  | "--address" :: v :: rest =>
      parseArgsLoop rest { acc with address := v }
  | "--help" :: _ =>
      printHelp
      pure none
  | x :: _ =>
      throw <| IO.userError s!"unknown argument: {x}"

def parseArgs (argv : List String) : IO (Option ServerArgs) :=
  parseArgsLoop argv {}

def deviceToString : Device → String
  | .CPU => "cpu"
  | .MPS => "mps"
  | .CUDA idx => s!"cuda:{idx}"

def resolveDevice (arg : String) : IO Device := do
  let requested := arg.trimAscii.toString.toLower
  match requested with
  | "auto" => getBestDevice
  | "cpu" => pure .CPU
  | "mps" => pure .MPS
  | "cuda" =>
      if ← cuda_is_available then pure (.CUDA 0) else pure .CPU
  | _ =>
      if requested.startsWith "cuda:" then
        match (requested.drop 5).toNat? with
        | some idx =>
            if ← cuda_is_available then pure (.CUDA idx.toUInt64) else pure .CPU
        | none => pure .CPU
      else
        pure .CPU

private def moveModelToDevice [TensorStruct α] (device : Device) (x : α) : IO α :=
  TensorStruct.mapM (fun t => pure (t.to device)) x

private def unixSocketPath? (address : String) : Option System.FilePath :=
  if address.startsWith "unix:" then
    some ⟨(address.drop 5).toString⟩
  else
    none

private def removeUnixSocketIfPresent (address : String) : IO (Option System.FilePath) := do
  match unixSocketPath? address with
  | some path =>
      if ← path.pathExists then
        try
          IO.FS.removeFile path
        catch _ =>
          pure ()
      pure (some path)
  | none =>
      pure none

private def buildInputTensor (tokenIds : Array UInt32) : IO (Sigma (fun seq => T #[1, seq])) := do
  let seq := tokenIds.size.toUInt64
  if seq == 0 then
    throw <| IO.userError "tokenization produced an empty prompt"
  let flat : Array Int64 := tokenIds.map (fun tok => Int64.ofNat tok.toNat)
  pure ⟨seq, reshape (data.fromInt64Array flat) #[1, seq]⟩

private def decodeGeneratedText
    (tok : tokenizer.qwen36.QwenTokenizer)
    (promptLen : Nat)
    {seq : UInt64}
    (ids : T #[1, seq]) : IO (String × Array UInt32) := do
  let flat : T #[seq] := reshape (data.toLong ids) #[seq]
  let vals ← data.tensorToUInt64Array flat
  let generatedU64 :=
    if vals.size <= promptLen then
      #[]
    else
      vals.extract promptLen vals.size
  let generated := generatedU64.map (fun tok => tok.toUInt32)
  pure (tokenizer.qwen36.decodeText tok generated, generated)

private def samplingStrategyOfRequest (req : ModelGateway.generate_Params) :
    torch.Model.SamplingStrategy :=
  if req.temperature <= 0.0 then
    .greedy
  else
    .multinomial req.temperature req.topK.toUInt64 req.topP

private def renderPromptFromRequest (req : ModelGateway.generate_Params) : IO String := do
  match renderQwenConversation req.messages req.enableThinking with
  | .ok prompt => pure prompt
  | .error err => throw <| IO.userError err

def LoadedQwen36Model.load (args : ServerArgs) : IO LoadedQwen36Model := do
  let device ← resolveDevice args.device
  let modelDir ← hub.resolvePretrainedDir args.source {
    revision := args.revision
    cacheDir := args.cacheDir
    includeTokenizer := true
  }
  let tok ← tokenizer.qwen36.loadTokenizer modelDir
  let ⟨cfg, modelCpu⟩ ← Qwen36ForCausalLM.loadFromPretrained args.source Config.qwen36_35B_A3B args.revision args.cacheDir
  let model ← moveModelToDevice device modelCpu
  pure {
    source := args.source
    revision := args.revision
    modelDir := modelDir
    device := device
    tokenizer := tok
    cfg := cfg
    model := model
  }

def LoadedQwen36Model.infoResults (loaded : LoadedQwen36Model) : ModelGateway.info_Results :=
  { modelId := "Qwen/Qwen3.6-35B-A3B"
    source := loaded.source
    modelDir := loaded.modelDir
    device := deviceToString loaded.device
    maxContextTokens := UInt32.ofNat loaded.cfg.max_position_embeddings.toNat
    vocabSize := UInt32.ofNat loaded.cfg.vocab_size.toNat
    supportsThinking := true
    supportsVision := false }

def LoadedQwen36Model.generateResults
    (loaded : LoadedQwen36Model)
    (req : ModelGateway.generate_Params) : IO ModelGateway.generate_Results := do
  let promptText ← renderPromptFromRequest req
  let promptTokens := tokenizer.qwen36.encodeText loaded.tokenizer promptText
  let promptLen := promptTokens.size
  let ⟨_seq, inputIdsCpu⟩ ← buildInputTensor promptTokens
  let inputIds := inputIdsCpu.to loaded.device
  let eos :=
    match loaded.cfg.eos_token_id with
    | some tok => #[tok]
    | none => #[]
  let maxNewTokens := req.maxNewTokens.toUInt64
  let strategy := samplingStrategyOfRequest req
  let ⟨_outSeq, outIds⟩ ←
    Qwen36ForCausalLM.generate loaded.cfg loaded.model inputIds maxNewTokens strategy eos
  let (text, generatedTokens) ← decodeGeneratedText loaded.tokenizer promptLen outIds
  let stopReason :=
    if generatedTokens.size < req.maxNewTokens.toNat then "eos" else "max_new_tokens"
  pure
    { text := text
      generatedTokens := generatedTokens
      promptTokenCount := UInt32.ofNat promptLen
      generatedTokenCount := UInt32.ofNat generatedTokens.size
      stopReason := stopReason }

def LoadedQwen36Model.tokenizeResults
    (loaded : LoadedQwen36Model)
    (req : ModelGateway.tokenize_Params) : ModelGateway.tokenize_Results :=
  { tokens := tokenizer.qwen36.encodeText loaded.tokenizer req.text }

def LoadedQwen36Model.decodeResults
    (loaded : LoadedQwen36Model)
    (req : ModelGateway.decode_Params) : ModelGateway.decode_Results :=
  { text := tokenizer.qwen36.decodeText loaded.tokenizer req.tokens }

def LoadedQwen36Model.typedServer (loaded : LoadedQwen36Model) : ModelGateway.TypedServer :=
  { info := fun _ _ _ =>
      pure (buildInfoResponse loaded.infoResults)
    generate := fun _ request _ => do
      let req := ModelGateway.generate_Params.ofReader request
      pure (buildGenerateResponse (← loaded.generateResults req))
    tokenize := fun _ request _ => do
      let req := ModelGateway.tokenize_Params.ofReader request
      pure (buildTokenizeResponse (loaded.tokenizeResults req).tokens)
    decode := fun _ request _ => do
      let req := ModelGateway.decode_Params.ofReader request
      pure (buildDecodeResponse (loaded.decodeResults req).text) }

def RunningServer.start (loaded : LoadedQwen36Model) (address : String) : IO RunningServer := do
  let socketPath? ← removeUnixSocketIfPresent address
  let runtime ← Capnp.Rpc.Runtime.init
  let bootstrap ← ModelGateway.registerTypedTarget runtime loaded.typedServer
  let server ← runtime.newServer bootstrap
  let listener ← server.listen address
  server.accept listener
  pure {
    runtime := runtime
    bootstrap := bootstrap
    server := server
    listener := listener
    address := address
    socketPath? := socketPath?
  }

def RunningServer.shutdown (running : RunningServer) : IO Unit := do
  running.server.release
  running.runtime.releaseListener running.listener
  running.runtime.releaseTarget running.bootstrap
  running.runtime.shutdown
  match running.socketPath? with
  | some path =>
      if ← path.pathExists then
        try
          IO.FS.removeFile path
        catch _ =>
          pure ()
  | none =>
      pure ()

partial def waitForever : IO PUnit := do
  IO.sleep (UInt32.ofNat 3_600_000)
  waitForever

def runMain (argv : List String) : IO UInt32 := do
  match (← parseArgs argv) with
  | none =>
      pure 0
  | some args =>
      let loaded ← LoadedQwen36Model.load args
      let running ← RunningServer.start loaded args.address
      IO.println s!"serving {loaded.source} on {args.address} using {deviceToString loaded.device}"
      try
        waitForever
        pure 0
      finally
        running.shutdown

end TyrModelServer
