@0xc0fec8d32d4535a1;

struct ChatMessage {
  role @0 :Text;
  content @1 :Text;
}

interface ModelGateway {
  info @0 () -> (
    modelId :Text,
    source :Text,
    modelDir :Text,
    device :Text,
    maxContextTokens :UInt32,
    vocabSize :UInt32,
    supportsThinking :Bool,
    supportsVision :Bool
  );

  generate @1 (
    messages :List(ChatMessage),
    maxNewTokens :UInt32,
    temperature :Float32,
    topK :UInt32,
    topP :Float32,
    enableThinking :Bool
  ) -> (
    text :Text,
    generatedTokens :List(UInt32),
    promptTokenCount :UInt32,
    generatedTokenCount :UInt32,
    stopReason :Text
  );

  tokenize @2 (text :Text) -> (tokens :List(UInt32));
  decode @3 (tokens :List(UInt32)) -> (text :Text);
}
