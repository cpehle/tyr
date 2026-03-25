/-
  Tyr/Model/KittenTTS/Weights.lean

  Pretrained weight loading for the Lean KittenTTS / Kokoro port.
  Loads converted SafeTensors checkpoints produced from upstream PyTorch
  checkpoints so single-utterance synthesis can use real weights.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Log
import Tyr.Model.KittenTTS.Checkpoint
import Tyr.Model.KittenTTS.Model

namespace torch.kittentts

open torch.Log

private structure LoadedCheckpointWeights where
  bert : KokoroCheckpoint.WeightsBert
  bertEncoder : KokoroCheckpoint.WeightsBertencoder
  predictor : KokoroCheckpoint.WeightsPredictor
  textEncoder : KokoroCheckpoint.WeightsTextencoder
  decoder : KokoroCheckpoint.WeightsDecoder

private def reqGradFalse {s : Shape} (t : T s) : T s :=
  autograd.set_requires_grad (toFloat' t) false

private def getOrThrow {α} (xs : Array α) (idx : Nat) (msg : String) : IO α :=
  match xs[idx]? with
  | some x => pure x
  | none => throw <| IO.userError msg

private def tensorDimOrThrow {s : Shape} (t : T s) (idx : Nat) (name : String) : IO UInt64 :=
  getOrThrow t.runtimeShape idx s!"Missing dimension {idx} for {name}"

private def warnShapeOverride (log : Handlers) (field : String) (cfgValue actualValue : UInt64) : IO Unit := do
  if cfgValue != actualValue then
    log.onWarn s!"Checkpoint overrides {field}: config={cfgValue}, checkpoint={actualValue}"
  pure ()

private def loadTensorTarget (handle : safetensors.SafeTensorsHandle) (name : String) (s : Shape) : IO (T s) := do
  safetensors.loadFromHandle handle name s

private def tryLoadTensorTarget (handle : safetensors.SafeTensorsHandle) (name : String) (s : Shape)
    : IO (Option (T s)) := do
  try
    pure (some (← loadTensorTarget handle name s))
  catch _ =>
    pure none

private def loadTensorCandidates (handle : safetensors.SafeTensorsHandle) (names : Array String) (s : Shape)
    : IO (T s) := do
  let mut lastErr : Option IO.Error := none
  for name in names do
    try
      return ← loadTensorTarget handle name s
    catch err =>
      lastErr := some err
  match lastErr with
  | some err => throw err
  | none => throw <| IO.userError "No tensor candidates provided."

private def loadOptionalTensorCandidates
    (handle : safetensors.SafeTensorsHandle)
    (names : Array String)
    (s : Shape)
    : IO (Option (T s)) := do
  for name in names do
    if let some t ← tryLoadTensorTarget handle name s then
      return some t
  pure none

private def normDim0_3d {d0 d1 d2 : UInt64}
    (x : T #[d0, d1, d2])
    : T #[d0] :=
  let sq : T #[d0, d1, d2] := x * x
  let s2 : T #[d0, d1, 1] := nn.sumDim sq 2 true
  let s1 : T #[d0, 1, 1] := nn.sumDim s2 1 true
  reshape (nn.sqrt (s1 + (1e-12 : Float))) #[d0]

private def loadWeightGConv1d (handle : safetensors.SafeTensorsHandle) (namePrefix : String) (outC : UInt64)
    : IO (T #[outC]) := do
  match (← tryLoadTensorTarget handle s!"{namePrefix}.weight_g" #[outC]) with
  | some g => pure g
  | none =>
    let g3 : T #[outC, 1, 1] ← loadTensorTarget handle s!"{namePrefix}.weight_g" #[outC, 1, 1]
    pure (reshape g3 #[outC])

private def loadWeightGTransConv1d (handle : safetensors.SafeTensorsHandle) (namePrefix : String) (inC : UInt64)
    : IO (T #[inC]) := do
  match (← tryLoadTensorTarget handle s!"{namePrefix}.weight_g" #[inC]) with
  | some g => pure g
  | none =>
    let g3 : T #[inC, 1, 1] ← loadTensorTarget handle s!"{namePrefix}.weight_g" #[inC, 1, 1]
    pure (reshape g3 #[inC])

private def materializeWeightNormConv1d {outC inC kernel : UInt64}
    (weightV : T #[outC, inC, kernel])
    (weightG : T #[outC])
    : T #[outC, inC, kernel] :=
  let denom : T #[outC, 1, 1] := reshape (normDim0_3d weightV) #[outC, 1, 1]
  let denomE : T #[outC, inC, kernel] := nn.expand denom #[outC, inC, kernel]
  let gain : T #[outC, 1, 1] := reshape weightG #[outC, 1, 1]
  let gainE : T #[outC, inC, kernel] := nn.expand gain #[outC, inC, kernel]
  nn.div (weightV * gainE) denomE

private def materializeWeightNormTransConv1d {inC outC kernel : UInt64}
    (weightV : T #[inC, outC, kernel])
    (weightG : T #[inC])
    : T #[inC, outC, kernel] :=
  let denom : T #[inC, 1, 1] := reshape (normDim0_3d weightV) #[inC, 1, 1]
  let denomE : T #[inC, outC, kernel] := nn.expand denom #[inC, outC, kernel]
  let gain : T #[inC, 1, 1] := reshape weightG #[inC, 1, 1]
  let gainE : T #[inC, outC, kernel] := nn.expand gain #[inC, outC, kernel]
  nn.div (weightV * gainE) denomE

private def linearFromTensors {inDim outDim : UInt64}
    (weight : T #[outDim, inDim])
    (bias : T #[outDim])
    : LinearNorm inDim outDim := {
      weight := reqGradFalse weight
      bias := reqGradFalse bias
    }

private def biLSTMFromTensors {inputSize hiddenSize : UInt64}
    (wxForward : T #[4 * hiddenSize, inputSize])
    (whForward : T #[4 * hiddenSize, hiddenSize])
    (biasIHForward : T #[4 * hiddenSize])
    (biasHHForward : T #[4 * hiddenSize])
    (wxBackward : T #[4 * hiddenSize, inputSize])
    (whBackward : T #[4 * hiddenSize, hiddenSize])
    (biasIHBackward : T #[4 * hiddenSize])
    (biasHHBackward : T #[4 * hiddenSize])
    : BiLSTM inputSize hiddenSize := {
      wxForward := reqGradFalse wxForward
      whForward := reqGradFalse whForward
      biasIHForward := reqGradFalse biasIHForward
      biasHHForward := reqGradFalse biasHHForward
      wxBackward := reqGradFalse wxBackward
      whBackward := reqGradFalse whBackward
      biasIHBackward := reqGradFalse biasIHBackward
      biasHHBackward := reqGradFalse biasHHBackward
    }

private def conv1dFromWeightNorm {inC outC kernel stride padding dilation : UInt64}
    (weightV : T #[outC, inC, kernel])
    (weightG3 : T #[outC, 1, 1])
    (bias : T #[outC])
    : Conv1dParams inC outC kernel stride padding dilation := {
      weight := reqGradFalse <| materializeWeightNormConv1d weightV (reshape weightG3 #[outC])
      bias := reqGradFalse bias
    }

private def wnConv1dFromWeightNorm {inC outC kernel stride padding dilation : UInt64}
    (weightV : T #[outC, inC, kernel])
    (weightG3 : T #[outC, 1, 1])
    (bias : T #[outC])
    : WNConv1dParams inC outC kernel stride padding dilation := {
      weightV := reqGradFalse weightV
      weightG := reqGradFalse (reshape weightG3 #[outC])
      bias := reqGradFalse bias
    }

private def wnDepthwiseTransConv1dFromWeightNorm
    {channels kernel stride padding outputPadding dilation : UInt64}
    (weightV : T #[channels, 1, kernel])
    (weightG3 : T #[channels, 1, 1])
    (bias : T #[channels])
    : WNDepthwiseTransConv1dParams channels kernel stride padding outputPadding dilation := {
      weightV := reqGradFalse weightV
      weightG := reqGradFalse (reshape weightG3 #[channels])
      bias := reqGradFalse bias
    }

private def wnConvTranspose1dFromWeightNorm
    {inC outC kernel stride padding outputPadding dilation : UInt64}
    (weightV : T #[inC, outC, kernel])
    (weightG3 : T #[inC, 1, 1])
    (bias : T #[outC])
    : WNConvTranspose1dParams inC outC kernel stride padding outputPadding dilation := {
      weightV := reqGradFalse weightV
      weightG := reqGradFalse (reshape weightG3 #[inC])
      bias := reqGradFalse bias
    }

private def adaLayerNormFromLinear {styleDim channels : UInt64}
    (weight : T #[2 * channels, styleDim])
    (bias : T #[2 * channels])
    : AdaLayerNorm styleDim channels := {
      fc := linearFromTensors weight bias
    }

private def adaIN1dFromLinear {styleDim channels : UInt64}
    (weight : T #[2 * channels, styleDim])
    (bias : T #[2 * channels])
    : AdaIN1d styleDim channels := {
      fc := linearFromTensors weight bias
      normWeight := reqGradFalse (torch.ones #[channels] false Device.CPU)
      normBias := reqGradFalse (torch.zeros #[channels] false Device.CPU)
    }

private def adaIN1dDynFromLinear {styleDim channels : UInt64}
    (weight : T #[2 * channels, styleDim])
    (bias : T #[2 * channels])
    : AdaIN1dDyn styleDim := {
      fcWeight := nn.eraseShape (reqGradFalse weight)
      fcBias := nn.eraseShape (reqGradFalse bias)
      normWeight := nn.eraseShape (torch.ones #[channels] false Device.CPU)
      normBias := nn.eraseShape (torch.zeros #[channels] false Device.CPU)
    }

private def wnConv1dDynFromWeightNorm {inC outC kernel : UInt64}
    (weightV : T #[outC, inC, kernel])
    (weightG3 : T #[outC, 1, 1])
    (bias : T #[outC])
    (stride padding dilation : UInt64)
    : WNConv1dDyn := {
      weightV := nn.eraseShape (reqGradFalse weightV)
      weightG := nn.eraseShape (reqGradFalse (reshape weightG3 #[outC]))
      bias := nn.eraseShape (reqGradFalse bias)
      kernel := kernel
      stride := stride
      padding := padding
      dilation := dilation
    }

private def wnConv1dDynFromPlain {inC outC kernel : UInt64}
    (weight : T #[outC, inC, kernel])
    (bias : T #[outC])
    (stride padding dilation : UInt64)
    : WNConv1dDyn := {
      weightV := nn.eraseShape (reqGradFalse weight)
      weightG := nn.eraseShape (reqGradFalse (normDim0_3d weight))
      bias := nn.eraseShape (reqGradFalse bias)
      kernel := kernel
      stride := stride
      padding := padding
      dilation := dilation
    }

private def wnConvTranspose1dDynFromWeightNorm {inC outC kernel : UInt64}
    (weightV : T #[inC, outC, kernel])
    (weightG3 : T #[inC, 1, 1])
    (bias : T #[outC])
    (stride padding outputPadding dilation : UInt64)
    : WNConvTranspose1dDyn := {
      weightV := nn.eraseShape (reqGradFalse weightV)
      weightG := nn.eraseShape (reqGradFalse (reshape weightG3 #[inC]))
      bias := nn.eraseShape (reqGradFalse bias)
      kernel := kernel
      stride := stride
      padding := padding
      outputPadding := outputPadding
      dilation := dilation
    }

private def alphaDyn {channels : UInt64} (x : T #[1, channels, 1]) : T #[] :=
  nn.eraseShape (reqGradFalse x)

private def loadBiasOrZero
    (handle : safetensors.SafeTensorsHandle)
    (name : String)
    (channels : UInt64)
    : IO (T #[channels]) := do
  match (← tryLoadTensorTarget handle name #[channels]) with
  | some b => pure b
  | none => pure (torch.zeros #[channels] false Device.CPU)

private def loadAdaINNormWeightOrDefault
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (channels : UInt64)
    : IO (T #[channels]) := do
  match (← tryLoadTensorTarget handle s!"{namePrefix}.norm.weight" #[channels]) with
  | some w => pure w
  | none => pure (torch.ones #[channels] false Device.CPU)

private def loadAdaINNormBiasOrDefault
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (channels : UInt64)
    : IO (T #[channels]) := do
  match (← tryLoadTensorTarget handle s!"{namePrefix}.norm.bias" #[channels]) with
  | some b => pure b
  | none => pure (torch.zeros #[channels] false Device.CPU)

private def loadLinear
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (inDim outDim : UInt64)
    : IO (LinearNorm inDim outDim) := do
  let weight ←
    loadTensorCandidates handle
      #[s!"{namePrefix}.weight", s!"{namePrefix}.linear_layer.weight"]
      #[outDim, inDim]
  let bias ←
    loadTensorCandidates handle
      #[s!"{namePrefix}.bias", s!"{namePrefix}.linear_layer.bias"]
      #[outDim]
  pure { weight := reqGradFalse weight, bias := reqGradFalse bias }

private def loadConv1dEffective
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (inC outC kernel : UInt64)
    (allowMissingBias : Bool := false)
    : IO (Conv1dParams inC outC kernel 1 ((kernel - 1) / 2) 1) := do
  let bias ←
    if allowMissingBias then
      loadBiasOrZero handle s!"{namePrefix}.bias" outC
    else
      loadTensorTarget handle s!"{namePrefix}.bias" #[outC]
  let weight? ← tryLoadTensorTarget handle s!"{namePrefix}.weight" #[outC, inC, kernel]
  let weight ←
    match weight? with
    | some w => pure w
    | none => do
      let weightV : T #[outC, inC, kernel] ← loadTensorTarget handle s!"{namePrefix}.weight_v" #[outC, inC, kernel]
      let weightG ← loadWeightGConv1d handle namePrefix outC
      pure (materializeWeightNormConv1d weightV weightG)
  pure { weight := reqGradFalse weight, bias := reqGradFalse bias }

private def loadWNConv1d
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (inC outC kernel stride padding dilation : UInt64)
    (allowMissingBias : Bool := false)
    : IO (WNConv1dParams inC outC kernel stride padding dilation) := do
  let bias ←
    if allowMissingBias then
      loadBiasOrZero handle s!"{namePrefix}.bias" outC
    else
      loadTensorTarget handle s!"{namePrefix}.bias" #[outC]
  let weightV? ← tryLoadTensorTarget handle s!"{namePrefix}.weight_v" #[outC, inC, kernel]
  let (weightV, weightG) ←
    match weightV? with
    | some wv =>
      let wg ← loadWeightGConv1d handle namePrefix outC
      pure (wv, wg)
    | none => do
      let weight : T #[outC, inC, kernel] ← loadTensorTarget handle s!"{namePrefix}.weight" #[outC, inC, kernel]
      pure (weight, normDim0_3d weight)
  pure { weightV := reqGradFalse weightV, weightG := reqGradFalse weightG, bias := reqGradFalse bias }

private def loadWNConvTranspose1d
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (inC outC kernel stride padding outputPadding dilation : UInt64)
    : IO (WNConvTranspose1dParams inC outC kernel stride padding outputPadding dilation) := do
  let bias ← loadBiasOrZero handle s!"{namePrefix}.bias" outC
  let weightV? ← tryLoadTensorTarget handle s!"{namePrefix}.weight_v" #[inC, outC, kernel]
  let (weightV, weightG) ←
    match weightV? with
    | some wv =>
      let wg ← loadWeightGTransConv1d handle namePrefix inC
      pure (wv, wg)
    | none => do
      let weight : T #[inC, outC, kernel] ← loadTensorTarget handle s!"{namePrefix}.weight" #[inC, outC, kernel]
      pure (weight, normDim0_3d weight)
  pure { weightV := reqGradFalse weightV, weightG := reqGradFalse weightG, bias := reqGradFalse bias }

private def loadWNDepthwiseTransConv1d
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (channels kernel stride padding outputPadding dilation : UInt64)
    : IO (WNDepthwiseTransConv1dParams channels kernel stride padding outputPadding dilation) := do
  let bias ← loadBiasOrZero handle s!"{namePrefix}.bias" channels
  let weightV? ← tryLoadTensorTarget handle s!"{namePrefix}.weight_v" #[channels, 1, kernel]
  let (weightV, weightG) ←
    match weightV? with
    | some wv =>
      let wg ← loadWeightGTransConv1d handle namePrefix channels
      pure (wv, wg)
    | none => do
      let weight : T #[channels, 1, kernel] ← loadTensorTarget handle s!"{namePrefix}.weight" #[channels, 1, kernel]
      pure (weight, normDim0_3d weight)
  pure { weightV := reqGradFalse weightV, weightG := reqGradFalse weightG, bias := reqGradFalse bias }

private def loadBiLSTM
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (inputSize hiddenSize : UInt64)
    : IO (BiLSTM inputSize hiddenSize) := do
  let wxForward ← loadTensorTarget handle s!"{namePrefix}.weight_ih_l0" #[4 * hiddenSize, inputSize]
  let whForward ← loadTensorTarget handle s!"{namePrefix}.weight_hh_l0" #[4 * hiddenSize, hiddenSize]
  let biasIHForward ← loadTensorTarget handle s!"{namePrefix}.bias_ih_l0" #[4 * hiddenSize]
  let biasHHForward ← loadTensorTarget handle s!"{namePrefix}.bias_hh_l0" #[4 * hiddenSize]
  let wxBackward ← loadTensorTarget handle s!"{namePrefix}.weight_ih_l0_reverse" #[4 * hiddenSize, inputSize]
  let whBackward ← loadTensorTarget handle s!"{namePrefix}.weight_hh_l0_reverse" #[4 * hiddenSize, hiddenSize]
  let biasIHBackward ← loadTensorTarget handle s!"{namePrefix}.bias_ih_l0_reverse" #[4 * hiddenSize]
  let biasHHBackward ← loadTensorTarget handle s!"{namePrefix}.bias_hh_l0_reverse" #[4 * hiddenSize]
  pure {
    wxForward := reqGradFalse wxForward
    whForward := reqGradFalse whForward
    biasIHForward := reqGradFalse biasIHForward
    biasHHForward := reqGradFalse biasHHForward
    wxBackward := reqGradFalse wxBackward
    whBackward := reqGradFalse whBackward
    biasIHBackward := reqGradFalse biasIHBackward
    biasHHBackward := reqGradFalse biasHHBackward
  }

private def loadAlbertEmbeddings (handle : safetensors.SafeTensorsHandle) (cfg : KittenTTSConfig)
    : IO (AlbertEmbeddings cfg.nToken cfg.plbert) := do
  let wordEmbeddings ← loadTensorTarget handle "bert.embeddings.word_embeddings.weight" #[cfg.nToken, cfg.plbert.embeddingSize]
  let positionEmbeddings ← loadTensorTarget handle "bert.embeddings.position_embeddings.weight" #[cfg.plbert.maxPositionEmbeddings, cfg.plbert.embeddingSize]
  let tokenTypeEmbeddings ← loadTensorTarget handle "bert.embeddings.token_type_embeddings.weight" #[cfg.plbert.typeVocabSize, cfg.plbert.embeddingSize]
  let layerNormWeight ← loadTensorTarget handle "bert.embeddings.LayerNorm.weight" #[cfg.plbert.embeddingSize]
  let layerNormBias ← loadTensorTarget handle "bert.embeddings.LayerNorm.bias" #[cfg.plbert.embeddingSize]
  pure {
    wordEmbeddings := reqGradFalse wordEmbeddings
    positionEmbeddings := reqGradFalse positionEmbeddings
    tokenTypeEmbeddings := reqGradFalse tokenTypeEmbeddings
    layerNormWeight := reqGradFalse layerNormWeight
    layerNormBias := reqGradFalse layerNormBias
  }

private def loadKittenAlbertSelfAttention
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (cfg : AlbertConfig)
    : IO (KittenAlbertSelfAttention cfg) := do
  let query ← loadLinear handle s!"{namePrefix}.attention.query" cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  let key ← loadLinear handle s!"{namePrefix}.attention.key" cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  let value ← loadLinear handle s!"{namePrefix}.attention.value" cfg.hiddenSize (AlbertConfig.allHeadSize cfg)
  let dense ← loadLinear handle s!"{namePrefix}.attention.dense" (AlbertConfig.allHeadSize cfg) cfg.hiddenSize
  let layerNormWeight ← loadTensorTarget handle s!"{namePrefix}.attention.LayerNorm.weight" #[cfg.hiddenSize]
  let layerNormBias ← loadTensorTarget handle s!"{namePrefix}.attention.LayerNorm.bias" #[cfg.hiddenSize]
  pure {
    query, key, value, dense
    layerNormWeight := reqGradFalse layerNormWeight
    layerNormBias := reqGradFalse layerNormBias
  }

private def loadKittenAlbertLayer
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (cfg : AlbertConfig)
    : IO (KittenAlbertLayer cfg) := do
  let attention ← loadKittenAlbertSelfAttention handle namePrefix cfg
  let fullLayerNormWeight ← loadTensorTarget handle s!"{namePrefix}.full_layer_layer_norm.weight" #[cfg.hiddenSize]
  let fullLayerNormBias ← loadTensorTarget handle s!"{namePrefix}.full_layer_layer_norm.bias" #[cfg.hiddenSize]
  let ffn ← loadLinear handle s!"{namePrefix}.ffn" cfg.hiddenSize cfg.intermediateSize
  let ffnOutput ← loadLinear handle s!"{namePrefix}.ffn_output" cfg.intermediateSize cfg.hiddenSize
  pure {
    attention
    fullLayerNormWeight := reqGradFalse fullLayerNormWeight
    fullLayerNormBias := reqGradFalse fullLayerNormBias
    ffn
    ffnOutput
  }

private def loadKittenAlbertEncoder (handle : safetensors.SafeTensorsHandle) (cfg : AlbertConfig)
    : IO (KittenAlbertEncoder cfg) := do
  let embeddingHiddenMappingIn ←
    loadLinear handle "bert.encoder.embedding_hidden_mapping_in" cfg.embeddingSize cfg.hiddenSize
  let mut groups : Array (KittenAlbertLayerGroup cfg) := #[]
  for gi in [:cfg.numHiddenGroups.toNat] do
    let mut layers : Array (KittenAlbertLayer cfg) := #[]
    for li in [:cfg.innerGroupNum.toNat] do
      let namePrefix := s!"bert.encoder.albert_layer_groups.{gi}.albert_layers.{li}"
      layers := layers.push (← loadKittenAlbertLayer handle namePrefix cfg)
    groups := groups.push { layers }
  pure { embeddingHiddenMappingIn, groups }

private def loadKittenAlbert (handle : safetensors.SafeTensorsHandle) (cfg : KittenTTSConfig)
    : IO (KittenAlbert cfg.nToken cfg.plbert) := do
  let embeddings ← loadAlbertEmbeddings handle cfg
  let encoder ← loadKittenAlbertEncoder handle cfg.plbert
  let pooler ← loadLinear handle "bert.pooler" cfg.plbert.hiddenSize cfg.plbert.hiddenSize
  pure { embeddings, encoder, pooler }

private def loadTextConvLayer
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (cfg : KittenTTSConfig)
    : IO (TextConvLayer cfg) := do
  let conv ←
    loadConv1dEffective handle s!"{namePrefix}.0" cfg.hiddenDim cfg.hiddenDim cfg.textEncoderKernelSize
  let lnWeight ← loadTensorTarget handle s!"{namePrefix}.1.gamma" #[cfg.hiddenDim]
  let lnBias ← loadTensorTarget handle s!"{namePrefix}.1.beta" #[cfg.hiddenDim]
  pure { conv, lnWeight := reqGradFalse lnWeight, lnBias := reqGradFalse lnBias }

private def loadTextEncoder (handle : safetensors.SafeTensorsHandle) (cfg : KittenTTSConfig)
    : IO (TextEncoder cfg) := do
  let embedding ← loadTensorTarget handle "text_encoder.embedding.weight" #[cfg.nToken, cfg.hiddenDim]
  let mut convs : Array (TextConvLayer cfg) := #[]
  for i in [:cfg.nLayer.toNat] do
    convs := convs.push (← loadTextConvLayer handle s!"text_encoder.cnn.{i}" cfg)
  let lstm ← loadBiLSTM handle "text_encoder.lstm" cfg.hiddenDim (cfg.hiddenDim / 2)
  pure { embedding := reqGradFalse embedding, convs, lstm }

private def loadAdaLayerNorm
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (styleDim channels : UInt64)
    : IO (AdaLayerNorm styleDim channels) := do
  let fc ← loadLinear handle s!"{namePrefix}.fc" styleDim (2 * channels)
  pure { fc }

private def loadAdaIN1d
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (styleDim channels : UInt64)
    : IO (AdaIN1d styleDim channels) := do
  let fc ← loadLinear handle s!"{namePrefix}.fc" styleDim (2 * channels)
  let normWeight ← loadAdaINNormWeightOrDefault handle namePrefix channels
  let normBias ← loadAdaINNormBiasOrDefault handle namePrefix channels
  pure {
    fc
    normWeight := reqGradFalse normWeight
    normBias := reqGradFalse normBias
  }

private def loadDurationEncoder (handle : safetensors.SafeTensorsHandle) (cfg : KittenTTSConfig)
    : IO (DurationEncoder cfg) := do
  let mut lstms : Array (BiLSTM (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2)) := #[]
  let mut norms : Array (AdaLayerNorm cfg.styleDim cfg.hiddenDim) := #[]
  for i in [:cfg.nLayer.toNat] do
    lstms := lstms.push (← loadBiLSTM handle s!"predictor.text_encoder.lstms.{2 * i}" (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2))
    norms := norms.push (← loadAdaLayerNorm handle s!"predictor.text_encoder.lstms.{2 * i + 1}" cfg.styleDim cfg.hiddenDim)
  pure { lstms, norms }

private def loadAdainResBlk1dSame
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (inC outC styleDim : UInt64)
    : IO (AdainResBlk1dSame inC outC styleDim) := do
  let conv1 ← loadWNConv1d handle s!"{namePrefix}.conv1" inC outC 3 1 1 1
  let conv2 ← loadWNConv1d handle s!"{namePrefix}.conv2" outC outC 3 1 1 1
  let norm1 ← loadAdaIN1d handle s!"{namePrefix}.norm1" styleDim inC
  let norm2 ← loadAdaIN1d handle s!"{namePrefix}.norm2" styleDim outC
  let shortcut ←
    if inC == outC then
      pure none
    else
      pure <| some (← loadWNConv1d handle s!"{namePrefix}.conv1x1" inC outC 1 1 0 1 true)
  pure { conv1, conv2, norm1, norm2, shortcut }

private def loadAdainResBlk1dUp
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (inC outC styleDim : UInt64)
    : IO (AdainResBlk1dUp inC outC styleDim) := do
  let pool ← loadWNDepthwiseTransConv1d handle s!"{namePrefix}.pool" inC 3 2 1 1 1
  let conv1 ← loadWNConv1d handle s!"{namePrefix}.conv1" inC outC 3 1 1 1
  let conv2 ← loadWNConv1d handle s!"{namePrefix}.conv2" outC outC 3 1 1 1
  let norm1 ← loadAdaIN1d handle s!"{namePrefix}.norm1" styleDim inC
  let norm2 ← loadAdaIN1d handle s!"{namePrefix}.norm2" styleDim outC
  let shortcut ←
    if inC == outC then
      pure none
    else
      pure <| some (← loadWNConv1d handle s!"{namePrefix}.conv1x1" inC outC 1 1 0 1 true)
  pure { pool, conv1, conv2, norm1, norm2, shortcut }

private def loadAdaINResBlock1
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (channels kernel styleDim : UInt64)
    : IO (AdaINResBlock1 channels kernel styleDim) := do
  let mut convs1 : Array (Conv1dParams channels channels kernel 1 ((kernel - 1) / 2) 1) := #[]
  let mut convs2 : Array (Conv1dParams channels channels kernel 1 ((kernel - 1) / 2) 1) := #[]
  let mut norms1 : Array (AdaIN1d styleDim channels) := #[]
  let mut norms2 : Array (AdaIN1d styleDim channels) := #[]
  let mut alpha1 : Array (T #[1, channels, 1]) := #[]
  let mut alpha2 : Array (T #[1, channels, 1]) := #[]
  for i in [:3] do
    convs1 := convs1.push (← loadConv1dEffective handle s!"{namePrefix}.convs1.{i}" channels channels kernel)
    convs2 := convs2.push (← loadConv1dEffective handle s!"{namePrefix}.convs2.{i}" channels channels kernel)
    norms1 := norms1.push (← loadAdaIN1d handle s!"{namePrefix}.adain1.{i}" styleDim channels)
    norms2 := norms2.push (← loadAdaIN1d handle s!"{namePrefix}.adain2.{i}" styleDim channels)
    let a1 : T #[1, channels, 1] ← loadTensorTarget handle s!"{namePrefix}.alpha1.{i}" #[1, channels, 1]
    let a2 : T #[1, channels, 1] ← loadTensorTarget handle s!"{namePrefix}.alpha2.{i}" #[1, channels, 1]
    alpha1 := alpha1.push (reqGradFalse a1)
    alpha2 := alpha2.push (reqGradFalse a2)
  pure { convs1, convs2, norms1, norms2, alpha1, alpha2 }

private def loadProsodyPredictor (handle : safetensors.SafeTensorsHandle) (cfg : KittenTTSConfig) (log : Handlers := {})
    : IO (ProsodyPredictor cfg) := do
  log.onInfo "  predictor.text_encoder..."
  let durationEncoder ← loadDurationEncoder handle cfg
  log.onInfo "  predictor.lstm..."
  let lstm ← loadBiLSTM handle "predictor.lstm" (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2)
  log.onInfo "  predictor.duration_proj..."
  let durationProj ← loadLinear handle "predictor.duration_proj" cfg.hiddenDim cfg.maxDur
  log.onInfo "  predictor.shared..."
  let shared ← loadBiLSTM handle "predictor.shared" (cfg.hiddenDim + cfg.styleDim) (cfg.hiddenDim / 2)
  log.onInfo "  predictor.F0 blocks..."
  let f0Blk0 ← loadAdainResBlk1dSame handle "predictor.F0.0" cfg.hiddenDim cfg.hiddenDim cfg.styleDim
  let f0Blk1 ← loadAdainResBlk1dUp handle "predictor.F0.1" cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim
  let f0Blk2 ← loadAdainResBlk1dSame handle "predictor.F0.2" (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim
  log.onInfo "  predictor.N blocks..."
  let nBlk0 ← loadAdainResBlk1dSame handle "predictor.N.0" cfg.hiddenDim cfg.hiddenDim cfg.styleDim
  let nBlk1 ← loadAdainResBlk1dUp handle "predictor.N.1" cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim
  let nBlk2 ← loadAdainResBlk1dSame handle "predictor.N.2" (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim
  log.onInfo "  predictor output projections..."
  let f0ProjWeight : T #[1, cfg.hiddenDim / 2, 1] ← loadTensorTarget handle "predictor.F0_proj.weight" #[1, cfg.hiddenDim / 2, 1]
  let f0ProjBias : T #[1] ← loadTensorTarget handle "predictor.F0_proj.bias" #[1]
  let nProjWeight : T #[1, cfg.hiddenDim / 2, 1] ← loadTensorTarget handle "predictor.N_proj.weight" #[1, cfg.hiddenDim / 2, 1]
  let nProjBias : T #[1] ← loadTensorTarget handle "predictor.N_proj.bias" #[1]
  pure {
    durationEncoder
    lstm
    durationProj
    shared
    f0Blk0
    f0Blk1
    f0Blk2
    nBlk0
    nBlk1
    nBlk2
    f0Proj := { weight := reqGradFalse f0ProjWeight, bias := reqGradFalse f0ProjBias }
    nProj := { weight := reqGradFalse nProjWeight, bias := reqGradFalse nProjBias }
  }

private def loadAdaIN1dDyn
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (styleDim channels : UInt64)
    : IO (AdaIN1dDyn styleDim) := do
  let fc ← loadLinear handle s!"{namePrefix}.fc" styleDim (2 * channels)
  let normWeight ← loadAdaINNormWeightOrDefault handle namePrefix channels
  let normBias ← loadAdaINNormBiasOrDefault handle namePrefix channels
  pure {
    fcWeight := nn.eraseShape (reqGradFalse fc.weight)
    fcBias := nn.eraseShape (reqGradFalse fc.bias)
    normWeight := nn.eraseShape (reqGradFalse normWeight)
    normBias := nn.eraseShape (reqGradFalse normBias)
  }

private def loadWNConv1dDyn
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (inC outC kernel stride padding dilation : UInt64)
    : IO WNConv1dDyn := do
  let m ← loadWNConv1d handle namePrefix inC outC kernel stride padding dilation
  pure {
    weightV := nn.eraseShape m.weightV
    weightG := nn.eraseShape m.weightG
    bias := nn.eraseShape m.bias
    kernel, stride, padding, dilation
  }

private def loadWNConvTranspose1dDyn
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (inC outC kernel stride padding outputPadding dilation : UInt64)
    : IO WNConvTranspose1dDyn := do
  let m ← loadWNConvTranspose1d handle namePrefix inC outC kernel stride padding outputPadding dilation
  pure {
    weightV := nn.eraseShape m.weightV
    weightG := nn.eraseShape m.weightG
    bias := nn.eraseShape m.bias
    kernel, stride, padding, outputPadding, dilation
  }

private def loadGeneratorAdaINResBlock
    (handle : safetensors.SafeTensorsHandle)
    (namePrefix : String)
    (styleDim channels : UInt64)
    (kernel : UInt64)
    (dilations : Array UInt64)
    : IO (GeneratorAdaINResBlock styleDim) := do
  let mut convs1 : Array WNConv1dDyn := #[]
  let mut convs2 : Array WNConv1dDyn := #[]
  let mut norms1 : Array (AdaIN1dDyn styleDim) := #[]
  let mut norms2 : Array (AdaIN1dDyn styleDim) := #[]
  let mut alpha1 : Array (T #[]) := #[]
  let mut alpha2 : Array (T #[]) := #[]
  for i in [:3] do
    let dilation := dilations.getD i 1
    let padding := (kernel * dilation - dilation) / 2
    convs1 := convs1.push (← loadWNConv1dDyn handle s!"{namePrefix}.convs1.{i}" channels channels kernel 1 padding dilation)
    convs2 := convs2.push (← loadWNConv1dDyn handle s!"{namePrefix}.convs2.{i}" channels channels kernel 1 ((kernel - 1) / 2) 1)
    norms1 := norms1.push (← loadAdaIN1dDyn handle s!"{namePrefix}.adain1.{i}" styleDim channels)
    norms2 := norms2.push (← loadAdaIN1dDyn handle s!"{namePrefix}.adain2.{i}" styleDim channels)
    let a1 : T #[1, channels, 1] ← loadTensorTarget handle s!"{namePrefix}.alpha1.{i}" #[1, channels, 1]
    let a2 : T #[1, channels, 1] ← loadTensorTarget handle s!"{namePrefix}.alpha2.{i}" #[1, channels, 1]
    alpha1 := alpha1.push (nn.eraseShape (reqGradFalse a1))
    alpha2 := alpha2.push (nn.eraseShape (reqGradFalse a2))
  pure { convs1, convs2, norms1, norms2, alpha1, alpha2 }

private def loadSourceModuleHnNSF (handle : safetensors.SafeTensorsHandle) (cfg : KittenTTSConfig)
    : IO (SourceModuleHnNSF cfg) := do
  let linear ←
    loadLinear handle "decoder.generator.m_source.l_linear" (cfg.generator.harmonicCount + 1) 1
  pure { linear }

private def loadGenerator (handle : safetensors.SafeTensorsHandle) (cfg : KittenTTSConfig)
    : IO (Generator cfg) := do
  let source ← loadSourceModuleHnNSF handle cfg
  let numKernels := cfg.generator.resblockKernelSizes.size
  let mut stages : Array (GeneratorStage cfg.styleDim) := #[]
  for i in [:cfg.generator.upsampleRates.size] do
    let inC := GeneratorConfig.stageInChannels cfg.generator i
    let outC := GeneratorConfig.stageOutChannels cfg.generator i
    let stride ← getOrThrow cfg.generator.upsampleRates i s!"Missing upsample rate for stage {i}"
    let kernel ← getOrThrow cfg.generator.upsampleKernelSizes i s!"Missing upsample kernel for stage {i}"
    let up ← loadWNConvTranspose1dDyn handle s!"decoder.generator.ups.{i}" inC outC kernel stride ((kernel - stride) / 2) 0 1
    let noiseKernel := GeneratorConfig.noiseKernel cfg.generator i
    let noiseStride := GeneratorConfig.noiseStride cfg.generator i
    let noisePadding := GeneratorConfig.noisePadding cfg.generator i
    let noiseConv ← loadWNConv1dDyn handle s!"decoder.generator.noise_convs.{i}" (GeneratorConfig.stftFeatureChannels cfg.generator) outC noiseKernel noiseStride noisePadding 1
    let noiseResKernel := if i + 1 < cfg.generator.upsampleRates.size then 7 else 11
    let noiseRes ← loadGeneratorAdaINResBlock handle s!"decoder.generator.noise_res.{i}" cfg.styleDim outC noiseResKernel #[1, 3, 5]
    let mut resBlocks : Array (GeneratorAdaINResBlock cfg.styleDim) := #[]
    for j in [:numKernels] do
      let blockKernel ←
        getOrThrow cfg.generator.resblockKernelSizes j s!"Missing resblock kernel {j}"
      let dilations ←
        getOrThrow cfg.generator.resblockDilationSizes j s!"Missing resblock dilation set {j}"
      let flatIdx := i * numKernels + j
      resBlocks := resBlocks.push (← loadGeneratorAdaINResBlock handle s!"decoder.generator.resblocks.{flatIdx}" cfg.styleDim outC blockKernel dilations)
    stages := stages.push { up, noiseConv, noiseRes, resBlocks }
  let convPost ←
    loadWNConv1dDyn handle "decoder.generator.conv_post"
      (GeneratorConfig.finalChannels cfg.generator)
      (cfg.generator.genIstftNFft + 2)
      7
      1
      3
      1
  pure {
    source
    stages
    convPost
    window := signal.hannWindow cfg.generator.genIstftNFft
  }

private def loadDecoder (handle : safetensors.SafeTensorsHandle) (cfg : KittenTTSConfig)
    : IO (Decoder cfg) := do
  let f0Conv ← loadWNConv1d handle "decoder.F0_conv" 1 1 3 2 1 1
  let nConv ← loadWNConv1d handle "decoder.N_conv" 1 1 3 2 1 1
  let encode ← loadAdainResBlk1dSame handle "decoder.encode" (cfg.hiddenDim + 2) cfg.maxConvDim cfg.styleDim
  let asrRes ← loadWNConv1d handle "decoder.asr_res.0" cfg.hiddenDim cfg.asrResDim 1 1 0 1
  let decode0 ← loadAdainResBlk1dSame handle "decoder.decode.0" (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  let decode1 ← loadAdainResBlk1dSame handle "decoder.decode.1" (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  let decode2 ← loadAdainResBlk1dSame handle "decoder.decode.2" (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim
  let decode3 ← loadAdainResBlk1dUp handle "decoder.decode.3" (cfg.maxConvDim + cfg.asrResDim + 2) (KittenTTSConfig.decoderChannels cfg) cfg.styleDim
  let generator ← loadGenerator handle cfg
  pure { f0Conv, nConv, encode, asrRes, decode0, decode1, decode2, decode3, generator }

private def generatorAdaINResBlockFromParts {styleDim : UInt64}
    (convs1 : Array WNConv1dDyn)
    (convs2 : Array WNConv1dDyn)
    (norms1 : Array (AdaIN1dDyn styleDim))
    (norms2 : Array (AdaIN1dDyn styleDim))
    (alpha1 : Array (T #[]))
    (alpha2 : Array (T #[]))
    : GeneratorAdaINResBlock styleDim := {
      convs1, convs2, norms1, norms2, alpha1, alpha2
    }

private def mapArrayM {α β} (xs : Array α) (f : α → IO β) : IO (Array β) := do
  let mut ys : Array β := #[]
  for x in xs do
    ys := ys.push (← f x)
  pure ys

private def mapArrayIdxM {α β} (xs : Array α) (f : Nat → α → IO β) : IO (Array β) := do
  let mut ys : Array β := #[]
  for i in [:xs.size] do
    if let some x := xs[i]? then
      ys := ys.push (← f i x)
  pure ys

private def loadGeneratorAdaINResBlockFromCheckpointArrays
    {styleDim : UInt64}
    (convs1 : Array α)
    (convs2 : Array β)
    (adain1 : Array γ)
    (adain2 : Array δ)
    (alpha1 : Array ε)
    (alpha2 : Array ζ)
    (kernel : UInt64)
    (dilations : Array UInt64)
    (mkConv1 : α → UInt64 → UInt64 → WNConv1dDyn)
    (mkConv2 : β → WNConv1dDyn)
    (mkAdaIN1 : γ → AdaIN1dDyn styleDim)
    (mkAdaIN2 : δ → AdaIN1dDyn styleDim)
    (mkAlpha1 : ε → T #[])
    (mkAlpha2 : ζ → T #[])
    : IO (GeneratorAdaINResBlock styleDim) := do
  let convs1 ←
    mapArrayIdxM convs1 fun i node => do
      let dilation := dilations.getD i 1
      let padding := (kernel * dilation - dilation) / 2
      pure (mkConv1 node padding dilation)
  let convs2 ← mapArrayM convs2 fun node => pure (mkConv2 node)
  let norms1 ← mapArrayM adain1 fun node => pure (mkAdaIN1 node)
  let norms2 ← mapArrayM adain2 fun node => pure (mkAdaIN2 node)
  pure <|
    generatorAdaINResBlockFromParts
      convs1
      convs2
      norms1
      norms2
      (alpha1.map mkAlpha1)
      (alpha2.map mkAlpha2)

private def loadKittenAlbertFromCheckpoint
    (weights : KokoroCheckpoint.WeightsBert)
    (cfg : KittenTTSConfig)
    : IO (KittenAlbert cfg.nToken cfg.plbert) := do
  let embeddings : AlbertEmbeddings cfg.nToken cfg.plbert := {
    wordEmbeddings := reqGradFalse weights.embeddings.word_embeddings.weight
    positionEmbeddings := reqGradFalse weights.embeddings.position_embeddings.weight
    tokenTypeEmbeddings := reqGradFalse weights.embeddings.token_type_embeddings.weight
    layerNormWeight := reqGradFalse weights.embeddings.layernorm.weight
    layerNormBias := reqGradFalse weights.embeddings.layernorm.bias
  }
  let groups : Array (KittenAlbertLayerGroup cfg.plbert) :=
    weights.encoder.albert_layer_groups.map fun group => {
      layers := group.albert_layers.map fun layer => {
        attention := {
          query := linearFromTensors layer.attention.query.weight layer.attention.query.bias
          key := linearFromTensors layer.attention.key.weight layer.attention.key.bias
          value := linearFromTensors layer.attention.value.weight layer.attention.value.bias
          dense := linearFromTensors layer.attention.dense.weight layer.attention.dense.bias
          layerNormWeight := reqGradFalse layer.attention.layernorm.weight
          layerNormBias := reqGradFalse layer.attention.layernorm.bias
        }
        fullLayerNormWeight := reqGradFalse layer.full_layer_layer_norm.weight
        fullLayerNormBias := reqGradFalse layer.full_layer_layer_norm.bias
        ffn := linearFromTensors layer.ffn.weight layer.ffn.bias
        ffnOutput := linearFromTensors layer.ffn_output.weight layer.ffn_output.bias
      }
    }
  let encoder : KittenAlbertEncoder cfg.plbert := {
    embeddingHiddenMappingIn :=
      linearFromTensors
        weights.encoder.embedding_hidden_mapping_in.weight
        weights.encoder.embedding_hidden_mapping_in.bias
    groups := groups
  }
  let pooler : LinearNorm cfg.plbert.hiddenSize cfg.plbert.hiddenSize :=
    linearFromTensors weights.pooler.weight weights.pooler.bias
  pure { embeddings, encoder, pooler }

private def loadTextEncoderFromCheckpoint
    (weights : KokoroCheckpoint.WeightsTextencoder)
    (cfg : KittenTTSConfig)
    : IO (TextEncoder cfg) := do
  let convs : Array (TextConvLayer cfg) :=
    weights.cnn.map fun layer => {
      conv := conv1dFromWeightNorm layer.i0.weight_v layer.i0.weight_g layer.i0.bias
      lnWeight := reqGradFalse layer.i1.gamma
      lnBias := reqGradFalse layer.i1.beta
    }
  let lstm :=
    biLSTMFromTensors
      weights.lstm.weight_ih_l0
      weights.lstm.weight_hh_l0
      weights.lstm.bias_ih_l0
      weights.lstm.bias_hh_l0
      weights.lstm.weight_ih_l0_reverse
      weights.lstm.weight_hh_l0_reverse
      weights.lstm.bias_ih_l0_reverse
      weights.lstm.bias_hh_l0_reverse
  pure {
    embedding := reqGradFalse weights.embedding.weight
    convs
    lstm
  }

private def loadDurationEncoderFromCheckpoint
    (weights : KokoroCheckpoint.WeightsPredictorTextencoder)
    (cfg : KittenTTSConfig)
    : DurationEncoder cfg := {
      lstms := #[
        biLSTMFromTensors
          weights.lstms.i0.weight_ih_l0
          weights.lstms.i0.weight_hh_l0
          weights.lstms.i0.bias_ih_l0
          weights.lstms.i0.bias_hh_l0
          weights.lstms.i0.weight_ih_l0_reverse
          weights.lstms.i0.weight_hh_l0_reverse
          weights.lstms.i0.bias_ih_l0_reverse
          weights.lstms.i0.bias_hh_l0_reverse,
        biLSTMFromTensors
          weights.lstms.i2.weight_ih_l0
          weights.lstms.i2.weight_hh_l0
          weights.lstms.i2.bias_ih_l0
          weights.lstms.i2.bias_hh_l0
          weights.lstms.i2.weight_ih_l0_reverse
          weights.lstms.i2.weight_hh_l0_reverse
          weights.lstms.i2.bias_ih_l0_reverse
          weights.lstms.i2.bias_hh_l0_reverse,
        biLSTMFromTensors
          weights.lstms.i4.weight_ih_l0
          weights.lstms.i4.weight_hh_l0
          weights.lstms.i4.bias_ih_l0
          weights.lstms.i4.bias_hh_l0
          weights.lstms.i4.weight_ih_l0_reverse
          weights.lstms.i4.weight_hh_l0_reverse
          weights.lstms.i4.bias_ih_l0_reverse
          weights.lstms.i4.bias_hh_l0_reverse
      ]
      norms := #[
        adaLayerNormFromLinear weights.lstms.i1.fc.weight weights.lstms.i1.fc.bias,
        adaLayerNormFromLinear weights.lstms.i3.fc.weight weights.lstms.i3.fc.bias,
        adaLayerNormFromLinear weights.lstms.i5.fc.weight weights.lstms.i5.fc.bias
      ]
    }

private def loadProsodyPredictorFromCheckpoint
    (weights : KokoroCheckpoint.WeightsPredictor)
    (cfg : KittenTTSConfig)
    (log : Handlers := {})
    : IO (ProsodyPredictor cfg) := do
  log.onInfo "  predictor.text_encoder..."
  let durationEncoder := loadDurationEncoderFromCheckpoint weights.text_encoder cfg
  log.onInfo "  predictor.lstm..."
  let lstm :=
    biLSTMFromTensors
      weights.lstm.weight_ih_l0
      weights.lstm.weight_hh_l0
      weights.lstm.bias_ih_l0
      weights.lstm.bias_hh_l0
      weights.lstm.weight_ih_l0_reverse
      weights.lstm.weight_hh_l0_reverse
      weights.lstm.bias_ih_l0_reverse
      weights.lstm.bias_hh_l0_reverse
  log.onInfo "  predictor.duration_proj..."
  let durationProj : LinearNorm cfg.hiddenDim cfg.maxDur :=
    linearFromTensors
      weights.duration_proj.linear_layer.weight
      weights.duration_proj.linear_layer.bias
  log.onInfo "  predictor.shared..."
  let shared :=
    biLSTMFromTensors
      weights.shared.weight_ih_l0
      weights.shared.weight_hh_l0
      weights.shared.bias_ih_l0
      weights.shared.bias_hh_l0
      weights.shared.weight_ih_l0_reverse
      weights.shared.weight_hh_l0_reverse
      weights.shared.bias_ih_l0_reverse
      weights.shared.bias_hh_l0_reverse
  log.onInfo "  predictor.F0 blocks..."
  let f0Blk0 : AdainResBlk1dSame cfg.hiddenDim cfg.hiddenDim cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm weights.f0.i0.conv1.weight_v weights.f0.i0.conv1.weight_g weights.f0.i0.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.f0.i0.conv2.weight_v weights.f0.i0.conv2.weight_g weights.f0.i0.conv2.bias
    norm1 := adaIN1dFromLinear weights.f0.i0.norm1.fc.weight weights.f0.i0.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.f0.i0.norm2.fc.weight weights.f0.i0.norm2.fc.bias
    shortcut := none
  }
  let f0Blk1 : AdainResBlk1dUp cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim := {
    pool := wnDepthwiseTransConv1dFromWeightNorm weights.f0.i1.pool.weight_v weights.f0.i1.pool.weight_g weights.f0.i1.pool.bias
    conv1 := wnConv1dFromWeightNorm weights.f0.i1.conv1.weight_v weights.f0.i1.conv1.weight_g weights.f0.i1.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.f0.i1.conv2.weight_v weights.f0.i1.conv2.weight_g weights.f0.i1.conv2.bias
    norm1 := adaIN1dFromLinear weights.f0.i1.norm1.fc.weight weights.f0.i1.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.f0.i1.norm2.fc.weight weights.f0.i1.norm2.fc.bias
    shortcut := some <| wnConv1dFromWeightNorm weights.f0.i1.conv1x1.weight_v weights.f0.i1.conv1x1.weight_g (torch.zeros #[cfg.hiddenDim / 2] false Device.CPU)
  }
  let f0Blk2 : AdainResBlk1dSame (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm weights.f0.i2.conv1.weight_v weights.f0.i2.conv1.weight_g weights.f0.i2.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.f0.i2.conv2.weight_v weights.f0.i2.conv2.weight_g weights.f0.i2.conv2.bias
    norm1 := adaIN1dFromLinear weights.f0.i2.norm1.fc.weight weights.f0.i2.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.f0.i2.norm2.fc.weight weights.f0.i2.norm2.fc.bias
    shortcut := none
  }
  log.onInfo "  predictor.N blocks..."
  let nBlk0 : AdainResBlk1dSame cfg.hiddenDim cfg.hiddenDim cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm weights.n.i0.conv1.weight_v weights.n.i0.conv1.weight_g weights.n.i0.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.n.i0.conv2.weight_v weights.n.i0.conv2.weight_g weights.n.i0.conv2.bias
    norm1 := adaIN1dFromLinear weights.n.i0.norm1.fc.weight weights.n.i0.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.n.i0.norm2.fc.weight weights.n.i0.norm2.fc.bias
    shortcut := none
  }
  let nBlk1 : AdainResBlk1dUp cfg.hiddenDim (cfg.hiddenDim / 2) cfg.styleDim := {
    pool := wnDepthwiseTransConv1dFromWeightNorm weights.n.i1.pool.weight_v weights.n.i1.pool.weight_g weights.n.i1.pool.bias
    conv1 := wnConv1dFromWeightNorm weights.n.i1.conv1.weight_v weights.n.i1.conv1.weight_g weights.n.i1.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.n.i1.conv2.weight_v weights.n.i1.conv2.weight_g weights.n.i1.conv2.bias
    norm1 := adaIN1dFromLinear weights.n.i1.norm1.fc.weight weights.n.i1.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.n.i1.norm2.fc.weight weights.n.i1.norm2.fc.bias
    shortcut := some <| wnConv1dFromWeightNorm weights.n.i1.conv1x1.weight_v weights.n.i1.conv1x1.weight_g (torch.zeros #[cfg.hiddenDim / 2] false Device.CPU)
  }
  let nBlk2 : AdainResBlk1dSame (cfg.hiddenDim / 2) (cfg.hiddenDim / 2) cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm weights.n.i2.conv1.weight_v weights.n.i2.conv1.weight_g weights.n.i2.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.n.i2.conv2.weight_v weights.n.i2.conv2.weight_g weights.n.i2.conv2.bias
    norm1 := adaIN1dFromLinear weights.n.i2.norm1.fc.weight weights.n.i2.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.n.i2.norm2.fc.weight weights.n.i2.norm2.fc.bias
    shortcut := none
  }
  log.onInfo "  predictor output projections..."
  pure {
    durationEncoder
    lstm
    durationProj
    shared
    f0Blk0
    f0Blk1
    f0Blk2
    nBlk0
    nBlk1
    nBlk2
    f0Proj := {
      weight := reqGradFalse weights.f0_proj.weight
      bias := reqGradFalse weights.f0_proj.bias
    }
    nProj := {
      weight := reqGradFalse weights.n_proj.weight
      bias := reqGradFalse weights.n_proj.bias
    }
  }

private def loadGeneratorFromCheckpoint
    (weights : KokoroCheckpoint.WeightsDecoderGenerator)
    (cfg : KittenTTSConfig)
    : IO (Generator cfg) := do
  let source : SourceModuleHnNSF cfg := {
    linear := linearFromTensors weights.m_source.l_linear.weight weights.m_source.l_linear.bias
  }
  let noiseRes0Dilations : Array UInt64 := #[1, 3, 5]
  let noiseRes1Dilations : Array UInt64 := #[1, 3, 5]
  let stage0Out := GeneratorConfig.stageOutChannels cfg.generator 0
  let stage1Out := GeneratorConfig.stageOutChannels cfg.generator 1
  let stage0Stride ← getOrThrow cfg.generator.upsampleRates 0 "Missing upsample rate for stage 0"
  let stage0Kernel ← getOrThrow cfg.generator.upsampleKernelSizes 0 "Missing upsample kernel for stage 0"
  let stage1Stride ← getOrThrow cfg.generator.upsampleRates 1 "Missing upsample rate for stage 1"
  let stage1Kernel ← getOrThrow cfg.generator.upsampleKernelSizes 1 "Missing upsample kernel for stage 1"
  let resblockKernel0 ← getOrThrow cfg.generator.resblockKernelSizes 0 "Missing generator resblock kernel 0"
  let resblockKernel1 ← getOrThrow cfg.generator.resblockKernelSizes 1 "Missing generator resblock kernel 1"
  let resblockKernel2 ← getOrThrow cfg.generator.resblockKernelSizes 2 "Missing generator resblock kernel 2"
  let resblockDilations0 ← getOrThrow cfg.generator.resblockDilationSizes 0 "Missing generator dilation set 0"
  let resblockDilations1 ← getOrThrow cfg.generator.resblockDilationSizes 1 "Missing generator dilation set 1"
  let resblockDilations2 ← getOrThrow cfg.generator.resblockDilationSizes 2 "Missing generator dilation set 2"
  let noiseRes0 ←
    loadGeneratorAdaINResBlockFromCheckpointArrays
      weights.noise_res.i0.convs1
      weights.noise_res.i0.convs2
      weights.noise_res.i0.adain1
      weights.noise_res.i0.adain2
      weights.noise_res.i0.alpha1
      weights.noise_res.i0.alpha2
      (if 0 + 1 < cfg.generator.upsampleRates.size then 7 else 11)
      noiseRes0Dilations
      (fun node padding dilation =>
        wnConv1dDynFromWeightNorm node.weight_v node.weight_g node.bias 1 padding dilation)
      (fun node =>
        wnConv1dDynFromWeightNorm
          node.weight_v
          node.weight_g
          node.bias
          1
          (((if 0 + 1 < cfg.generator.upsampleRates.size then 7 else 11) - 1) / 2)
          1)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage0Out) node.fc.weight node.fc.bias)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage0Out) node.fc.weight node.fc.bias)
      alphaDyn
      alphaDyn
  let noiseRes1 ←
    loadGeneratorAdaINResBlockFromCheckpointArrays
      weights.noise_res.i1.convs1
      weights.noise_res.i1.convs2
      weights.noise_res.i1.adain1
      weights.noise_res.i1.adain2
      weights.noise_res.i1.alpha1
      weights.noise_res.i1.alpha2
      (if 1 + 1 < cfg.generator.upsampleRates.size then 7 else 11)
      noiseRes1Dilations
      (fun node padding dilation =>
        wnConv1dDynFromWeightNorm node.weight_v node.weight_g node.bias 1 padding dilation)
      (fun node =>
        wnConv1dDynFromWeightNorm
          node.weight_v
          node.weight_g
          node.bias
          1
          (((if 1 + 1 < cfg.generator.upsampleRates.size then 7 else 11) - 1) / 2)
          1)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage1Out) node.fc.weight node.fc.bias)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage1Out) node.fc.weight node.fc.bias)
      alphaDyn
      alphaDyn
  let res0 ←
    loadGeneratorAdaINResBlockFromCheckpointArrays
      weights.resblocks.i0.convs1
      weights.resblocks.i0.convs2
      weights.resblocks.i0.adain1
      weights.resblocks.i0.adain2
      weights.resblocks.i0.alpha1
      weights.resblocks.i0.alpha2
      resblockKernel0
      resblockDilations0
      (fun node padding dilation =>
        wnConv1dDynFromWeightNorm node.weight_v node.weight_g node.bias 1 padding dilation)
      (fun node =>
        wnConv1dDynFromWeightNorm node.weight_v node.weight_g node.bias 1 ((resblockKernel0 - 1) / 2) 1)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage0Out) node.fc.weight node.fc.bias)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage0Out) node.fc.weight node.fc.bias)
      alphaDyn
      alphaDyn
  let res1 ←
    loadGeneratorAdaINResBlockFromCheckpointArrays
      weights.resblocks.i1.convs1
      weights.resblocks.i1.convs2
      weights.resblocks.i1.adain1
      weights.resblocks.i1.adain2
      weights.resblocks.i1.alpha1
      weights.resblocks.i1.alpha2
      resblockKernel1
      resblockDilations1
      (fun node padding dilation =>
        wnConv1dDynFromWeightNorm node.weight_v node.weight_g node.bias 1 padding dilation)
      (fun node =>
        wnConv1dDynFromWeightNorm node.weight_v node.weight_g node.bias 1 ((resblockKernel1 - 1) / 2) 1)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage0Out) node.fc.weight node.fc.bias)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage0Out) node.fc.weight node.fc.bias)
      alphaDyn
      alphaDyn
  let res2 ←
    loadGeneratorAdaINResBlockFromCheckpointArrays
      weights.resblocks.i2.convs1
      weights.resblocks.i2.convs2
      weights.resblocks.i2.adain1
      weights.resblocks.i2.adain2
      weights.resblocks.i2.alpha1
      weights.resblocks.i2.alpha2
      resblockKernel2
      resblockDilations2
      (fun node padding dilation =>
        wnConv1dDynFromWeightNorm node.weight_v node.weight_g node.bias 1 padding dilation)
      (fun node =>
        wnConv1dDynFromWeightNorm node.weight_v node.weight_g node.bias 1 ((resblockKernel2 - 1) / 2) 1)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage0Out) node.fc.weight node.fc.bias)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage0Out) node.fc.weight node.fc.bias)
      alphaDyn
      alphaDyn
  let res3 ←
    loadGeneratorAdaINResBlockFromCheckpointArrays
      weights.resblocks.i3.convs1
      weights.resblocks.i3.convs2
      weights.resblocks.i3.adain1
      weights.resblocks.i3.adain2
      weights.resblocks.i3.alpha1
      weights.resblocks.i3.alpha2
      resblockKernel0
      resblockDilations0
      (fun node padding dilation =>
        wnConv1dDynFromWeightNorm node.weight_v node.weight_g node.bias 1 padding dilation)
      (fun node =>
        wnConv1dDynFromWeightNorm node.weight_v node.weight_g node.bias 1 ((resblockKernel0 - 1) / 2) 1)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage1Out) node.fc.weight node.fc.bias)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage1Out) node.fc.weight node.fc.bias)
      alphaDyn
      alphaDyn
  let res4 ←
    loadGeneratorAdaINResBlockFromCheckpointArrays
      weights.resblocks.i4.convs1
      weights.resblocks.i4.convs2
      weights.resblocks.i4.adain1
      weights.resblocks.i4.adain2
      weights.resblocks.i4.alpha1
      weights.resblocks.i4.alpha2
      resblockKernel1
      resblockDilations1
      (fun node padding dilation =>
        wnConv1dDynFromWeightNorm node.weight_v node.weight_g node.bias 1 padding dilation)
      (fun node =>
        wnConv1dDynFromWeightNorm node.weight_v node.weight_g node.bias 1 ((resblockKernel1 - 1) / 2) 1)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage1Out) node.fc.weight node.fc.bias)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage1Out) node.fc.weight node.fc.bias)
      alphaDyn
      alphaDyn
  let res5 ←
    loadGeneratorAdaINResBlockFromCheckpointArrays
      weights.resblocks.i5.convs1
      weights.resblocks.i5.convs2
      weights.resblocks.i5.adain1
      weights.resblocks.i5.adain2
      weights.resblocks.i5.alpha1
      weights.resblocks.i5.alpha2
      resblockKernel2
      resblockDilations2
      (fun node padding dilation =>
        wnConv1dDynFromWeightNorm node.weight_v node.weight_g node.bias 1 padding dilation)
      (fun node =>
        wnConv1dDynFromWeightNorm node.weight_v node.weight_g node.bias 1 ((resblockKernel2 - 1) / 2) 1)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage1Out) node.fc.weight node.fc.bias)
      (fun node =>
        adaIN1dDynFromLinear (channels := stage1Out) node.fc.weight node.fc.bias)
      alphaDyn
      alphaDyn
  let stage0 : GeneratorStage cfg.styleDim := {
    up :=
      wnConvTranspose1dDynFromWeightNorm
        weights.ups.i0.weight_v
        weights.ups.i0.weight_g
        weights.ups.i0.bias
        stage0Stride
        ((stage0Kernel - stage0Stride) / 2)
        0
        1
    noiseConv :=
      wnConv1dDynFromPlain
        weights.noise_convs.i0.weight
        weights.noise_convs.i0.bias
        (GeneratorConfig.noiseStride cfg.generator 0)
        (GeneratorConfig.noisePadding cfg.generator 0)
        1
    noiseRes := noiseRes0
    resBlocks := #[res0, res1, res2]
  }
  let stage1 : GeneratorStage cfg.styleDim := {
    up :=
      wnConvTranspose1dDynFromWeightNorm
        weights.ups.i1.weight_v
        weights.ups.i1.weight_g
        weights.ups.i1.bias
        stage1Stride
        ((stage1Kernel - stage1Stride) / 2)
        0
        1
    noiseConv :=
      wnConv1dDynFromPlain
        weights.noise_convs.i1.weight
        weights.noise_convs.i1.bias
        (GeneratorConfig.noiseStride cfg.generator 1)
        (GeneratorConfig.noisePadding cfg.generator 1)
        1
    noiseRes := noiseRes1
    resBlocks := #[res3, res4, res5]
  }
  let convPost := wnConv1dDynFromWeightNorm weights.conv_post.weight_v weights.conv_post.weight_g weights.conv_post.bias 1 3 1
  pure {
    source
    stages := #[stage0, stage1]
    convPost
    window := signal.hannWindow cfg.generator.genIstftNFft
  }

private def loadDecoderFromCheckpoint
    (weights : KokoroCheckpoint.WeightsDecoder)
    (cfg : KittenTTSConfig)
    : IO (Decoder cfg) := do
  let asrResNode ← getOrThrow weights.asr_res 0 "Missing decoder.asr_res.0"
  let generator ← loadGeneratorFromCheckpoint weights.generator cfg
  let encode : AdainResBlk1dSame (cfg.hiddenDim + 2) cfg.maxConvDim cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm weights.encode.conv1.weight_v weights.encode.conv1.weight_g weights.encode.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.encode.conv2.weight_v weights.encode.conv2.weight_g weights.encode.conv2.bias
    norm1 := adaIN1dFromLinear weights.encode.norm1.fc.weight weights.encode.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.encode.norm2.fc.weight weights.encode.norm2.fc.bias
    shortcut := some <| wnConv1dFromWeightNorm weights.encode.conv1x1.weight_v weights.encode.conv1x1.weight_g (torch.zeros #[cfg.maxConvDim] false Device.CPU)
  }
  let decode0 : AdainResBlk1dSame (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm weights.decode.i0.conv1.weight_v weights.decode.i0.conv1.weight_g weights.decode.i0.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.decode.i0.conv2.weight_v weights.decode.i0.conv2.weight_g weights.decode.i0.conv2.bias
    norm1 := adaIN1dFromLinear weights.decode.i0.norm1.fc.weight weights.decode.i0.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.decode.i0.norm2.fc.weight weights.decode.i0.norm2.fc.bias
    shortcut := some <| wnConv1dFromWeightNorm weights.decode.i0.conv1x1.weight_v weights.decode.i0.conv1x1.weight_g (torch.zeros #[cfg.maxConvDim] false Device.CPU)
  }
  let decode1 : AdainResBlk1dSame (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm weights.decode.i1.conv1.weight_v weights.decode.i1.conv1.weight_g weights.decode.i1.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.decode.i1.conv2.weight_v weights.decode.i1.conv2.weight_g weights.decode.i1.conv2.bias
    norm1 := adaIN1dFromLinear weights.decode.i1.norm1.fc.weight weights.decode.i1.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.decode.i1.norm2.fc.weight weights.decode.i1.norm2.fc.bias
    shortcut := some <| wnConv1dFromWeightNorm weights.decode.i1.conv1x1.weight_v weights.decode.i1.conv1x1.weight_g (torch.zeros #[cfg.maxConvDim] false Device.CPU)
  }
  let decode2 : AdainResBlk1dSame (cfg.maxConvDim + cfg.asrResDim + 2) cfg.maxConvDim cfg.styleDim := {
    conv1 := wnConv1dFromWeightNorm weights.decode.i2.conv1.weight_v weights.decode.i2.conv1.weight_g weights.decode.i2.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.decode.i2.conv2.weight_v weights.decode.i2.conv2.weight_g weights.decode.i2.conv2.bias
    norm1 := adaIN1dFromLinear weights.decode.i2.norm1.fc.weight weights.decode.i2.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.decode.i2.norm2.fc.weight weights.decode.i2.norm2.fc.bias
    shortcut := some <| wnConv1dFromWeightNorm weights.decode.i2.conv1x1.weight_v weights.decode.i2.conv1x1.weight_g (torch.zeros #[cfg.maxConvDim] false Device.CPU)
  }
  let decode3 : AdainResBlk1dUp (cfg.maxConvDim + cfg.asrResDim + 2) (KittenTTSConfig.decoderChannels cfg) cfg.styleDim := {
    pool := wnDepthwiseTransConv1dFromWeightNorm weights.decode.i3.pool.weight_v weights.decode.i3.pool.weight_g weights.decode.i3.pool.bias
    conv1 := wnConv1dFromWeightNorm weights.decode.i3.conv1.weight_v weights.decode.i3.conv1.weight_g weights.decode.i3.conv1.bias
    conv2 := wnConv1dFromWeightNorm weights.decode.i3.conv2.weight_v weights.decode.i3.conv2.weight_g weights.decode.i3.conv2.bias
    norm1 := adaIN1dFromLinear weights.decode.i3.norm1.fc.weight weights.decode.i3.norm1.fc.bias
    norm2 := adaIN1dFromLinear weights.decode.i3.norm2.fc.weight weights.decode.i3.norm2.fc.bias
    shortcut := some <| wnConv1dFromWeightNorm weights.decode.i3.conv1x1.weight_v weights.decode.i3.conv1x1.weight_g (torch.zeros #[KittenTTSConfig.decoderChannels cfg] false Device.CPU)
  }
  pure {
    f0Conv := wnConv1dFromWeightNorm weights.f0_conv.weight_v weights.f0_conv.weight_g weights.f0_conv.bias
    nConv := wnConv1dFromWeightNorm weights.n_conv.weight_v weights.n_conv.weight_g weights.n_conv.bias
    encode
    asrRes := wnConv1dFromWeightNorm asrResNode.weight_v asrResNode.weight_g asrResNode.bias
    decode0
    decode1
    decode2
    decode3
    generator
  }

private def resolveKittenDevice (log : Handlers := {}) : IO Device := do
  let requested := (← IO.getEnv "TYR_DEVICE").map String.toLower
  match requested with
  | some "cpu" => pure Device.CPU
  | some "cuda" =>
    if ← cuda_is_available then
      pure (Device.CUDA 0)
    else
      log.onWarn "TYR_DEVICE=cuda requested but CUDA is unavailable; falling back to auto."
      getBestDevice
  | some "mps" =>
    if ← mps_is_available then
      pure Device.MPS
    else
      log.onWarn "TYR_DEVICE=mps requested but MPS is unavailable; falling back to auto."
      getBestDevice
  | some "auto" => getBestDevice
  | some _ => getBestDevice
  | none =>
    if ← cuda_is_available then
      pure (Device.CUDA 0)
    else
      pure Device.CPU

private def loadCheckpointWeights (path : String) (log : Handlers := {}) : IO LoadedCheckpointWeights := do
  log.onInfo "Loading ALBERT backbone..."
  let bert ← KokoroCheckpoint.bert.load path
  log.onInfo "Loading ALBERT projection..."
  let bertEncoder ← KokoroCheckpoint.bert_encoder.load path
  log.onInfo "Loading prosody predictor..."
  let predictor ← KokoroCheckpoint.predictor.load path
  log.onInfo "Loading text encoder..."
  let textEncoder ← KokoroCheckpoint.text_encoder.load path
  log.onInfo "Loading decoder/vocoder..."
  let decoder ← KokoroCheckpoint.decoder.load path
  pure { bert, bertEncoder, predictor, textEncoder, decoder }

private def reconcileConfigWithCheckpoint
    (cfg : KittenTTSConfig)
    (weights : LoadedCheckpointWeights)
    (log : Handlers := {})
    : IO KittenTTSConfig := do
  let textConv0 ← getOrThrow weights.textEncoder.cnn 0 "Missing text_encoder.cnn.0"
  let asrRes0 ← getOrThrow weights.decoder.asr_res 0 "Missing decoder.asr_res.0"
  let hiddenDim ← tensorDimOrThrow weights.bertEncoder.bias 0 "bert_encoder.bias"
  let nToken ← tensorDimOrThrow weights.textEncoder.embedding.weight 0 "text_encoder.embedding.weight"
  let textEncoderKernelSize ← tensorDimOrThrow textConv0.i0.weight_v 2 "text_encoder.cnn.0.0.weight_v"
  let maxDur ← tensorDimOrThrow weights.predictor.duration_proj.linear_layer.bias 0 "predictor.duration_proj.linear_layer.bias"
  let styleDim ← tensorDimOrThrow weights.decoder.encode.norm1.fc.weight 1 "decoder.encode.norm1.fc.weight"
  let asrResDim ← tensorDimOrThrow asrRes0.bias 0 "decoder.asr_res.0.bias"
  let decoderBlockDim ← tensorDimOrThrow weights.decoder.encode.conv1.bias 0 "decoder.encode.conv1.bias"
  let decoderOutDim ← tensorDimOrThrow weights.decoder.decode.i3.conv2.bias 0 "decoder.decode.3.conv2.bias"
  let upsampleInitialChannel ← tensorDimOrThrow weights.decoder.generator.ups.i0.weight_g 0 "decoder.generator.ups.0.weight_g"
  let genIstftNFft ← do
    let convPostOut ← tensorDimOrThrow weights.decoder.generator.conv_post.bias 0 "decoder.generator.conv_post.bias"
    if convPostOut < 2 then
      throw <| IO.userError s!"Invalid decoder.generator.conv_post output dimension: {convPostOut}"
    pure (convPostOut - 2)
  let harmonicCount ← do
    let harmonicInputs ← tensorDimOrThrow weights.decoder.generator.m_source.l_linear.weight 1 "decoder.generator.m_source.l_linear.weight"
    if harmonicInputs == 0 then
      throw <| IO.userError "Invalid decoder.generator.m_source.l_linear.weight: expected at least one harmonic input"
    pure (harmonicInputs - 1)

  warnShapeOverride log "hiddenDim" cfg.hiddenDim hiddenDim
  warnShapeOverride log "nToken" cfg.nToken nToken
  warnShapeOverride log "textEncoderKernelSize" cfg.textEncoderKernelSize textEncoderKernelSize
  warnShapeOverride log "nLayer" cfg.nLayer weights.textEncoder.cnn.size.toUInt64
  warnShapeOverride log "maxDur" cfg.maxDur maxDur
  warnShapeOverride log "styleDim" cfg.styleDim styleDim
  warnShapeOverride log "asrResDim" cfg.asrResDim asrResDim
  warnShapeOverride log "maxConvDim" cfg.maxConvDim decoderBlockDim
  warnShapeOverride log "decoderOutDim" (KittenTTSConfig.decoderChannels cfg) decoderOutDim
  warnShapeOverride log "generator.upsampleInitialChannel" cfg.generator.upsampleInitialChannel upsampleInitialChannel
  warnShapeOverride log "generator.genIstftNFft" cfg.generator.genIstftNFft genIstftNFft
  warnShapeOverride log "generator.harmonicCount" cfg.generator.harmonicCount harmonicCount

  let generator : GeneratorConfig := {
    cfg.generator with
    upsampleInitialChannel := upsampleInitialChannel
    genIstftNFft := genIstftNFft
    harmonicCount := harmonicCount
  }
  pure {
    cfg with
    hiddenDim := hiddenDim
    maxConvDim := decoderBlockDim
    maxDur := maxDur
    nLayer := weights.textEncoder.cnn.size.toUInt64
    nToken := nToken
    styleDim := styleDim
    textEncoderKernelSize := textEncoderKernelSize
    asrResDim := asrResDim
    decoderOutDim := decoderOutDim
    generator := generator
  }

private def buildModelFromCheckpointWeights
    (weights : LoadedCheckpointWeights)
    (cfg : KittenTTSConfig)
    (log : Handlers := {})
    : IO (Model cfg) := do
  let bert ← loadKittenAlbertFromCheckpoint weights.bert cfg
  let bertEncoder : LinearNorm cfg.plbert.hiddenSize cfg.hiddenDim :=
    linearFromTensors weights.bertEncoder.weight weights.bertEncoder.bias
  let predictor ← loadProsodyPredictorFromCheckpoint weights.predictor cfg log
  let textEncoder ← loadTextEncoderFromCheckpoint weights.textEncoder cfg
  let decoder ← loadDecoderFromCheckpoint weights.decoder cfg
  log.onInfo "Resolving target device..."
  let targetDevice ← resolveKittenDevice log
  let baseModel : Model cfg := { bert, bertEncoder, predictor, textEncoder, decoder }
  let model : Model cfg :=
    match targetDevice with
    | Device.CPU => baseModel
    | _ => TensorStruct.map (fun t => t.to targetDevice) baseModel
  log.onInfo s!"KittenTTS target device: {repr targetDevice}"
  pure model

namespace Model

structure Loaded where
  cfg : KittenTTSConfig
  model : Model cfg

def load (path : String) (cfg : KittenTTSConfig := {}) (log : Handlers := {})
    : IO (Model cfg) := do
  log.onInfo s!"Loading KittenTTS weights from {path}..."
  let weights ← loadCheckpointWeights path log
  buildModelFromCheckpointWeights weights cfg log

def loadAutoConfig (path : String) (cfg : KittenTTSConfig := {}) (log : Handlers := {})
    : IO Loaded := do
  log.onInfo s!"Loading KittenTTS weights from {path}..."
  let weights ← loadCheckpointWeights path log
  let cfg' ← reconcileConfigWithCheckpoint cfg weights log
  let model ← buildModelFromCheckpointWeights weights cfg' log
  pure { cfg := cfg', model := model }

end Model

end torch.kittentts
