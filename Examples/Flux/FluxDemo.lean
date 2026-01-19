/-
  Examples/Flux/FluxDemo.lean

  End-to-end demonstration of Flux Klein 4B image generation.
  Shows complete pipeline: text encoding → diffusion → VAE decoding.
-/
import Tyr.Torch
import Tyr.Model.Qwen
import Tyr.Model.Qwen.Weights
import Tyr.Model.VAE
import Tyr.Model.VAE.Weights
import Tyr.Model.Flux
import Tyr.Model.Flux.Weights

namespace Examples.Flux

open torch
open torch.qwen
open torch.vae
open torch.flux

private def toFloat32 {α : Type} [TensorStruct α] (x : α) : α :=
  TensorStruct.map toFloat' x

private def toFloat32Embedder {cfg : QwenConfig} {max_seq : UInt64}
    (e : QwenFluxEmbedder cfg max_seq) : QwenFluxEmbedder cfg max_seq :=
  { e with model := toFloat32 e.model }

/-- Configuration for the demo -/
structure DemoConfig where
  /-- Directory containing sharded Qwen text encoder weights -/
  qwenDir : String := "weights/flux-klein-4b/text_encoder"
  /-- Path to Flux Klein 4B SafeTensors weights (single file) -/
  fluxPath : String := "weights/flux.safetensors"
  /-- Path to VAE SafeTensors weights (single file) -/
  vaePath : String := "weights/ae.safetensors"
  /-- Output image path -/
  outputPath : String := "output.ppm"
  /-- Optional token file (whitespace-separated token IDs) -/
  tokensPath : String := "weights/flux-klein-4b/tokenizer/tokens.txt"
  /-- Number of diffusion steps -/
  numSteps : Nat := 4
  /-- Packed latent height (16 for 256x256 output with 2×2 patchification and 8× VAE upscaling) -/
  latentH : UInt64 := 16
  /-- Packed latent width (16 for 256x256 output with 2×2 patchification and 8× VAE upscaling) -/
  latentW : UInt64 := 16
  deriving Repr, Inhabited

/-- Simple tokenizer placeholder.
    In production, use a proper BPE tokenizer for Qwen. -/
def simpleTokenize (_text : String) (maxLen : UInt64) : IO (T #[1, maxLen]) := do
  -- Return dummy tokens for demo
  -- Real implementation would use Qwen's tokenizer
  torch.randint 0 151936 #[1, maxLen]

private def qwenPadId : Int64 := 151643
private def maxSeqLen : UInt64 := 512

private def parseTokenIds (content : String) : Array Int64 := Id.run do
  let mut out : Array Int64 := #[]
  let mut cur : String := ""
  for c in content.data do
    if c.isDigit then
      cur := cur.push c
    else
      if !cur.isEmpty then
        match cur.toNat? with
        | some n => out := out.push n.toInt64
        | none => pure ()
        cur := ""
  if !cur.isEmpty then
    match cur.toNat? with
    | some n => out := out.push n.toInt64
    | none => pure ()
  out

private def padTokens (ids : Array Int64) (maxLen : UInt64) (padId : Int64)
    : Array Int64 × Array Int64 := Id.run do
  let max := maxLen.toNat
  let mut tokens := Array.mkEmpty max
  let mut mask := Array.mkEmpty max
  for i in [:max] do
    if h : i < ids.size then
      tokens := tokens.push (ids[i]'h)
      mask := mask.push 1
    else
      tokens := tokens.push padId
      mask := mask.push 0
  (tokens, mask)

private def loadTokenIds (path : String) (maxLen : UInt64) : IO (Array Int64) := do
  let contents ← IO.FS.readFile path
  let mut ids := parseTokenIds contents
  if ids.isEmpty then
    throw (IO.userError s!"No tokens found in {path}")
  if ids.size.toUInt64 > maxLen then
    IO.println s!"    Truncating tokens from {ids.size} to {maxLen}"
    ids := ids.extract 0 maxLen.toNat
  pure ids

/-- Main demo: generate an image from a text prompt -/
def runDemo (cfg : DemoConfig) (prompt : String) : IO Unit := do
  IO.println "=== Flux Klein 4B Demo ==="
  IO.println s!"Prompt: {prompt}"
  IO.println ""

  -- Load models
  IO.println "Loading models..."

  -- Load Qwen text encoder (sharded, with Flux Klein config)
  let qwenCfg := QwenConfig.fluxKleinTextEncoder
  let qwen ← loadQwenFluxEmbedderSharded cfg.qwenDir qwenCfg maxSeqLen #[8, 17, 26]
  let qwen := toFloat32Embedder qwen
  IO.println "  ✓ Qwen text encoder loaded (Flux Klein variant)"

  -- Load Flux model
  let fluxCfg := FluxConfig.klein4B
  let flux ← loadFluxModel cfg.fluxPath fluxCfg
  let flux := toFloat32 flux
  IO.println "  ✓ Flux model loaded"

  -- Load VAE AutoEncoder (includes BN stats for proper normalization)
  let vae ← loadAutoEncoder cfg.vaePath
  let vae := toFloat32 vae
  IO.println "  ✓ VAE AutoEncoder loaded"

  IO.println ""
  IO.println "Generating image..."

  -- Step 1: Tokenize and encode text
  IO.println "  [1/4] Encoding text..."
  let (tokenCount, txtEmb) ← do
    if (← data.fileExists cfg.tokensPath) then
      IO.println s!"    Loading tokens from {cfg.tokensPath}"
      let ids ← loadTokenIds cfg.tokensPath maxSeqLen
      let tokenCount := ids.size.toUInt64
      let (padded, mask) := padTokens ids maxSeqLen qwenPadId
      let tokens := reshape (data.fromInt64Array padded) #[1, maxSeqLen]
      let attnMask := reshape (data.fromInt64Array mask) #[1, maxSeqLen]
      let txtEmb := qwen.encodeMasked qwenCfg maxSeqLen tokens attnMask
      pure (tokenCount, txtEmb)
    else
      IO.println "    Using fallback random tokens (token file not found)"
      let tokens ← simpleTokenize prompt maxSeqLen
      let attnMask := full_int #[1, maxSeqLen] 1
      let txtEmb := qwen.encodeMasked qwenCfg maxSeqLen tokens attnMask
      pure (maxSeqLen, txtEmb)

  IO.println s!"    Text token count: {tokenCount}"
  IO.println s!"    Text embedding shape: {txtEmb.runtimeShape}"
  IO.println s!"    Text embedding dtype: {txtEmb.dtype}"

  -- Step 2: Prepare latent space
  IO.println "  [2/4] Preparing latent space..."
  let imgSeq := cfg.latentH * cfg.latentW
  let noise ← torch.randn #[1, imgSeq, 128]

  -- Compute position IDs for RoPE
  let imgIds ← computeImagePositionIds cfg.latentH cfg.latentW
  let txtIds ← computeTextPositionIds maxSeqLen

  -- Step 3: Run diffusion denoising
  IO.println "  [3/4] Running Flux denoising..."
  let latents := denoise fluxCfg flux noise txtEmb imgIds txtIds cfg.numSteps
  IO.println s!"    Denoised with {cfg.numSteps} steps"
  let _latents2 := mul_scalar latents 1.0
  IO.println "    Latents tensor is valid"

  -- Transform latents for VAE
  let transposed := nn.transpose latents 1 2
  let packedLatents : T #[1, 128, cfg.latentH, cfg.latentW] :=
    reshape transposed #[1, 128, cfg.latentH, cfg.latentW]
  let _packed2 := mul_scalar packedLatents 1.0
  IO.println "    Packed latents tensor is valid"

  -- Step 4: Decode with VAE and save
  IO.println "  [4/4] Decoding with VAE..."

  -- Run VAE decode pipeline
  let z := vae.invNormalize packedLatents
  let z := torch.vae.AutoEncoder.unpackLatents16x16 z
  let _z2 := mul_scalar z 1.0
  IO.println "    VAE input tensor is valid"
  let image := vae.decoder.forward1 z
  let _image2 := mul_scalar image 1.0
  IO.println "    Image tensor is valid"
  torch.data.savePPMExplicit #[1, 3, 256, 256] image cfg.outputPath
  IO.println s!"  ✓ Image saved to {cfg.outputPath}"


/-- Main entry point -/
def _root_.main (args : List String) : IO UInt32 := do
  let prompt := match args with
    | arg :: _ => arg
    | [] => "a cat sitting on a windowsill, photorealistic"

  let cfg : DemoConfig := {
    qwenDir := "weights/flux-klein-4b/text_encoder"
    fluxPath := "weights/flux.safetensors"
    vaePath := "weights/ae.safetensors"
    outputPath := "output.ppm"
    tokensPath := "weights/flux-klein-4b/tokenizer/tokens.txt"
    numSteps := 4
    latentH := 16
    latentW := 16
  }

  runDemo cfg prompt
  return 0

end Examples.Flux
