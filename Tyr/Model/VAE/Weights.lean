/-
  Tyr/Model/VAE/Weights.lean

  Weight loading for Flux VAE decoder from SafeTensors (BFL format).
  Maps ae.safetensors weight names to Tyr structure.
-/
import Tyr.Torch
import Tyr.Model.VAE.Decoder

namespace torch.vae

/-- Load GroupNorm parameters -/
def loadGroupNorm (path : String) (name : String) (num_groups channels : UInt64)
    : IO (GroupNormParams num_groups channels) := do
  let weight ← safetensors.loadTensor path s!"{name}.weight" #[channels]
  let bias ← safetensors.loadTensor path s!"{name}.bias" #[channels]
  pure {
    weight := autograd.set_requires_grad weight false
    bias := autograd.set_requires_grad bias false
  }

/-- Load Conv2d parameters -/
def loadConv2d (path : String) (name : String)
    (in_ch out_ch kernel_size padding : UInt64)
    : IO (Conv2dParams in_ch out_ch kernel_size padding) := do
  let weight ← safetensors.loadTensor path s!"{name}.weight" #[out_ch, in_ch, kernel_size, kernel_size]
  let bias ← safetensors.loadTensor path s!"{name}.bias" #[out_ch]
  pure {
    weight := autograd.set_requires_grad weight false
    bias := autograd.set_requires_grad bias false
  }

/-- Load ResnetBlock (same in/out channels) -/
def loadResnetBlockSame (path : String) (name : String) (ch : UInt64)
    : IO (ResnetBlock ch ch) := do
  let norm1 ← loadGroupNorm path s!"{name}.norm1" 32 ch
  let conv1 ← loadConv2d path s!"{name}.conv1" ch ch 3 1
  let norm2 ← loadGroupNorm path s!"{name}.norm2" 32 ch
  let conv2 ← loadConv2d path s!"{name}.conv2" ch ch 3 1
  pure { norm1, conv1, norm2, conv2, nin_shortcut := none }

/-- Load ResnetBlock with channel change (has nin_shortcut) -/
def loadResnetBlockChange (path : String) (name : String) (in_ch out_ch : UInt64)
    : IO (ResnetBlock in_ch out_ch) := do
  let norm1 ← loadGroupNorm path s!"{name}.norm1" 32 in_ch
  let conv1 ← loadConv2d path s!"{name}.conv1" in_ch out_ch 3 1
  let norm2 ← loadGroupNorm path s!"{name}.norm2" 32 out_ch
  let conv2 ← loadConv2d path s!"{name}.conv2" out_ch out_ch 3 1
  let shortcut ← loadConv2d path s!"{name}.nin_shortcut" in_ch out_ch 1 0
  pure { norm1, conv1, norm2, conv2, nin_shortcut := some shortcut }

/-- Load AttnBlock -/
def loadAttnBlock (path : String) (name : String) (channels : UInt64)
    : IO (AttnBlock channels) := do
  let norm ← loadGroupNorm path s!"{name}.norm" 32 channels
  let q ← loadConv2d path s!"{name}.q" channels channels 1 0
  let k ← loadConv2d path s!"{name}.k" channels channels 1 0
  let v ← loadConv2d path s!"{name}.v" channels channels 1 0
  let proj_out ← loadConv2d path s!"{name}.proj_out" channels channels 1 0
  pure { norm, q, k, v, proj_out }

/-- Load Upsample -/
def loadUpsample (path : String) (name : String) (channels : UInt64)
    : IO (Upsample channels) := do
  let conv ← loadConv2d path s!"{name}.conv" channels channels 3 1
  pure { conv }

/-- Load full VAE Decoder from SafeTensors (BFL format). -/
def loadDecoder (path : String) : IO Decoder := do
  IO.println s!"Loading VAE Decoder from {path}..."

  let pfx := "decoder"

  -- Post-quant conv
  let post_quant_conv ← loadConv2d path s!"{pfx}.post_quant_conv" 32 32 1 0

  -- Initial conv
  let conv_in ← loadConv2d path s!"{pfx}.conv_in" 32 512 3 1
  IO.println "  Loaded conv_in"

  -- Middle block
  let mid_block_1 ← loadResnetBlockSame path s!"{pfx}.mid.block_1" 512
  let mid_attn ← loadAttnBlock path s!"{pfx}.mid.attn_1" 512
  let mid_block_2 ← loadResnetBlockSame path s!"{pfx}.mid.block_2" 512
  IO.println "  Loaded mid block"

  -- Up block 3: 512→512, 3 blocks, upsample
  let up_3_block_0 ← loadResnetBlockSame path s!"{pfx}.up.3.block.0" 512
  let up_3_block_1 ← loadResnetBlockSame path s!"{pfx}.up.3.block.1" 512
  let up_3_block_2 ← loadResnetBlockSame path s!"{pfx}.up.3.block.2" 512
  let up_3_upsample ← loadUpsample path s!"{pfx}.up.3.upsample" 512
  IO.println "  Loaded up block 3"

  -- Up block 2: 512→512, 3 blocks, upsample
  let up_2_block_0 ← loadResnetBlockSame path s!"{pfx}.up.2.block.0" 512
  let up_2_block_1 ← loadResnetBlockSame path s!"{pfx}.up.2.block.1" 512
  let up_2_block_2 ← loadResnetBlockSame path s!"{pfx}.up.2.block.2" 512
  let up_2_upsample ← loadUpsample path s!"{pfx}.up.2.upsample" 512
  IO.println "  Loaded up block 2"

  -- Up block 1: 512→256, 3 blocks, upsample
  let up_1_block_0 ← loadResnetBlockChange path s!"{pfx}.up.1.block.0" 512 256
  let up_1_block_1 ← loadResnetBlockSame path s!"{pfx}.up.1.block.1" 256
  let up_1_block_2 ← loadResnetBlockSame path s!"{pfx}.up.1.block.2" 256
  let up_1_upsample ← loadUpsample path s!"{pfx}.up.1.upsample" 256
  IO.println "  Loaded up block 1"

  -- Up block 0: 256→128, 3 blocks, no upsample
  let up_0_block_0 ← loadResnetBlockChange path s!"{pfx}.up.0.block.0" 256 128
  let up_0_block_1 ← loadResnetBlockSame path s!"{pfx}.up.0.block.1" 128
  let up_0_block_2 ← loadResnetBlockSame path s!"{pfx}.up.0.block.2" 128
  IO.println "  Loaded up block 0"

  -- Final layers
  let norm_out ← loadGroupNorm path s!"{pfx}.norm_out" 32 128
  let conv_out ← loadConv2d path s!"{pfx}.conv_out" 128 3 3 1
  IO.println "  Loaded final layers"

  IO.println "VAE Decoder loaded successfully!"
  pure {
    post_quant_conv
    conv_in
    mid_block_1, mid_attn, mid_block_2
    up_3_block_0, up_3_block_1, up_3_block_2, up_3_upsample
    up_2_block_0, up_2_block_1, up_2_block_2, up_2_upsample
    up_1_block_0, up_1_block_1, up_1_block_2, up_1_upsample
    up_0_block_0, up_0_block_1, up_0_block_2
    norm_out
    conv_out
  }

end torch.vae
