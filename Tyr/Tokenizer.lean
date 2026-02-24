/-
  Tokenizer

  BPE tokenizer for NanoProof theorem prover.
-/
import Tyr.Tokenizer.Types
import Tyr.Tokenizer.ByteLevel
import Tyr.Tokenizer.Pretokenize
import Tyr.Tokenizer.Encode
import Tyr.Tokenizer.Decode
import Tyr.Tokenizer.SpecialTokens
import Tyr.Tokenizer.TokenBytes
import Tyr.Tokenizer.Training
import Tyr.Tokenizer.IO
import Tyr.Tokenizer.Qwen3

/-!
# `Tyr.Tokenizer`

Tokenizer entrypoint that re-exports byte-level processing, encoding/decoding, and training utilities.

## Overview
- Part of the core `Tyr` library surface.
- Uses markdown module docs so `doc-gen4` renders a readable module landing section.
-/

