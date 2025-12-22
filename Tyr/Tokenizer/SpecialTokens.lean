/-
  Special Tokens

  NanoProof special tokens for theorem proving.
-/
import Tyr.Tokenizer.Types

namespace tokenizer

/-- Core special tokens -/
def coreSpecialTokens : Array String :=
  #[ "<|pad|>"
   , "<|eos|>"
   , "<|bos|>"
   , "<|tactic|>"
   , "<|value|>"
   , "<|goal|>"
   , "<|proof|>"
   , "<|state|>"
   ]

/-- Value bin tokens (for value head) -/
def valueBinTokens : Array String := Id.run do
  let mut tokens := Array.mkEmpty 64
  for i in [:64] do
    let padded := if i < 10 then s!"0{i}" else s!"{i}"
    tokens := tokens.push s!"<|bin_{padded}|>"
  return tokens

/-- Greek letters for mathematical notation -/
def greekLetters : Array String :=
  #[ "α", "β", "γ", "δ", "ε", "ζ", "η", "θ"
   , "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π"
   , "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω"
   , "Α", "Β", "Γ", "Δ", "Ε", "Ζ", "Η", "Θ"
   , "Ι", "Κ", "Λ", "Μ", "Ν", "Ξ", "Ο", "Π"
   , "Ρ", "Σ", "Τ", "Υ", "Φ", "Χ", "Ψ", "Ω"
   ]

/-- Mathematical operators -/
def mathOperators : Array String :=
  #[ "∀", "∃", "∄", "∈", "∉", "⊂", "⊃", "⊆", "⊇"
   , "∪", "∩", "∅", "∧", "∨", "¬", "⊕", "⊗", "⊙"
   , "→", "←", "↔", "⇒", "⇐", "⇔", "↦", "↣"
   , "≡", "≠", "≤", "≥", "≪", "≫", "≈", "≅", "∼"
   , "∞", "∂", "∇", "∫", "∑", "∏", "√", "∛"
   , "⊢", "⊣", "⊤", "⊥", "⊦", "⊧"
   , "⟨", "⟩", "⟪", "⟫", "⌊", "⌋", "⌈", "⌉"
   , "∘", "⋅", "×", "÷", "±", "∓", "′", "″"
   ]

/-- Lean-specific tokens -/
def leanTokens : Array String :=
  #[ "theorem", "lemma", "def", "structure", "class", "instance"
   , "inductive", "where", "with", "match", "by", "exact"
   , "apply", "intro", "intros", "refl", "rfl", "simp"
   , "rw", "rewrite", "have", "let", "show", "calc"
   , "sorry", "admit", "axiom", "variable", "namespace"
   , "open", "section", "end", "import", "export"
   , "#check", "#eval", "#print", "#reduce"
   , ":=", "=>", "->", "<-", "∘", "∘"
   , "Prop", "Type", "Sort", "Nat", "Int", "Bool"
   , "True", "False", "And", "Or", "Not", "Iff"
   ]

/-- All special tokens combined -/
def allSpecialTokens : Array String :=
  coreSpecialTokens ++ valueBinTokens ++ greekLetters ++ mathOperators ++ leanTokens

/-- Standard token IDs for core tokens -/
def padTokenId : TokenId := 0
def eosTokenId : TokenId := 1
def bosTokenId : TokenId := 2

/-- Add special tokens to a tokenizer -/
def addSpecialTokens (tok : BPETokenizer) (tokens : Array String) (startId : TokenId)
    : BPETokenizer := Id.run do
  let mut result := tok
  let mut id := startId
  for token in tokens do
    result := { result with
      specialTokens := result.specialTokens.insert token id
      idToSpecial := result.idToSpecial.insert id token
    }
    id := id + 1
  return result

end tokenizer
