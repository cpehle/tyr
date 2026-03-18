import Tyr.Audio.FloatBuffer

namespace Tyr.Audio.AppleInput

@[extern "lean_tyr_audio_input_start"]
opaque start (sampleRate : UInt64 := 16000) (channels : UInt64 := 1) (bufferMs : UInt64 := 100) : IO Unit

@[extern "lean_tyr_audio_input_read"]
opaque read (maxSamples : UInt64) (blockMs : UInt64 := 250) : IO (Array Float)

/-- Read audio samples into an unboxed FloatBuffer — no per-element boxing. -/
@[extern "lean_tyr_audio_input_read_buffer"]
opaque readBuffer (maxSamples : UInt64) (blockMs : UInt64 := 250) : IO FloatBuffer

@[extern "lean_tyr_audio_input_stop"]
opaque stop : IO Unit

/-- Compute RMS of a boxed float audio buffer via native loop. -/
@[extern "lean_tyr_audio_rms"]
opaque rms (xs : @& Array Float) : IO Float

end Tyr.Audio.AppleInput
