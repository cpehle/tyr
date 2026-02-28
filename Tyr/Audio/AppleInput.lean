namespace Tyr.Audio.AppleInput

@[extern "lean_tyr_audio_input_start"]
opaque start (sampleRate : UInt64 := 16000) (channels : UInt64 := 1) (bufferMs : UInt64 := 100) : IO Unit

@[extern "lean_tyr_audio_input_read"]
opaque read (maxSamples : UInt64) (blockMs : UInt64 := 250) : IO (Array Float)

@[extern "lean_tyr_audio_input_stop"]
opaque stop : IO Unit

end Tyr.Audio.AppleInput
