#!/usr/bin/env python3
"""
Synthesize Kokoro / KittenTTS audio from converted safetensors assets.

This is a debugging/reference path: it uses the upstream Python Kokoro modules,
but loads weights from the same `model.safetensors` and `voice.safetensors`
artifacts that the Lean port consumes.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from array import array
from pathlib import Path


def _resolve_device(device: str) -> str:
    import torch

    requested = device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
    if requested not in {"cpu", "cuda", "mps"}:
        raise RuntimeError(f"Unsupported device '{device}'")
    return requested


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_model(config: dict, repo_id: str, disable_complex: bool):
    import torch
    from transformers import AlbertConfig
    from kokoro.istftnet import Decoder
    from kokoro.model import KModel
    from kokoro.modules import CustomAlbert, ProsodyPredictor, TextEncoder

    model = KModel.__new__(KModel)
    torch.nn.Module.__init__(model)
    model.repo_id = repo_id
    model.vocab = config["vocab"]
    model.bert = CustomAlbert(AlbertConfig(vocab_size=config["n_token"], **config["plbert"]))
    model.bert_encoder = torch.nn.Linear(model.bert.config.hidden_size, config["hidden_dim"])
    model.context_length = model.bert.config.max_position_embeddings
    model.predictor = ProsodyPredictor(
        style_dim=config["style_dim"],
        d_hid=config["hidden_dim"],
        nlayers=config["n_layer"],
        max_dur=config["max_dur"],
        dropout=config["dropout"],
    )
    model.text_encoder = TextEncoder(
        channels=config["hidden_dim"],
        kernel_size=config["text_encoder_kernel_size"],
        depth=config["n_layer"],
        n_symbols=config["n_token"],
    )
    model.decoder = Decoder(
        dim_in=config["hidden_dim"],
        style_dim=config["style_dim"],
        dim_out=config["n_mels"],
        disable_complex=disable_complex,
        **config["istftnet"],
    )
    return model


def _load_grouped_state_dict(model_path: Path) -> dict[str, dict[str, object]]:
    from safetensors import safe_open

    groups: dict[str, dict[str, object]] = {}
    with safe_open(str(model_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            top, rest = key.split(".", 1)
            groups.setdefault(top, {})[rest] = f.get_tensor(key).contiguous()
    return groups


def _load_voice_style(voice_path: Path, phoneme_count: int):
    from safetensors import safe_open

    voice_index = 0 if phoneme_count <= 0 else min(509, phoneme_count - 1)
    with safe_open(str(voice_path), framework="pt", device="cpu") as f:
        table = f.get_tensor("voice")
    ref_s = table[voice_index]
    return ref_s, voice_index


def _save_wav(path: Path, audio, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = array("h")
    for x in audio:
        xf = float(x)
        if xf > 1.0:
            xf = 1.0
        elif xf < -1.0:
            xf = -1.0
        pcm.append(int(xf * 32767.0))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _is_allowed_missing_key(key: str) -> bool:
    return key.endswith(".norm.weight") or key.endswith(".norm.bias")


def _tensor_stats(tensor) -> dict[str, object]:
    flat = tensor.detach().cpu().flatten().float()
    if flat.numel() == 0:
        return {
            "shape": list(tensor.shape),
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
        }
    return {
        "shape": list(tensor.shape),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--phonemes", required=True)
    parser.add_argument("--output-wav", required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repo-id", default="hexgrad/Kokoro-82M")
    parser.add_argument("--disable-complex", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn.functional as F
        from loguru import logger
    except Exception as exc:
        raise RuntimeError("Missing Python dependencies for Kitten reference bridge") from exc

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    config_path = Path(args.config).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    voice_path = Path(args.voice).expanduser().resolve()
    output_wav = Path(args.output_wav).expanduser().resolve()

    config = _load_json(config_path)
    model = _build_model(config, repo_id=args.repo_id, disable_complex=args.disable_complex)
    grouped = _load_grouped_state_dict(model_path)

    required_groups = ("bert", "bert_encoder", "predictor", "text_encoder", "decoder")
    for group in required_groups:
        if group not in grouped:
            raise RuntimeError(f"Missing checkpoint group '{group}' in {model_path}")

    for group in required_groups:
        target = getattr(model, group)
        missing, unexpected = target.load_state_dict(grouped[group], strict=False)
        bad_missing = [key for key in missing if not _is_allowed_missing_key(key)]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"State-dict mismatch for {group}: missing={bad_missing}, unexpected={unexpected}"
            )

    device = _resolve_device(args.device)
    model = model.to(device).eval()

    ref_s, voice_index = _load_voice_style(voice_path, len(args.phonemes))
    ref_s = ref_s.to(device)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    vocab = model.vocab
    input_ids = [0] + [vocab[p] for p in args.phonemes if p in vocab] + [0]
    input_ids_t = torch.LongTensor([input_ids]).to(model.device)
    input_lengths = torch.full(
        (input_ids_t.shape[0],),
        input_ids_t.shape[-1],
        device=input_ids_t.device,
        dtype=torch.long,
    )
    text_mask = torch.arange(input_lengths.max(), device=input_ids_t.device).unsqueeze(0).expand(
        input_lengths.shape[0], -1
    )
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(model.device)

    with torch.no_grad():
        bert_dur = model.bert(input_ids_t, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        predictor_style = ref_s[:, 128:]
        dur_enc = model.predictor.text_encoder(d_en, predictor_style, input_lengths, text_mask)
        x, _ = model.predictor.lstm(dur_enc)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / args.speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids_t.shape[1], device=model.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids_t.shape[1], indices.shape[0]), device=model.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0], device=model.device)] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)
        prosody = dur_enc.transpose(-1, -2) @ pred_aln_trg
        f0_pred, n_pred = model.predictor.F0Ntrain(prosody, predictor_style)
        text_embed = model.text_encoder.embedding(input_ids_t)
        text_x = text_embed.transpose(1, 2)
        text_mask_1 = text_mask.unsqueeze(1)
        text_x.masked_fill_(text_mask_1, 0.0)
        conv_block_stats = []
        conv_param_stats = []
        for block in model.text_encoder.cnn:
            conv_param_stats.append(
                {
                    "weight": _tensor_stats(block[0].weight),
                    "ln_weight": _tensor_stats(block[1].gamma),
                    "ln_bias": _tensor_stats(block[1].beta),
                }
            )
            text_x = block(text_x)
            text_x.masked_fill_(text_mask_1, 0.0)
            conv_block_stats.append(_tensor_stats(text_x))
        text_conv = text_x
        text_enc = model.text_encoder(input_ids_t, input_lengths, text_mask)
        asr = text_enc @ pred_aln_trg
        decoder_style = ref_s[:, :128]
        decoder_f0 = model.decoder.F0_conv(f0_pred.unsqueeze(1))
        decoder_n = model.decoder.N_conv(n_pred.unsqueeze(1))
        decoder_x = torch.cat([asr, decoder_f0, decoder_n], axis=1)
        decoder_encode = model.decoder.encode(decoder_x, decoder_style)
        decoder_asr_res = model.decoder.asr_res(asr)
        decoder_decode_stats = []
        decoder_x = decoder_encode
        decoder_res = True
        for block in model.decoder.decode:
            if decoder_res:
                decoder_x = torch.cat([decoder_x, decoder_asr_res, decoder_f0, decoder_n], axis=1)
            decoder_x = block(decoder_x, decoder_style)
            decoder_decode_stats.append(_tensor_stats(decoder_x))
            if block.upsample_type != "none":
                decoder_res = False

        generator = model.decoder.generator
        f0_up = generator.f0_upsamp(f0_pred[:, None]).transpose(1, 2)
        har_source, _, _ = generator.m_source(f0_up)
        har_source_1d = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = generator.stft.transform(har_source_1d)
        har = torch.cat([har_spec, har_phase], dim=1)

        generator_stage_stats = []
        generator_x = decoder_x
        for i in range(generator.num_upsamples):
            generator_x_act = F.leaky_relu(generator_x, negative_slope=0.1)
            generator_x_source = generator.noise_convs[i](har)
            generator_x_source = generator.noise_res[i](generator_x_source, decoder_style)
            generator_x_up = generator.ups[i](generator_x_act)
            if i == generator.num_upsamples - 1:
                generator_x_up = generator.reflection_pad(generator_x_up)
            generator_x_mix = generator_x_up + generator_x_source
            generator_xs = None
            for j in range(generator.num_kernels):
                block = generator.resblocks[i * generator.num_kernels + j]
                if generator_xs is None:
                    generator_xs = block(generator_x_mix, decoder_style)
                else:
                    generator_xs = generator_xs + block(generator_x_mix, decoder_style)
            generator_x = generator_xs / generator.num_kernels
            generator_stage_stats.append(
                {
                    "x_source": _tensor_stats(generator_x_source),
                    "x_up": _tensor_stats(generator_x_up),
                    "x_mix": _tensor_stats(generator_x_mix),
                    "x_out": _tensor_stats(generator_x),
                }
            )

        generator_post = generator.conv_post(F.leaky_relu(generator_x, negative_slope=0.01))
        freq_bins = generator.post_n_fft // 2 + 1
        generator_spec_log = generator_post[:, :freq_bins, :]
        generator_phase_raw = generator_post[:, freq_bins:, :]
        generator_spec = torch.exp(generator_spec_log)
        generator_phase = torch.sin(generator_phase_raw)
        audio = generator.stft.inverse(generator_spec, generator_phase).squeeze().detach().cpu().flatten()
        pred_dur = pred_dur.detach().cpu().flatten()

    sample_rate = int(config.get("sample_rate", args.sample_rate))
    _save_wav(output_wav, audio.tolist(), sample_rate)

    summary = {
        "audio_shape": list(audio.shape),
        "audio_num_samples": int(audio.numel()),
        "pred_durations": [int(x) for x in pred_dur.tolist()],
        "sample_rate": sample_rate,
        "voice_index": int(voice_index),
        "embedding_stats": _tensor_stats(text_embed),
        "conv_param_stats": conv_param_stats,
        "conv_block_stats": conv_block_stats,
        "text_conv_stats": _tensor_stats(text_conv),
        "text_enc_stats": _tensor_stats(text_enc),
        "asr_stats": _tensor_stats(asr),
        "f0_stats": _tensor_stats(f0_pred),
        "n_stats": _tensor_stats(n_pred),
        "decoder_f0_stats": _tensor_stats(decoder_f0),
        "decoder_n_stats": _tensor_stats(decoder_n),
        "decoder_encode_stats": _tensor_stats(decoder_encode),
        "decoder_asr_res_stats": _tensor_stats(decoder_asr_res),
        "decoder_decode_stats": decoder_decode_stats,
        "generator_har_source_stats": _tensor_stats(har_source),
        "generator_har_stats": _tensor_stats(har),
        "generator_stage_stats": generator_stage_stats,
        "generator_post_stats": _tensor_stats(generator_post),
        "generator_spec_log_stats": _tensor_stats(generator_spec_log),
        "generator_phase_raw_stats": _tensor_stats(generator_phase_raw),
        "generator_spec_stats": _tensor_stats(generator_spec),
        "audio_stats": _tensor_stats(audio),
    }
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
