#!/usr/bin/env python3
"""qwen3_transcribe.py — Qwen3 ASR inference via direct OpenVINO models.

Qwen3 ASR (Qwen3ASRForConditionalGeneration) uses a multimodal architecture:
  1. Audio encoder: mel spectrogram → audio features [1, T, 1024] (includes
     the multi-modal projector that maps audio hidden state to text dimension)
  2. Thinker embeddings: input_ids → token embeddings [1, N, 1024]
  3. Decoder: text token embeddings with audio features spliced into the
     audio_token positions → logits [1, N, vocab_size]

The audio features replace AUDIO_TOKEN_ID (151676) positions in the embedding
sequence, using a masked_scatter equivalent in numpy.

Args:
  --model-dir PATH        Qwen3 ASR model directory (contains IR files)
  --device NAME           CPU / GPU / NPU
  --audio PATH            Float32 mono 16kHz WAV (or int16)

Stdout: a single JSON object {"text": "...", "load_seconds": ..., "transcribe_seconds": ...}
Exit: 0 on success, 1 on error.
"""
import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("OPENVINO_LOG_LEVEL", "ERROR")
os.environ.setdefault("OV_GPU_HELPERS", "0")
os.environ.setdefault("HF_HOME", "/home/shlok/.cache/huggingface-npu")

# Qwen3 ASR special token IDs
AUDIO_START_TOKEN_ID = 151669
AUDIO_END_TOKEN_ID = 151670
AUDIO_TOKEN_ID = 151676
EOC_TOKEN_ID = 151643

# Mel constants (from thinker_config audio_config)
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 128
MEL_NORM_MEAN = -4.6667
MEL_NORM_STD = 3.3165
ENCODER_FRAME_STRIDE = 100  # encoder requires mel frames divisible by 100

# Models cache
_models = {}


def read_f32_wav(path: str):
    """Read WAV file, return (float32 samples, sample_rate)."""
    import struct, array, numpy as np
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file")
    p = 12
    fmt = None
    audio = None
    while p < len(data) - 8:
        ck_id = data[p:p+4]
        p += 4
        ck_sz = struct.unpack("<I", data[p:p+4])[0]
        p += 4
        if ck_id == b"fmt ":
            audio_fmt, ch, sr, br, ba, sw = struct.unpack("<HHIIHH", data[p:p+16])
            fmt = dict(fmt=audio_fmt, ch=ch, sr=sr, sw=sw)
            p += ck_sz
        elif ck_id == b"data":
            audio = data[p:p+ck_sz]
            p += ck_sz
        else:
            p += ck_sz
    if not (fmt and audio):
        raise ValueError(f"{path} missing fmt/data chunks")

    if fmt["fmt"] == 3 and fmt["sw"] == 32:
        samples = np.frombuffer(audio, dtype=np.float32)
    elif fmt["fmt"] == 1 and fmt["sw"] == 16:
        raw = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        samples = raw
    else:
        raw = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        samples = raw
    return samples, fmt["sr"]


def compute_mel(wav, sr: int, n_frames_target: int = None):
    """Compute log-mel spectrogram from audio waveform."""
    import numpy as np
    # Resample if needed
    if sr != 16000:
        # Simple linear interpolation resample
        n_orig = len(wav)
        n_new = int(n_orig * 16000 / sr)
        indices = np.linspace(0, n_orig - 1, n_new)
        wav = np.interp(indices, np.arange(n_orig), wav).astype(np.float32)
        sr = 16000

    window = np.hamming(WIN_LENGTH)
    pad = N_FFT // 2
    wav = np.concatenate([np.zeros(pad, dtype=np.float32), wav])
    n_frames = 1 + (len(wav) - WIN_LENGTH) // HOP_LENGTH

    # STFT
    frames = np.array([wav[i * HOP_LENGTH:i * HOP_LENGTH + WIN_LENGTH] for i in range(n_frames)])
    frames *= window
    spec = np.abs(np.fft.rfft(frames, n=N_FFT))[:, :N_FFT // 2 + 1]

    # Mel filterbank
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    mel_lo = hz_to_mel(0.0)
    mel_hi = hz_to_mel(8000.0)
    mel_pts = np.linspace(mel_lo, mel_hi, N_MELS + 2)
    hz_pts = 700.0 * (10 ** (mel_pts / 2595.0) - 1.0)
    bins = np.floor((N_FFT + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1))
    for m in range(1, N_MELS + 1):
        fl, fc, fr = bins[m - 1], bins[m], bins[m + 1]
        for k in range(fl, fc):
            fb[m - 1, k] = (k - fl) / (fc - fl)
        for k in range(fc, fr):
            fb[m - 1, k] = (fr - k) / (fr - fc)
    mel_spec = spec @ fb.T  # [n_frames, n_mels]
    mel_spec = np.log(np.maximum(mel_spec, 1e-6))
    mel_spec = (mel_spec - MEL_NORM_MEAN) / MEL_NORM_STD
    mel_spec = mel_spec.T  # [n_mels, n_frames]

    # Pad to multiple of ENCODER_FRAME_STRIDE
    n_pad = (ENCODER_FRAME_STRIDE - n_frames % ENCODER_FRAME_STRIDE) % ENCODER_FRAME_STRIDE
    if n_pad > 0:
        mel_padded = np.zeros((N_MELS, n_frames + n_pad), dtype=np.float32)
        mel_padded[:, :n_frames] = mel_spec
        mel_spec = mel_padded
        n_frames_padded = n_frames + n_pad
    else:
        n_frames_padded = n_frames

    return mel_spec, n_frames, n_frames_padded


def get_models(model_dir: str, device: str):
    """Load and compile Qwen3 ASR sub-models."""
    key = (model_dir, device)
    if key in _models:
        return _models[key]

    import openvino as ov
    core = ov.Core()

    # Audio encoder (mel → audio features, includes multi-modal projector)
    enc_xml = os.path.join(model_dir, "audio_encoder_model.xml")
    enc = core.compile_model(core.read_model(enc_xml), device)

    # Thinker embeddings (input_ids → embeddings)
    thinker_xml = os.path.join(model_dir, "thinker_embeddings_model.xml")
    thinker = core.compile_model(core.read_model(thinker_xml), device)

    # Text decoder (embeddings + position_ids → logits)
    dec_xml = os.path.join(model_dir, "decoder_model.xml")
    dec = core.compile_model(core.read_model(dec_xml), device)

    _models[key] = (enc, thinker, dec)
    return _models[key]


def load_vocab(model_dir: str):
    """Load tokenizer vocabulary."""
    with open(os.path.join(model_dir, "vocab.json")) as f:
        vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}

    # Load added_tokens.json for special tokens
    added_path = os.path.join(model_dir, "added_tokens.json")
    if os.path.exists(added_path):
        with open(added_path) as f:
            added = json.load(f)
        for tok_str, tok_id in added.items():
            id_to_token[tok_id] = tok_str

    return id_to_token


def transcribe(model_dir: str, device: str, audio_path: str) -> str:
    """Run Qwen3 ASR transcription."""
    import numpy as np

    enc, thinker, dec = get_models(model_dir, device)
    id_to_token = load_vocab(model_dir)

    # Read audio
    wav, sr = read_f32_wav(audio_path)

    # Compute mel
    mel, n_frames, n_frames_padded = compute_mel(wav, sr)
    print(f"[qwen3] mel: {n_frames} frames (padded to {n_frames_padded})", file=sys.stderr)

    # Run audio encoder
    mel_input = mel.reshape(1, N_MELS, n_frames_padded).astype(np.float32)
    audio_res = enc([mel_input])
    audio_feats = audio_res[0]  # [1, T_enc, 1024]
    n_audio_tokens = audio_feats.shape[1]
    print(f"[qwen3] audio features: {audio_feats.shape}, audio tokens: {n_audio_tokens}", file=sys.stderr)

    # Build the input_ids sequence:
    # [AUDIO_START, AUDIO_TOKEN × n_audio_tokens, AUDIO_END]
    # The AUDIO_TOKEN positions will be replaced with audio features
    input_ids = [AUDIO_START_TOKEN_ID] + [AUDIO_TOKEN_ID] * n_audio_tokens + [AUDIO_END_TOKEN_ID]
    print(f"[qwen3] input sequence: {len(input_ids)} tokens", file=sys.stderr)

    # Get text embeddings for all tokens
    input_ids_arr = np.array([input_ids], dtype=np.int64)
    embeddings = thinker([input_ids_arr])[0]  # [1, seq_len, 1024]
    print(f"[qwen3] embeddings: {embeddings.shape}", file=sys.stderr)

    # Replace AUDIO_TOKEN positions with audio features (masked_scatter equivalent)
    audio_mask = (input_ids_arr[0] == AUDIO_TOKEN_ID)
    n_tokens_check = audio_mask.sum()
    audio_feats_flat = audio_feats.reshape(-1, 1024)
    assert n_tokens_check == audio_feats_flat.shape[0], \
        f"Audio tokens ({n_tokens_check}) != audio features ({audio_feats_flat.shape[0]})"
    embeddings[0][audio_mask] = audio_feats_flat
    print(f"[qwen3] spliced audio features into {n_tokens_check} positions", file=sys.stderr)

    # Greedy generation
    max_gen = 256
    generated_ids = []
    eos_reached = False

    for step in range(max_gen):
        # Position IDs: 0, 1, 2, ... for the full sequence
        positions = np.arange(len(input_ids), dtype=np.int64).reshape(1, -1)

        # Run decoder — only need the last position's logits
        logits = dec([embeddings, positions])[0]  # [1, seq_len, vocab_size]
        row = logits[0, -1, :]  # last position logits

        next_id = int(np.argmax(row))

        if step < 10:
            tok = id_to_token.get(next_id, f'<{next_id}>')
            print(f"[qwen3] t={step} token={next_id} ({tok[:40]}) logit={row[next_id]:.3f}", file=sys.stderr)

        # Check for end
        if next_id == EOC_TOKEN_ID:
            eos_reached = True
            break

        generated_ids.append(next_id)

        # Append new token embedding to the sequence
        new_emb = thinker([np.array([[next_id]], dtype=np.int64)])[0]  # [1, 1, 1024]
        embeddings = np.concatenate([embeddings, new_emb], axis=1)
        input_ids.append(next_id)

        # Early termination if generating too many blanks
        if len(generated_ids) > 50 and generated_ids[-50:].count(next_id) == 50:
            print(f"[qwen3] stuck on token {next_id}, stopping", file=sys.stderr)
            break

    print(f"[qwen3] generated {len(generated_ids)} tokens, eos_reached={eos_reached}", file=sys.stderr)

    # Decode tokens to text
    text_parts = []
    for tid in generated_ids:
        tok = id_to_token.get(tid, '')
        if tok and not tok.startswith('<') and tok not in ('', '[UNK]'):
            # Handle byte-level BPE: tokens with ▁ are word starts
            text_parts.append(tok)
        elif tok.startswith('<') and tok.endswith('>'):
            # Special token, skip
            continue
        else:
            text_parts.append(tok)

    # Qwen3 uses a BPE tokenizer with byte fallback
    # Tokens might be byte sequences or unicode chars
    raw = ''.join(text_parts)

    # Handle byte-level tokens (like the Qwen3 BPE)
    # Qwen3 BPE uses ▁ for word boundaries
    raw = raw.replace('▁', ' ')

    # Decode any remaining byte sequences
    try:
        if 'Ã' in raw or '\\x' in raw:
            # Try utf-8 decode
            raw_bytes = raw.encode('latin-1')
            raw = raw_bytes.decode('utf-8', errors='replace')
    except:
        pass

    return raw.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--audio", required=True)
    args = ap.parse_args()

    try:
        t0 = time.time()
        text = transcribe(args.model_dir, args.device, args.audio)
        total_dt = time.time() - t0

        out = {
            "text": text,
            "load_seconds": round(total_dt, 3),
            "transcribe_seconds": round(total_dt, 3),
            "audio_seconds": 0,
            "device": args.device,
            "model": os.path.basename(args.model_dir),
        }
        sys.stdout.write(json.dumps(out, ensure_ascii=False))
        sys.stdout.write("\n")
        sys.stdout.flush()
        return 0
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        out = {"text": "", "error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(out, ensure_ascii=False))
        sys.stdout.write("\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
