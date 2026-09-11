#!/usr/bin/env python3
"""indicwav2vec_transcribe.py — Nepali ASR via OpenVINO IR + CTC decode.

Calls the same argv/JSON contract as whisper_transcribe.py so dictate.cpp
can dispatch on backend == "indicwav2vec" without changes to the daemon.

Args:
  --model-dir PATH        Directory containing model.xml + model.bin
                         (OpenVINO IR) and a `decoder/` subdir with
                         vocab.json + lm_head.bin (float16 LM head weights
                         in numpy format). The daemon copies these in.
  --device NAME           CPU | GPU   (NPU rejected at the daemon level;
                         dynamic-shape ONNX export won't compile on the
                         NPU VCL compiler right now)
  --audio PATH            Float32 mono 16kHz WAV

Stdout: a single JSON object {"text": "...", "load_seconds": ...,
       "transcribe_seconds": ...}
Exit:   0 on success, 1 on error. On error, stdout gets
       {"text":"", "error":"..."} and stderr gets the traceback.

Decoding: hidden @ lm_head.T -> argmax -> collapse repeats + drop blank (80).
The `|` word-boundary token (id 78) is replaced with a single space and
runs of whitespace are collapsed, so "आज|मौसम" becomes "आज मौसम" cleanly.
ensure_ascii=False everywhere so Devanagari is round-tripped byte-identical
(see devnagari-paste-bug-report.md).
"""

import argparse
import json
import os
import sys
import time
import traceback

# Suppress the chatty OpenVINO / OpenCL log noise from stderr.
os.environ.setdefault("OV_LOG_LEVEL", "WARNING")
os.environ.setdefault("OPENVINO_LOG_LEVEL", "0")
os.environ.setdefault("GLOG_minloglevel", "3")

import numpy as np
import openvino as ov


# --- model cache so repeat calls in the same recording don't re-load ---
_CACHE: dict = {}


def _load_lm_head(model_dir: str) -> np.ndarray:
    dec = os.path.join(model_dir, "decoder")
    npy = os.path.join(dec, "lm_head.npy")
    raw = os.path.join(dec, "lm_head.bin")
    if os.path.exists(npy):
        return np.load(npy)
    if os.path.exists(raw):
        return np.fromfile(raw, dtype=np.float32).reshape(81, 768)
    sys.stderr.write(f"[indicwav2vec] missing {npy} (or .bin); cannot decode\n")
    sys.exit(1)


def _load_vocab(model_dir: str) -> dict:
    p = os.path.join(model_dir, "decoder", "vocab.json")
    if not os.path.exists(p):
        sys.stderr.write(f"[indicwav2vec] missing {p}; cannot decode\n")
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_wav_mono_f32(path: str) -> np.ndarray:
    """Read a 16k mono float32 (or int16) WAV. Stdlib wave refuses IEEE_FLOAT
    (tag 3) on Python 3.12, so we parse the RIFF header manually."""
    with open(path, "rb") as f:
        head = f.read(64)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise SystemExit(f"not a RIFF/WAVE file: {path}")
    fmt_tag = bits = rate = nch = 0
    data_off = data_len = 0
    pos = 12
    with open(path, "rb") as f:
        body = f.read()
    while pos + 8 <= len(body):
        cid = body[pos:pos + 4]
        csz = int.from_bytes(body[pos + 4:pos + 8], "little")
        cbody = pos + 8
        if cid == b"fmt ":
            fmt_tag = int.from_bytes(body[cbody:cbody + 2], "little")
            nch = int.from_bytes(body[cbody + 2:cbody + 4], "little")
            rate = int.from_bytes(body[cbody + 4:cbody + 8], "little")
            bits = int.from_bytes(body[cbody + 14:cbody + 16], "little")
        elif cid == b"data":
            data_off = cbody
            data_len = csz
            break
        pos = cbody + csz + (csz & 1)
    if nch != 1:
        raise SystemExit(f"only mono supported (got nch={nch})")
    if rate != 16000:
        raise SystemExit(f"need 16kHz (got {rate})")
    blob = body[data_off:data_off + data_len]
    if fmt_tag == 3 and bits == 32:
        return np.frombuffer(blob, dtype="<f4").astype(np.float32)
    if fmt_tag == 1 and bits == 16:
        return np.frombuffer(blob, dtype="<i2").astype(np.float32) / 32768.0
    raise SystemExit(f"unsupported fmt_tag={fmt_tag} bits={bits}")


def _get_compiled(model_dir: str, device: str):
    key = (model_dir, device)
    if key in _CACHE:
        return _CACHE[key]
    core = ov.Core()
    ir_xml = os.path.join(model_dir, "model.xml")
    if not os.path.exists(ir_xml):
        raise SystemExit(f"missing OV IR: {ir_xml}")
    comp = core.compile_model(ir_xml, device)
    _CACHE[key] = comp
    return comp


def _ctc_decode(ids, vocab: dict) -> str:
    """Greedy CTC decode + post-process: blank=80, word-boundary `|` (78) -> space."""
    inv = {v: k for k, v in vocab.items()}
    blank_id = 80
    word_boundary = 78  # `|` in the model's vocab; replace with a space
    out_chars = []
    prev = -1
    for i in ids:
        if i == prev or i == blank_id:
            prev = i
            continue
        tok = inv.get(int(i), "")
        if i == word_boundary:
            tok = " "
        out_chars.append(tok)
        prev = i
    raw = "".join(out_chars)
    # collapse runs of whitespace (the `|` boundaries can double up) and trim
    return " ".join(raw.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="ignored — CTC has no generator step")
    ap.add_argument("--language", default="",
                    help="ignored — indicwav2vec is Nepali-only")
    args = ap.parse_args()

    try:
        t0 = time.time()
        comp = _get_compiled(args.model_dir, args.device)
        lm_head = _load_lm_head(args.model_dir)
        vocab = _load_vocab(args.model_dir)
        load_seconds = time.time() - t0

        t0 = time.time()
        pcm = _load_wav_mono_f32(args.audio)
        # model expects [1, T] float32 + [1, T] int64 attention mask (all-1s
        # because we don't pad — VAD/segmentation is the daemon's job)
        input_values = pcm[np.newaxis, :].astype(np.float32)
        attn_mask = np.ones_like(input_values, dtype=np.int64)
        hidden = comp([input_values, attn_mask])[0][0]
        # hidden: [T, 768]   lm_head: [81, 768]
        logits = hidden @ lm_head.T
        ids = logits.argmax(axis=-1).tolist()
        text = _ctc_decode(ids, vocab)
        transcribe_seconds = time.time() - t0

        out = {"text": text,
               "load_seconds": round(load_seconds, 3),
               "transcribe_seconds": round(transcribe_seconds, 3)}
        # ensure_ascii=False so Devanagari is round-tripped byte-identical
        sys.stdout.write(json.dumps(out, ensure_ascii=False))
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        err = {"text": "", "error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(err, ensure_ascii=False))
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
