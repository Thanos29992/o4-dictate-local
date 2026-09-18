# MODELS_NEPALI — shortlist + bench

The Nepali transcription model in the live lineup is
[`sumanpaudel1997/nepali-asr-indicwav2vec`](https://huggingface.co/sumanpaudel1997/nepali-asr-indicwav2vec),
which is itself a fine-tune of AI4Bharat's
[`indicwav2vec_v1_hindi`](https://huggingface.co/ai4bharat/indicwav2vec_v1_hindi)
on OpenSLR SLR54 (165h Nepali).

## Why this one

Shortlist evaluated 2026-09-05, ranked by WER on OpenSLR SLR54 (the
training distribution, so lower-bound):

| Model | Size | WER (SLR54) | WER (FLEURS) | WER (CV25) | Verdict |
|---|---|---|---|---|---|
| **`sumanpaudel1997/nepali-asr-indicwav2vec`** | 378M | **14.89%** | 40.68% | 51.65% | **Picked** |
| ai4bharat/indicconformer | 1.1G | — | — | — | Heavy, no NPU path |
| kriti-asr/nepali-asr | ~1.2G | — | — | — | Same |
| OpenSLR/GAM30-Nepali | small | weak | — | — | Not competitive |

Compared to Whisper Large V3 Turbo INT4 (the prior leader, 25% WER on a
real clip) — indicwav2vec wins on Devanagari correctness.

## How the live IR was built

The conversion scripts are in `tools/nepali-convert/`:

```
HF model (safetensors) ──► ONNX (dynamic) ──► OV IR (encoder + lm_head + vocab)
                                            └─ decoder/lm_head.npy
                                            └─ vocab.json
```

Run `python3 tools/nepali-convert/convert_onnx_then_ov.py` to reproduce.
The output `models/nepali-indicwav2vec_ov/` is gitignored.

Smoke test audio: `tools/nepali-convert/audio/nep_trim1.wav` (500K) and
`tools/nepali-convert/audio/nepali_codex.wav` (975K).

## Decode notes (the bug that bit us)

The CTC decode needs to:
1. Collapse repeated tokens.
2. Drop the blank token (id 80).
3. Treat the `|` (id 78) token as a word boundary — but emit a SPACE,
   not the character.

Previous bug: `|` was being emitted as a literal pipe character. The
helper strips it. Don't regress.

## Verdict

`nepali-indicwav2vec` is the only sub-500MB Nepali model that:
- Transcribes real-world Nepali audio correctly.
- Runs on iGPU via OpenVINO.
- Ships a working Python helper we can integrate.

NPU is blocked by dynamic-shape ONNX. The static-shape retry is in
`docs/HISTORY_HYBRID_DECODER.md`.
