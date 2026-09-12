# MODELS

All three live models, with HF sources, sizes, devices, and a link to
the conversion/build notes for each.

| Name | Language | Size | Devices | Type | Source |
|---|---|---|---|---|---|
| `whisper-base-en` | English | 61M | CPU, iGPU, NPU | OpenVINO IR | [OpenVINO/whisper-base.en-int4-ov](https://huggingface.co/OpenVINO/whisper-base.en-int4-ov) |
| `parakeet-v3` | English | 698M | iGPU (Vulkan) | GGUF (Q8_0) | [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) → Q8_0 GGUF via [Handy](https://github.com/cjpais/Handy) |
| `nepali-indicwav2vec` | Nepali | 181M | CPU, iGPU | OpenVINO IR | [sumanpaudel1997/nepali-asr-indicwav2vec](https://huggingface.co/sumanpaudel1997/nepali-asr-indicwav2vec) |
| `vad` (shared) | — | 1.8M | CPU | ONNX | [Silero VAD v4](https://github.com/snakers4/silero-vad) |

See `MODELS_NEPALI.md` for the Nepali shortlist + bench.

## Why each is here

**Whisper Base** is the safe English default. Works on NPU (fast),
iGPU (fast), and CPU (acceptable). Multilingual base lost to indicwav2vec
on Nepali and to Parakeet on English, so it stays English-only.

**Parakeet V3 Streaming** is the streaming-capable iGPU English model.
Currently invoked in one-shot mode by the daemon. Live-mic streaming
is the next milestone.

**Nepali ASR** is the only model that ships real Devanagari output
without the whisper-base-multilingual round-trip.

## What was dropped

- Whisper Large V3 Turbo (456M, multilingual) — lost to indicwav2vec
  on a real Nepali clip. Replaced.
- Whisper Base Nepali (alphaedge trim) — 0–4% word accuracy on the same
  clip. Removed.
- Qwen3-ASR-0.6B — never wired into the daemon. Helper kept under
  `helpers/` for credit.
- Whisper Tiny, Small — too inaccurate to ship.
