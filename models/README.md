# Models

Three live models, plus shared Silero VAD. All weights are gitignored
and fetched on demand — see `tools/fetch-models.sh` and
`tools/fetch-vad.sh`.

## whisper-base-en (English)

- **HF source:** [`OpenVINO/whisper-base.en-int4-ov`](https://huggingface.co/OpenVINO/whisper-base.en-int4-ov)
- **Type:** OpenVINO IR (`openvino_encoder_model.xml/.bin` + decoder + tokenizer)
- **Size:** ~61M
- **Devices:** CPU, iGPU, NPU
- **Helper:** `helpers/whisper_transcribe.py`
- **Notes:** English-only. Fastest path on NPU. Drops `repetition_penalty=1.2` automatically because turbo rejects it.

## parakeet-v3 (English, streaming-capable)

- **HF source:** [`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) → converted to Q8_0 GGUF
- **Type:** GGUF (Q8_0), run via libtranscribe + ggml Vulkan
- **Size:** ~698M
- **Devices:** iGPU (Vulkan)
- **Helper:** `daemon/parakeet_stream_transcribe.cpp` (linked into `~/.local/bin/parakeet_stream_transcribe`)
- **Notes:** True streaming on the helper side. The daemon currently uses
  the one-shot path (records to WAV, then transcribes). Live-mic streaming
  integration is the next milestone.

## nepali-indicwav2vec (Nepali)

- **HF source:** [`sumanpaudel1997/nepali-asr-indicwav2vec`](https://huggingface.co/sumanpaudel1997/nepali-asr-indicwav2vec)
- **Base:** [`ai4bharat/indicwav2vec_v1_hindi`](https://huggingface.co/ai4bharat/indicwav2vec_v1_hindi) (Hindi pre-training)
- **Fine-tune:** Suman Paudel, on OpenSLR SLR54 (165h Nepali)
- **Type:** OpenVINO IR (encoder + `lm_head.npy` + `vocab.json`)
- **Size:** ~181M
- **Devices:** CPU, iGPU
- **Helper:** `helpers/indicwav2vec_transcribe.py`
- **Notes:** Real Devanagari output. `ensure_ascii=False` on the JSON
  response (do not regress — the previous bug escaped Devanagari as
  `\uXXXX`). NPU blocked by dynamic-shape ONNX; static-shape retry
  pending.

## vad (shared)

- **Source:** Silero VAD v4 ONNX (downloaded by `tools/fetch-vad.sh`)
- **Size:** ~1.8M
- **Devices:** CPU
- **Notes:** Used by the daemon for silence detection.

## PARKED

- `qwen3_transcribe.py` — kept for credit; the Qwen3 model is not in the
  current lineup and the helper is not invoked by `dictate.cpp`.
