# o4-dictate-local

Local speech-to-text on Omarchy Linux. Quickshell bar panel (`plugin/`) + C++ daemon (`daemon/dictate.cpp`) + Python helpers (`helpers/`). Push-to-talk (Copilot key); transcribe on CPU/GPU/NPU; paste into focused field.

## Structure (docs/ excluded)
- `plugin/` — `manifest.json`, `Panel.qml`, `Model.js`, `assets/` (icons/logos). Bar-widget: on/off, device/model/offload.
- `daemon/` — `dictate.cpp` (OpenVINO, VAD, load-on-demand), `feats.h`, `build.sh`, `install.sh`, `ref_*.py`.
- `helpers/` — `indicwav2vec_transcribe.py` (Nepali ASR), `qwen3_transcribe.py`, `whisper_transcribe.py`.
- `models/` — `nepali-indicwav2vec/`, `parakeet-v3/`, `vad/`, `whisper-base-en/` (metadata; weights fetched separately).
- `systemd/` — `omarchy-npu-dictate.service`.
- `tools/` — `fetch-vad.sh`, `nepali-convert/` (audio excluded).
- `README.md`, `LICENSE`.

## Models
- Whisper Base — English (CPU/GPU/NPU)
- Parakeet V3 — English (iGPU, GGUF)
- Nepali ASR — Nepali (`indicwav2vec`, CPU/GPU)

## Install / Use
`./daemon/install.sh` → daemon + systemd + plugin. `tools/fetch-models.sh` pulls weights. Copilot key toggles record/transcribe.
