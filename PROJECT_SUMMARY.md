# Project Tree & Group Summary

## Tree
```
LICENSE
README.md / README.full.md
daemon/ (dictate.cpp, feats.h, build.sh, install.sh, ref_*.py)
docs/ (MODELS.md, NEPALI_LAB_README.md, PLAN.md, TALLY.md, etc.)
helpers/ (transcribe scripts)
models/ (nepali-indicwav2vec, parakeet-v3, vad, whisper-base-en)
plugin/ (manifest.json, Model.js, Panel.qml, assets/)
systemd/ (omarchy-npu-dictate.service)
tools/ (fetch-vad.sh, nepali-convert/)
```

## Groups (30)
1. Plugin (pushed) — manifest.json, Panel.qml, Model.js, assets/ (minus 3). Defines the Omarchy bar-widget UI and model selection.
2. Daemon core — dictate.cpp, feats.h. Core ASR inference engine.
3. Build/install — build.sh, install.sh. Compilation and deployment.
4. Reference scripts — ref_asr.py, ref_fbank.py, ref_nemo.py, ref_prep_nemo.py. Prototyping/reference implementations.
5. Systemd service — omarchy-npu-dictate.service. Auto-start daemon.
6. Models metadata — devices.txt, READMEs. Hardware/model config.
7. Nepali model files — nepali-indicwav2vec/. Nepali ASR weights/config.
8. Parakeet v3 files — parakeet-v3/. English ASR model.
9. VAD files — vad/. Voice activity detection.
10. Whisper base files — whisper-base-en/. Base English model.
11. Helper transcribers — indicwav2vec_transcribe.py, qwen3_transcribe.py, whisper_transcribe.py. Transcription helpers.
12. Tools — fetch-vad.sh. Fetch VAD assets.
13. Nepali convert — convert_and_test_v2.py, convert_onnx_then_ov.py, audio/ (excluded). Model conversion pipeline.
14. Docs — MODELS.md. Model documentation.
15. Docs — NEPALI_LAB_README.md. Nepali lab notes.
16. Docs — PLAN.md. Project plan.
17. Docs — TALLY.md. Tracking/tally.
18. Docs — BUILD_NOTES_PARAKEET.md. Parakeet build notes.
19. Docs — HISTORY_HYBRID_DECODER.md. Hybrid decoder history.
20. Docs — lab-results.md. Experimental results.
21. README.md (pushed) — Project overview.
22. README.full.md (defer) — Extended docs.
23. LICENSE — License file.
24. Plugin assets — huggingface (not pushed) — HF branding/assets.
25. Plugin assets — NVIDIA (not pushed) — NVIDIA branding/assets.
26. Plugin assets — OpenAI (not pushed) — OpenAI branding/assets.
27. Plugin assets — other (pushed) — Remaining assets.
28. State/runtime (ignored) — Runtime state/PII.
29. Cache/build outputs (ignored) — Compiled artifacts.
30. Personal audio (ignored) — Recorded mic audio.

## Chronology
1. Plugin + README (pushed)
2. Daemon + build
3. Reference scripts
4. Systemd
5. Models + helpers
6. Tools + docs
7. Assets (partial)

## Group Details
1. Plugin (pushed) — manifest.json, Panel.qml, Model.js, assets/ (minus 3). Defines the Omarchy bar-widget UI and model selection.
2. Daemon core — dictate.cpp, feats.h. Core ASR inference engine.
3. Build/install — build.sh, install.sh. Compilation and deployment.
4. Reference scripts — ref_asr.py, ref_fbank.py, ref_nemo.py, ref_prep_nemo.py. Prototyping/reference implementations.
5. Systemd service — omarchy-npu-dictate.service. Auto-start daemon.
6. Models metadata — devices.txt, READMEs. Hardware/model config.
7. Nepali model files — nepali-indicwav2vec/. Nepali ASR weights/config.
8. Parakeet v3 files — parakeet-v3/. English ASR model.
9. VAD files — vad/. Voice activity detection.
10. Whisper base files — whisper-base-en/. Base English model.
11. Helper transcribers — indicwav2vec_transcribe.py, qwen3_transcribe.py, whisper_transcribe.py. Transcription helpers.
12. Tools — fetch-vad.sh. Fetch VAD assets.
13. Nepali convert — convert_and_test_v2.py, convert_onnx_then_ov.py, audio/ (excluded). Model conversion pipeline.
14. Docs — MODELS.md. Model documentation.
15. Docs — NEPALI_LAB_README.md. Nepali lab notes.
16. Docs — PLAN.md. Project plan.
17. Docs — TALLY.md. Tracking/tally.
18. Docs — BUILD_NOTES_PARAKEET.md. Parakeet build notes.
19. Docs — HISTORY_HYBRID_DECODER.md. Hybrid decoder history.
20. Docs — lab-results.md. Experimental results.
21. README.md (pushed) — Project overview.
22. README.full.md (defer) — Extended docs.
23. LICENSE — License file.
24. Plugin assets — huggingface (not pushed) — HF branding/assets.
25. Plugin assets — NVIDIA (not pushed) — NVIDIA branding/assets.
26. Plugin assets — OpenAI (not pushed) — OpenAI branding/assets.
27. Plugin assets — other (pushed) — Remaining assets.
28. State/runtime (ignored) — Runtime state/PII.
29. Cache/build outputs (ignored) — Compiled artifacts.
30. Personal audio (ignored) — Recorded mic audio.
