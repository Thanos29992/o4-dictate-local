# Nepali Lab — Results (2026-09-05)

## TL;DR
**`sumanpaudel1997/nepali-asr-indicwav2vec` is real, converts to OpenVINO IR, and produces genuine Devanagari on this machine's iGPU.** Smoke-tested on `nep_trim1.wav` (8s Nepali, 16k mono); the transcription **matches the truth text** on the clip. CPU result is identical. NPU compile failed (dynamic-shape ONNX export — not surprising; we'd need to fix shapes for NPU).

## Smoked transcript (vs truth)
- Truth (your "nep_trim1.wav", 8 s): "आज मौसम निकै राम्रो छ त्यसैले बिहानै बाहिर गएर केही समय हिँड्न चाह ..."
- GPU & CPU output (CTC `|` is the model's word-boundary token):
  `आज|मौसम|निकै|राम्रो|छ|त्यसैले||बिहानै|बाहिर|गएर|केही|समय|हिँड्न|चाह`
- Mapping: `आज` = today, `मौसम` = weather, `निकै` = really, `राम्रो` = good, `छ` = is, `त्यसैले` = so, `बिहानै` = morning, `बाहिर` = outside, `गएर` = going, `केही` = some, `समय` = time, `हिँड्न` = walk, `चाह` = want. **12 of the 12 content words match.** ✓

## Performance
- **GPU (Intel Arc 130V via Level-Zero):** 2119 ms encoder + 7 ms head+decode = 2.1 s end-to-end
- **CPU (Ultra 5 226V):** 824 ms encoder + 3 ms head+decode = **0.8 s end-to-end** (CPU faster on this clip)
- NPU: compile failed (`ZE_RESULT_ERROR_INVALID_ARGUMENT` from `vclAllocatedExecutableCreate2` — dynamic-shape ONNX export doesn't compile on the NPU compiler as-is)

## Artifacts
- `models/indicwav2vec/` — 377 MB safetensors + config + vocab + preprocessor
- `models/indicwav2vec_enc.onnx` — 377.7 MB ONNX export (encoder only)
- `models/indicwav2vec_ov/model.xml` + `.bin` — OpenVINO IR (FP16-compressed, ready to run)
- `smoke/convert_onnx_then_ov.py` — reproducible conversion + smoke test
- `smoke/nep_trim1.wav`, `smoke/nepali_codex.wav` — test audio
- `onnx_then_ov.log` — full stdout/stderr

## Build + run
```bash
# one-time: install onnx into the lab venv (nep-base-lab venv, shared with other Nepali work)
VENV=/home/shlok/Projects/omarchy-asr/nep-base-lab/.venv
VIRTUAL_ENV=$VENV uv pip install onnx

# convert + smoke
$VENV/bin/python smoke/convert_onnx_then_ov.py
```

## What this means
- The top pick (`sumanpaudel1997/nepali-asr-indicwav2vec`, 378 MB, 14.89% WER) is **genuinely viable**. It runs on iGPU and CPU today, and produces real Devanagari.
- To use it in the live ASR daemon, the next step is a small C++ helper (e.g. `indicwav2vec_transcribe.cpp`) that:
  1. reads a 16k mono wav
  2. runs the OpenVINO IR on the user's chosen device (CPU/GPU)
  3. runs the LM-head (one matmul, `[T,768] @ [768,81].T`) — can stay in C++/numpy
  4. does CTC greedy decode (collapse repeats + remove blank 80) and prints Devanagari
  5. reuses the existing `whisper_transcribe.py`-style invocation pattern
- NPU: requires reshaping the ONNX to a static input shape (`[1, T_FIXED]`) for the VCL compiler to accept it. Possible follow-up; not needed for GPU/CPU which work now.
- For real-time streaming use, this model is a CTC encoder, not a streaming model — would need chunked inference (overlap-add or VAD-segmented single-shot). For "stream while transcribing", the live libtranscribe streaming path is the cleaner choice (English-only on Parakeet Unified; the Nepali equivalent would need a separate streaming-capable model, e.g. the `sumanpaudel` indicwav2vec would need a streaming variant — doesn't exist on the public Hub yet).

## Open follow-ups (not done; queue for next session)
- Build the `indicwav2vec_transcribe` C++ helper for the live daemon.
- Add a `devices.txt` with `CPU\nGPU\n` to `models/indicwav2vec_ov/` so the daemon's `resolve_model_dir` validates it.
- Wire it into the panel as a new model entry `nepali-indicwav2vec`.
- (Optional) NPU: static-shape ONNX export for the VCL compiler.