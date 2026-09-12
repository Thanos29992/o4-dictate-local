# NPU-ASR Implementation Summary

## What Works ✓

**Hardware**: Intel Core Ultra 5 (Lunar Lake), Intel Arc 130V GPU / Intel NPU ("AI Boost" on `/dev/accel0`)

**Pipeline** (Hybrid NPU+CPU):
- **Encoder**: Parakeet TDT 0.6B INT8 model runs on NPU (~1s inference)
- **Decoder**: decoder_joint runs on CPU (~<10ms per step, correct INT8 logits)
- **Frontend**: LogMel128 feature extraction (native C++/fftw)
- **TDT Decode**: Greedy TDT with duration tokens, state maintenance

**Verified Output** (sample16k.wav = Narsil/asr_dummy/1.flac):
```
"He hoped there would be stew for dinner, turnips and carrots and bruised potatoes, and fat mutton pieces to be ladled out in thick, peppered, flour fattened sauce."
```

**Daily Use**:
```bash
omarchy-npu-dictate status   # GET (outputs JSON for bar)
omarchy-npu-dictate toggle   # TAP: record → transcribe → type + copy
omarchy-npu-dictate start    # START: warm daemon on login
omarchy-npu-dictate start-daemon  # START NOW (compile NPU models)
```

## The Key Fix (Why Hybrid?)

- Encoder compiled to NPU: **Works fine** — static shapes, accurate INT8
- Decoder compiled to NPU: **FAILS** — INT8 LSTM states produce constant outputs (always blank on NPU)
- Decoder compiled to CPU: **Works** — original intent, fast (18MB model), correct

Result: NPU encoder runs on hardware, CPU decoder runs on correct INT8 accuracy.

## Files

```
~/.local/share/npu-asr/
  binaries: dictate, install.sh
  cache/ (ov::cache_dir) — NPU compiled blobs (~2GB)
  state/
    daemon.pid, status.json, capture.wav, err.log
  models/parakeet-tdt-0.6b-v3-onnx/ — ONNX INT8 models
  src/
    dictate.cpp — main NPU+CPU hybrid implementation
    feats.h — LogMel128 frontend (native fftw)

~/.local/bin/
  dictate — main binary
  omarchy-npu-dictate — Omarchy wrapper (executes daemon or one-shot)

~/.config/
  hypr/bindings.lua — Copilot key (SUPER+SHIFT+F23) → omarchy-npu-dictate toggle
  systemd/user/omarchy-npu-dictate.service — auto-start daemon on login
```

## Troubleshooting

**Bar shows "idle"** → daemon not running: `omarchy-npu-dictate start`

**First run hangs ~2-4 min** → NPU encoder compiles then caches to `~/.local/share/npu-asr/cache/` — subsequent runs warm from cache

**Copilot key does nothing** → run `hyprctl reload` after bindings.lua change

**Transcript looks wrong** → encoder on CPU instead: `dictate --device CPU`

## Performance

- **Encoder (NPU)**: ~1s per 30s utterance (first compile 2-4 min, then cached)
- **Decoder (CPU)**: ~<10ms per decoder step (131 steps for 30s → ~1.5s total)
- **Typing**: `wtype` (XWayland), ~0.5s for ~50 words
- **Total**: ~3-4s end-to-end (daemon warm), <20s cold

## Architecture Notes

- **Static shapes required on NPU**: models reshaped to fixed frame counts (3000 frames = 30s @ 16kHz hop 160)
- **Feature layout**: `audio_signal [1, 128, T]` (features not raw audio)
- **Dynamic frame count**: `length` input = `N/160` (valid frames; encoder masks zero-padded tail)
- **TDT decode**: tokens[i], durations[i] from `outputs [?, ?, ?, num_tokens]` where last dim = 8193 vocab + 5 duration
- **No hybrid model swap**: dictate binary is hybrid at compile time (encoder device setter, decoder CPU-compiled)
- **Daemon hot-persistence**: encode models stay resident (~200MB share), no per-tap recompile
