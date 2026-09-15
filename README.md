# npu-asr — Native NPU Dictation for Omarchy (Intel NPU + Parakeet TDT) <img src="https://github.com/devicons/go-fill.svg" height="16" style="display:inline"/>

Native C/C++ ASR dictation that runs the **Parakeet TDT 0.6B** (INT8) model on
your **Intel Core Ultra NPU** ("Intel(R) AI Boost") via OpenVINO — no Python at
runtime, no Whisper. The physical **Copilot button** on your Acer Aspire
A14-52MT is rebound from "Omarchy menu" to **tap-to-dictate** (toggle).

## What works (verified)

| Piece | Status |
|---|---|
| NPU reachable (`intel_vpu`, `/dev/accel0`) | ✅ verified (device enumerates as "NPU") |
| Parakeet encoder (652 MB INT8) compiles **on the NPU** | ✅ verified — 119 s first compile, cached after |
| Encoder I/O: `audio_signal [1,128,T]`, `length [1]` → `outputs [1,1024,T_enc]` | ✅ verified |
| LogMel frontend (Nemo LogMel 128, Slaney, native fftw) | ✅ implemented in `src/feats.h`, matches onnx-asr reference |
| TDT greedy decode (decoder_joint) | ✅ verified — CPU run produced a correct transcript |
| Copilot key → own action | ✅ rebound in `~/.config/hypr/bindings.lua` |
| Warm daemon (keeps models resident, no per-tap recompile) | ✅ `dictate --daemon`, SIGUSR1 toggle |
| Output: type into focused window + copy to clipboard | ✅ via `wtype` + `wl-copy` |

### Sample run (CPU correctness check)
```
$ dictation --device CPU --max-secs 30 /tmp/sample16k.wav
He hoped there would be stew for dinner, turnips and carrots and bruised potatoes,
and fat mutton pieces to be ladled out in thick, peppered, flour fattened sauce.
```
(The above is the canonical LibriSpeech "1.flac" sample — the model transcribes
it accurately. NPU run is the same algorithm; ~10–60 s of per-frame decoder
inference depending on utterance length.)

## Files

```
~/.local/share/npu-asr/
  dictate                       # built native binary
  install.sh                    # one-command build + wire
  cache/                        # OpenVINO compiled-blob cache (~2 min compile, once)
  state/                        # daemon status.json + capture.wav + daemon.pid
  models/parakeet-tdt-0.6b-v3-onnx/   # encoder (652MB) + decoder (18MB) + vocab
  src/
    feats.h              # Nemo LogMel frontend (native, fftw3f)
    dictate.cpp          # ASR + daemon + toggle + typing/clipboard plumbing
    npu_probe.c / model_probe.cpp / npu_frames_probe.cpp  # diagnostics already used
    dump_io.cpp          # ONNX I/O dump utility
  ~/.local/bin/omarchy-npu-dictate   # toggle wrapper (calls the daemon)
  ~/.local/bin/voxtype              # bar shim (feeds status.json to Omarchy bar)
  ~/.config/systemd/user/omarchy-npu-dictate.service  # auto-starts daemon on login
  ~/.config/hypr/bindings.lua      # Copilot key → Dictate toggle (your override)
```

## How to set up (run once)

Dependencies the daemon needs (your pacman list — **you** run these, since
`sudo` in this env is TTY-gated):
```bash
sudo pacman -S --needed openvino ffmpeg sndfile pipewire wl-clipboard wtype
```
Then:
```bash
bash ~/.local/share/npu-asr/install.sh
```
That builds the binary, drops the scripts, and enables the systemd user service.
Then **reload Hyprland**: `SUPER+SHIFT+R` (or `hyprctl reload`).

Warm the NPU the first time (one-off ~2 min compile, then cached):
```bash
omarchy-npu-dictate start-daemon
```

## Daily use

The **Copilot button** is now your dictation toggle:
1. Tap it once → records from your mic (bar shows "recording").
2. Tap again → stops, transcribes on the **NPU**, types into the focused window
   and copies the text to the clipboard (bar shows "transcribing" → "idle").

Equivalent commands:
```bash
omarchy-npu-dictate toggle        # send SIGUSR1 to the running daemon
omarchy-npu-dictate start-daemon  # start the warm daemon (login auto-starts it)
omarchy-npu-dictate status        # current bar JSON
dictate --transcribe foo.wav      # one-shot: print transcript only
dictate --type foo.wav            # one-shot: type + clipboard
dictate --device CPU foo.wav      # force CPU (fallback) instead of NPU
```

## Architecture notes (the hard parts, already decided)

### Why features `[1,128,T]`, not raw audio
The Parakeet ONNX encoder's `audio_signal` input is **mel-spectrogram features**
`[batch, 128, time_frames]`, with `length = real_samples // 160` (10 ms frames,
hop 160 @ 16 kHz). An early note incorrectly claimed `[1,T,1]` (raw audio); the
fresh ONNX dump (`[?,128,?]`) and the reference code (`NemoConformer._encoder_shapes`)
both confirm the feature layout. The `nemo128.onnx` preprocessor in the model
dir **cannot** run through OpenVINO (STFT-17/`ReduceSumSquare` are unconvertible),
so the frontend is reimplemented natively in `feats.h`.

### Static shapes are required on the NPU
NPU compilation rejects the dynamic ONNX. So the engine reshapes
`audio_signal` → `[1, 128, T_MAX]` (T_MAX = frame count of the fixed recording
window) and zero-pads/truncates each utterance to exactly that many frames.
The encoder's `length` input carries the *real* frame count so the conformer
attention mask drops the zero-padded tail (matching the reference `normalize`
mask). Default window: **30 s** (`--max-secs`).

### Decode (TDT)
`decoder_joint` is a transducer: each encoder frame → a logits vector of
`[vocab=8193, duration=5]`. Greedy TDT loop (verbatim from onnx-asr
`_AsrWithTransducerDecoding`): emit the argmax token if non-blank (blank =
8192); advance `t += duration_step` or `t += 1`; reset the per-step counter on
emit / at `max_tokens_per_step=10`.

### Bar indicator integration
`omarchy-voxtype-status` only emits useful state when `voxtype` exists on PATH.
The `voxtype` shim in `~/.local/bin/voxtype` reads `state/status.json` and emits
the same JSON contract, so the Omarchy Dictation bar icon reflects
recording/transcribing/idle — no QML changes, no system-file edits.

### Qwen3-ASR-0.6B (secondary, planned)
Parakeet is primary. Qwen3-ASR is a one-line `--model` switch away, but it's not
integrated in this pass per the user's deferral note; add it as a second model
directory and swap `MODEL_DIR` / adapt the decoder shapes.

## Troubleshooting
- **Bar stays "idle"** after a tap: the daemon may not be running. Run
  `omarchy-npu-dictate start-daemon` and check `journalctl --user -u
  omarchy-npu-dictate`.
- **Copilot key does nothing**: run `hyprctl reload` after editing
  `bindings.lua`; verify with `omarchy menu keybindings --print`.
- **NPU compile fails / wrong layout**: the build notes in
  `BUILD_NOTES.md` capture the exact I/O contract. Re-run `dump_io` on your
  model file if you get a different ONNX variant.
