# o4-dictate-local

Local speech-to-text for my Omarchy desktop. I tap the Copilot key to start
recording, tap it again to stop, and the transcript gets typed straight into
whatever I'm focused on. No cloud, no account, the audio never leaves the
machine.

The daemon is a small C++ process that stays resident, loads a model only when
I push the button, transcribes, then pastes the result (clipboard plus
synthetic keypresses, so it lands in any app). The panel is an Omarchy bar
widget that picks device, model and offload policy.

## Models

Three models behind the toggle, each pinned to the hardware it runs best on:

| model | language | device |
|---|---|---|
| Whisper Base, INT4 OpenVINO IR | English | CPU, iGPU, NPU |
| Parakeet V3 Streaming, Q8_0 GGUF | English | GPU |
| Nepali ASR (indicwav2vec), OpenVINO IR | Nepali | CPU + iGPU |

Silero VAD v4 sits in front for silence detection. Weights are fetched by a
script rather than committed.

- Whisper Base int4 runs on CPU, iGPU or NPU. English-only; the NPU path is
  the fastest, and it's the only model that pins to it.
- Parakeet V3 Streaming runs on the GPU through ggml (Vulkan backend). This
  is the streaming-capable one.
- Nepali ASR is a fine-tune of indicwav2vec exported to OpenVINO IR. Real
  Devanagari output, runs on CPU or iGPU.

Hardware I run this on: an Acer Aspire 14 AI.

- Intel Core Ultra 5 226V
- Intel ARC 130V iGPU
- Intel AI Boost NPU, 40 TOPS

Arch with Omarchy. OpenVINO enumerates CPU and NPU; the iGPU gets used
through ggml's Vulkan backend.

## What works

- tap-to-dictate through the Copilot key (record, transcribe, paste)
- whisper base int4 on the NPU — fast, noise-free english dictation
- parakeet v3 streaming on the GPU, one-shot and live streaming
- nepali asr on CPU / iGPU with real devanagari in the transcript
- silero vad so long recordings don't feed dead air to the model
- offload policy: drop the weights after a take, or keep them resident
- floating recorder overlay with a timer and vu bars while recording

## Repo layout

- daemon/ — dictate.cpp, feats.h (log-mel frontend), build.sh, install.sh,
  the ref_* probes, parakeet_stream_transcribe_rt.cpp
- plugin/ — the Quickshell/Omarchy panel, the recorder popup, assets
- helpers/ — python transcribers: whisper, qwen3, indicwav2vec
- models/ — metadata per model; the weights are gitignored
- systemd/ — omarchy-npu-dictate.service
- tools/ — fetch-vad.sh, nepali-convert/

## Setup

Dependencies on Arch: openvino, fftw, sndfile (or ffmpeg), pipewire,
wl-clipboard, wtype. Then run daemon/install.sh — it builds the daemon, drops
the systemd user unit and enables it. Reload Hyprland so the Copilot key
binding takes effect. The first NPU compile takes a couple of minutes, then
it's cached.

## Day to day

Tap Copilot to start recording, tap again to transcribe and paste. The bar
shows recording / transcribing / idle, and the floating popup mirrors it live.

Two gotchas I hit:

- whisper and nepali shell out to the python helpers; parakeet is pure C++
  through ggml. All helper output goes through json with ensure_ascii=False,
  so devanagari (and german, and anything non-ascii) round-trips correctly
  into the clipboard instead of coming out as escaped \u sequences.
- the systemd user service doesn't inherit the wayland display, so it sets
  WAYLAND_DISPLAY explicitly. Without it wl-copy can't reach the compositor
  and nothing gets pasted.

## Troubleshooting

- nothing pastes after a take: check WAYLAND_DISPLAY is set for the service
- bar stays idle: the daemon may not be running, journalctl --user -u
  omarchy-npu-dictate
- transcript full of \uXXXX escapes: an older helper without ensure_ascii=False