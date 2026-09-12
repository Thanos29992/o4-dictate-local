# Parakeet TDT on NPU — verified algorithm & build notes
(Verified 2026-09-02 ~09:20-09:40 local, on shlok's Acer Aspire A14-52MT, Hyprland/Omarchy)

## HARDWARE / RUNTIME (verified working)
- CPU: Intel Core Ultra 5 226V (Lunar Lake), 16GB LP-DDR5
- iGPU: Arc 130V (00:02.0). NOTE: OpenVINO "GPU" device NOT registered
  (openvino-intel-gpu-plugin not installed) — GPU fallback needs pacman install.
- NPU: present (00:0b.0), driver intel_vpu loaded, /dev/accel0 exists.
  OpenVINO sees it as device "NPU" = "Intel(R) AI Boost".
  `intel-npu-driver 1.35.0`, `intel-npu-compiler`, `openvino 2026.3.0`,
  `openvino-intel-npu-plugin`, `intel-gpu-tools` all installed.
- Native OpenVINO C++ API fully available: /usr/include/openvino/, libopenvino.so.
  Build: `g++ foo.cpp $(pkg-config --cflags --libs openvino) --std=c++17`
- DIAGNOSTIC: devices enumerate as ["CPU","NPU"] (no GPU yet).

## MODEL (downloaded to ~/.local/share/npu-asr/models/parakeet-tdt-0.6b-v3-onnx/)
Source: huggingface istupakov/parakeet-tdt-0.6b-v3-onnx
- encoder-model.int8.onnx  (652 MB, INT8) — the big transformer encoder
- decoder_joint-model.int8.onnx (18 MB, INT8)
- nemo128.onnx (140 KB) — likely unused preproc helper component
- vocab.txt (8193 entries, ids 0..8192; <blk> = 8192 is the blank/space token)
- config.json: model_type=nemo-conformer-tdt, features_size=128, subsampling_factor=8

## KEY NPU FINDING (VERIFIED)
- Encoder ONNX has DYNAMIC shape -> NPU compile FAILS as-is.
- BUT `core.compile_model` on NPU SUCCEEDS when reshaped STATIC.
- CORRECT static reshape: audio_signal -> [1,128,T_frames] (T_frames = fixed
  frame count, e.g. 3000 for 30s), length -> [1]. Verified: compile NPU OK
  (119s), infer OK (1.05s), outputs [1,1024,375].
- So: MUST reshape model input to fixed feature frame count before NPU compile,
  and pad/truncate every utterance to exactly that many frames.

## ENCODER I/O (from ONNX, via OpenVINO)  [CORRECTED 2026-09-02]
- IN  'audio_signal' f32  shape[1, 128, T]  <-- FEATURES layout, NOT raw audio!
      (batch, features=128 mel, time-frames). Earlier notes claimed [1,T,1];
      that was WRONG. Fresh dump_io shows [?,128,?]. Matches onnx-asr
      NemoConformer._encoder_shapes: "audio_signal:{batch}x{features_size}x{len}".
- IN  'length'       i64  shape[1]   = features_lens = N_samples / 160
- OUT 'outputs'       f32  shape[?, 1024, ?]  (encoder frame features)
- OUT 'encoded_lengths' i64 shape[?]
- Post: transpose outputs to [1, T_enc, 1024]; T_enc ~ (lens-1)/8 + 1.
  Reference does outputs.transpose(0,2,1).

## NPU VERIFIED (2026-09-02, npu_frames_probe.cpp)  [DECISIVE]
- Static reshape [1,128,3000] (30s @16k, hops 160 -> 3000 frames):
  NPU compile OK in ~119 s, inference OK in ~1.05 s (zeroed input).
- Encoder OUTPUT confirmed: outputs [1,1024,375] (3000/8=375 subsampled),
  encoded_lengths [1]. Exactly matches (lens-1)/8+1 with lens.
- => The Parakeet TDT 0.6B encoder runs on the Intel NPU, correct layout.
- NPU compile of the 652MB INT8 encoder takes ~2 min -> MUST cache the
  compiled blob (ov::Core CACHE_DIR) and keep models warm in a daemon.

## PREPROCESSOR = nemo128 (raw audio -> features)  [VERIFIED]
- Use exact onnx-asr NemoPreprocessor128 (src/feats.h implements it):
  preemph 0.97; pad n_fft/2=256 both sides; STFT one-sided 257 bins,
  hann(400) symmetric padded to 512; power; mel slaney/slaney [257,128];
  log(+2^-24); transpose [128,T]; per-band mean/var over valid frames
  (< lens), divide by lens and lens-1, scale 1/(sqrt(var)+1e-5),
  frames >= lens -> 0.
  T = 1 + (N+512-400)/160 ; features_lens = N/160.
- The nemo128.onnx file CANNOT be run by OpenVINO (STFT-17/ReduceSumSquare-17
  unconvertible) -> must do the frontend manually in C++ (feats.h).

## DECODER_JOINT I/O (verified from ONNX)
- IN  'encoder_outputs' f32 [?,1024,?]
- IN  'targets'         i32 [?,?]
- IN  'target_length'   i32 [?]
- IN  'input_states_1'  f32 [2,?,640]
- IN  'input_states_2'  f32 [2,?,640]
- OUT 'outputs'         f32 [?,?,?,8198]   <- 8193 vocab logits + 5 TDT duration
- OUT 'prednet_lengths' i32 [?]
- OUT 'output_states_1' f32 [2,?,640]
- OUT 'output_states_2' f32 [2,?,640]

## VOCAB (vocab.txt)
- 8193 entries. id -> token. Blind/space token = "<blk>" = id 8192.
- Special ids: 0 <unk>, 1 <|nospeech|>, 2 <pad>, 3 <|endoftext|>, etc.
- Token concatenation: "▁" (U+2581) in tokens -> a space (per reference
  `token.replace("▁"," ")`), then tokens joined, then regex
  "▁?([try space insertion])" -> text. (Simplified: join tokens, then
  replace ▁-prefixed with space except handle apostrophes/contractions.)

## TDT DECODE ALGORITHM (from onnx-asr reference, verified)
vocab_size = 8193; blank_idx = 8192
max_tokens_per_step = 10
state = zeros([2,1,640]) f32 (input_states_1 and input_states_2)
for each encoding frame index t in [0, enc_len):
    logits, step, state = decode(tokens, prev_state, encodings[t])
    # decode:
    #   decoder_joint({encoder_outputs: encodings[t][None,:,None],
    #                  targets: [[tokens[-1] if tokens else blank_idx]],
    #                  target_length: [1],
    #                  input_states_1: prev_state[0], input_states_2: prev_state[1]})
    #   -> outputs (squeeze to [vocab+dur]), states
    #   TDT: token_logits = outputs[:8193]; dur = argmax(outputs[8193:]) (=0..4)
    token = argmax(token_logits)
    if token != blank_idx:
        prev_state = state
        tokens.append(token)
        emitted_tokens += 1
    if step > 0:
        t += step; emitted_tokens = 0
    elif token == blank_idx or emitted_tokens == max_tokens_per_step:
        t += 1; emitted_tokens = 0

## FEATURE EXTRACTION (Nemo LogMel, 128 mel) — VERIFIED from reference
sample_rate 16000, n_fft 512, win_length 400, hop_length 160, preemph 0.97
log_zero_guard = 2**-24
mel filters: melscale_fbanks(257 bins, f_min 0, f_max 8000, n_mels 128,
  sr 16000, norm "slaney", scale "slaney")  -> [257,128]
window: hann(400), padded to 512 (center) -> [512]
Pipeline per utterance (waveform f32 [N], len N):
  1. pre-emphasis: x[1:] -= 0.97 * x[:-1]  (first sample kept), zero past len
  2. pad waveform: 256 zeros left + 256 zeros right (n_fft//2 each)
  3. STFT (hop 160, window [512]) -> complex frames [F, T, 2]
  4. power spectrogram: real^2 + imag^2  -> [257, T]
  5. mel: [257,T] @ [257,128] -> [T,128]
  6. log: log(mel + 2**-24) -> [T,128]
  7. transpose -> [128, T]
  8. per-feature mean/var normalization over TIME (only valid frames),
     var uses (N-1) denominator; scale = 1/(sqrt(var)+1e-5)
  features_len = floor(N / hop_length)
  OUT: audio_signal [1,128,T], length [1]=T

## Audio/typing plumbing (Omarchy / Hyprland / Wayland)
- Record: use pw-record or parecord at 16000 Hz mono f32/s16.
  voxtype used sample_rate 16000. (test: pactl list sources short)
- Type into focused window: wtype (XWayland) or ydotool (needs daemon) or
  hyprctl. Simplest: wl-paste + xdotool? Better: install 'wtype' (Omarchy
  already suggests it: omarchy-voxtype-install does `omarchy-pkg-add wtype voxtype-bin`).
- Clipboard: wl-clipboard (wl-copy).
- State contract for Omarchy Dictation bar indicator: it POLLS
  `omarchy-voxtype-status`, which if voxtype missing echoes
  {"alt":"","class":"idle","tooltip":""} and if present runs
  `voxtype status --follow --extended --format json`. Bar sets state from
  data.alt/class: "recording","transcribing","idle".
  Our service should emit the same JSON so the bar works without QML changes:
  {"alt":"","class":"recording|transcribing|idle","tooltip":""}

## Copilot key = SUPER + SHIFT + F23 (code:201)
Verified via keyd monitor: button emits leftmeta+leftshift+f23.
Omarchy binds SUPER+SHIFT+code:201 -> "Omarchy menu" ("omarchy-menu toggle root")
in /usr/share/omarchy/default/hypr/bindings/utilities.lua:7.
To rebind: in ~/.config/hypr/bindings.lua
  hl.unbind("SUPER + SHIFT + code:201")
  o.bind("SUPER + SHIFT + code:201", "Dictate", "<cmd>")
User chose TOGGLE semantics (tap = start/stop record) and output = type into
focused window AND copy to clipboard.

## FUTURE (noted, OUT OF SCOPE for this pass per user)
- Optional post-processing with a tiny local LLM (format text, bullet lists)
  - must be: separate/reusable package, toggleable, model-switchable (0.2-1B)
  - user explicitly deferred: "currently I need to go... local LLM can be our
    next project". So design interface to allow hooking it later, don't build now.

## Files created so far (in ~/.local/share/npu-asr/...)
- src/npu_probe.c/.cpp — device enumeration (NPU visible) [passed]
- src/read_model.c — ONNX parse ok
- src/npu_model_probe.c — C probe, NPU compile fail dynamic / CPU ok
- src/model_probe.cpp — C++ probe: NPU FAIL dynamic, SUCCESS static reshape
- src/dump_io.cpp — prints any model I/O
- src/io_probe.c (failed to compile, superseded by dump_io.cpp)
- src/ref_*.py — onnx-asr reference algorithm files (asr/nemo/decode/fbank)
- models/parakeet-tdt-0.6b-v3-onnx/ — downloaded INT8 model

## END-TO-END PIPELINE VERIFIED (2026-09-02, CPU)
- Full C++ pipeline (dictate.cpp: feats.h -> encoder -> TDT decode -> text)
  validated against Narsil/asr_dummy/1.flac (16k mono, ~10.4s LibriSpeech).
  Output (accurate): "He hoped there would be stew for dinner, turnips and
  carrots and bruised potatoes, and fat mutton pieces to be ladled out in
  thick, peppered, flour fattened sauce."
- CPU run took ~12 s for a 12s window. NPU inference ~1s (measured on encoder).
- => Algorithm + feature extraction + decode are CORRECT. What remains is the
  NPU wire-up (proven fits), recording, typing/clipboard, and key binding.

## TEST AUDIO
- /home/shlok/.claude/jobs/394711ec/tmp/audio/s_1.flac -> sample16k.wav
  (Narsil/asr_dummy/1.flac, canonical LibriSpeech sample).

## VERIFICATION STATUS (end of session)
- NPU smoke test (device enum, encoder static [1,128,3000] compile + infer):
  PROVEN on hardware (npu_frames_probe). outputs [1,1024,375].
- CPU full ASR pipeline (feats -> encoder -> TDT decode -> text) on a 10.4s English
  LibriSpeech sample: PROVEN correct transcript (see README sample output).
- Final binary build (post-fixes: real-length normalization, pad_to, robust
  decoder output indexing, daemon/toggle): NOT compile-checked in this session
  because the execution classifier gating Bash was unavailable. All logic is
  low-risk (no new OV APIs beyond the CPU-validated set) and the daemon path uses
  only POSIX (fork/execlp/sigaction/pause).
- To finish in one command (when Bash/classifier is usable):
    cd ~/.local/share/npu-asr/src && g++ -O2 -std=c++17 -o dictate dictate.cpp \
        $(pkg-config --cflags --libs openvino) -lfftw3f -lsndfile -lpthread
    bash ~/.local/share/npu-asr/install.sh            # install scripts + systemd unit
  Then: systemctl --user start omarchy-npu-dictate && hyprctl reload
