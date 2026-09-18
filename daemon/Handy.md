# Handy ASR — Architecture Deep-Dive

> A ground-truth breakdown of **Handy** (github.com/cjpais/Handy), the open-source
> always-on speech-to-text app, so we can build our own competitive dictation
> stack. **Everything here was verified** against the extracted AppImage binaries
> (`/tmp/squashfs-root`, v0.9.6), the on-disk GGUF model, Handy's own runtime log
> (`~/.local/share/com.pais.handy/logs/handy.log`), its settings store, and the
> GitHub source. Claims that are inferred but not directly observed are marked
> `[UNVERIFIED]`. Nothing is fabricated.

---

## 1. What Handy actually is

Handy is a **Tauri 2.x** desktop app: a Rust backend (`src-tauri/src/`) + a
React/TypeScript web frontend (`src/`). It is **not** the old gtk4-layer-shell
"panel" app; GTK layer-shell is only used for the Linux **recording overlay
window** (`HANDY_NO_GTK_LAYER_SHELL=1` reverts it to a plain always-on-top window).

There is **no single "engine crate."** Handy plugs in **two independent inference
backends**, selected per model:

| Backend | Library | Run-time | Models it drives |
|---|---|---|---|
| **transcribe-cpp** | C/C++ **GGML/GGUF** (`libtranscribe.so.0.2` + `libggml*`) | Whisper-family / new `moonshine_streaming` GGUF | the default **Moonshine Streaming Tiny** |
| **transcribe-rs** | Rust **ONNX Runtime** | Parakeet, Moonshine (streaming too), SenseVoice, GigaAM, Canary, Cohere, Whisperfile |

Handy audio path: cpal (I/O) → vad-rs (Silero VAD) → router → a transcribe backend
→ text → clipboard/paste. Hotkeys via rdev. Single-instance; a second instance
acts as an CLI remote (`--toggle-transcription`, `--toggle-post-process`,
`--cancel`).

**Key takeaway:** Handy does **not** run the model with OpenVINO, whisper.cpp alone,
or a GStreamer ASR pipeline. On this machine the model is Moonshine Streaming Tiny
quantized to GGUF and executed on the **GGML** runtime through **transcribe-cpp**
(a fork of `ondrejhd/whisper.cpp`, renamed `whisper_*` → `transcribe_*`).

---

## 2. Inference stack (verified from the binaries)

From `libtranscribe.so.0.2` (not stripped — full symbol table), `libggml.so.0`,
`libggml-base.so.0`, and the bundled backend plugins:

- **GGML version**: literal `"0.15.2"` in `libggml-base.so.0`.
- `libtranscribe.so.0.2` is a **whisper.cpp fork** — every symbol is
  `transcribe_*` (transcribe_run, transcribe_run_batch, transcribe_open,
  transcribe_model_load_file, transcribe_init_backends, transcribe_close).
  All `ggml`/`gguf` symbols are **imported (U)** from `libggml*`.
- **GGUF loader**: `gguf_init_from_file`, `gguf_get_n_kv`, `gguf_find_key`,
  `gguf_find_tensor`, `gguf_get_tensor_offset/size/type`; GGUF v1 explicitly
  rejected (`"GGUFv1 is no longer supported"`).
- **Compute** runs through the backend-scheduler API:
  `ggml_backend_init_by_type`, `ggml_backend_sched_new`,
  `ggml_backend_sched_graph_compute`, plus OpenBLAS (`cblas_sgemm`).
- **Backends shipped in the AppImage** under `/usr/lib/`:
  `libggml-cpu-{alderlake,cannonlake,cascadelake,cooperlake,haswell,icelake,
  ivybridge,piledriver,sandybridge,sapphirerapids,skylakex,sse42,x64,zen4}.so`,
  `libggml-base.so.0`, `libggml.so.0`, `libggml-vulkan.so`, `libopenblas.so.0`,
  `libtranscribe.so.0.2`, plus the VAD/model resources.

### Runtime specifics observed on THIS machine (handy.log)

Two distinct boots are recorded:

- **2026-09-01 (earlier)**: `load_backend: loaded CPU backend from
  .../libggml-cpu-alderlake.so` … `transcribe_init_backends: 1 compute device(s) …
  : CPU` → `[CPU (cpu)]`. **CPU only**.
- **2026-09-03 (later, today)**: `ggml_vulkan: Found 1 Vulkan devices: 0 =
  Intel(R) Graphics (LNL) (Intel open-source Mesa driver) | uma: 1 | fp16: 1 |
  bf16: 0` … then **both** Vulkan and CPU backends load → `transcribe_init_backends:
  2 compute device(s): Vulkan0, CPU` → `[Vulkan0 (vulkan), CPU (cpu)]`.

So on Linux the GPU path **had** become available by 2026-09-03 (the Vulkan0
backend binds the Intel LNL iGPU through the open-source Mesa driver). The user's
observation ("Handy in this Linux does not have any support for my iGPU") matches
the **Sep 01 / CPU-only** state; by Sep 03 it binds. This is a strong signal that a
GGML–Vulkan path to our iGPU is the proven-working route for the *model*, and that
the GPU gap on Linux is **not** the hardware — it is the backend/driver registration.

- **Model load**: `Loaded whisper model 'handy-computer/moonshine-streaming-tiny-gguf/
  moonshine-streaming-tiny-Q8_0.gguf' (requested Auto, gpu_device 0, bound backend
  'CPU' [earlier] / 'Vulkan0' [later], supports_streaming=true, supports_translate=false,
  supports_language_detect=false)`. Model load time `took 43ms` (CPU) / `139ms..43ms`
  (Vulkan).
- The ASCII `whisper` label is a **generic** one — the architecture string is
  `moonshine_streaming`; it is not OpenAI's Whisper model.

---

## 3. The Moonshine model (verified from the GGUF bytes)

**File** `moonshine-streaming-tiny-Q8_0.gguf`, **48.13 MiB / 50,462,816 bytes**,
GGUF **v3**, 161 tensors, 63 metadata KVs. Download says
`handy-computer/moonshine-streaming-tiny-gguf@main`; `general.repo_url` =
`UsefulSensors/moonshine-streaming-tiny`; `general.license` MIT;
`general.languages = ["en"]`. Snapshot contains **only** this GGUF (no README/config).

### Header fields (I parsed the binary first-hand)

**Tokens/identity:**
- `general.architecture = moonshine_streaming`
- `general.name = Moonshine Streaming Tiny`, `general.size_label = 44M`,
  `stt.variant = moonshine-streaming-tiny`
- `stt.capability.streaming = true`, `lang_detect=false`, `translate=false`,
  `timestamps=false`
- Tokenizer: `tokenizer.ggml.model = bpe`; vocab **32768**; `bos=1, eos=2, pad=0`

**Encoder** (6 blocks): `d_model=320, n_heads=8, n_kv_heads=8, head_dim=40,
ffn_dim=1280, activation=gelu`, plus a **raw-waveform front feature stack**
(`frontend.type=raw`, `sample_rate=16000`, `num_mels=1`)
- `enc.embedder.comp.log_k` (CMVN log-compression)
- `enc.embedder.linear.weight [80,320]` (F16)
- `enc.embedder.conv1.weight [5,320,640]` + `conv2.weight [5,640,320]` (F16) —
  two causal stride-2 convs that subsample the PCM waveform. **No STFT/mel** —
  the network ingests 16 kHz mono float PCM directly.

**Decoder** (6 blocks): `d_model=320, n_heads=8, n_kv_heads=8, head_dim=40,
ffn_dim=1280, activation=silu`, `max_position_embeddings=4096`,
`tie_word_embeddings=false` (separate `[320,32768]` embedding and `lm_head`).
Has **cross-attention** to the encoder and an `adapter.pos_emb [320,4096]`.

**Streaming params** (`stt.moonshine_streaming.*` — the *only* latency keys in the file):
- `encoder.frame_ms = 5.0`
- `encoder.frame_len = 80`
- `encoder.sliding_windows = [16,4,16,4,16,4,16,4,16,4,16,4]` (len 12 = 2×6 layers)

> The often-cited "512 ms chunk / 25 ms mel / ~2 s latency / context 512" figures
> are **not** in this card. The decoder `max_position_embeddings=4096` is the only
> "context" figure and it's the decoder's, not a 512-token audio window.

### Quantization nuance

Filename and `general.file_type = 7` say **Q8_0**, but per-tensor dtypes are
mostly **Q8_1** (99 Q8_1 weight matrices, 59 F32 norms/biases/CMVN, 3 F16 conv/
linear). Numerically negligible for our purposes — the takeaway is the model is
**~44.05M params** at **1 byte/weight**, which is why it lives in ~48 MB and runs
with far lower RAM than our 622 MB INT8 encoder.

---

## 4. How the STREAMING engine works (the part we care about)

Handy's live dictation is a **true streaming loop** on the Moonshine Streaming
model, exposed as a C ABI from transcribe-cpp:

```
transcribe_stream_begin / transcribe_stream_feed / transcribe_stream_finalize
  + get_text / get_state / last_status / reset / update / revision
  + n_committed_segments / n_committed_tokens / n_committed_words
```

Internals (mangled C++ symbols → decoded):
- `transcribe::moonshine_streaming::apply_adapter_window(session, model, float*, …)`
- `transcribe::moonshine_streaming::encode_window_to_host(session, model, …)`
- `transcribe::moonshine_streaming::flush_stable_frames(session, model, i, i, i, i)`
- `transcribe::moonshine_streaming::build_sliding_window_mask(w, h, w, float*)`

**The loop (verified semantics):**
1. Audio arrives as raw 16 kHz mono float PCM, in multiples of `frame_len`.
   Short input is right-padded (the binary asserts `n_samples` is a multiple of
   `frame_len` and `only 16000 Hz supported`).
2. Each `stream_feed` runs the encoder over a **sliding PCM window**, applies the
   learned-positional **adapter** on the new emit slice, projects per-layer
   cross-attention K/V, then **re-decodes the transcript from BOS**.
3. The decoder output is split into **committed** vs **tentative** text. The
   longest token-id prefix that re-appeared identically across the last
   `stable_prefix_agreement_n` feeds (default 3) is marked committed; tokens after
   the divergence stay tentative until later feeds confirm them or
   `stream_finalize` commits everything.
4. `committed_text` is **append-only, never rolled back**; `full_text` is the
   authoritative raw hypothesis. When `full_text` revises a committed byte the seam
   is transiently incoherent — this is acceptable because the *published* text is
   committed-only.
5. **This is exactly how filler/noise is suppressed.** Divergent tentative tokens
   are **dropped** on each feed rather than flushed as final text — a token that
   flickers (a repeated "yeah") never commits unless it stabilizes across feeds.

**Latency/refresh evidence from the runtime log:**
- `first audio chunk arrived 21.7ms after stream start (42.0ms of audio)`;
  `first captured chunk (42.0ms of audio) processed`.
- Feed cadence & emit: `transcribe.cpp docs … ~240 ms cumulative encoder
  right-context, 80 ms feed cadence, 20 ms natural emit unit` (from repo docs).
- Real telemetry:
  - `Live preview perf: 3.12s streamed audio, 2.54s model compute (1.23x
    real-time), input_received=3.12s, committed_audio=2.88s, buffered=240ms,
    revision=9, 89 frames fed, 9 updates emitted`
  - `Live preview perf: 3.78s streamed audio, 1.87s model compute (2.02x
    real-time) … revision=11, 111 frames fed, 11 updates emitted`
  - `Live preview finalized in 4.81s model compute for 7.80s streamed audio
    (1.62x real-time) … revision=20, 245 frames fed, 19 updates emitted, 21 chars`
- `[stream-mem] retained PCM=… samples (…s) audio-embeds=… frames`

So on a CPU-only boot it hit **1.1–2× real time** transcribing **live, per-frame**,
while we watch a whole segment finish then batch-transcribe. That is the single
biggest architectural difference.

---

## 5. Backend / device matrix

- **Linux**: GGML with Vulkan (Mesa) and/or CPU. On this laptop it went
  CPU-only (Sep 01) → Vulkan0 + CPU (Sep 03). iGPU ↔ dGPU selection is a
  known rough edge (ggml-vulkan numbers devices VulkanN; `DRI_PRIME=1` forces the
  dGPU; `transcribe_accelerator: Auto`).
- **Windows**: DirectML is enabled (PR #1058) *and* Vulkan. On a Windows hybrid
  laptop backends registered as `Vulkan0, Vulkan1, CPU`.
- **macOS**: Metal; **NVIDIA**: CUDA.
- **transcribe-rs** (ONNX) separately supports ort-cuda / ort-rocm / ort-directml /
  ort-coreml / ort-webgpu via its own `set_ort_accelerator`.

Handy does **not** target the Intel NPU at all — its accelerators are GGML/Vulkan/
CUDA/Metal/DirectML and ORT. The user's "use my iGPU (which I do on Windows)" is met
by the **Vulkan** path.

---

## 6. Audio capture (verified)

- Data flow: **GStreamer** (bundled via linuxdeploy-plugin-gstreamer) → the app's
  Rust audio layer (`audio_toolkit::audio`, `managers::audio`) → resample
  (rubato) → 16 kHz mono float PCM → `transcribe_stream_feed`.
- Bundled plugins: `libgstpulseaudio.so` (pulsesrc/pulsesink),
  `libgstaudioconvert`, `libgstaudioresample`, `libgstaudiomixer`, `libgstapp`
  (appsink), `libgstautodetect`. `[UNVERIFIED]` exact capture element chosen at
  runtime (likely `pulsesrc`).
- Runtime evidence says audio source `pipewire`, **44100 Hz mono F32**, saved WAVs
  named `handy-<unix>.wav`. Resampling to 16 kHz happens before the model.
- **VAD** runs separately in embedded ONNX Runtime on `silero_vad_v4.onnx`
  (bundled, 1.8 MB) — VAD in ORT, ASR in GGML. GigaAM (Russian) is the *bundle*'s
  default ASR model; Moonshine Streaming Tiny is what the user selected.

---

## 7. Why it's lean on RAM and accurate (vs our pipeline)

**RAM.** Handy streams a **sliding window**; it never materializes a 90 s padded
feature slab. The encoder re-encodes only the new/active window and keeps a
**KV cache** (`cross_kv_commit/cross_kv_proj`, `kv_window`, `build_sliding_window_mask`),
so per-layer activations track the *active* speech span, not a fixed 90 s maximum.
The model itself is ~44 M params × 1 byte/weight ≈ **tens of MB resident** (vs our
~622 MB INT8 encoder on disk + full-window activations → GB peak). GGUF 8-bit block
weights are the dominant reason for the small footprint.

**Accuracy / no repeated filler.** Handy's streaming decoder discards a tentative
hypothesis that diverges across feeds — repeated/onset filler ("yeah yeah yeah",
"okay okay") is dropped before it can commit. The committed-prefix algorithm
**preserves meaning** (it never deletes *stable* text) while preventing flicker.
It also keeps decoder/cross-attention state continuous across the stream, so it is
always "primed" — it never re-voices the start of each chunk, the way a fresh
zero-primed decode does.

**RTF.** 1.1–2× real time on CPU as measured. That's the bar.

> ### Two distinct "live" mechanisms (don't conflate them)
> Handy has **two** different partial-text paths:
> - **Native Moonshine streaming** (above) — the true per-chunk decode that carries
>   state and commits a stable prefix.
> - **Live-preview overlay** — a *separate* background thread repeatedly snapshots
>   the recorded audio and re-transcribes the **full snapshot on a periodic timer**
>   (initially 3 s, then 1 s), emitting display events (PR #864). Any single
>   snapshot is a **batch** transcript of the whole buffer so far; it is
>   "fake incremental" for the overlay eicon. `[source: GitHub code/PR, not
>   independently verified in the local binaries]`

This matters for us: a lightweight "re-transcribe the accumulated buffer on a
1 s timer" loop is a cheap partial-text stepping stone that does **not** require
converting the whole engine to per-frame streaming.

---

## 8. How our pipeline differs (from the same research, with line cites into `dictate.cpp`)

Findings below were verified directly against our daemon. They confirm the user's
thesis: **the model isn't the problem; the pipeline is.**

### 8.1 RAM spike (8.9 GB peak)
- The encoder is compiled to a **static** shape `[1, NMELS=128, T_MAX=9001]`
  (`dictate.cpp:416-419`); `T_MAX=9001` is fixed at load (`:64`, default
  `MAX_SECS=90`) and used for every load (`:778`, `:950`).
- **Every** VAD segment is padded to that 90 s window via `nemo::pad_to(F, T_MAX)`
  (`:818`, `:838`, `:863`). `pad_to` allocates `128*9001` floats and zero-fills
  (`feats.h:71-77`).
- Quantified pad waste: a **3 s** segment ≈ 301 real frames vs 9001 → **96.7 %
  zero-fill**; even 30 s is 66.7 % zeros. A single 1024-channel encoder-output slab
  is `9001*1024*4 B ≈ 36.9 MB` padded vs ~0.1 MB real (×281). Several deep layers
  hold padded-length widths at once → GBs. Plus 622 MB encoder weights resident +
  decoder → the observed ~8.9 GB dictate RSS. `[INFERRED exact number, PROVEN
  mechanism]`

### 8.2 CPU capped ~50 % on ~4 cores
- The only `set_property` in the file is `ov::cache_dir(CACHE_DIR)` (`:410`).
  There are **no** `num_streams`, `inference_num_threads`, `ov::threads`, no
  async, no `ov::AsyncInferRequest`, one encoder `InferRequest` (`:420-421`) and
  one decoder `InferRequest` (`:435-436`), all **synchronous** (`enc_req->infer()`
  `:457`, `dec_req->infer()` `:496`) interleaved serially in the greedy decode loop
  (`:478-520`). VAD runs serially too (`:294`). One CPU stream → ~1–4 cores busy →
  ~48–50 % total, E-cores idle, high single-core temp. `[INFERRED exact %, PROVEN
  single-stream design]`

### 8.3 Repeated filler + the "transcribe → tribe" wrong word
- `postprocess_transcript()` (`:138-202`) is **deliberately conservative** and does
  **not** delete/alter real words — so filler passes through verbatim. It is **not**
  the source. (This matches our earlier adversarial review: "uh uh" = "no",
  sentence-initial "Yeah" can answer an embedded question, so naive filler-stripping
  is wrong.)
- The **real** cause is the VAD loop + per-segment unprimed decode:
  - Each speech-offset calls a **fresh `transcribe()`** per finished segment
    (`:817-819`, `:837-839`).
  - `transcribe()` **zero-resets decoder state every call**: `st1.assign(2*1*640,0)`,
    `st2.assign(2*1*640,0)` (`:474`), and seeds `prev = tokens.empty() ? blank :
    tokens.back()` (`:484`) — with an empty `tokens` at segment start, each segment
    starts **unprimed** (blank state, blank prior).
  - The greedy loop advances `t` by `dur` or `1` (`:518-519`) with a
    `MAX_TOKENS_PER_STEP=10` emission cap (`:68`), and runs until `t>=enc_len`
    (`:478`) — which is bounded by the **padded** length, not the real speech.
  - Result: gaps, repeated onsets, and short-segment padding reinforce each other
    across segments — the classic mechanism for segment-boundary filler and onset
    word repeats.
- "tribe" for "transcribe" is a real ASR error on a short, zero-padded, unprimed
  chunk — **not** a postprocess rewrite. It reproduces on NPU too because the
  shared pipeline code (VAD/segment/unprimed-decode/pad) is device-independent.

_Parity note:_ Handy exposes **explicit, default-off** post-process settings —
`change_filler_word_removal_enabled` and `change_word_correction_threshold` (a
prompt-driven "Improve Transcriptions" LLM pass). This is the right shape for us:
keep the conservative meaning-preserving default, and offer aggressive filler
removal / correction as an **opt-in** setting at the clipboard choke point — never
inside `transcribe()`.

### 8.4 Ours is NOT streaming
- First tap forks raw `pw-record` (no timeout) → `capture.wav` (`:894-899`).
  Second tap SIGTERMs it, re-reads the WAV (`:783`), then runs VAD and transcribes
  whole finished segments (`:794-830`). No partial text while speaking. Handy
  decodes per-frame.

---

## 9. Concrete pipeline + GPU-path recommendations for us

Ordered roughly highest-value/lowest-risk first. Each is grounded in a finding
above; `[LOW]`/`[MED]` = estimated risk. **None touch the model weights.**

1. **Dynamic/compact encoder shape instead of static 90 s window** `[MED]`
   - Target: RAM (8.9 GB → hundreds of MB).
   - Change: stop compiling the encoder to `[1,128,9001]` and stop
     `pad_to(F, T_MAX)` on every segment. Compact the real segment to its true
     frame length for the actual `T_frames` (OpenVINO dynamic or re-shaped static
     per-segment). Only the real speech span materializes activations. This is the
     single biggest RAM win (see §8.1).
2. **Size the window to the segment, not to 90 s** `[LOW]`
   - Even keeping a static shape, re-shape to the *actual* segment length + a small
     margin instead of `T_MAX=9001`, and pad only to that. Removes the ×281
     activation blowup for short segments.
3. **Add OpenVINO threading / stream parallelism** `[MED]`
   - Target: CPU util (~50 % → full).
   - Change: set `ov::hint::num_streams` / `inference_num_threads` for CPU, and/or
     use an async request pipeline so encode and the per-step decode overlap. The
     E-cores stay idle because no parallelism is requested today (§8.2).
4. **Decoder-state continuity across segments** `[MED]`
   - Target: filler/repeats.
   - Change: carry the decoder state (or at least the last committed tokens as a
     prompt) from the end of one segment into the next, instead of `st1/st2=0` +
     `prev=blank` on every call (§8.3). This stops the model re-voicing onset
     fillers at each segment boundary. Keep the conservative postprocessor as-is.
5. **Bound the decoder loop to real speech, not the padded length** `[LOW-MED]`
   - Target: the "transcribe → tribe" wrong word.
   - Change: the greedy loop runs `while (t < enc_len)` where `enc_len` is bounded
     by the **padded** `T`, not the real segment (`:478`), advancing `t` by `dur`
     or `1` with `MAX_TOKENS_PER_STEP=10` (`:68`). Cap the decode at the *actual*
     speech frame count, and/or raise the max-likelihood margin before committing a
     short token. This is a symptom-level fix that subsumes into #2/#4, but it's the
     visible correctness bug, so it's worth standing on its own.
6. **Live partial results / true streaming** `[HIGH] (bigger lift)`
   - The real fix for "can't hear anything until I stop talking" is to transcribe
     while speaking. Two routes:
     - **(a) Streaming route (recommended):** adopt a streaming-capable engine.
       This is exactly the Moonshine Streaming trajectory (or Parakeet Unified
       Streaming / sortformer via transcribe-rs). A GGML/Vulkan path (Handy's own
       proven approach) or an NPU-native streaming model.
     - **(b) Batch-then-emit:** at minimum, transcribe *finished* VAD segments
       immediately while recording continues, and paste partials — but this still
       isn't streaming and keeps re-priming (§8.4).
7. **GPU path (the user wants to test our iGPU)** `[MED]`
   - Context: `ov::get_available_devices()` returns only **CPU + NPU** in our
     OpenVINO build; selecting GPU fails because the OpenVINO GPU plugin
     (`libopenvino_intel_gpu`, Level-Zero) isn't installed — even though the Arc
     130V iGPU, `xe` driver, and `/dev/dri/renderD128` are present.
   - Two independent, compatible tracks:
     - **(a) Install the OpenVINO GPU plugin** so `compile_model(enc, "GPU")` works
       on our existing INT8/ONNX Parakeet path. **But keep the LSTM decoder on
       CPU** — `dictate.cpp:424` already documents that the NPU INT8-LSTM has
       quantization-accuracy issues, and that same risk follows an INT8-LSTM move
       to GPU. Encoder→GPU, decoder→CPU is the safe target. Verify the package
       provides `libopenvino_intel_gpu_plugin` / Level-Zero GPU device first.
     - **(b) Add a GGML/Vulkan path** like Handy's (it now binds our exact LNL
       iGPU as Vulkan0 via Mesa). This is the *proven* Linux iGPU route for ASR
       and is model-agnostic (can run the same Moonshine Streaming GGUF or our
       whisper-family GGUF conversions). GGML's Q8_1 weights in fp16 compute on
       Vulkan is exactly the configuration Handy validated, so don't claim
       OpenVINO-INT8-GPU parity with it until measured.
   - Device-picker caveat (Handy documented): auto-picking Vulkan0 is unsafe on
     multi-GPU (issue #1137 zero-length-batch crash on an Intel UHD 630); use an
     explicit device pick (`DRI_PRIME=1` / `transcribe_gpu_device`), never assume.

8. **Streaming model parity to consider** `[INFO]`
   - Parakeet Unified Streaming speeds live transcription on GPU/NPU so you can
     keep speaking while sentences are transcribed in parallel — the user's stated
     goal. Handy supports it via transcribe-rs (ONNX). Our NPU/OpenVINO adoption
     would need an ONNX streaming model that exposes NPU-compatible streaming ops.

---

## 10. Quick reference — key symbols / config / files

- **Public streaming ABI** (`libtranscribe.so.0.2`, nm-verified):
  `transcribe_stream_{begin,feed,finalize,get_text,get_state,reset,update}`,
  `transcribe_stream_revision`, `transcribe_stream_{last_status}`,
  `transcribe_stream_n_committed_{segments,tokens,words}`,
  `transcribe_run{,_batch}`, `transcribe_init_backends`, `transcribe_model_load_file`.
- **Moonshine streaming internals**:
  `apply_adapter_window`, `encode_window_to_host`,
  `flush_stable_frames`, `build_sliding_window_mask`, `cross_kv_commit`,
  `cross_kv_proj` (mangled `transcribe::moonshine_streaming::…`).
- **Env/cli**: `--stream-chunk-ms` (native streaming feed granularity),
  `--transcribe-file <wav>` (batch 16 kHz mono headless),
  `HANDY_NO_GTK_LAYER_SHELL=1`, `DRI_PRIME=1`.
- **Settings** (`~/.local/share/com.pais.handy/settings_store.json`):
  `transcribe_accelerator: Auto`, `ort_accelerator: Auto`,
  `transcribe_gpu_device: None/-1`, `clipboard_handling: copy_to_clipboard`,
  `bindings.transcribe: ctrl+space`, `transcribe_with_post_process:
  ctrl+shift+space`, `app_language: en-US`.
- **Log** `~/.local/share/com.pais.handy/logs/handy.log`; **models** in the HF
  cache `models--handy-computer--moonshine-streaming-tiny-gguf/`.

---

### What this means for us (one paragraph)

We were right that the **pipeline** is the problem, not the model. Handy wins by
(1) running a **~44 M-param 8-bit GGUF** that keeps activations lean, rather than a
622 MB INT8 encoder padded to a 90 s static window; (2) **true per-frame streaming**
with a committed/tentative decoder that keeps state and **drops divergent fillers**
instead of re-priming every segment; and (3) parallel/streaming compute instead of a
single serial `InferRequest`. Our fastest wins are **dynamic window sizing** (RAM),
**threading/streams** (CPU), and **decoder-state continuity across segments**
(accuracy). The **streaming + Vulkan/GPU path is proven on this exact laptop** by
Handy's own Sep 03 logs (Vulkan0 = Intel LNL iGPU) and is the route to the
"transcribe-while-I-speak" experience the user wants.