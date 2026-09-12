# ASR Project — VIP (Very Important Plan)

> **What this file is:** the single source of truth for the offline ASR project.
> Every instruction below has a detailed description. We work **one instruction at a time,
> only when Shlok says so**. Nothing here runs without explicit go-ahead.
> Location: `~/Projects/omarchy-asr/VIP.md` (which is `/mnt/shared/Projects/omarchy-asr/VIP.md`,
> shared with Windows 11 on the dual-boot). Created 2026-09-04.

## 0. Current truth (snapshot, 2026-09-04 ~16:20 +0545)

- **Active selection:** `state/model.txt = whisper-tiny`, `state/device.txt = NPU` (user is back on tiny for short clips).
- **`state/devices.json`:** parakeet-tdt-v3-ov [CPU,GPU], parakeet-v3 [CPU,GPU], qwen3 [CPU,GPU,NPU], whisper-base/small/tiny/turbo [NPU].
- **Model sizes (measured):** parakeet-tdt-v3-ov **1.2 GB** (Intel OV IR, the active Parakeet) · whisper-large-v3-turbo **566 MB** · qwen3-asr-0.6b-int8 **1.3 GB** · parakeet-v3 GGUF **706 MB** (symlink into HF cache).

  **2026-09-05 update — two new live models wired in (per user request "plug those models into my ASR"):**
  - `parakeet-stream-unified-en-0.6b` (Q8_0 GGUF, **698 MB**) — streaming-capable parakeet via libtranscribe + ggml Vulkan on iGPU. One-shot path proven through the daemon dispatch (English, "What is up", 263ms transcribe). True stream-mic integration (200ms ring + VAD 30s auto-stop) is follow-up work.
  - `nepali-indicwav2vec` (OV IR, **181 MB** with decoder/lm_head.npy 248KB + vocab.json) — `sumanpaudel1997/nepali-asr-indicwav2vec` ONNX-exported then ovc'd. Proven real Devanagari on GPU and CPU through the full daemon dispatch on `nep_trim1.wav`. NPU compile failed (dynamic-shape ONNX; static-shape retry queued).
  - `indicwav2vec_transcribe.py`: `f09e76a225bf4dee8e6845ab307d02d8` (2026-09-05, 181M helper; `|` token stripped to space, ensure_ascii=False, CTC greedy decode).
  - `parakeet_stream_transcribe` (binary): `a628ccf75dee075e4a45f7bcd02ecb42` (2026-09-05, C++17 + libtranscribe 0.2 streaming API; L=5600/C=1040/R=1040ms chunked-attention ext; one-shot and --stream both verified).
  - `state/devices.json`: `75534dfc01c41b7543596281bf2e8d02` (2026-09-05, added parakeet-stream + nepali-indicwav2vec entries [CPU,GPU]).
  - `Model.js`: `8c862d22af3925433605b292ca86045e` (2026-09-05, MODEL_DISPLAY_NAMES extended: "Parakeet Unified (Streaming)" + "IndicWav2Vec (Nepali)"; glyphForModel extended for nepali/indicwav2vec).
  - `~/.local/bin/dictate`: `857a049012fc958bfa7df08cae5e014b` (2026-09-05, model_backend() + dispatch + new transcribe_via_helper_bin() for the C++ helper).
- **Daemon:** `omarchy-npu-dictate.service` runs `%h/.local/bin/dictate --max-secs 300 --daemon`. `MAX_SECS = 300` in source + `--max-secs 300` in the systemd unit (both updated 2026-09-05). 5-minute cap protects against accidental long holds while allowing natural longer dictation.
- **Model lineup (2026-09-05, user order "delete every other model"):** live dir holds `whisper-base-en` (61M), `whisper-large-v3-turbo-int4` (456M), `parakeet-v3` (⚠️ GGUF symlink is BROKEN — blob missing from HF cache, not loadable as-is), `vad`. Deleted: turbo-int8 `whisper-large-v3-turbo` (broken `beam_idx`), `small-en`, `tiny`, `qwen3-asr-0.6b-int8`, `parakeet-tdt-v3-ov`. Panel labels updated in Model.js to: Whisper Base / Whisper Large V3 Turbo / Parakeet V3 Streaming. devices.json rebuilt to match survivors.
- **Nepali trim experiment (2026-09-05):** downloaded + converted `alphaedge-ai/whisper-base-nep-{16384,32768}` to OpenVINO (FP16/INT8/INT4) in isolated lab `~/Projects/omarchy-asr/nep-base-lab/`. RESULT (RESULTS.md): trim is NOT an accuracy win over turbo — 0–4% word acc vs turbo INT4's 25% on the real clip. `repetition_penalty=1.2` IS supported on whisper-base (fixes loop) but doesn't rescue accuracy. Turbo INT4 stays the best Nepali transcriber. Lab is isolated; live ASR untouched.
- **INT8 Nepali base — INTEGRATED then REMOVED (2026-09-05).** The int8 base (`whisper-base-nep-int8`, 63MB) was integrated as a live panel model ("Whisper Base Nepali"), adding `repetition_penalty=1.2` for any `whisper-base*` model + the `--language`/`lang_to_id` force. **Then removed (same day).** User verdict: turbo-int4 is "(SORTA) works" — the only Nepali model worth keeping; the base's Devanagari is accurate but its low 0–4% word-accuracy isn't usable. Removal: deleted `models/whisper-base-nep-int8/`, stripped the `devices.json` entry + `Model.js` "Whisper Base Nepali" label, and retargeted `state/model.txt` (which pointed at it) → `whisper-large-v3-turbo-int4`. Surviving lineup: **base-en / turbo-int4 / parakeet-v3**. The lab (`~/Projects/omarchy-asr/nep-base-lab/`) still holds the trim for future experiments. Key lesson kept: on OpenVINO genai, `repetition_penalty` and a forced `language` are NOT mutually exclusive (passing BOTH gives clean non-looping Devanagari); only English-only base-en rejects any `language`.
- **GGUF Q8_0 iGPU experiment (2026-09-05):** NEGATIVE. Q8_0 quantization of nep-base-32768 works (64MB `ggml-model-q8_0.bin` via whisper-quantize, with a tensor-name shim because whisper.cpp loaders drifted from their own converter). But **whisper.cpp cannot load a 24881-vocab trim** — its hardcoded `num_languages()=n_vocab−51765` gives `n_langs=−26884`, decode aborts `GGML_ASSERT(i01>=0 && i01<ne01)` in get_rows. Also the iGPU/Vulkan build needs the system Vulkan SDK (headers+glslc+SPIRV-Headers) = root-only. Verdict: OpenVINO INT8 is the only viable runtime for this trim. See `~/Projects/omarchy-asr/nep-base-lab/RESULTS_GGUF.md`.
- **Panel:** plugin `shlok.asr` v1.0.0, entry `Panel.qml` (bar-widget). Single copy confirmed (see §4).
- **Qwen status:** PARKED by user order. Zero Qwen processes running. No Qwen work until user re-opens it.
- **Nothing heavy running:** verified no python test jobs alive; daemon idle (~164K RAM, 25ms CPU).
- **DEVANAGARI `\u` PASTE BUG — ROOT CAUSE + FIXED (2026-09-05).** Dictated Devanagari (e.g. `बेकार`) reached `wl-copy`/paste as literal `\uXXXX` text. Root cause: **both Python helpers call `json.dumps(out)` with default `ensure_ascii=True`**, serializing every non-ASCII codepoint as `\uXXXX` ASCII bytes; the C++ `transcribe_via_helper` raw `find`/`substr` extractor (and `transcribe_ggml`) then copies those bytes verbatim into `wl-copy`. Why invisible in English: ASCII round-trips identically; only non-ASCII scripts (Devanagari **and German** — `über` → `über`) break. Backends affected: `whisper`, `qwen3`, **and `parakeet-ggml`** (it shells to `whisper_transcribe.py` as its bridge). `parakeet-ov` native C++ `ids_to_text()` bypasses JSON entirely = unaffected. **Fix:** `ensure_ascii=False` on all `json.dumps(out)` in both helpers → raw UTF-8, which the verbatim slice+`fwrite` round-trips byte-for-byte. Verified: turbo/ne emits `आज़ा मौसम…` directly, C++ slice returns byte-identical UTF-8 (no `\u`). No daemon rebuild needed (helpers are spawned per transcription). Report: `/home/shlok/Downloads/devnagari-pate-bug-report.md`. C++ JSON-parser hardening (nlohmann/json) is optional future work, NOT done.

## 1. Folder inventory — answering "how many folders do we really have?"

| Role | Path | Notes |
|---|---|---|
| LIVE plugin (what the bar shows) | `~/.config/omarchy/plugins/shlok.asr/` (`Panel.qml`, `Model.js`, `manifest.json`, `assets/*.svg`) | Only ONE `Panel.qml` on the whole system — verified via find. |
| LIVE daemon workdir (state + models + helpers) | `~/.local/share/npu-asr/` (`state/`, `models/`, `whisper_transcribe.py`, `qwen3_transcribe.py`, `src/`, `ggml/`, `cache/`) | This is what the running daemon reads. |
| LIVE binary (what systemd runs) | `~/.local/bin/dictate` (md5 `f8570ed2…`, built Sep 4 13:18) | ⚠️ DIFFERS from `~/.local/share/npu-asr/dictate` (`145d2840…`) and `~/…/npu-asr/src/dictate` (`8da602ec…`) — three different binaries. |
| LIVE source | `~/.local/share/npu-asr/src/dictate.cpp` (+ `build.sh`, probes, `Handy.md`, old `.bak`) | |
| STALE / DANGER copies | `~/.local/share/npu-asr-vad-fix/dictate.cpp`, `~/.local/share/npu-asr/src/.claude/worktrees/typed-snacking-quokka/dictate.cpp` | Old experiments. Never deployed, but confusing. Delete or archive during Task 4. |
| Shared, publishable home (NEW) | `~/Projects/omarchy-asr/` → `/mnt/shared/Projects/omarchy-asr/` (NTFS, visible from Windows) | Currently holds only this VIP.md. Destination for the consolidated repo (Task 4). |
| Docs already in workdir | `~/.local/share/npu-asr/{README.md, BUILD_NOTES.md, IMPLEMENTATION_SUMMARY.md}` | To be merged into the consolidated repo, not kept as three separate truths. |

**On the "theme-aware for a short span, then butchered again" suspicion:** for the PANEL the suspicion is cleared — there is only one `Panel.qml`, so no old panel file overlapped it. What likely happened: the SVG recolor edit landed while quickshell still held the old SVGs/textures, then a shell restart re-decoded the SVGs and exposed that the recolor approach (white-base + MultiEffect colorization) only works on light themes. For the DAEMON the suspicion is CONFIRMED in spirit — three different `dictate` binaries exist and only `~/.local/bin/dictate` is live, so rebuilding in `src/` without reinstalling changes nothing the user hears.

## 2. Instructions (work one at a time, on Shlok's call)

### INSTRUCTION 1 — Control-panel UI only (UI ≠ full control unit). STATUS: COMPLETE — 1a LOCKED · 1b LOCKED · 1c LOCKED · 1d LOCKED (dark Aether + light both user-verified 2026-09-04).

**1a. ACCELERATOR header row — "Powered by" on the same line.**
Today: `Panel.qml` (~L377) renders `PanelSectionHeader "ACCELERATOR"`, then a SEPARATE right-aligned Row below it with intel.svg + text "Powered by Intel" (opacity 0.5). User calls this weird/dangling.
Target: ONE horizontal row — `ACCELERATOR` left-aligned; on the right side of the SAME row, small dimmed text `Powered by` + the Intel SVG immediately to its right. No explicit "Intel" word (the logo says it). Intel SVG height MUST equal the `Powered by` font height (cap-height match), width scales from its wide viewBox (22.27×9.5) via `PreserveAspectFit`. Same baseline, vertically centered with the header text.

**1b. Accelerator buttons — icon + label as one centered group.**
Today: three `BorderSurface` cells (CPU/GPU/NPU) each with an inner Row(icon Item `title*2` + Text `bodySmall`). Complaint: icons dominate, labels look small, neither horizontally nor vertically centered; "Inte" truncation seen in light theme; GPU/NPU/CPU logo-letter collisions.
Target: keep the icon size (user says icon size is perfect). Fix by (i) making the label vertically centered to the icon (`anchors.verticalCenter`, `Text.AlignVCenter`, matching line height), (ii) treating icon+label as ONE group and centering the GROUP in the cell (`anchors.centerIn`), (iii) giving the group breathing room so nothing truncates (cell `clip: false`, no fixed narrow width on the label, `elide: ElideRight` only as last resort), (iv) uniform `sourceSize` for crispness, with GPU's wider viewBox compensated, not stretched. Sketch of intent: `[ ○ CPU ] [ ◉ GPU ] [ 8 NPU ]` — icon left, label right, group centered.

**1c. Theme-aware icons — restore the "perfect for a short span" state, permanently.**
Today: all 7 SVGs have `<svg fill="#ffffff">` root + `fill="currentColor"` paths, colorized via `MultiEffect { colorization: 1.0; colorizationColor: root.bar.foreground }` (vendor chips use `Qt.lighter(foreground, 1.25)`). Works on light, blows out on dark. Reference pattern (Omarchy Tray.qml): base `Image` hidden + `MultiEffect` sampling it as texture, `colorizationColor: root.foreground`, `visible` gating for symbolic icons; device SVGs must be single-color white-base silhouettes for this to work. Intel/NVIDIA/OpenAI/Qwen vendor logos are NOT silhouettes — decide per logo: silhouette-colorize (CPU/GPU/NPU) vs original-brand-colors (vendor logos), and apply consistently in both ACCELERATOR buttons and MODEL chips (which today also have the `chipRow.supported.indexOf(modelData)` scoping bug — inner Repeater shadows `modelData`; must use the delegate's `dev` variable).
Acceptance: screenshots in dark (Aether) AND light themes, icons legible, no blowout, no truncation.

**1d. Prove no stale-file overlap for the panel.**
Do after 1a–1c: `find` for Panel.qml copies, `md5sum` panel + assets, restart quickshell, re-screenshot both themes. Record hashes in Appendix B so any future "it was fixed then broke again" can be settled in seconds.

### INSTRUCTION 2 — Prove the panel actually controls the daemon (real-time control integrity). STATUS: pending.

User's doubt: "I do not trust the control panel… what if despite choosing different models it's running the same version?"
Meaning: switching device/model must deterministically change what the daemon does on the NEXT record press (nothing preloads until record is pressed — that part stays). Work: (i) map every panel write (`state/enabled`, `state/device.txt`, `state/model.txt`, `state/offload`) to the exact daemon read point (`resolve_routing` / model_backend dispatch in `dictate.cpp`); (ii) add a visible confirmation (e.g. status/tooltip echoing `model + device` the daemon actually loaded, plus a log line per toggle); (iii) test matrix: change panel → press record → say 5s → confirm log + transcription used the newly selected pair; (iv) fix `devices.json` generation so it is WRITTEN by the daemon/probe, not hand-edited (today it is stale-prone and drives all panel dimming via `Model.devicesForModel`).

### INSTRUCTION 3 — Model + pipeline fixes (the transcription problems). STATUS: OPEN — user order 2026-09-04: (1) Whisper Base chunking → (2) Whisper Turbo → (3) Parakeet OV then streaming GGUF → (4) Qwen3 last. HF identities confirmed this session (see 3i).

Whisper family (NPU, via `whisper_transcribe.py` + `openvino_genai.WhisperPipeline`):
- **3a. small-en TRUNCATES long speech.** Evidence: user read a ~30s paragraph with pauses; got first ~3 sentences + looping tail ("It felt like the perfect / It felt like the perfect moment for the world was still asleep. It felt like the perfect"). NPU monitor shows 20–30% during runs.
- **3b. tiny SEEMED to survive longer once, then showed the same truncation.** One long test passed with minor accuracy loss ("heart" for "hard"); later the user confirmed tiny also truncates. Hypothesis to test: max-window differs by model size (e.g. small ~30s, tiny ~60s) — verify with timed recordings, don't guess.
- **3c. base-en SAME truncation class** ("said much more but it couldn't transcribe after that"). User also doubts the panel (see Instruction 2): verify base/small/tiny actually load different blobs.
- **3d. large-v3-turbo (566 MB, the user's pick — sub-1GB, ~500MB target) transcribes NOTHING** (`[]`). Highest priority inside Instruction 3 once opened: get one clean turbo transcription on NPU first, then probe CPU/GPU and update `devices.json` if they work.
- **3e. Prime suspect (user's own diagnosis, very plausible): the pipeline feeds the NPU everything in one batch.** NPU util SPIKES a few times then sits at 0% — consistent with one big single-shot inference capped at the model's max window, the tail beyond the window silently dropped. Fix direction: chunked feeding — split long recordings on VAD/silence into ≤ model-window segments, transcribe each, stitch with overlap handling, then paste once. Also surface per-chunk NPU util so the monitor shows sustained use. (Daemon already records up to 90s and has VAD segmentation for Parakeet — reuse the pattern.)

Parakeet family:
- **3f. parakeet-v3 (non-OV GGUF, 706 MB): outputs only `,`** on BOTH GPU and CPU no matter what is said. Backend suspect (ggml/libtranscribe path), not the acoustics.
- **3g. parakeet-tdt-v3-ov (1.2 GB OV IR): runs on CPU AND GPU but transcription is trash.** Evidence GPU: "open, oursion of the racket" (openvino parakeet), "GUE" (GPU), "do" for "does". Evidence CPU ("CUU"): same class — "samodel", "C U.Fill will take more time than GU. U". Hardware runs, words come out, quality is bad. Suspects: feature/mel mismatch or decoder params on the OV path. Note: earlier the non-OV/OV naming confused things — OV = the 1.2 GB Intel-converted one (active), non-OV = the 706 MB GGUF (broken `,` output).

Qwen3-ASR-0.6B-int8 (1.3 GB) — REVISIT LAST (#18):
- **3h. CPU: unicode garbage + RAM 5 GB → 13 GB.** Sample: `-ĊĊĊĊĊĊĊ!ĊĊĊ…#…`. **GPU: also garbage**. **NPU: nothing** — bar shows Recording, never flips to Transcribing. Identity: `Echo9Zulu/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO` (mixed: encoder FP16, thinker+decoder INT8, OpenArc; README silent on NPU). Author notes "stateful" KV-cache-on-device — our qwen3_transcribe.py may mishandle it. Wanted partly for the logo (Whisper/OpenAI + Parakeet/NVIDIA + Qwen = full logo trio).

**3i. HF identities confirmed 2026-09-04:** base-en=`OpenVINO/whisper-base.en-int4-ov` (INT4_ASYM, 61 MB); turbo=`OpenVINO/whisper-large-v3-turbo-int8-ov` (INT8_ASYM, 566 MB, d1280/enc32/dec4, 128 mels, MULTILINGUAL forced `[1,null],[2,50360]` — null language slot!); parakeet-OV=`FluidInference/parakeet-tdt-0.6b-v3-ov` (README: 1.1B V3 FastConformer-RNNT, 10s chunks+3s overlap, NPU 25.7×/CPU 5-8× on 155H, blank id 8192); parakeet-GGUF=`handy-computer/parakeet-tdt-0.6b-v3-gguf` Q8_0 (706 MB; daemon's transcribe_ggml wrongly shells to whisper helper — explains `,`). Env: OpenVINO 2026.3.1 + genai 2026.3.1 (turbo needs ≥2026.1 — OK). Monitor tools: btop / nvtop / NPU busy_time_us loop.
**3j. Turbo-specific suspect:** daemon never passes `--language`, turbo's null language slot may sit undecided → `[]`. First test: `language="en"` explicitly.
**3j2. RESOLVED 2026-09-05 — turbo rejects `language=` kwarg entirely.** OpenVINO WhisperPipeline's C++ `lang_to_id` is populated from the EXPORT runtime, NOT generation_config.json. turbo-int4's JSON lists `<|ne|>': 50313` but passing `language="ne"` CRASHES: `RuntimeError: Check 'lang_to_id.count(*language)' failed ('language' ne must be provided in 'lang_to_id' map)`. FIX (whisper_transcribe.py): resolve id from lang_to_id ∪ added_tokens.json, load from a SYMLINKED RUNTIME COPY with forced_decoder_ids `[[1,None],[2,<id>]]` — bypasses the C++ check entirely. ids: en 50259, de 50261, ne 50313, gu 50333. Panel Language section (en/ne/de/Auto) writes state/language.txt.
**3k. Chunking root cause (user's gut CONFIRMED by docs):** daemon feeds up to 90s (N_MAX=1440000) into ONE `generate()`; genai WhisperPipeline is short-form oriented (streamer only <30s), native window 30s/3000 frames → tail dropped, NPU spikes then 0%. Fix: Silero-VAD segments (450ms pre-roll/hangover exist) → ≤30s chunks → transcribe → stitch + trailing `" "` (user request). Same principle as Parakeet README's 10s+3s chunks. Whisper CPU/GPU also to be tried (same helper + `--device`, then devices.json update).

Constraints for all of Instruction 3: keep the winner **< 1 GB, ideally ~500 MB** (turbo fits); **GPU acceleration genuinely wanted**; CPU nice-to-have for UI completeness ("looks neat giving every option"); offline, free, battery-friendly — the whole point (see §3 Why).

### INSTRUCTION 4 — One home, shared with Windows, publishable. STATUS: pending.

- Consolidate: ONE repo (this folder) containing plugin (`Panel.qml`, `Model.js`, `manifest.json`, `assets/`), daemon (`src/dictate.cpp`, `build.sh`, helpers), `models/` manifest (not blobs), `state/` contract doc, install + systemd unit, and this VIP.md. Archive `npu-asr-vad-fix/`, the `.claude/worktrees/` copy, `*.bak`, and one-off probes out of the live path.
- Keep it Windows-visible: repo lives here (`/mnt/shared/Projects/omarchy-asr/`, NTFS) so Windows 11 sees it too; keep filenames + scripts portable (no hardcoded `/home/shlok` in shipped code — env/`%h` only).
- Publish-ready: README with Intel-chip prerequisites (CPU/GPU/NPU via OpenVINO), model download table with sizes, install/uninstall, theme screenshots (dark+light), license. Goal: another Intel laptop owner can clone and run.
- Install rule going forward: `src/` builds → installs to `~/.local/bin/dictate` → `systemctl --user restart omarchy-npu-dictate.service` → hash recorded. No more mystery binaries.

## 3. Why this project exists (user's words, condensed)

NPU sits idle; Windows uses ~5% NPU for mic noise-filtering ("studio mic" on a stock laptop mic) — want that class of local-AI utility on Omarchy. Cloud transcription (ChatGPT) now rate-limits free use — want free, offline, battery-friendly voice typing. ASR is the first workload; audio filtering and more later.

## Appendix A — verbatim model evidence (user-pasted, kept for regression tests)

- whisper-small-en (NPU), prompt ~30s with pauses → got: "The morning sun began to clear the low hills casting long shadows across the empty road. A quiet breeze moved through the trees making the green leaves dance softly in the light. Down by the water a few birds were calling out to each other while the rest of the world was still asleep. It felt like the perfect It felt like the perfect moment for the world was still asleep. It felt like the perfect" (FULL prompt in §2-3a context; tail missing + loop).
- whisper-tiny (NPU), long bracket test → full-length output with small errors ("heart" vs "hard"); later same truncation observed on longer takes.
- whisper-base (NPU) → cuts off like small; user: "I said much more but it couldnt transcribe after that."
- parakeet-v3 non-OV (GPU and CPU) → `[, ]` regardless of speech.
- parakeet-tdt-v3-ov (GPU) → "So, Yeah. Ye And now I am using the open, oursion of the racket, right? … GUE do run…"; (CPU) → "Well Now I have switched the samodel but kept the acceleration to CUU… C U.Fill will take more time than GU. U".
- whisper-large-v3-turbo → `[]` (nothing).
- qwen (CPU) → `-ĊĊĊĊĊĊĊ!Ċ…` + RAM spike; (GPU) → ```Ġå…```; (NPU) → `[]`, stuck on Recording.

## Appendix B — hashes & versions (fill after each change)

- `Panel.qml`: `ea4b7d2fa43659bdc48b39c41102bfd6` (2026-09-05, post-truncated-lineage; labels + language section)
- `Model.js`: `07fd061742b4ded38462064e7dd33d98` (2026-09-05, MODEL_DISPLAY_NAMES = Base/V3Turbo/Parakeet-Streaming)
- `whisper_transcribe.py`: `aba3b7e823942c4e4aebfbe278007918` (2026-09-05, added `ensure_ascii=False` on both `json.dumps`; prior `bf550220…` was rpen=1.2 base + en→auto-detect)
- `qwen3_transcribe.py`: `c912977f0a4a7f59069ae0227834c066` (2026-09-05, added `ensure_ascii=False` on both `json.dumps`)
- Model `whisper-base-nep-int8`: **REMOVED 2026-09-05** (was enc `fbbfbca…` / dec `05f405c…`, 63MB, INT8_ASYM). Trim still in `~/Projects/omarchy-asr/nep-base-lab/` for experiments.
- assets: cpu `6f4c8a8b…` · gpu `0571a33f…` · intel `ede275ed…` · npu `0fd61f19…` · nvidia `95b5f7d7…` · openai `63333c40…` · qwen `a79d8a8c…` (all white-base `#ffffff`, no currentColor — intel recipe)
- `~/.local/bin/dictate`: `a3b0b14a5de6c862fdbca93febafa682` (2026-09-05 15:57) — live, `--max-secs 300`.
- `~/.local/share/npu-asr/dictate`: `145d28402a9f879393a6358043985c89` — stale?
- `~/.local/share/npu-asr/src/dictate`: `8da602ecf0e81d046f9075609a389966` — stale?
- Theme when screenshots taken: dark `Aether` + light, BOTH user-verified 2026-09-04 (1d). Single Panel.qml confirmed; shell ping `ok`.
- Models discussion (next, per user 2026-09-04): supervise EACH model one at a time — Whisper family (tiny/base/small/turbo-V3 truncation + turbo `[]`), Parakeet OV quality + GGUF `,`, Qwen PARKED. Open Instruction 2 (control integrity) and 3 (pipelines) only on call.

**3j3. RESOLVED 2026-09-05 (final) — language FORCE via lang_to_id map.**
Passing forced_decoder_ids slot 2 via a runtime copy does NOT steer turbo's language.
The WORKING force is `generate(language="<|ne|>", lang_to_id=<map>)` where map =
generation_config.json["lang_to_id"]. The C++ WhisperPipeline's internal lang_to_id is
EMPTY on turbo-int4, so supply it explicitly. VERIFIED on real Nepali: ne->Devanagari,
en->English, de->German, auto->Gujarati loop. Helper updated. Panel Language=Nepali
means Devanagari now. (3j2's runtime-copy approach was superseded.)
