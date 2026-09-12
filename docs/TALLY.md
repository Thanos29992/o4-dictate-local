# Nepali Lab TALLY

Isolated working dir for the **new Nepali-only ASR model** search (README.md has scope).
Research-only — nothing installed, no live-ASR files touched.

## Nepali Model Shortlist (agent date 2026-09-05)

**Headline finding:** the only *controlled, normalized* multi-model Nepali benchmark that exists
is [Paudel & Sayami, "Comparative Analysis of Multilingual Pre-trained Models for Nepali ASR"
(arXiv 2608.12327)](https://arxiv.org/abs/2608.12327), released May 2026. It fine-tunes six
multilingual pretrained models on the **same** OpenSLR SLR54 (~165 h) corpus under an identical
protocol and reports WER/CER on three test sets (OpenSLR / FLEURS / Common Voice). Its two top
models: **Whisper-Large-v3-Turbo (14.76% WER, 809M)** and **IndicWav2Vec (14.89% WER, 94.4M)** —
a 9× parameter gap, tied, because language-family proximity in pretraining substitutes for scale.
All six checkpoints are on HF under `sumanpaudel1997/`. Benchmarks matter: several "best on HF"
Nepali Whisper cards (e.g. Dragneel small "26.69%") report on the same OpenSLR *test split used in
training*, and the `kriti` 2026 benchmark.json shows the naive `openai/whisper-large-v3-turbo`
zero-shot run pinned to `ne` scores ~111% WER on mixed audio — i.e. the user's ~25% on the turbo
int4 is roughly consistent with real, uncontrolled mic audio and is **not** representative of what
a *tuned* model can do on read speech.

**Verdict on the previous trim:** the tested `alphaedge-ai/whisper-base-nep-{16384,32768}` trim
stays dead — pity (base = 74M, could run anywhere) but its 0–4% on real clips is reproduced-family
bad; nothing in this search changes that call.

### Ranked shortlist

**1. `sumanpaudel1997/nepali-asr-indicwav2vec`** — *top pick.*
- **Size:** 94.4M params; `model.safetensors` **378 MB** (F32) — comfortably sub-500MB.
- **Format:** HF safetensors, `Wav2Vec2ForCTC` (transformers); **no prebuilt OV IR export exists** —
  conversion is a known one-shot (`optimum-intel`/`ovc`), and the audio encoder (conv1d → transformer) is
  a standard openvino exportable graph. CPU/GPU yes; NPU plausible (single CTC pass, no KV cache, no
  autoregressive loop — friendliest possible decode for the NPU one-pass model).
- **Quality:** **14.89% WER / 3.07% CER on OpenSLR** (publicly documented in README model-index); FLEURS
  40.68%, CommonVoice 51.65%. ~400× real-time (RTF 0.0025 vs turbo 0.076 — **29× faster than the user's
  turbo**). Devanagari CTC vocab **verified in the Hub vocab.json** (81 + pad/unk rows: अ..ह, all matras,
  ँ/ं/ः/्/।, ०-९). **License `cc-by-nc-4.0`** (non-commercial).
- **Red flag:** FC4 confusable `र`/`ऱ`; possessive-`को`-word-splitting is the known WER inflation
  (CER 3.07% is the real phonetic accuracy). NC license — fine for a personal dictation system.

**2. `harrrshall/kriti`** (Naamche Labs) — *best NPU-friendliness story, MIT.*
- **Size:** 119M live params (98 MB at F32); shipped `kriti.nemo` **498 MB** (F32 NeMo bundle, incl. what
  looks like fp16 + tokenizers). Sub-500MB on disk as shipped.
- **Format:** NeMo `.nemo` RNNT; runs via the loader (Python, needs torch + AI4Bharat NeMo fork, hashes
  verified). No GPU/NPU native path without a NeMo→ONNX/OV export — the repo is a research-grade loader,
  Do-It-Yourself export.
- **Quality:** frozen 2026 snapshot (3,630 dev utterances, mixed FLEURS/IndicVoices/OpenSLR), MIT-licensed:
  **24.08% punctuation-insensitive WER / 8.29% CER — exact tie with the full ai4bharat Nepali IndićConformer
  RNNT (24.08%) and better than its CTC (25.31%)**, and the only open system to beat everything on raw WER.
  (Note: its snapshot is a *development* view; the `sumanpaudel` numbers are the stricter standardized test.)
- **Red flag:** needs an ONNX/OpenVINO export for GPU/NPU; niche, brand-new (Aug 2026), single-author.
  Still — the Nepali-only 257-row prediction head is a deliberate **anti-Hindi-fication** fix, the exact
  failure the user hit on auto language.

**3. `ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large`** — *easiest offline GPU-to-NPU pipeline.*
- **Size:** 120M params; `.nemo` **523 MB** F32.
- **Format:** NeMo hybrid CTC+RNNT, **dual decoder in one checkpoint** — take the CTC head and you get a
  one-pass, non-autoregressive, KV-free graph identical in spirit to the live Parakeet-TDT 0.6B the daemon
  already drives (already proven: a 0.6B Parakeet ≈ 1.2GB OV runs in ~4GB peak RAM; this is 1/5 the params).
  MIT. `gam30` even publishes **ready ONNX exports** of this exact encoder (FP 493 MB / INT8 **134 MB**) —
  see Honorable #6.
- **Quality:** per `kriti` snapshot: RNNT 24.08% / CTC 25.31% mixed pi-WER; ai4bharat reports ~15–18% WER
  on their own read-speech eval. Natively emits Devanagari (SentencePiece vocab).
- **Red flag:** NeMo→ONNX/OV needs the CTC branch wiring (Kaldi-style `w2l_ctc` beam), and the full RNNT
  decode is heavier than Parakeet's. But this is the least exotic "proven-by-the-stack" path to *actual* Devanagari.

**4. `sumanpaudel1997/nepali-asr-whisper-turbo`** — *drop-in GPU/NPU upgrade to what the user already has.*
- **Size:** 809M params F32 (**3.05 GB** soft — 1.47 GB at bf16/turbo fp16, ~450–500 MB OV-int8).
  Over the 500 MB preference, but within/at the 1 GB cap once compressed.
- **Format:** stock Whisper Pipeline (same genai path as the live turbo-int4 — NPU stateful decoder family).
  It is still a stock multilingual turbo, so forcing the **`ne` language token (`<|ne|>` 50313) via
  `language`+`lang_to_id`** works exactly like the existing E13/E14 helper flow. **License `cc-by-nc-4.0`.**
- **Quality:** **14.76% WER / 3.48% CER (OpenSLR)** — near-identical to IndicWav2Vec but ~29× slower (RTF
  0.076). FLEURS 39.56%, CV 48.35% (best of the whole six-model suite on the hard CV crowdsourced set too).
- **Red flag:** the 14.76% is in-domain read speech; it carries the same ~25%-on-real-mic degradation the
  current turbo has.

**5. `tonibirat/whisper-large-v3-turbo-ne-mixed`** — *anti-hallucination / anti-Hindi-fication turbo.*
- **Size:** 809M turbo fine-tune; ships CT2-INT8 in `ct2/` (**~809 MB**), which is right at the 1 GB cap;
  GGUF/OV conversion is same-as-whisper path (CT2→whisper.cpp via `convert-h5-to-ggml.py` or ONNX).
- **Format:** transformers/CT2. **Multilingual turbo, pinnable to `ne`** via `language`+`lang_to_id`.
- **Quality:** self-reported WER 0.82 / **CER 0.39** on FLEURS(Nepali)+mixed (they correctly flag WER is
  inflated by Devanagari whitespace segmentation; CER is the phonetic signal). Explicitly trained against
  **Hindi-fication (`और`→`र`), 5–10 s truncation, and hallucination loops** — the exact three failure modes
  the user's turbo exhibits on auto.
- **Red flag:** self-reported, small dev-set, no controlled benchmark; QLoRA fine-tune quality is unproven
  against the Paudel standard.

**Honorable:**
- **6. `gam30/nepali-automatic-speech-recognition`** — literally `ai4bharat indicconformer_stt_ne` punched
  to ONNX already: `model_ctc.onnx` **493 MB** FP or **134.5 MB INT8**. License "other" (must credit gam30
  + AI4Bharat), CC-style source usage. Red flag: the ONNX BPE vocab is a Bengali/Assamese-tuned
  SentencePiece — it *contains* 6.9k Devanagari tokens but with 1.5k wrong-script entries mixed in; needs
  a real Nepali-smoke-test before trust.
- **7. `sumanpaudel1997/nepali-asr-whisper-medium`** — 15.57% OpenSLR WER, same as turbo within 1 pp, and
  769M F32 = **3.06 GB** (bf16 ≈ 1.5 GB, OV-int8 ≈ 650 MB — inside cap compressed, just misses 500 MB).
  CC-BY-NC.
- **8. `kiranpantha/whisper-large-v3-nepali`** — 18.73% WER on OpenSLR54, apache-2.0 (the *only* permissive
  license among the tuned whisper family), 1.55B F32 → **3.06 GB** (too big except as fp16 ≈ 1.5 GB / int8
  ≈ 800 MB). Pinnable to `ne`.
- **9. `Dragneel/whisper-small-nepali-openslr`** — small (244M) → F32 **967 MB** is already over; the only
  "small" Nepali whisper in the family and its 26.69% WER is self-reported on the Same-OpenSLR-Training-
  split — treat carefully.
- **10. `Harveenchadha/vakyansh-wav2vec2-nepali-nem-130`** — the Vakyansh 130 h Nepali model, ~15.7% WER
  + 9.4% with KenLM on the OS evaluation (2019 paper), but raw pt/TS weights (~1.1 GB) with zero HF
  integrations and a BENGALI-alphabet BPE vocab (BoomNera `jar.online` legacy) → script mismatch. Demo
  only.

**Not recommended / excluded:** `sumanpaudel1997/nepali-asr-xlsr-53` (1.26 GB, 26.85% WER);
`nepali-asr-mms-1b` (965M, 3.7 GB F32, 27.28% — best FLEURS CER but too big and medium WER);
`ampixa/nepali-conformer-streaming` (485 MB .nemo, but 59.87% real-call WER — a streaming/telephony model,
not a dictation model; NC); kiranpantha turbo CT2 (3.09 GB — turbo is already produced smaller elsewhere);
`vakyansh` (see above); the tested `alphaedge whisper-base-nep-*` (dead, 0–4%); the naive
`openai/whisper-large-v3-turbo` pinned `ne` (already the user's weak baseline).

### Recommendation (one paragraph)

**Go with `sumanpaudel1997/nepali-asr-indicwav2vec` first** — it is the only candidate that
is simultaneously (a) Proven at the top of the single controlled Nepali benchmark (14.89% WER,
equal to the 809M tuned turbo), (b) 94.4M/378 MB — comfortably sub-500MB, under a third of the
user's current turbo, (c) CTC single-pass, KV-free, so the friendliest possible shape for the Intel
NPU and the GPU/CPU OpenVINO stack (the daemon already proved a 0.6B Parakeet encoder runs fine — 1/3
the size), and (d) natively Devanagari — verified Devanagari CTC vocab, no transliteration, no language
token needed because it is Nepali-only. Costs: it is a `.safetensors` transformers checkpoint, so the
lab must convert to OpenVINO IR (one `optimum-intel` export; off-the-shelf, not research), and the
license is non-commercial (fine for personal dictation). Backstop **2:** `tonibirat/whisper-large-v3-turbo-ne-mixed`
is a pin-the-language multilingual with an explicit anti-Hindiification/anti-loop objective — try it as the
GPU harness if you want to stay on the turbo path; backstop **3:** `ai4bharat/indicconformer_stt_ne_hybrid`
(strongly MIT, dual CTC/RNNT) if the whole family proves out on your mic. If the lab wants a first
GPU-only smoke test today with zero conversion: `gam30` ONNX INT8 (134 MB) is 1 download away.
Note on GPU vs NPU data: the README probed the live stack GPU + NPU; every candidate except
`kriti` (Python native) lands in the existing `OpenVINO/whisper-*`-style pipeline without touching the live daemon.

### Source links

- Benchmark paper (arXiv 2608.12327): https://arxiv.org/abs/2608.12327 — six-model controlled suite + WER/CER/RTF
- Chosen model: https://huggingface.co/sumanpaudel1997/nepali-asr-indicwav2vec — vocab verified in-repo
- kriti: https://huggingface.co/harrrshall/kriti + https://github.com/Naamche-Labs/kriti (benchmark.json)
- ai4bharat IndićConformer Nepali: https://huggingface.co/ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large
- tonibirat turbo-Nepali: https://huggingface.co/tonibirat/whisper-large-v3-turbo-ne-mixed
- gam30 ONNX: https://huggingface.co/gam30/nepali-automatic-speech-recognition
- Vakyansh Nepali: https://github.com/Open-Speech-EkStep/vakyansh-models + https://huggingface.co/Harveenchadha/vakyansh-wav2vec2-nepali-nem-130