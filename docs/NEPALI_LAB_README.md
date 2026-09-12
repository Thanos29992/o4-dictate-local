# Nepali Lab (nepali-lab)

Isolated working directory for the **new Nepali-only ASR model** search.
Cordoned from the live ASR module and from `../nep-base-lab` (the old trim experiment —
keep those separate).

## Scope (from Shlok, 2026-09-05)
- Find a **sub-1GB** Nepali-capable ASR model (prefer **sub-500MB**, smaller is better).
- **GPU-first** transcription/acceleration; **NPU** is a luxury target but keep a pathway.
- Focus **only Nepali** for now (German is fine if it comes free; English can wait).
- If a multilingual Whisper *(not the base-nep trim we tried)* works with a CUSTOM handler that
  pins it to Nepali (even though it knows 98+ languages), that's a valid interim path.
- Hard-coding the language codebase/handler to support one model across language configs = **LATER**.
  First: find a Whisper (or anything) that ACTUALLY produces good Nepali, ideally also on NPU.

## Isolation rules (do NOT interfere with `~/.local/share/npu-asr/`)
- Model downloads/copies live HERE. Never write into live `models/` or `state/`.
- Live ASR module = `~/.local/share/npu-asr/` + `~/.config/omarchy/plugins/shlok.asr/`.
- If a candidate is proven, propose it to the user — don't hotwire the live daemon.
- Scripts here must set their own `--model-dir`/`--device` — no ambient state.
- Don't overlap with `../nep-base-lab` (the alphaedge base-nep trim). New candidates only.

## Environment (probed 2026-09-05)
- OpenVINO 2026.3.1.0 (genai); devices `['CPU','GPU','NPU']`; **GPU works**.
- NPU is available (luxury target). GPU is the pragmatic first target.

## Search thresholds (rank candidates for the shortlist)
1. Sub-500MB preferred; hard cap 1GB.
2. Real Nepali quality (not transliterated English).
3. GPU-accelerable; NPU a bonus.
4. Multilingual-but-pinnable handler = acceptable fallback.

## Status
- [ ] HuggingFace / internet candidate shortlist (sub-1GB, Nepali, GPU-first) — see `TALLY.md`
- [ ] Pick 1–2 top candidates toward GPU+NPU
- [ ] Verify offline transcription on iGPU in-lab
- [ ] (later) custom language-pin handler if multilingual model chosen