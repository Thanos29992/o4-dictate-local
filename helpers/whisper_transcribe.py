#!/usr/bin/env python3
"""whisper_transcribe.py — NPU/GPU/CPU Whisper inference via openvino_genai.

Called by the dictate.cpp daemon as a subprocess for the NPU path.

Args:
  --model-dir PATH        Whisper model directory (contains openvino_*_model.xml)
  --device NAME           NPU / GPU / CPU
  --audio PATH            Float32 mono 16kHz WAV
  --max-new-tokens INT    Cap on generated tokens (default 128)
  --language CODE         Optional Whisper language hint (e.g. "en")

Stdout: a single JSON object {"text": "...", "load_seconds": ..., "transcribe_seconds": ...}
Exit: 0 on success, 1 on error. On error, stderr gets the traceback and stdout
gets {"text":"", "error":"..."}.

Long audio is split on silence into <=25s chunks (the model's native window is
30s; a single generate() over longer audio fills the KV-cache and the tail is
dropped with a "stuck (KVcache is full)" warning). Each chunk is transcribed
separately and stitched with single spaces; a trailing space is appended so the
next dictation continues naturally after a space.

A single pipeline is cached per (model_dir, device) inside the same process so
the daemon can call us many times per recording without re-loading the model.
"""

# Chunking tuning (see TALLY.md E3/E5 for the evidence).
CHUNK_SECONDS = 25.0    # comfortably inside the 30s native window
CHUNK_OVERLAP = 1.0     # seconds of back-off when a hard cut is unavoidable
SILENCE_RMS = 0.02      # below this RMS (100ms window) counts as silence
MIN_SILENCE = 0.4       # seconds of continuous silence needed for a cut
MIN_CHUNK = 2.0         # never emit a cut before this many seconds (except EOF)
import argparse
import json
import os
import sys
import time
import traceback

# Suppress the very chatty OpenVINO / OpenCL log noise from stderr.
os.environ.setdefault("OPENVINO_LOG_LEVEL", "ERROR")
os.environ.setdefault("OV_GPU_HELPERS", "0")
# Caches we don't need.
os.environ.setdefault("HF_HOME", "/home/shlok/.cache/huggingface-npu")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Cache pipelines globally.
_PIPELINES = {}


def get_pipeline(model_dir: str, device: str):
    key = (model_dir, device)
    if key not in _PIPELINES:
        import openvino_genai as og
        _PIPELINES[key] = og.WhisperPipeline(model_dir, device=device)
    return _PIPELINES[key]




def read_f32_wav(path: str):
    """Read a float32 mono 16k WAV into a list[float]."""
    import struct, array
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file")
    p = 12
    fmt = None
    audio = None
    while p < len(data) - 8:
        ck_id = data[p:p+4]
        p += 4
        ck_sz = struct.unpack("<I", data[p:p+4])[0]
        p += 4
        if ck_id == b"fmt ":
            audio_fmt, ch, sr, br, ba, sw = struct.unpack("<HHIIHH", data[p:p+16])
            fmt = dict(fmt=audio_fmt, ch=ch, sr=sr, sw=sw)
            p += ck_sz
        elif ck_id == b"data":
            audio = data[p:p+ck_sz]
            p += ck_sz
        else:
            p += ck_sz
    if not (fmt and audio):
        raise ValueError(f"{path} missing fmt/data chunks")
    if fmt["fmt"] != 3 or fmt["sw"] != 32 or fmt["ch"] != 1:
        raise ValueError(
            f"{path} must be float32 mono; got fmt={fmt['fmt']} sw={fmt['sw']} ch={fmt['ch']}"
        )
    samples = array.array("f", audio)
    return [s for s in samples], fmt["sr"]


def _rms(seg):
    if not seg:
        return 1.0
    return (sum(v * v for v in seg) / len(seg)) ** 0.5


def split_on_silence(data, sr):
    """Split samples into (start, end) index ranges of <= CHUNK_SECONDS,
    cutting inside silence where possible. Returns [(a, b), ...]."""
    n = len(data)
    win = int(0.1 * sr)
    min_sil = int(MIN_SILENCE * sr)
    max_len = int(CHUNK_SECONDS * sr)
    bounds = [0]
    i = 0
    while i < n:
        end = min(n, i + max_len)
        if end == n:
            bounds.append(n)
            break
        # Search backwards up to 3s for a silence pocket to cut inside.
        s0 = max(i + int(MIN_CHUNK * sr), end - int(3 * sr))
        best = -1
        j = s0
        while j < end - min_sil:
            if _rms(data[j:j + min_sil]) < SILENCE_RMS:
                best = j
                j += min_sil
            else:
                j += win
        if best > 0:
            bounds.append(best)
            i = best
        else:
            # No silence found: hard cut, backing off by the overlap so no
            # word is sliced exactly once without context on either side.
            bounds.append(end)
            i = end - int(CHUNK_OVERLAP * sr)
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


def _result_text(res):
    # genai >= 2026 returns WhisperDecodedResults (no .text attr);
    # older builds returned an object with .text. str() works for both.
    t = getattr(res, "text", None)
    if isinstance(t, str):
        return t
    texts = getattr(res, "texts", None)
    if texts:
        try:
            return " ".join(texts)
        except TypeError:
            pass
    return str(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=448)
    ap.add_argument("--language", default=None)
    args = ap.parse_args()

    try:
        data, sr = read_f32_wav(args.audio)
        if sr != 16000:
            # We could resample with librosa but the daemon always writes 16k.
            # Warn but proceed; the pipeline itself will resample.
            sys.stderr.write(f"[whisper_transcribe] warning: sr={sr} (expected 16000)\n")

        kwargs = {"max_new_tokens": args.max_new_tokens}
        # rpen for whisper-BASE-family models only (INT8 Nepali base and EN
        # base): 1.2 kills their repetition loop. Turbo (d_model 1280) rejects
        # any non-1.0 value, so it must never receive rpen — keyed off the model
        # dir basename.
        if "whisper-base" in os.path.basename(args.model_dir):
            kwargs["repetition_penalty"] = 1.2
        # Language override (daemon's state/language.txt, or CLI --language).
        # VERIFIED 2026-09-05: to force a language on a multilingual whisper
        # export you must pass BOTH the bracketed token `language="<|ne|>"` AND
        # the explicit `lang_to_id` map read from generation_config.json. The
        # C++ WhisperPipeline's internal lang_to_id is EMPTY on the turbo-int4
        # export, so a bare `language="ne"` crashes ("'language' ne must be
        # provided in 'lang_to_id' map") and forced_decoder_ids slot 2 has NO
        # effect (output stays the auto-detect script). Supplying lang_to_id
        # explicitly drives the model to the requested language, verified:
        #   ne -> Devanagari, en -> English, de -> German, auto -> Gujarati loop.
        # Pass `lang_to_id` (plus the bracketed token) as generate() kwargs.
        # NOTE: rpen and a forced language are NOT mutually exclusive — probed
        # on this build 2026-09-05 (GPU + NPU): passing BOTH `repetition_penalty`
        # and `language`+`lang_to_id` is OK, exits 0, and is exactly what yields
        # the clean non-looping Devanagari. The base-nep model therefore gets
        # BOTH rpen and the forced language when the panel sets one.
        _lang_to_id_map = None
        if args.language:
            _configured_lang = args.language.strip().lower()
            _model_dir = args.model_dir
            _conf_path = os.path.join(_model_dir, "generation_config.json")
            try:
                with open(_conf_path) as _f:
                    _lang_to_id_map = json.load(_f).get("lang_to_id")
            except Exception:
                _lang_to_id_map = None
            # Fall back to added_tokens.json if the config has no lang_to_id
            # (some exports keep the map only there).
            if not _lang_to_id_map:
                _at_path = os.path.join(_model_dir, "added_tokens.json")
                try:
                    with open(_at_path) as _f:
                        _lang_to_id_map = json.load(_f)
                except Exception:
                    _lang_to_id_map = {}

            _wanted = f"<|{_configured_lang}|>"
            _is_base = "whisper-base" in os.path.basename(args.model_dir)
            if _wanted in (_lang_to_id_map or {}):
                # For the trim Nepali base, the (multilingual) lang_to_id also
                # contains English; but the base cannot lay down proper English —
                # forcing `<|en|>` yields English-transliteration garbage. A
                # Nepali speaker on the base should get Devanagari. So: only
                # honor ne/de (any language the base can reproduce in its own
                # script); if the requested language is one the base would mangle
                # (like en from an overrides file), fall back to AUTO-DETECT for
                # this base rather than mis-forcing. rpen still suppresses the loop.
                if (_is_base and _configured_lang in ("ne", "de")) or not _is_base:
                    # Pass the bracketed token + the explicit map so generate()
                    # has a populated lang_to_id to honor. (Export map is empty.)
                    kwargs["language"] = _wanted
                    kwargs["lang_to_id"] = _lang_to_id_map
                    # NOTE: rpen and language are NOT mutually exclusive —
                    # verified on this build (GPU+NPU, probe 2026-09-05):
                    # passing BOTH is OK and clean, and is what suppresses the
                    # base repetition loop. Do NOT pop rpen here.
                else:
                    sys.stderr.write(
                        f"[whisper_transcribe] base model can't render language "
                        f"'{_configured_lang}'; auto-detect (rpen stays, kills loop)\n"
                    )

        t0 = time.time()
        m = get_pipeline(args.model_dir, args.device)
        load_dt = time.time() - t0

        # NOTE: do NOT pass initial_prompt/hotwords here: on NPU builds the
        # prompt tensor path fails with `roi_end <= max_dim` for prompts
        # longer than ~1 word. Chunks start/end on silence, so plain
        # per-chunk generate is clean without cross-chunk context.
        chunks = split_on_silence(data, sr)
        if len(chunks) > 1:
            sys.stderr.write(
                f"[whisper_transcribe] chunked {len(data)/sr:.1f}s into "
                f"{len(chunks)} segments\n"
            )

        t0 = time.time()
        parts = []
        for (a, b) in chunks:
            res = m.generate(data[a:b], **kwargs)
            t = _result_text(res).strip()
            # Drop pure-hallucination tails: single stray tokens ("you", ".")
            # that some chunks emit after the real speech. A real chunk of
            # speech transcribes to multiple words; a lone filler word with
            # no sentence content is the model idling, not transcribing.
            if t and not (len(chunks) > 1 and len(t.split()) <= 1):
                parts.append(t)
        dt = time.time() - t0

        # Stitch with single spaces + trailing space for the next dictation.
        text = (" ".join(parts) + " ") if parts else ""

        out = {
            "text": text,
            "load_seconds": round(load_dt, 3),
            "transcribe_seconds": round(dt, 3),
            "audio_seconds": round(len(data) / sr, 3),
            "device": args.device,
            "model": os.path.basename(args.model_dir),
        }
        # ensure_ascii=False: emit raw UTF-8 for non-ASCII scripts (Devanagari,
        # German). ensure_ascii=True would serialize बेकार as literal ब...
        # escape-ascii-bytes, which the C++ find/substr extractor then copies
        # verbatim into wl-copy — the paste shows \uXXXX garbage. ASCII text is
        # unchanged either way, so this is a zero-risk change for English.
        sys.stdout.write(json.dumps(out, ensure_ascii=False))
        sys.stdout.write("\n")
        sys.stdout.flush()
        return 0
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        out = {"text": "", "error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(out, ensure_ascii=False))
        sys.stdout.write("\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
