# Production ASR — agent workflow (this file stays)

This directory (`/home/shlok/.local/share/npu-asr/`) is the PRODUCTION
dictation setup. It works. Keep it working.

## Production layout

- Source: `src/` (master branch) — `dictate.cpp`,
  `parakeet_stream_transcribe_rt.cpp`, `build.sh`
- Binaries: `/home/shlok/.local/bin/dictate` + `parakeet_stream_transcribe_rt`
  (shared streaming helper; prod `build.sh` builds both)
- Service: `omarchy-npu-dictate.service` — `dictate --max-secs 600 --daemon`
- State: `/home/shlok/.local/share/npu-asr/state/` — `status.json`
  (`{"class": recording|transcribing|idle, "stream":1?, "since":epoch}`),
  `level` (mic RMS, ~8x/sec), `model.txt`, `device.txt`, `enabled`, `volume`,
  rolling `stream1-5.wav/txt` archive of raw streaming takes
- Sounds: `/home/shlok/.local/share/npu-asr/sounds/` (`start.wav`, `stop.wav`
  — Handy pop pair; blip on mic-open and at the exact record-stop instant)
- Plugins: `shlok.asr` (bar widget, product name "ASR") + `shlok.asr-popup`
  (floating overlay: glyph + timer + bars while recording, Transcribing… +
  X-cancel while draining)
- Behavior contract: streaming = unlimited duration, stops on 2nd Copilot tap
  or 10s continuous silence; non-streaming = 600s cap; paste = one BAM shot
  via clipboard, byte-identical to model output (no punctuation
  post-processing, per user request); taps during transcribe are ignored.
- Idle top-bar glyph = U+F198. Any test copy MUST use a different idle glyph
  so prod and test are distinguishable at a glance.

## Rules for any agent asked to change ASR behavior

1. DO NOT edit anything under this directory first — unless the user reports
   production itself is broken AND explicitly asks for a production fix.
2. Instead, build a TEST copy and work only there:
   - Fresh git worktree off master, e.g.
     `git worktree add src/.claude/worktrees/asr-test -b asr-test`.
   - Clone the plugins to `shlok.asr-test` (bar) + `shlok.asr-test-popup`
     (overlay): own `stateDir` (`npu-asr-test/state/`), own product name,
     own idle glyph (NOT U+F198).
   - Build to `/home/shlok/.local/bin/dictate-asr-test` (+ test sounds dir);
     test service `omarchy-npu-dictate-test.service` with a short
     `--max-secs` (30s) so caps are easy to verify.
   - STOP + DISABLE the prod service first, route Copilot to the test daemon
     (`bindings.lua`), register the test widgets in `shell.json`. Prod and
     test must never run together — whichever daemon is alive eats the taps.
3. Port tested changes here ONLY with explicit user approval
   ("green light" / "update prod with that"). Keep prod's own idle glyph.
4. After the port, once the user confirms prod works: restart the prod
   service, restore `bindings.lua` + `shell.json` to prod-only, then delete
   the ENTIRE test stack — plugins, binaries, service unit, state dir,
   worktree + branch. No souvenirs. Append a changelog entry below. THIS
   file is the only doc that stays.

Why: if anything breaks mid-experiment, it breaks the disposable copy —
never the setup the user relies on every day.

## Known limitations (do not "fix" without being asked)

- `bindings.lua` points Copilot at `omarchy-npu-dictate toggle`. Any test
  round that re-points it must restore the prod binding during teardown.
- Panels only read `enabled` + `status.json`; they cannot detect a dead
  daemon.

## Changelog

- 2026-09-10 — Glyph swap (FINAL, per user; supersedes the earlier EBC5/F2A2 note):
  top-bar recording icon = U+F07C5 (md-ear_hearing, 󰟅), non-stream transcribing =
  U+EC21 (cod-sparkle_filled, ), streaming (record+transcribe together) = U+F2A2
  (fa-ear_listen, ). Applied to `shlok.asr/Panel.qml` (recordingGlyph / streamGlyph
  / transcribingGlyph) and the matching waybar `omarchy.npu-mic` format-icons
  (`~/.config/waybar/config.jsonc`, recording=U+F07C5, transcribing=U+EC21). Panel
  QML is the source of truth; daemon keeps ASCII status classes only (plus "stream":1
  while real-time streaming). No prod daemon rebuild required.

- 2026-09-10 — Stage 1 port from asr-test: unlimited streaming, 15s fed-audio
  silence auto-stop (streaming only), 1s audio tails on all stop paths,
  NaN/Inf mic sanitize, helper-first fork + tap-instant spool + 1MB pipes,
  full-drain EOF handshake, whole-file `rt_final.txt`, longest-wins fallback,
  tee `stream.wav` fallback transcription, BAM clipboard paste, verbatim model
  output, `g_transcribing` tap-ignore guard. Non-streaming cap stays 300s. No
  5-take archive in prod. Production LOCKED after this — further experiments
  stay in asr-test.

- 2026-09-11 — Full promote from asr-test (user-verified working, test stack
  retired afterwards): streaming live phase reports `recording` + `stream:1`
  so the popup shows stream glyph + timer + bars; post-tap drain reports
  `transcribing`; top bar shows the stream glyph (U+F2A2) in both phases via
  the `stream` flag while non-streaming keeps rec U+F07C5 / transcribe U+EC21.
  Silence auto-stop 10s (fed-audio seconds), non-streaming cap 600s
  (`--max-secs 600`). Audio feedback: Handy pop start/stop pair, async
  double-fork playback, 0–100 volume slider (`state/volume`, default 15) with
  piecewise gain curve, stop blip at the record-stop instant. Popup promoted
  as `shlok.asr-popup`: 10 VU bars from `state/level`, X-cancel via
  `cancel-requested` (daemon polls every 100ms, discards the take, no paste);
  non-stream path tees sanitized f32 for live level; 5-take stream archive
  kept in prod state. Prod idle glyph unchanged (U+F198). Entire test stack
  (plugins, `dictate-asr-test`, service unit, `npu-asr-test/`, worktree +
  branch) removed after the user's green light — this file is the only doc
  kept, and future changes go through the test-copy workflow above.
