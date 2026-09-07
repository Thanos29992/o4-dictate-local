// Model.js — pure, testable helpers for the shlok.asr panel.
// Reads/writes small text files under ~/.local/share/npu-asr/state/ that the
// dictate daemon reads on every Copilot toggle. Imports into Panel.qml as `Model`.

// --- read/write helpers ---------------------------------------------------

function boolFromFile(value) {
  // value: "on"/"off"/""/null. Absent => enabled by default.
  if (value === null || value === undefined || value === "") return true
  var s = String(value).trim().toLowerCase()
  return !(s === "off" || s === "false" || s === "0" || s === "disabled")
}

function deviceFromFile(value) {
  var s = String(value == null ? "" : value).trim().toUpperCase()
  if (s === "CPU" || s === "GPU" || s === "NPU") return s
  return "NPU"
}

// --- model listing ---------------------------------------------------------

// Parse `ls -1 <modelsDir>` output into a sorted list of model directory names.
function parseModelList(raw) {
  var lines = String(raw || "").split("\n")
  var out = []
  for (var i = 0; i < lines.length; i++) {
    var t = lines[i].trim()
    // Skip hidden, files (we want dirs), and the shared non-model dirs.
    if (!t || t.charAt(0) === ".") continue
    if (t === "vad") continue
    if (t === "cache") continue
    out.push(t)
  }
  out.sort()
  return out
}

// Display names: original model dir -> beautiful label shown in the panel.
// The daemon ALWAYS receives the original dir name (state/model.txt);
// this map is display-only. Keep in sync when adding models.
var MODEL_DISPLAY_NAMES = {
  "parakeet-v3": "Parakeet V3 Streaming",
  "nepali-indicwav2vec": "Nepali ASR",
  "whisper-base-en": "Whisper Base"
}

function modelLabel(name) {
  var s = String(name || "")
  if (MODEL_DISPLAY_NAMES[s]) return MODEL_DISPLAY_NAMES[s]
  // Fallback for future models: strip -onnx, underscore->space.
  return s
    .replace(/-onnx$/i, "")
    .replace(/_/g, " ")
}

// --- device chips ---------------------------------------------------------

// Nerd Font glyphs for each accelerator (match the panel's accelerator row).
var DEVICE_GLYPHS = {
  "CPU": "󰍺",
  "GPU": "󰍪",
  "NPU": "󰘚"
}

// --- "not live yet" marker -----------------------------------------------
//
// A small dim glyph shown next to a model's display name when its live-mic
// streaming pipeline isn't fully built yet (e.g. the model works for
// one-shot dictation but the daemon can't yet feed it from the mic in
// real-time with parallel transcription glyph + VAD auto-stop).
//
// Add a model here when the name is FINAL but the live-mic path is still
// on the to-do list. Remove the entry when the live-mic integration
// ships — the panel will then render the model name without the marker.
var NOT_LIVE_GLYPH = "󰀦"   // nf-md-progress-clock (work-in-progress)
var NOT_LIVE_YET = {
  // "parakeet-v3": true,            // ← uncomment when live-mic ships
  "nepali-indicwav2vec": true        // one-shot works; live-mic streaming TODO
}

function notLiveMarker(name) {
  if (!name) return ""
  return NOT_LIVE_YET[name] ? NOT_LIVE_GLYPH : ""
}

// --- model "vendor" glyph --------------------------------------------------

// Small icon to the left of each model name, indicating who built the
// model. Only used as a FALLBACK when the SVG asset for a family isn't
// present (Panel.qml prefers the SVG via assetForModel()).
//   - parakeet-v3  -> nf-md-chip        (NVIDIA Parakeet TDT — silicon)
//   - whisper-*    -> nf-md-microphone  (OpenAI Whisper — speech)
//   - indicwav2vec -> nf-md-account-voice (Nepali ASR, generic voice)
//   - default      -> nf-md-cube-outline (generic)
var MODEL_VENDOR_GLYPHS = {
  "parakeet": "󰭲",
  "whisper": "󰍬",
  "indicwav2vec": "󰦭"
}

function glyphForModel(name) {
  var s = String(name || "")
  if (s.indexOf("parakeet") === 0) return MODEL_VENDOR_GLYPHS.parakeet
  if (s.indexOf("whisper")  === 0) return MODEL_VENDOR_GLYPHS.whisper
  if (s.indexOf("nepali")   === 0 || s.indexOf("indicwav2vec") !== -1) return MODEL_VENDOR_GLYPHS.indicwav2vec
  return "󰆧"
}

// Read the supported accelerators for a model. The daemon writes
// `state/devices.json` (a flat map of model name -> devices array) at
// startup; the panel reads it instead of doing N filesystem reads. The
// caller passes the parsed JSON object as the second argument. The third
// argument is the `modelsDir` (kept for API stability).
function devicesForModel(name, devicesJson, modelsDir) {
  if (!name) return ["CPU", "GPU", "NPU"]
  // devicesJson is the contents of state/devices.json ({"models":{...}}).
  if (devicesJson && devicesJson.models && devicesJson.models[name]) {
    return devicesJson.models[name]
  }
  // Fallback: infer from the name. Same rules as the daemon uses.
  // Note: state/devices.json is the source of truth; this only runs when
  // the daemon hasn't written the file yet. Per user (2026-09-05):
  //   parakeet-v3            -> GPU only
  //   nepali-indicwav2vec    -> CPU + GPU (no NPU until proven)
  var s = String(name)
  if (s === "parakeet-v3") return ["GPU"]
  if (s === "nepali-indicwav2vec") return ["CPU", "GPU"]
  if (s.indexOf("whisper-") === 0) return ["NPU"]
  if (s.indexOf("parakeet-") === 0) return ["CPU", "GPU"]
  if (s.indexOf("-gguf") !== -1)   return ["CPU", "GPU"]
  if (s.indexOf("-onnx") !== -1)   return ["CPU", "GPU", "NPU"]
  return ["CPU", "GPU", "NPU"]
}

// Build a single string of device glyphs in the canonical CPU/GPU/NPU order,
// for the row label. Dim/active states are handled in QML via opacity.
function glyphsForDevices(devs) {
  if (!devs || !devs.length) return ""
  var out = []
  // Always render in the same order so the chips don't jump around.
  var order = ["CPU", "GPU", "NPU"]
  for (var i = 0; i < order.length; i++) {
    if (devs.indexOf(order[i]) !== -1) {
      out.push(DEVICE_GLYPHS[order[i]])
    }
  }
  return out.join(" ")
}

// --- status icon -----------------------------------------------------------

// Map daemon status.json (alt/class) to a bar glyph.
function iconForStatus(clsOrAlt) {
  var s = String(clsOrAlt == null ? "idle" : clsOrAlt)
  if (s === "recording") return "󰙃"     // mic
  if (s === "transcribing") return "󰦜"  // processing dots
  return "󰍬"                            // idle mic
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    boolFromFile: boolFromFile,
    deviceFromFile: deviceFromFile,
    parseModelList: parseModelList,
    modelLabel: modelLabel,
    devicesForModel: devicesForModel,
    glyphsForDevices: glyphsForDevices,
    glyphForModel: glyphForModel,
    notLiveMarker: notLiveMarker,
    NOT_LIVE_YET: NOT_LIVE_YET,
    NOT_LIVE_GLYPH: NOT_LIVE_GLYPH,
    DEVICE_GLYPHS: DEVICE_GLYPHS,
    iconForStatus: iconForStatus
  }
}