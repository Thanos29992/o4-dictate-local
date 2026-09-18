// shlok.asr — NPU speech-to-text control panel.
//
// A right-side bar-widget, styled like the built-in Bluetooth/Power panels.
// Pressing Copilot starts/stops recording and transcribes on the selected
// device/model (the daemon reads the state files below fresh on every toggle).
// device/model (the daemon reads the state files below fresh on every toggle).
// This panel is purely settings:
//   - Master on/off       state/enabled      (when off, Copilot does nothing)
//   - Inference device    state/device.txt   (CPU / GPU / NPU)
//   - Model               state/model.txt    (picked from models/ dir)
//   - Offload policy      state/offload      (immediate vs keep-in-RAM seconds)
//   - Animated status line                   (cycling phrases, like WiFi's)
//
// State files live under ~/.local/share/npu-asr-test/state/.
import QtQuick
import QtQuick.Effects
import QtQuick.Layouts
import Quickshell
import Quickshell.Io
import qs.Ui
import qs.Commons
import "Model.js" as Model

Panel {
  id: root
  moduleName: "shlok.asr"
  ipcTarget: "shlok.asr"

  readonly property string stateDir: "/home/shlok/.local/share/npu-asr/state"
  readonly property string modelsDir: "/home/shlok/.local/share/npu-asr/models"

  // Current settings, mirrored from the state files on disk.
  property bool dictationEnabled: true
  property string activeDevice: "NPU"
  property var modelList: []
  property var devicesJson: ({} )  // {models: {name: [devs...]}} from state/devices.json
  property string activeModel: ""
  property int offloadIndex: 0
  property int feedbackVolume: 15
  property string statusClass: "idle"
  // True when the daemon is in real-time STREAMING mode (parakeet-stream).
  // The daemon sets "stream":1 for streaming takes: live phase reports class
  // "recording" + stream (so the popup shows timer+bars), post-tap drain
  // reports class "transcribing" (+stream until finalized). Either way the
  // bar shows the single streaming glyph while root.streaming is true.
  property bool streaming: false

  // ---- cursor navigation state (for slider hover borders) ----
  property string focusSection: "offload"
  property int selectedIndex: -1
  property bool cursorActive: false

  function ensureCursorVisible(item) { /* short panel, no scroll needed */ }

  // ---- animated status phrases (mirror WiFi's hero rotation) ----
  property int phraseIndex: 0
  readonly property var phrases: [
    "Warming the NPU",
    "Listening for your voice",
    "Awaiting the mic",
    "Coaxing the chips",
    "Tuning the LSTM",
    "Hushing the fans",
    "Aligning 16 kHz",
    "Polishing phonemes",
    "Ready when you are",
  ]
  readonly property string heroStatusText:
    phrases[phraseIndex % phrases.length]

  // Model offload slider. Discrete stops, left->right:
  //   Immediate, 30s, 1m, 2m, 5m, 10m, 15m, Never
  // "Immediate" writes "1" to state/offload; every other stop writes "0 <secs>"
  // (keep-in-RAM). A huge secs on the last stop makes the timed unload never
  // fire in practice, so "Never" keeps the model resident.
  readonly property var offloadStops: ["Immediate", "30s", "1m", "2m", "5m", "10m", "15m", "Never"]
  readonly property var offloadSecs:  [30, 30, 60, 120, 300, 600, 900, 31536000]

  // ---- product name ----
  // THE name shown next to the bar icon (and in the header). This is the single
  // place to rename the product: change this one string and every widget label
  // follows. "ASR" is a temporary placeholder name.
  property string productName: "ASR"

  // ---- glyphs (all verified present in JetBrainsMonoNerdFont v3) ----
  // Primary bar/hero icon (idle, enabled) — user-picked.
  readonly property string idleGlyph: ""      // U+F198  (primary)
  // Shown in the bar while WE are recording to transcribe (not global recording).
  readonly property string recordingGlyph: "󰟅" // U+F07C5 (md-ear_hearing)
  // Streaming: record + transcribe run together — ONE glyph.
  readonly property string streamGlyph: ""    // U+F2A2 (fa-ear_listen)
  // Shown while a NON-streaming model is transcribing (model use).
  readonly property string transcribingGlyph: "" // U+EC21 (cod-sparkle_filled)
  // Disabled state (distinct so on/off is legible at a glance).
  // Device glyphs (CPU / GPU / NPU), user-picked.
  readonly property string cpuGlyph: ""       // U+F4BC  CPU
  readonly property string gpuGlyph: ""       // U+F2DB  GPU (microchip)
  readonly property string npuGlyph: "󰘚"      // U+F061A NPU (SPUA-B)
  // Model-selector glyph.
  readonly property string modelGlyph: ""     // U+EEF6

  // The menu-bar + hero glyph, driven by current state.
  // When disabled we keep the SAME idle glyph, but dim it (barDimmed) so it
  // reads as "off" without swapping to a different icon.
  // NOTE: live streaming reports class "recording" + stream:1 (so the popup
  // can show its recording phase with timer+bars), so the stream branch must
  // be checked on BOTH recording and transcribing classes.
  readonly property string barGlyph:
    root.statusClass === "recording"    ? (root.streaming ? root.streamGlyph : root.recordingGlyph) :
    root.statusClass === "transcribing" ? (root.streaming ? root.streamGlyph : root.transcribingGlyph) :
                                          root.idleGlyph

  // True when ASR is disabled — bar + hero show the idle glyph at reduced opacity.
  readonly property bool barDimmed: !root.dictationEnabled

  function glyphForDevice(d) {
    if (d === "CPU") return root.cpuGlyph
    if (d === "GPU") return root.gpuGlyph
    return root.npuGlyph                // NPU
  }

  // Helper: return asset path for accelerator SVG. Defined on root so that
  // Repeater delegates (which run in their own scope) can call it.
  function assetForDevice(dev) {
    if (dev === "CPU") return "assets/cpu.svg"
    if (dev === "GPU") return "assets/gpu.svg"
    return "assets/npu.svg"  // NPU
  }

  // ---- shared write process (the daemon reads these same files) ----
  Process {
    id: actionProc
    onExited: root.refresh()
  }

  // ---- read state files ----
  function refresh() {
    if (!procEnabled.running) procEnabled.running = true
    if (!procDevice.running) procDevice.running = true
    if (!procStatus.running) procStatus.running = true
    if (!procModels.running) procModels.running = true
    if (!procModelSel.running) procModelSel.running = true
    if (!procOffload.running) procOffload.running = true
    if (!procDevices.running) procDevices.running = true
    if (!procVolume.running) procVolume.running = true
  }

  Process {
    id: procEnabled
    command: ["cat", root.stateDir + "/enabled"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: {
      root.dictationEnabled = Model.boolFromFile(text)
    } }
  }
  Process {
    id: procDevice
    command: ["cat", root.stateDir + "/device.txt"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: {
      root.activeDevice = Model.deviceFromFile(text)
    } }
  }
  Process {
    id: procStatus
    command: ["cat", root.stateDir + "/status.json"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: {
      // status.json = {"alt":"…","class":"…","tooltip":"","stream":1,"since":…}
      var m = /"class"\s*:\s*"([^"]*)"/.exec(String(text))
      root.statusClass = m ? m[1] : "idle"
      root.streaming = (String(text || "").indexOf('"stream":1') >= 0)
    } }
  }
  Process {
    id: procModels
    command: ["bash", "-c", "ls -1 " + root.modelsDir]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: {
      root.modelList = Model.parseModelList(text)
    } }
  }
  Process {
    id: procModelSel
    command: ["cat", root.stateDir + "/model.txt"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: {
      root.activeModel = String(text).trim()
    } }
  }
  Process {
    id: procOffload
    command: ["bash", "-c", "cat " + root.stateDir + "/offload 2>/dev/null || echo none"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: {
      var t = String(text).trim().split(/\s+/)
      if (t.length >= 1 && t[0] !== "none") {
        root.offloadIndex = root.offloadIndexFromPolicy(t[0] === "1", t.length >= 2 ? (parseInt(t[1]) || 30) : 30)
      }
    } }
  }
  Process {
    id: procDevices
    command: ["cat", root.stateDir + "/devices.json"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: {
      try {
        var j = JSON.parse(String(text || "{}"))
        root.devicesJson = (j && j.models) ? j : { models: {} }
      } catch (e) {
        root.devicesJson = { models: {} }
      }
    } }
  }
  Process {
    id: procVolume
    command: ["bash", "-c", "cat " + root.stateDir + "/volume 2>/dev/null || echo 15"]
    stdout: StdioCollector { waitForEnd: true; onStreamFinished: {
      var v = parseInt(String(text || "15").trim())
      if (isNaN(v) || v < 0 || v > 100) v = 15
      root.feedbackVolume = v
    } }
  }

  // ---- actions (write files; daemon reads them next Copilot toggle) ----
  function writeSetting(shellCmd) {
    if (actionProc.running) return
    actionProc.command = ["bash", "-c", shellCmd]
    actionProc.running = true
  }
  function setEnabled(on) {
    dictationEnabled = on
    writeSetting("mkdir -p " + root.stateDir + " && echo " + (on ? "on" : "off") + " > " + root.stateDir + "/enabled")
  }
  function setDevice(d) {
    activeDevice = d
    writeSetting("echo " + d + " > " + root.stateDir + "/device.txt")
  }
  function setModel(m) {
    activeModel = m
    writeSetting("echo '" + m + "' > " + root.stateDir + "/model.txt")
  }
  // Map a state-file policy ("1" immediate / "0" keep-in-RAM + secs) back to the
  // nearest slider index. An exact seconds match wins; otherwise we snap to the
  // closest keep-in-RAM stop.
  function offloadIndexFromPolicy(immediate, secs) {
    if (immediate) return 0  // state "1" -> "Immediate", whatever the secs value
    for (var i = 1; i < root.offloadSecs.length; i++) {
      if (root.offloadSecs[i] === secs) return i
    }
    var nearest = 1, best = Math.abs(root.offloadSecs[1] - secs)
    for (var j = 2; j < root.offloadSecs.length; j++) {
      var d = Math.abs(root.offloadSecs[j] - secs)
      if (d < best) { best = d; nearest = j }
    }
    return nearest
  }
  function setOffloadIndex(idx) {
    idx = Math.max(0, Math.min(root.offloadStops.length - 1, idx))
    root.offloadIndex = idx
    if (idx === 0) {
      writeSetting("echo '1 30' > " + root.stateDir + "/offload")
    } else {
      writeSetting("echo '0 " + root.offloadSecs[idx] + "' > " + root.stateDir + "/offload")
    }
  }
  function setFeedbackVolume(v) {
    root.feedbackVolume = v
    writeSetting("echo " + v + " > " + root.stateDir + "/volume")
  }

  // ---- LIVE status poll: always on, NOT gated on root.opened ----
  // This is what keeps the bar icon in sync with the daemon while the panel is
  // closed. The daemon writes status.json {"class": recording|transcribing|idle}
  // the instant a Copilot tap starts recording or transcription; we re-read it
  // continuously so barGlyph flips without needing the widget open. The settings
  // reads below stay gated on root.opened (they don't affect the bar glyph).
  Timer {
    id: liveStatusTimer
    interval: 800
    running: true
    repeat: true
    onTriggered: {
      if (!procStatus.running) procStatus.running = true
    }
  }

  // ---- polling while open + rotating status animation ----
  Timer { interval: 2500; running: root.opened; repeat: true; onTriggered: root.refresh() }
  Timer {
    interval: 2800
    running: root.opened
    repeat: true
    onTriggered: phraseSwap.restart()
  }
  SequentialAnimation {
    id: phraseSwap
    PropertyAnimation { target: heroStatus; property: "opacity"; to: 0.0; duration: 180; easing.type: Easing.OutQuad }
    ScriptAction { script: root.phraseIndex = (root.phraseIndex + 1) % root.phrases.length }
    PropertyAnimation { target: heroStatus; property: "opacity"; to: 1.0; duration: 260; easing.type: Easing.InQuad }
  }

  // ---- bar icon / active state ----
  readonly property bool barActive: root.statusClass === "recording" || root.statusClass === "transcribing"

  // Always occupy a visible bar slot. (Microphone/hidden-state widgets hide
  // themselves via `visible`, but ASR is always meaningful, so we stay visible
  // and size the entry to the button — otherwise the Panel collapses to 0×0
  // and the icon never renders.)
  visible: true
  implicitWidth: button.implicitWidth
  implicitHeight: button.implicitHeight

  BarIconButton {
    id: button
    anchors.fill: parent
    bar: root.bar
    text: root.barGlyph
    tooltipText: root.dictationEnabled
      ? (root.statusClass === "idle" ? root.productName + " ready · " + root.activeDevice : root.productName + " " + root.statusClass)
      : root.productName + " off"
    // Keep the glyph change when recording/transcribing but NOT tint the icon
    // color — the live accent tint on the icon reads as a visual bug to the user.
    useActiveColor: false
    active: root.barActive
    // Disabled → keep the idle glyph but dim it (WidgetButton.dimmed = 0.45 opacity).
    dimmed: root.barDimmed
    onPressed: function() { root.toggle() }
  }

  KeyboardPanel {
    id: panel
    anchorItem: button
    owner: root
    bar: root.bar
    open: root.opened
    contentWidth: panel.fittedContentWidth(Style.space(360))
    contentHeight: panel.fittedContentHeight(column.implicitHeight)

    PanelKeyCatcher { id: keyCatcher; anchors.fill: parent }

    Column {
      id: column
      anchors.left: parent.left
      anchors.right: parent.right
      anchors.top: parent.top
      spacing: Style.space(14)

      // ---------- hero: icon · product name + status · master on/off ----------
      // Mirrors the built-in WiFi/Audio/Bluetooth heroes: the icon is status-only,
      // the name sits bold right next to it, and the ToggleSwitch (which owns the
      // enable/disable) rides on the trailing edge of the same row.
      Item {
        width: parent.width
        implicitHeight: Math.max(heroIcon.implicitHeight, heroLabels.implicitHeight, masterSwitch.implicitHeight)

        Text {
          id: heroIcon
          textFormat: Text.PlainText
          text: root.barGlyph
          color: root.bar.foreground
          font.family: root.bar.fontFamily
          font.pixelSize: Style.font.display
          opacity: root.barDimmed ? 0.5 : 1.0
          anchors.left: parent.left
          anchors.verticalCenter: parent.verticalCenter
        }

        // Master on/off lives here, beside the icon (like WiFi/Audio/Bluetooth),
        // not buried in a separate section.
        ToggleSwitch {
          id: masterSwitch
          checked: root.dictationEnabled
          foreground: root.bar.foreground
          anchors.right: parent.right
          anchors.verticalCenter: parent.verticalCenter
          onToggled: root.setEnabled(!root.dictationEnabled)

          PanelToolTip {
            visible: masterSwitch.containsMouse
            text: root.dictationEnabled ? "Disable " + root.productName : "Enable " + root.productName
            fontFamily: root.bar.fontFamily
          }
        }

        Column {
          id: heroLabels
          anchors.left: heroIcon.right
          anchors.leftMargin: Style.space(14)
          anchors.right: parent.right
          anchors.rightMargin: masterSwitch.width + Style.space(12)
          anchors.verticalCenter: parent.verticalCenter
          spacing: Style.space(2)

          // Product name right next to the icon (the SSID analog).
          Text {
            text: root.productName
            color: root.bar.foreground
            font.family: root.bar.fontFamily
            font.pixelSize: Style.font.title
            font.bold: true
            elide: Text.ElideRight
            width: parent.width
          }

          Text {
            id: heroStatus
            textFormat: Text.PlainText
            text: root.dictationEnabled
              ? root.heroStatusText.toUpperCase()
              : "DISABLED"
            color: Qt.darker(root.bar.foreground, 1.4)
            font.family: root.bar.fontFamily
            font.pixelSize: Style.font.caption
            font.bold: true
            font.letterSpacing: 1.2
            elide: Text.ElideRight
            width: parent.width
          }
        }
      }

      // ---------- inference device selector ----------
      PanelSeparator { foreground: root.bar.foreground }
      Column {
        width: parent.width
        spacing: Style.space(10)

        // ACCELERATOR header + credit on the SAME row: header left-aligned,
        // "POWERED BY [intel logo]" right-aligned. No "Intel" word — the logo
        // says it. Logo height = 1.6x the text font (bodySmall) so it stays
        // focused; width follows intel's wide viewBox (24.24 x 9.552).
        // The new intel.svg has ~2px of transparent padding baked in at the
        // BOTTOM of its viewBox (measured: ink fills top 29/31 rows), so the
        // logo box is lifted by that baked padding to share the text's floor.
        Item {
          id: accHeaderRow
          width: parent.width
          readonly property real intelLogoH: Style.font.bodySmall * 1.6
          implicitHeight: Math.max(accHeader.implicitHeight, intelLogoH)

          PanelSectionHeader {
            id: accHeader
            text: "ACCELERATOR"
            foreground: root.bar.foreground
            fontFamily: root.bar.fontFamily
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
          }

          Item {
            id: poweredByRow
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            readonly property real logoW: Math.round(accHeaderRow.intelLogoH * 24.24 / 9.552)
            width: poweredByText.implicitWidth + Style.space(8) + logoW
            height: Math.max(poweredByText.implicitHeight, accHeaderRow.intelLogoH + 1)
            Text {
              id: poweredByText
              text: "POWERED BY"
              color: root.bar.foreground
              font.family: root.bar.fontFamily
              font.pixelSize: Style.font.bodySmall
              anchors.left: parent.left
              anchors.bottom: parent.bottom
            }
            Item {
              width: poweredByRow.logoW
              height: accHeaderRow.intelLogoH
              anchors.right: parent.right
              anchors.bottom: parent.bottom
              // Compensate the baked-in bottom padding of intel.svg
              // (2/31 of height) so ink bottom == text floor, plus the
              // user-tuned +2px nudge upward.
              anchors.bottomMargin: Math.round(accHeaderRow.intelLogoH * 2 / 31) + 1
              Image {
                id: intelSvg
                anchors.fill: parent
                fillMode: Image.PreserveAspectFit
                sourceSize.height: accHeaderRow.intelLogoH * 2
                source: Qt.resolvedUrl("assets/intel.svg")
              }
              MultiEffect {
                anchors.fill: intelSvg
                source: intelSvg
                colorization: 1.0
                colorizationColor: root.bar.foreground
              }
            }
          }
        }

        // Cross-check helper: does the active model support a given accelerator?
        // Used to dim accelerator buttons the current model can't run on.
        function modelSupports(dev) {
          if (!root.activeModel) return true
          var ds = Model.devicesForModel(root.activeModel, root.devicesJson, root.modelsDir)
          return ds.indexOf(dev) !== -1
        }

        Row {
          id: deviceRow
          width: parent.width
          spacing: Style.space(6)
          readonly property int count: 3
          readonly property real cellWidth: (width - spacing * (count - 1)) / count

          Repeater {
            model: ["CPU", "GPU", "NPU"]
            delegate: BorderSurface {
              property string dev: modelData
              width: deviceRow.cellWidth
              height: Style.font.title * 2 + Style.spacing.controlPaddingY * 2 + Style.space(2) * 2
              color: root.activeDevice === modelData
                ? Style.selectedFillFor(root.bar.foreground, root.bar.accent)
                : "transparent"
              borderSpec: Border.controlSpec("hover-cursor", root.bar.foreground, root.bar.accent)
              border.color: Border.canUseNative(borderSpec) ? Border.color(borderSpec) : "transparent"
              border.width: Border.canUseNative(borderSpec) ? Border.uniformWidth(borderSpec) : 0
              // Always full brightness — no dimming. Compatibility is shown
              // by FILTERING the model list below, not by dimming buttons.
              opacity: 1.0
              radius: Style.cornerRadius
              // Icon + label as ONE centered group. All three icons share
              // the SAME box height (title*2) so heights match; GPU's art
              // is wider (viewBox 753x540) so only its box is wider.
              // Label is subtitle size + bold, vertically centered to the
              // icon's middle, with a wider gap after the icon.
              Row {
                id: devBtnRow
                anchors.centerIn: parent
                spacing: Style.space(8)
                Item {
                  width: (modelData === "GPU") ? Style.font.title * 2.5 : Style.font.title * 2
                  height: Style.font.title * 2
                  anchors.verticalCenter: parent.verticalCenter
                  clip: true
                  Image {
                    id: deviceImg
                    anchors.fill: parent
                    source: Qt.resolvedUrl(assetForDevice(modelData))
                    fillMode: Image.PreserveAspectFit
                    sourceSize.height: Style.font.title * 4
                    sourceSize.width: (modelData === "GPU") ? Style.font.title * 5 : Style.font.title * 4
                  }
                  MultiEffect {
                    anchors.fill: parent
                    source: deviceImg
                    colorization: 1.0
                    colorizationColor: root.bar.foreground
                  }
                }
                Text {
                  text: modelData
                  color: root.bar.foreground
                  font.family: root.bar.fontFamily
                  font.pixelSize: Style.font.title
                  font.bold: true
                  anchors.verticalCenter: parent.verticalCenter
                  // Caps sit high in the line box — nudge down so the
                  // letters look centered next to the icon.
                  anchors.verticalCenterOffset: 1
                }
              }
              MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                hoverEnabled: true
                onClicked: {
                  // Allow the click unconditionally. The dim is purely visual;
                  // the user might be intentionally switching accelerator to
                  // pick a different model that runs on the new accelerator.
                  root.setDevice(modelData)
                }
              }
            }
          }
        }
      }

      // ---------- model selector ----------
      PanelSeparator { foreground: root.bar.foreground }
      Column {
        width: parent.width
        spacing: Style.space(10)

        PanelSectionHeader { text: "MODEL"; foreground: root.bar.foreground; fontFamily: root.bar.fontFamily }

        Column {
          width: parent.width
          spacing: Style.space(6)

          // Models filtered to the ACTIVE accelerator: only models whose
          // supported set contains root.activeDevice are shown. No dimming
          // anywhere — if you see a model, it runs on the selected chip.
          Repeater {
            model: root.modelList.filter(function(m) {
              return Model.devicesForModel(m, root.devicesJson, root.modelsDir).indexOf(root.activeDevice) !== -1
            })
            delegate: Item {
              id: modelRow
              required property string modelData
              width: parent.width
              implicitHeight: Style.font.title + Style.space(8)

              // Vendor icon (left edge) — theme-aware SVG logo based on model family.
              // parakeet* -> nvidia.svg, whisper* -> openai.svg,
              // nepali-indicwav2vec -> huggingface.svg (the model lives on HF),
              // fallback -> cube glyph via Model.glyphForModel().
              function assetForModel(name) {
                var s = String(name || "")
                if (s.indexOf("parakeet") === 0) return "assets/nvidia.svg"
                if (s.indexOf("whisper")  === 0) return "assets/openai.svg"
                if (s === "nepali-indicwav2vec" || s.indexOf("indicwav2vec") !== -1) return "assets/huggingface.svg"
                return ""
              }

              Item {
                id: vendorIcon
                anchors.left: parent.left
                anchors.leftMargin: Style.spacing.controlPaddingX + Style.space(6)
                anchors.verticalCenter: parent.verticalCenter
                // Bigger icon WITHOUT thickening the row: row height stays
                // (title + space(8)), the icon just fills more of it.
                width: Style.font.title * 1.4
                height: Style.font.title * 1.4
                Image {
                  id: vendorImg
                  anchors.fill: parent
                  source: assetForModel(modelData) ? Qt.resolvedUrl(assetForModel(modelData)) : ""
                  fillMode: Image.PreserveAspectFit
                  sourceSize.width: Style.font.title * 3
                  sourceSize.height: Style.font.title * 3
                  visible: assetForModel(modelData) !== ""
                }
                MultiEffect {
                  anchors.fill: vendorImg
                  source: vendorImg
                  visible: assetForModel(modelData) !== ""
                  colorization: 1.0
                  // Vendor/model logo: FULL brightness, never dimmed.
                  colorizationColor: root.bar.foreground
                }
                Text {
                  anchors.centerIn: parent
                  visible: vendorImg.source === "" || vendorImg.status !== Image.Ready
                  text: Model.glyphForModel(modelData)
                  color: root.bar.foreground
                  font.family: root.bar.fontFamily
                  font.pixelSize: Style.font.title
                }
              }

              // Model name — always full brightness (list is pre-filtered
              // to the active accelerator, so everything shown is valid).
              Text {
                id: label
                anchors.left: vendorIcon.right
                anchors.leftMargin: Style.space(8)
                anchors.verticalCenter: parent.verticalCenter
                text: Model.modelLabel(modelData)
                color: root.bar.foreground
                font.family: root.bar.fontFamily
                font.pixelSize: Style.font.body
              }

              // Right-aligned device chips as a Row of SVG icons (cpu/gpu/npu.svg).
              // Shows only the accelerators this model supports. Slightly
              // enlarged per user feedback. Matches the button SVGs above.
              Row {
                id: chipRow
                anchors.right: parent.right
                anchors.rightMargin: Style.spacing.controlPaddingX + Style.space(6)
                anchors.verticalCenter: parent.verticalCenter
                spacing: Style.space(4)
                property string modelName: modelRow.modelData
                property var supported: Model.devicesForModel(modelRow.modelData, root.devicesJson, root.modelsDir)
                Repeater {
                  model: ["CPU", "GPU", "NPU"]
                  delegate: Item {
                    required property string modelData
                    visible: chipRow.supported.indexOf(modelData) !== -1
                    // Slightly SMALLER than the vendor icon (vendor is the
                    // star, chips are supporting info) and slightly dimmed.
                    // GPU art is wider (viewBox 753x540) so only its chip box
                    // is wider. Height stays fixed, row never thickens.
                    width: (modelData === "GPU" ? Style.font.title * 1.25 : Style.font.title * 0.95)
                    height: Style.font.title * 0.95
                    opacity: 0.55
                    Image {
                      id: chipImg
                      anchors.fill: parent
                      source: Qt.resolvedUrl("assets/" + modelData.toLowerCase() + ".svg")
                      fillMode: Image.PreserveAspectFit
                      sourceSize.width: Style.font.title * 3
                      sourceSize.height: Style.font.title * 3
                    }
                    MultiEffect {
                      anchors.fill: parent
                      source: chipImg
                      colorization: 1.0
                      colorizationColor: Qt.lighter(root.bar.foreground, 1.25)
                    }
                  }
                }
              }

              // Active-model left edge.
              Rectangle {
                visible: root.activeModel === modelRow.modelData
                width: Style.space(2)
                height: parent.height
                color: root.bar.foreground
                opacity: 0.6
                anchors.left: parent.left
              }

              MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                hoverEnabled: true
                onClicked: {
                  // Always accept the click. The dim is purely visual —
                  // the user might be intentionally picking a model that
                  // requires a different accelerator. The daemon handles
                  // the model-incompatible case gracefully (the
                  // resolve_routing helper already falls back or refuses).
                  root.setModel(modelRow.modelData)
                }
              }
            }
          }

          Text {
            visible: root.modelList.length === 0
            text: "No models found"
            color: Qt.darker(root.bar.foreground, 1.4)
            font.family: root.bar.fontFamily
            font.pixelSize: Style.font.bodySmall
          }
        }
      }

      // ---------- model offload policy ----------
      PanelSeparator { foreground: root.bar.foreground }
      Column {
        width: parent.width
        spacing: Style.space(10)

        // Header + value on the SAME row: "MODEL OFFLOAD" left-aligned,
        // current stop value right-aligned (like monitor's text-size readout).
        Item {
          width: parent.width
          implicitHeight: Style.font.body * 1.4

          PanelSectionHeader {
            text: "MODEL OFFLOAD"
            foreground: root.bar.foreground
            fontFamily: root.bar.fontFamily
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
          }

          Text {
            id: offloadValue
            textFormat: Text.PlainText
            text: root.offloadStops[Math.round(offloadSlider.dragging ? offloadSlider.liveValue : root.offloadIndex)]
            color: Qt.darker(root.bar.foreground, 1.4)
            font.family: root.bar.fontFamily
            font.pixelSize: Style.font.caption
            font.bold: true
            anchors.right: parent.right
            anchors.rightMargin: Style.space(6)
            anchors.verticalCenter: parent.verticalCenter
          }
        }

        // Discrete ramp: Immediate / 30s / 1m / 2m / 5m / 10m / 15m / Never.
        // Maps to state/offload exactly as the old toggle + stepper did.
        // CursorSurface + outline + HoverHandler mirrors the display panel's
        // brightness slider hover border (subtle accent border on mouse-hover).
        CursorSurface {
          id: offloadRow
          width: parent.width
          height: offloadSlider.implicitHeight + Style.spacing.controlGap
          hasCursor: root.cursorActive && root.focusSection === "offload" && root.selectedIndex === -1
          onHasCursorChanged: if (hasCursor) root.ensureCursorVisible(offloadRow)
          foreground: root.bar.foreground
          outline: true

          PanelSlider {
            id: offloadSlider
            bar: root.bar
            anchors.fill: parent
            anchors.leftMargin: Style.space(6)
            anchors.rightMargin: Style.space(6)
            minimum: 0
            maximum: root.offloadStops.length - 1
            step: 1
            integer: true
            tickCount: root.offloadStops.length
            value: root.offloadIndex
            onReleased: function(v) { root.setOffloadIndex(Math.round(v)) }
          }

          HoverHandler {
            onHoveredChanged: if (hovered) {
              root.cursorActive = true
              root.focusSection = "offload"
              root.selectedIndex = -1
            }
          }
        }
      }

      // ---------- audio feedback ----------
      PanelSeparator { foreground: root.bar.foreground }
      Column {
        width: parent.width
        spacing: Style.space(10)

        // Header + value on the SAME row: "AUDIO FEEDBACK" left-aligned,
        // current volume right-aligned (matches monitor's brightness layout).
        Item {
          width: parent.width
          implicitHeight: Math.max(volumeHeader.implicitHeight, volumePercent.implicitHeight)

          PanelSectionHeader {
            id: volumeHeader
            text: "AUDIO FEEDBACK"
            foreground: root.bar.foreground
            fontFamily: root.bar.fontFamily
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
          }

          Text {
            id: volumePercent
            textFormat: Text.PlainText
            text: Math.round(volumeSlider.dragging ? volumeSlider.liveValue : root.feedbackVolume) + "%"
            color: Qt.darker(root.bar.foreground, 1.4)
            font.family: root.bar.fontFamily
            font.pixelSize: Style.font.caption
            font.bold: true
            anchors.right: parent.right
            anchors.rightMargin: Style.space(6)
            anchors.verticalCenter: parent.verticalCenter
          }
        }

        // Continuous slider 0–100, default 15. Mirrors the display panel's
        // brightness slider look: CursorSurface + outline + HoverHandler
        // gives the subtle accent border on mouse-hover.
        CursorSurface {
          id: volumeRow
          width: parent.width
          height: volumeSlider.implicitHeight + Style.spacing.controlGap
          hasCursor: root.cursorActive && root.focusSection === "volume" && root.selectedIndex === -1
          onHasCursorChanged: if (hasCursor) root.ensureCursorVisible(volumeRow)
          foreground: root.bar.foreground
          outline: true

          PanelSlider {
            id: volumeSlider
            bar: root.bar
            anchors.fill: parent
            anchors.leftMargin: Style.space(6)
            anchors.rightMargin: Style.space(6)
            minimum: 0
            maximum: 100
            step: 1
            integer: true
            value: root.feedbackVolume
            onReleased: function(v) { root.setFeedbackVolume(Math.round(v)) }
          }

          HoverHandler {
            onHoveredChanged: if (hovered) {
              root.cursorActive = true
              root.focusSection = "volume"
              root.selectedIndex = -1
            }
          }
        }
      }
    }
  }

  Component.onCompleted: root.refresh()
}