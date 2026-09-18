// shlok.asr — floating recording/streaming/transcribing popup.
//
// Bottom-center overlay (layer-shell Overlay) that shows what the daemon
// is doing right now, live, without having to open the settings dropdown:
//
//   recording (non-stream)  [rec-glyph     ●  0:00   ●  ||||||||||  ●  ✕]
//   recording (stream)      [stream-glyph  ●  00:00  ●  ||||||||||  ●  ✕]
//   transcribing (either)   [transcribe-glyph  ●  Transcribing...  ●  ✕]
//
// Data it reads (all under /home/shlok/.local/share/npu-asr/state/):
//   status.json   {"class": recording|transcribing|idle, "stream":1?, "since":epoch}
//   level         plain decimal RMS (0.0..1.0), written ~8x/sec while recording
//
// Cancel: the ✕ TOUCHES state/cancel-requested. The daemon polls that file
// every 200ms and discards the take (no transcribe, no paste; temps cleaned).
// This window is passive otherwise: the full-screen surface's input region is
// masked to the card so clicks pass through to the desktop.
//
// The popup instance is mounted as a `panel` entry point (keepLoaded) on the
// same plugin, so it floats from shell startup and polls even while the bar
// dropdown is closed.
import QtQuick
import QtQuick.Effects
import Quickshell
import Quickshell.Io
import Quickshell.Wayland
import qs.Commons
import qs.Ui

Item {
  id: root

  readonly property string stateDir: "/home/shlok/.local/share/npu-asr/state"

  // ---- live state (mirrored from the files) ----
  property string statusClass: "idle"     // idle | recording | transcribing
  property bool streaming: false          // "stream":1 in status.json
  property bool cancelPending: false      // true once we've asked the daemon to abort
  property double takeSince: 0            // record/stream start epoch (seconds), from status.json
  property double level: 0                // RMS 0..1, from state/level
  property int tick: 0                    // Forced refresh for timer bindings

  // Smoothed per-bar "volume", 0..1. Each bar holds its own target and we
  // animate toward the latest level; the animation gives the bars their VU
  // meter bounce rather than a hard fill. Rebuilt (not mutated) each update
  // so bindings on root.barLevels actually re-evaluate.
  readonly property int barCount: 10
  // Fixed but shuffled factors so bars don't form a clean left→right gradient
  readonly property var barFactors: [0.92, 0.61, 1.0, 0.74, 0.88, 0.55, 0.97, 0.68, 0.81, 0.79]
  property var barLevels: {
    var a = [];
    for (var i = 0; i < root.barCount; i++) a.push(0);
    return a
  }

  // ---- glyphs (user-provided / verified in JetBrainsMonoNerdFont v3) ----
  readonly property string recGlyph: "󰟅"      // U+F07C5 (md-ear_hearing)
  readonly property string streamGlyph: ""    // U+F2A2 (fa-ear_listen) one glyph, record+transcribe together
  readonly property string transcribeGlyph: "" // U+EC21 (cod-sparkle_filled)
  readonly property string cancelGlyph: ""    // U+F52F (oct-x-circle)

  readonly property bool active: root.statusClass === "recording" || root.statusClass === "transcribing"

  // TRUE for the transcribe phase, streaming or not.
  readonly property bool postTranscribing: root.statusClass === "transcribing"

  // ---------------- file readers ----------------
  Process {
    id: statusProc
    command: ["bash", "-c", "cat " + root.stateDir + "/status.json 2>/dev/null"]
    stdout: StdioCollector {
      waitForEnd: true
      onStreamFinished: {
        var raw = String(text || "")
        var m = /"class"\s*:\s*"([^"]*)"/.exec(raw)
        var cls = m ? m[1] : "idle"
        root.statusClass = cls
        root.streaming = (raw.indexOf('"stream":1') >= 0)
        var sm = /"since"\s*:\s*(\d+)/.exec(raw)
        if (sm) root.takeSince = parseFloat(sm[1])
        if (cls === "idle") root.cancelPending = false
      }
    }
  }
  Process {
    id: levelProc
    command: ["bash", "-c", "cat " + root.stateDir + "/level 2>/dev/null || echo 0"]
    stdout: StdioCollector {
      waitForEnd: true
      onStreamFinished: {
        var v = parseFloat(String(text || "0"))
        if (isNaN(v)) v = 0
        root.level = Math.max(0, Math.min(1, v))
        root.updateBarLevels(root.level)
      }
    }
  }

  Timer {
    id: pollTimer
    interval: root.active ? 100 : 1200
    running: true
    repeat: true
    onTriggered: {
      interval = root.active ? 100 : 1200
      if (!statusProc.running) statusProc.running = true
      if (root.active && !levelProc.running) levelProc.running = true
    }
  }

  // Rebuilds the whole array and reassigns root.barLevels so bindings that
  // read root.barLevels[i] actually re-evaluate (mutating in place does not
  // trigger QML's change notification).
  function updateBarLevels(lvl) {
    if (root.postTranscribing) return
    // Amplify so quiet stays low, loud punches harder (stronger wobble)
    var boosted = Math.min(1.0, lvl * 1.55)
    var arr = []
    for (var i = 0; i < root.barCount; i++) {
      var factor = root.barFactors[i] || 0.8
      var t = Math.max(0, Math.min(1, boosted * factor))
      t = Math.max(t, lvl > 0 ? 0.05 : 0)
      arr.push(t)
    }
    root.barLevels = arr
  }

  Timer {
    id: clockTimer
    interval: 200
    repeat: true
    running: root.active
    onTriggered: root.tick++
  }

  function formatTimer() {
    // root.tick forces the binding to re-evaluate every clock tick
    var _ = root.tick
    if (root.takeSince <= 0) return root.streaming ? "00:00" : "0:00"
    var now = Date.now() / 1000
    var secs = Math.max(0, Math.floor(now - root.takeSince))
    var m = Math.floor(secs / 60)
    var s = secs % 60
    if (root.streaming) {
      return (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s
    }
    return String(m) + ":" + (s < 10 ? "0" : "") + s
  }

  Component {
    id: dotComponent
    Rectangle {
      width: Style.space(7)
      height: Style.space(7)
      radius: width / 2
      color: Color.popups.text
      opacity: root.active ? 0.8 : 0
      layer.enabled: false
    }
  }

  function requestCancel() {
    if (root.cancelPending) return
    root.cancelPending = true
    cancelProc.command = ["bash", "-c", "echo \"$$\" > " + root.stateDir + "/cancel-requested"]
    cancelProc.running = true
  }
  Process {
    id: cancelProc
    onExited: { }
  }

  PanelWindow {
    id: win
    visible: root.active
    anchors { top: true; bottom: true; left: true; right: true }
    color: "transparent"
    WlrLayershell.namespace: "omarchy-asr-recorder-popup"
    WlrLayershell.layer: WlrLayer.Overlay
    WlrLayershell.keyboardFocus: WlrKeyboardFocus.None
    exclusionMode: ExclusionMode.Ignore
    mask: Region { item: cardContent }

    Item {
      anchors.fill: parent
      BorderSurface {
        id: cardContent
        width: Math.max(Style.space(220), contentRow.implicitWidth + Style.space(16) * 2)
        height: Style.space(48)
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 10
        color: Util.alpha(Color.background, 0.96)
        borderSpec: Border.surfaceSpec("popups", "border", Color.popups.border, Math.max(1, Style.space(2)))
        radius: Style.cornerRadius * 3.8
        padding: Style.space(8)
        opacity: root.active ? 1.0 : 0.0
        scale: root.active ? 0.9 : 0.81

        Behavior on opacity {
          NumberAnimation { duration: 130; easing.type: Easing.OutCubic }
        }
        Behavior on scale {
          NumberAnimation { duration: 130; easing.type: Easing.OutCubic }
        }

        // Positioners (Row) skip children with visible:false entirely, so the
        // bars/second-dot segment collapses cleanly during the transcribing
        // phase instead of leaving a gap.
        Row {
          id: contentRow
          anchors.centerIn: parent
          spacing: Style.space(10)

          Text {
            id: stateGlyph
            anchors.verticalCenter: parent.verticalCenter
            textFormat: Text.PlainText
            text: root.postTranscribing ? root.transcribeGlyph :
                  (root.streaming ? root.streamGlyph : root.recGlyph)
            color: Color.popups.text
            font.family: Style.font.family
            font.pixelSize: Style.font.display
          }

          Loader {
            sourceComponent: dotComponent
            anchors.verticalCenter: parent.verticalCenter
          }

          Text {
            id: statusText
            anchors.verticalCenter: parent.verticalCenter
            textFormat: Text.PlainText
            text: root.postTranscribing ? "Transcribing..." : root.formatTimer()
            color: root.postTranscribing ? Color.accent : Color.popups.text
            font.family: Style.font.family
            font.pixelSize: Style.font.title
            font.bold: true
            font.letterSpacing: 0.5
          }

          Loader {
            sourceComponent: dotComponent
            anchors.verticalCenter: parent.verticalCenter
            visible: !root.postTranscribing
          }

          Item {
            id: bars
            width: root.barCount * (Style.space(4) + Style.space(3)) - Style.space(3)
            height: Style.space(28)
            anchors.verticalCenter: parent.verticalCenter
            visible: !root.postTranscribing
            Repeater {
              model: root.barCount
              delegate: Rectangle {
                id: bar
                width: Style.space(4)
                required property int index
                property real target: root.barLevels[index] || 0
                anchors.verticalCenter: parent.verticalCenter
                x: index * (width + Style.space(3))
                // smaller base + bigger swing, expands both up & down from center
                height: Style.space(3) + Style.space(96) * bar.target
                radius: Style.space(1)
                color: bar.target >= 0.9
                  ? Color.accent
                  : Color.popups.border
                Behavior on height {
                  NumberAnimation { duration: 55 + (index * 18); easing.type: Easing.OutQuad }
                }
                Behavior on color {
                  ColorAnimation { duration: 55 + (index * 5) }
                }
              }
            }
          }

          Loader {
            sourceComponent: dotComponent
            anchors.verticalCenter: parent.verticalCenter
          }

          Item {
            id: cancelBtn
            width: cancelGlyphText.implicitWidth
            height: Style.space(30)
            anchors.verticalCenter: parent.verticalCenter
            Text {
              id: cancelGlyphText
              anchors.centerIn: parent
              textFormat: Text.PlainText
              text: root.cancelGlyph
              color: Color.popups.text
              opacity: cancelMouse.containsMouse ? 1.0 : 0.75
              font.family: Style.font.family
              font.pixelSize: Style.font.display - 2
              scale: cancelMouse.pressed ? 0.85 : 1.0
              Behavior on opacity { NumberAnimation { duration: 100 } }
              Behavior on scale { NumberAnimation { duration: 80 } }
            }
            MouseArea {
              id: cancelMouse
              anchors.fill: parent
              hoverEnabled: true
              cursorShape: Qt.PointingHandCursor
              onClicked: root.requestCancel()
            }
          }
        }
      }
    }
  }
}
