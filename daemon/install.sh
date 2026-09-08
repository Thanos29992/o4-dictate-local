#!/usr/bin/env bash
# install.sh — build the native NPU dictation runtime and wire it to the
# Copilot button on Omarchy/Hyprland. Run once (after your pacman installs).
#   bash ~/.local/share/npu-asr/install.sh
set -euo pipefail
SRC=$HOME/.local/share/npu-asr/src
BIN=$HOME/.local/bin/dictate

# dependencies (you install these; sudo may be TTY-gated for you too):
#   sudo pacman -S --needed openvino fftw sndfile pipewire wl-clipboard wtype
echo "[build] compiling native C++ Parakeet-TDT runtime (NPU encoder + CPU decoder)..."
g++ -O2 -o "$BIN" "$SRC/dictate.cpp" -I/usr/include/openvino -lopenvino -lfftw3f -lsndfile -lpthread
chmod +x "$BIN"

echo "[build] writing omarchy wrapper to $HOME/.local/bin/omarchy-npu-dictate"
cat > "$HOME/.local/bin/omarchy-npu-dictate" << 'WRAPPER_EOF'
#!/usr/bin/env bash
set -euo pipefail
DAEMON="/home/shlok/.local/bin/dictate"
STATUS_DIR="/home/shlok/.local/share/npu-asr/state"

toggle() {
    if [ -f "$STATUS_DIR/daemon.pid" ]; then
        pid=$(cat "$STATUS_DIR/daemon.pid")
        if kill -0 "$pid" 2>/dev/null; then
            kill SIGUSR1 "$pid"
            echo "[omarchy-npu-dictate] sent SIGUSR1 to daemon $pid"
            exit 0
        fi
    fi
    echo "[omarchy-npu-dictate] no daemon running" >&2
    exit 1
}

start() {
    if [ -f "$STATUS_DIR/daemon.pid" ]; then
        pid=$(cat "$STATUS_DIR/daemon.pid")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[omarchy-npu-dictate] daemon already running (pid=$pid)"
            exit 0
        fi
    fi
    "$DAEMON" --device NPU --max-secs 30 --daemon > "$STATUS_DIR/err.log" 2>&1 &
    echo $! > "$STATUS_DIR/daemon.pid"
    echo "[omarchy-npu-dictate] daemon started"
}

status() {
    if [ -f "$STATUS_DIR/daemon.pid" ]; then
        pid=$(cat "$STATUS_DIR/daemon.pid")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[omarchy-npu-dictate] daemon running (pid=$pid)"
            echo -n '{"alt":"","class":"'
            if [ -f "$STATUS_DIR/capture.wav" ]; then
                echo -n 'recording", "tooltip":"Recording"'
            else
                echo -n 'idle", "tooltip":"Ready"'
            fi
        else
            echo "[omarchy-npu-dictate] daemon not responding" >&2
        fi
    else
        echo "[omarchy-npu-dictate] no daemon running" >&2
    fi
}

case "${1:-}" in
    toggle) toggle ;;
    start)  start ;;
    status) status ;;
    *) echo "Usage: $0 {toggle|start|status}" >&2; exit 1 ;;
esac
WRAPPER_EOF
chmod +x "$HOME/.local/bin/omarchy-npu-dictate"

# systemd user unit (autostarts the warm daemon on login)
install -d "$HOME/.config/systemd/user"
install -m 0644 "$SRC/../omarchy-npu-dictate.service" \
    "$HOME/.config/systemd/user/omarchy-npu-dictate.service"
systemctl --user daemon-reload
systemctl --user enable --now omarchy-npu-dictate.service || true

echo
echo "=== DONE ==="
echo "Binary: $BIN"
echo "Copilot key (SUPER+SHIFT+F23) bound in ~/.config/hypr/bindings.lua -> omarchy-npu-dictate toggle"
echo "Reload Hyprland with  SUPER+SHIFT+R  (or: hyprctl reload)"
echo "Daemon commands: omarchy-npu-dictate {toggle|start|status}"
