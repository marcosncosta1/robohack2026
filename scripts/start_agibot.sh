#!/usr/bin/env bash
# Agibot X2 Startup Script
# Run on the robot via SSH from your Mac:
#   ssh agibot@<ROBOT_IP> 'bash -s' < scripts/start_agibot.sh
#
# Prerequisites on the robot:
#   - ROS2 Humble installed
#   - x2_description package built in ~/ros2_ws
#   - Node.js installed (for the bridge server)

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
ROBOT_IP="${ROBOT_IP:-0.0.0.0}"
BRIDGE_PORT="${BRIDGE_PORT:-8080}"
WEBRTC_PORT="${WEBRTC_PORT:-8443}"
ROS_WS="${ROS_WS:-$HOME/ros2_ws}"
URDF_MODEL="${URDF_MODEL:-x2_ultra.urdf}"
LOG_DIR="$HOME/.agibot_logs"

mkdir -p "$LOG_DIR"

# ─── Cleanup on exit ────────────────────────────────────────────────────────
cleanup() {
    echo "[$(date)] Shutting down all processes..."
    kill 0 2>/dev/null
    wait
    echo "[$(date)] All processes stopped."
}
trap cleanup EXIT INT TERM

# ─── Source ROS2 ─────────────────────────────────────────────────────────────
echo "[$(date)] Sourcing ROS2 environment..."
source /opt/ros/humble/setup.bash
if [ -f "$ROS_WS/install/setup.bash" ]; then
    source "$ROS_WS/install/setup.bash"
fi

# ─── Launch ROS2 hardware drivers ────────────────────────────────────────────
echo "[$(date)] Launching robot state publisher..."
ros2 launch x2_description display.launch.py \
    > "$LOG_DIR/ros2.log" 2>&1 &
ROS_PID=$!
echo "[$(date)] ROS2 launched (PID: $ROS_PID)"

# Wait for ROS2 to initialize
sleep 3

# ─── Launch WebSocket bridge (rosbridge) ─────────────────────────────────────
echo "[$(date)] Starting rosbridge WebSocket server on port $BRIDGE_PORT..."
ros2 launch rosbridge_server rosbridge_websocket_launch.xml \
    port:="$BRIDGE_PORT" \
    address:="$ROBOT_IP" \
    > "$LOG_DIR/rosbridge.log" 2>&1 &
BRIDGE_PID=$!
echo "[$(date)] Rosbridge started (PID: $BRIDGE_PID)"

# ─── Launch WebRTC video server ──────────────────────────────────────────────
echo "[$(date)] Starting WebRTC video server on port $WEBRTC_PORT..."
ros2 run web_video_server web_video_server \
    --ros-args -p port:="$WEBRTC_PORT" -p address:="$ROBOT_IP" \
    > "$LOG_DIR/webrtc.log" 2>&1 &
WEBRTC_PID=$!
echo "[$(date)] WebRTC video server started (PID: $WEBRTC_PID)"

# ─── Status ──────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Agibot X2 is ONLINE"
echo "═══════════════════════════════════════════════════════"
echo "  ROS2 PID:        $ROS_PID"
echo "  Rosbridge PID:   $BRIDGE_PID  (ws://$ROBOT_IP:$BRIDGE_PORT)"
echo "  Video PID:       $WEBRTC_PID  (http://$ROBOT_IP:$WEBRTC_PORT)"
echo "  Logs:            $LOG_DIR/"
echo ""
echo "  Quest browser → http://<ROBOT_IP>:$BRIDGE_PORT"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop all services (e-stop)"

# ─── Keep alive & monitor ────────────────────────────────────────────────────
while true; do
    for pid_name in "ROS2:$ROS_PID" "Rosbridge:$BRIDGE_PID" "WebRTC:$WEBRTC_PID"; do
        name="${pid_name%%:*}"
        pid="${pid_name##*:}"
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[$(date)] WARNING: $name (PID $pid) has died!"
        fi
    done
    sleep 5
done
