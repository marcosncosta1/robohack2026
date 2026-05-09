#!/usr/bin/env bash
# Deploy the yolo_person_detector package to the Agibot X2 robot and build it.
#
# Usage:
#   ./scripts/deploy_to_robot.sh <ROBOT_USER@ROBOT_IP> [--build]
#
# Example:
#   ./scripts/deploy_to_robot.sh agi@10.0.1.40
#   ./scripts/deploy_to_robot.sh agi@10.0.1.40 --build
#
# This script:
#   1. rsync's the package to ~/yolo_ws/src/ on the robot (separate workspace)
#   2. Optionally builds with colcon (sourcing aimdk for aimdk_msgs)
#   3. Installs Python dependencies (ultralytics) if not present
#
# Robot workspace layout:
#   /home/agi/aimdk/           <- existing aimdk with aimdk_msgs
#   /home/agi/yolo_ws/         <- our new workspace (created by this script)
#       src/yolo_person_detector/

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <user@host> [--build]"
    echo "Example: $0 agi@10.0.1.40 --build"
    exit 1
fi

REMOTE="$1"
BUILD_FLAG="${2:-}"

# Find this script's parent directory (the package root)
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Use a separate workspace to avoid conflicts with aimdk
REMOTE_WS="/home/agi/yolo_ws"
REMOTE_SRC="$REMOTE_WS/src/yolo_person_detector"

echo "═══════════════════════════════════════════════════════"
echo " Deploying yolo_person_detector to $REMOTE"
echo "═══════════════════════════════════════════════════════"
echo " Local:  $PACKAGE_DIR"
echo " Remote: $REMOTE_SRC"
echo ""

# Step 1: Ensure remote workspace exists
echo "[1/4] Ensuring remote workspace exists..."
ssh "$REMOTE" "mkdir -p $REMOTE_WS/src"

# Step 2: Sync files (exclude build artifacts)
echo "[2/4] Syncing package files..."
rsync -av --delete \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='build' \
    --exclude='install' \
    --exclude='log' \
    --exclude='*.mp4' \
    --exclude='output.mp4' \
    "$PACKAGE_DIR/" "$REMOTE:$REMOTE_SRC/"

# Step 3: Install Python dependencies
echo "[3/4] Installing Python dependencies on robot..."
ssh "$REMOTE" "pip3 install --user ultralytics opencv-python numpy" || {
    echo "WARNING: pip install failed. You may need to install dependencies manually."
}

# Step 4: Build (optional)
if [ "$BUILD_FLAG" = "--build" ]; then
    echo "[4/4] Building package on robot..."
    ssh "$REMOTE" "bash -l -c '
        source /opt/ros/humble/setup.bash &&
        source /home/agi/aimdk/install/setup.bash &&
        cd $REMOTE_WS &&
        colcon build --packages-select yolo_person_detector --symlink-install
    '"
    echo ""
    echo "✓ Build complete"
else
    echo "[4/4] Skipping build (pass --build to build remotely)"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo " Deployment complete!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "On the robot, run:"
echo ""
echo "  ssh $REMOTE"
echo "  source /opt/ros/humble/setup.bash"
echo "  source /home/agi/aimdk/install/setup.bash"
echo "  source /home/agi/yolo_ws/install/setup.bash"
echo ""
echo "Detection only (no robot movement):"
echo "  ros2 launch yolo_person_detector yolo_pipeline.launch.py"
echo ""
echo "Person follower (robot WILL move — safety first!):"
echo "  # 1. Set the robot to locomotion mode:"
echo "  ros2 run py_examples set_mc_action LD"
echo "  # 2. Launch with follower enabled:"
echo "  ros2 launch yolo_person_detector yolo_follower.launch.py follower_enabled:=true"
