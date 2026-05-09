# YOLO Person Detector & Follower for Agibot X2

Real-time person detection and follow-me behavior for the Agibot X2 humanoid robot, built on YOLOv8 and ROS2 Humble.

## What's in the box

| Node | Purpose |
|------|---------|
| `camera_selector_node` | Subscribes to one of the robot's cameras, republishes on a unified topic |
| `yolo_detector_node` | Runs YOLOv8 person detection, publishes bboxes + annotated image |
| `visualization_node` | Overlays detections on camera frames for monitoring (rqt / web video) |
| `person_follower_node` | Visual-servoing follower: robot centers on & approaches the detected person |

Two launch files:

- `yolo_pipeline.launch.py` — **detection only** (robot does not move)
- `yolo_follower.launch.py` — **detection + active following** (robot moves!)

---

## 1. Local testing on MacBook (no robot)

Quick sanity check against your webcam before touching the robot.

### Setup

```bash
# Create conda environment (one time)
conda create -n yolo_detector python=3.10 -y
conda activate yolo_detector
pip install ultralytics opencv-python numpy pytest
```

### Run against your webcam

From **Terminal.app** (macOS requires camera permission):

```bash
conda activate yolo_detector
cd ~/Desktop/ETH/robohack2026/yolo_person_detector

# List available cameras
python3 scripts/run_detector_standalone.py --list-cameras

# Run detection and save a debug video (index 1 = Continuity Camera on most Macs)
python3 scripts/run_detector_standalone.py --webcam --cam-index 1 --save-video output.mp4
```

Press `q` to quit. The annotated video is saved to `output.mp4` in the current directory.

### Run unit tests

```bash
pytest test/ -v
```

---

## 2. Deploy to the Agibot X2

### Prerequisites on the robot

- ROS2 Humble installed at `/opt/ros/humble`
- Workspace at `~/ros2_ws` with `aimdk_msgs` built (already present on the robot)
- Python 3 with pip
- Network access to pip for the YOLO model download (or pre-copy the `.pt` file)

### One-shot deploy (recommended)

From your Mac:

```bash
cd ~/Desktop/ETH/robohack2026/yolo_person_detector

# Copy + install deps + build
./scripts/deploy_to_robot.sh agi@10.0.1.40 --build
```

This rsyncs the package to `~/ros2_ws/src/yolo_person_detector/` on the robot, pip-installs `ultralytics`, and runs `colcon build --packages-select yolo_person_detector`.

### Manual deploy (if the script doesn't fit)

```bash
# From your Mac
rsync -av --exclude='__pycache__' --exclude='*.mp4' \
    yolo_person_detector/ agi@<ROBOT_IP>:~/ros2_ws/src/yolo_person_detector/

# On the robot
ssh agi@<ROBOT_IP>
pip3 install --user ultralytics opencv-python numpy
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
colcon build --packages-select yolo_person_detector --symlink-install
source install/setup.bash
```

---

## 3. Running on the robot

Always SSH in first and source the environments:

```bash
ssh agi@<ROBOT_IP>
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

### A. Detection only (safe, robot will not move)

```bash
ros2 launch yolo_person_detector yolo_pipeline.launch.py
```

Optional arguments:

```bash
ros2 launch yolo_person_detector yolo_pipeline.launch.py \
    camera:=rgbd_head_front \
    device:=cuda \
    confidence:=0.6
```

Monitor the output from another terminal:

```bash
# List detections at the terminal
ros2 topic echo /yolo/detections

# View annotated video (if rqt_image_view is available)
ros2 run rqt_image_view rqt_image_view /yolo/detection_image

# Or via the existing WebRTC server on the robot:
#   http://<ROBOT_IP>:8443  →  stream  → /yolo/detection_image
```

### B. Person follower (robot WILL move — follow safety steps)

**Safety checklist before enabling the follower:**

1. Robot is in an open, unobstructed area
2. Operator has the e-stop ready (Ctrl+C on the launch terminal stops publishing velocity; the watchdog stops the robot within 0.5s of lost detection)
3. Start with low gains and slow speeds (defaults are conservative)

**Start sequence:**

```bash
# 1. SSH to the robot (two terminals)
ssh agi@<ROBOT_IP>
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# 2. Put the robot in locomotion mode (required before velocity commands work)
ros2 run py_examples set_mc_action LD
#   LD = LOCOMOTION_DEFAULT

# 3. Launch the follower with active following enabled
ros2 launch yolo_person_detector yolo_follower.launch.py \
    follower_enabled:=true \
    device:=cuda
```

Stop it with `Ctrl+C` — this triggers the watchdog and zeros velocity within 50 ms.

### How the follower behaves

The follower uses **visual servoing** (no depth needed):

- **Horizontal**: keeps the target person's bounding box centered in the image by rotating the robot
- **Forward**: keeps the bounding box height at ~60% of the image height (i.e., target distance). If the person is far, bbox is small → robot walks forward. If the person is close, bbox is large → robot stops/backs off (up to the min_forward_speed threshold).
- **Safety**: stops immediately if no detection arrives for > 0.5 s (watchdog)
- **Target selection**: largest bounding box (closest person)

All parameters are tunable in `config/yolo_params.yaml` → `person_follower`:

| Param | Default | Purpose |
|-------|---------|---------|
| `target_bbox_height_ratio` | 0.6 | Target person size (fraction of image height) |
| `forward_gain` | 0.8 | P gain on forward error |
| `angular_gain` | 1.5 | P gain on lateral error |
| `max_forward_speed` | 0.6 m/s | Cap on forward velocity |
| `max_angular_speed` | 0.8 rad/s | Cap on rotation |
| `center_deadzone_px` | 50 | Ignore horizontal errors smaller than this |
| `watchdog_timeout_sec` | 0.5 | Stop after this long without detection |

---

## 4. Topics & Messages

### Inputs (subscribed)

| Topic | Type |
|-------|------|
| `/aima/hal/sensor/rgbd_head_front/rgb_image` | `sensor_msgs/Image` |
| `/aima/hal/sensor/rgb_head_rear/rgb_image` | `sensor_msgs/Image` |
| `/aima/hal/sensor/stereo_head_front_left/rgb_image` | `sensor_msgs/Image` |

### Outputs (published)

| Topic | Type | Description |
|-------|------|-------------|
| `/yolo/input_image` | `sensor_msgs/Image` | Selected camera stream |
| `/yolo/detections` | `vision_msgs/Detection2DArray` | Person detections (bbox + score) |
| `/yolo/detection_image` | `sensor_msgs/Image` | Image annotated with bboxes |
| `/yolo/inference_time` | `std_msgs/Float32` | Inference time per frame (ms) |
| `/yolo/visualization_image` | `sensor_msgs/Image` | Visualization node output |
| `/aima/mc/locomotion/velocity` | `aimdk_msgs/McLocomotionVelocity` | Velocity commands (follower mode only) |

---

## 5. Troubleshooting

**"aimdk_msgs not available"**
The follower node is running outside the robot's workspace. Source it:
`source ~/ros2_ws/install/setup.bash`

**Robot doesn't move even with follower_enabled:=true**
The robot isn't in locomotion mode. Run `ros2 run py_examples set_mc_action LD` first.
Also confirm the input-source registration log line: `Input source registered as "person_follower"`.

**Detection is slow (> 100 ms/frame) on CPU**
Use `device:=cuda` if the robot has an NVIDIA GPU, or switch to a smaller model (`yolov8n.pt` is already the smallest). Check `ros2 topic hz /yolo/detections` — should be 15+ Hz on modern CPUs.

**YOLO model download fails on the robot**
If the robot has no internet, download `yolov8n.pt` on your Mac first and copy it:
```bash
scp ~/.cache/ultralytics/yolov8n.pt agi@<ROBOT_IP>:~/ros2_ws/src/yolo_person_detector/
# Then launch with model:=./yolov8n.pt
```

**Camera not publishing anything**
Check the camera is running:
`ros2 topic hz /aima/hal/sensor/rgbd_head_front/rgb_image`
If silent, the sensor service may not be started on the robot.
