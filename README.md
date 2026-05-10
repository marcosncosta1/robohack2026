# YOLO Person Detector & Follower for Agibot X2

Real-time person detection and follow-me behavior for the Agibot X2 humanoid robot, built on YOLOv8 and ROS2 Humble.

**This repo contains the `yolo_person_detector` and `x2_motion_audio_tools` ROS2 packages.** On the robot, clone it into `~/ros2_ws/src/robohack2026/` and build with `colcon`.

`yolo_person_detector` owns perception and publishes the shared target
contract `/stereo_person/target_point`. `x2_motion_audio_tools` owns motion:
head yaw tracking, the high-level walking supervisor, and the chair-assist
arm pose with a head-pat release. The end-to-end chair-assist demo
(perception → head → base → arm raise → head-pat release) is in section
[6. End-to-end demo](#6-end-to-end-demo-perception--head--base--arm-integration)
below.

## What's in the box

| Node | Purpose |
|------|---------|
| `camera_selector_node` | Subscribes to one of the robot's cameras, republishes on a unified topic |
| `yolo_detector_node` | Runs YOLOv8 person detection, publishes bboxes + annotated image |
| `visualization_node` | Overlays detections on camera frames for monitoring (rqt / web video) |
| `person_follower_node` | Visual-servoing follower: robot centers on & approaches the detected person |
| `stereo_final_annotator_node` | Stereo YOLO + depth pipeline that publishes annotated images and a 3D person target point |

Core launch files:

- `yolo_pipeline.launch.py` — **detection only** (robot does not move)
- `yolo_follower.launch.py` — **detection + active following** (robot moves!)
- `stereo_person_pipeline.launch.py` — **stereo detection + 3D target point** (robot does not move)

---

## 1. Local testing on MacBook (no robot)

### Setup

```bash
conda create -n yolo_detector python=3.10 -y
conda activate yolo_detector
pip install -r requirements.txt
pip install pytest
```

### Run against your webcam

```bash
conda activate yolo_detector

# List available cameras
python3 yolo_person_detector/scripts/run_detector_standalone.py --list-cameras

# Run detection and save a debug video (index 1 = Continuity Camera on most Macs)
python3 yolo_person_detector/scripts/run_detector_standalone.py --webcam --cam-index 1 --save-video output.mp4
```

Press `q` to quit.

### Run unit tests

```bash
pytest yolo_person_detector/test/ -v
```

---

## 2. Deploy to the Agibot X2

The robot needs:
- ROS2 Humble (`/opt/ros/humble`)
- `aimdk_msgs` available — source whatever aimdk workspace you have on the robot
- `ros-humble-vision-msgs` (install with apt)
- Python packages: `ultralytics`, `opencv-python`, `numpy`

### Install missing dependencies on the robot

```bash
sudo apt update
sudo apt install -y ros-humble-vision-msgs ros-humble-cv-bridge
pip3 install --user ultralytics opencv-python numpy
```

### Clone + build the workspace

```bash
# On the robot (e.g. ssh agi@10.0.1.40)
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone git@github.com:alessiosalvatore1703-ops/robohack2026.git
cd ~/ros2_ws

# Source your environments (adjust aimdk path as needed)
source /opt/ros/humble/setup.bash
source ~/aimdk/install/setup.bash   # wherever aimdk_msgs lives on your robot

# Build (colcon discovers yolo_person_detector inside src/robohack2026/)
colcon build --packages-select yolo_person_detector --symlink-install
source install/setup.bash
```

> If `~/aimdk/install/setup.bash` doesn't exist, locate it with:
> `find ~ -name "setup.bash" -path "*/install/*" 2>/dev/null`

### Pull updates later

```bash
cd ~/ros2_ws/src/robohack2026
git pull
cd ~/ros2_ws
colcon build --packages-select yolo_person_detector --symlink-install
source install/setup.bash
```

If the build reports a missing launch file under
`src/robohack2026/install/yolo_person_detector/...`, a previous build was run
from inside the repository instead of the workspace root. Clean the stale nested
build products, then rebuild from `~/ros2_ws`:

```bash
cd ~/ros2_ws/src/robohack2026
rm -rf build install log
git pull

cd ~/ros2_ws
rm -rf build/yolo_person_detector install/yolo_person_detector
colcon build --packages-select yolo_person_detector --symlink-install
source install/setup.bash
```

---

## 3. Running on the robot

SSH in and source all environments:

```bash
source /opt/ros/humble/setup.bash
source ~/aimdk/install/setup.bash
source ~/ros2_ws/install/setup.bash
```

### A. Detection only (safe, robot will not move)

```bash
ros2 launch yolo_person_detector yolo_pipeline.launch.py
```

Optional arguments:

```bash
ros2 launch yolo_person_detector yolo_pipeline.launch.py \
    camera:=rgb_head_front_center \
    device:=cuda \
    confidence:=0.6
```

Monitor the output:

```bash
ros2 topic echo /yolo/detections                       # detection list
ros2 run rqt_image_view rqt_image_view /yolo/detection_image
```

### B. Stereo compressed person + depth view

This uses the front stereo head compressed JPEG pair, runs YOLO on the left
image, adds best-effort stereo depth labels, and publishes one final compressed
JPEG stream. It also publishes the selected person's 3D bbox-center estimate on
`/stereo_person/target_point` as `geometry_msgs/PointStamped`:

```bash
ros2 launch yolo_person_detector stereo_person_pipeline.launch.py \
    device:=cpu \
    confidence:=0.5
```

Use `device:=cuda` only if `python3 -c "import torch; print(torch.cuda.is_available())"`
prints `True` on the robot.

Monitor the final stream on the robot:

```bash
ros2 topic hz /stereo_person/final_annotated_image/compressed --qos-reliability best_effort
ros2 topic echo /stereo_person/target_point
```

On your laptop, after ROS networking is pointed at the robot:

```bash
ros2 run yolo_person_detector view_stereo
```

If `CameraInfo` does not provide a stereo baseline, pass the measured baseline:

```bash
ros2 launch yolo_person_detector stereo_person_pipeline.launch.py \
    device:=cuda \
    baseline_m:=0.06
```

The target point is the perception contract used by the X2 head, waist, and
stereo walking supervisor in `x2_motion_audio_tools`. See
`docs/HUMAN_FOLLOWING_AND_CONTROL.md` for the current control standards and
gantry/off-gantry workflow.

### C. Person follower (robot WILL move)

**Safety checklist before enabling the follower:**

1. Robot is in an open, unobstructed area
2. E-stop ready (Ctrl+C stops publishing velocity; watchdog stops robot within 0.5s of lost detection)
3. Start with default conservative gains

**Start sequence:**

```bash
# Launch with following enabled — the node will put the robot into
# STAND_DEFAULT itself (DD -> JD -> SD via SetMcAction).
ros2 launch yolo_person_detector yolo_follower.launch.py \
    follower_enabled:=true \
    device:=cuda
```

You can also leave `follower_enabled:=false` and flip the switch at runtime:

```bash
ros2 topic pub -1 /yolo/follower/enable std_msgs/Bool "data: true"
# ...and to stop following without killing the node:
ros2 topic pub -1 /yolo/follower/enable std_msgs/Bool "data: false"
```

If you prefer to drive the motion-mode state machine yourself (e.g. with
`ros2 run py_examples set_mc_action SD`), set `auto_enable_locomotion: false`
in `config/yolo_params.yaml` so the node won't issue `SetMcAction` calls.

Stop with `Ctrl+C` — watchdog zeros velocity within 50ms.

### How the follower behaves

Visual servoing (no depth required):

- **Horizontal**: rotates to keep the person's bbox centered
- **Forward**: walks toward the person until bbox height reaches ~60% of image height
- **Safety**: stops if no detection for > 0.5s
- **Target**: largest bounding box (closest person)

All parameters tunable in `config/yolo_params.yaml` → `person_follower`:

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
| `/aima/hal/sensor/rgb_head_front_center/rgb_image` | `sensor_msgs/Image` |
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

**`aimdk_msgs not found` during build**
You forgot to source the aimdk workspace. Find it with:
```bash
find ~ -name "setup.bash" -path "*/install/*" 2>/dev/null
```
Source that file before running `colcon build`.

**`vision_msgs` error during build**
```bash
sudo apt install -y ros-humble-vision-msgs
```

**`cv_bridge` crashes with NumPy 2.x**
ROS Humble's `cv_bridge` is usually built against NumPy 1.x. If you see
`AttributeError: _ARRAY_API not found`, either use the stereo launch files in
this repo, which avoid `cv_bridge`, or pin the robot Python environment back to
NumPy 1.x:
```bash
pip3 install --user "numpy<2"
```

**Robot doesn't move with `follower_enabled:=true`**
Set Stable Stand first: `ros2 run py_examples set_mc_action SD`. Also check the log for `Input source registered as "person_follower"`.

**Detection slow (>100ms per frame)**
Use `device:=cuda` if the robot has a GPU. Check frame rate: `ros2 topic hz /yolo/detections`.

**YOLO model download fails (no internet)**
Copy `yolov8n.pt` from your Mac:
```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
scp ~/.cache/ultralytics/yolov8n.pt agi@10.0.1.40:~/ros2_ws/src/robohack2026/yolo_person_detector/
# Then launch with model:=~/ros2_ws/src/robohack2026/yolo_person_detector/yolov8n.pt
```

**Camera topic silent**
Check the sensor is running: `ros2 topic hz /aima/hal/sensor/rgb_head_front_center/rgb_image`. If empty, the HAL sensor service isn't started on the robot.

---

## 6. End-to-end demo: perception + head + base + arm integration

This is the pipeline demonstration used to bring up the full chair-assist
behavior: stereo perception publishes `/stereo_person/target_point`, the head
tracker points the head at the target, the stereo walking supervisor drives
the base, and on first arrival into the stop band it fires a one-shot trigger
on `/x2/assist/raise_arms_trigger`. The arm node holds the assist pose until a
Bool `true` lands on `/x2/assist/head_pat`, which releases the arms and
resumes normal following.

Run each stage in order. Do not skip stages. Each stage only reads state or
adds one more degree of freedom compared to the previous one.

### 6.0 One-time setup per SSH session (robot)

Open one SSH session for the launch and a second one for inspection and the
head-pat reset. In every new terminal, source all three environments:

```bash
source /opt/ros/humble/setup.bash
source ~/aimdk/install/setup.bash
source ~/ros2_ws/install/setup.bash
```

Confirm the stereo image streams are alive before anything else:

```bash
ros2 topic hz /aima/hal/sensor/stereo_head_front_left/rgb_image/compressed \
  --qos-reliability best_effort
ros2 topic hz /aima/hal/sensor/stereo_head_front_right/rgb_image/compressed \
  --qos-reliability best_effort
```

### 6.1 Stage 1: dry run (no motion, read state only)

Goal: confirm perception, head tracker, walking supervisor, and arm node all
come up and stay in a valid state, without commanding any motion on the robot.

The walking supervisor runs in `follow_dry_run:=true`, so it logs the intended
forward/angular velocities and the `STOP_BAND` / `APPROACH` / `ALIGN` state
but does not publish on `/aima/mc/locomotion/velocity`. The arm node runs
with `auto_start:=false` via the launch: it subscribes to the trigger topic
and reads `/aima/hal/joint/arm/state` but does not publish arm commands until
a trigger arrives (and we do not send one in this stage). The head tracker is
also set to `dry_run:=true` so it only logs computed yaw targets.

```bash
ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py \
  device:=cpu \
  dry_run:=true \
  follow_enabled:=true \
  follow_dry_run:=true \
  assist_arm_pose_enabled:=true \
  assist_head_pat_enabled:=true
```

In the second SSH session, inspect pipeline state:

```bash
# Perception contract
ros2 topic hz   /stereo_person/target_point
ros2 topic echo /stereo_person/target_point --once

# Head tracker input / output topics exist and are connected
ros2 topic info /aima/hal/joint/head/state
ros2 topic info /aima/hal/joint/head/command

# Arm node is subscribed to the trigger and to arm state, but not publishing
ros2 node info /x2_raise_arms_pose
ros2 topic info /x2/assist/raise_arms_trigger
ros2 topic info /aima/hal/joint/arm/command

# Walking supervisor is alive, listening for the target and the enable switch
ros2 node info /x2_stereo_person_follow
ros2 topic info /aima/mc/locomotion/velocity
```

Expected in the launch log:

- `stereo_final_annotator` publishing annotated JPEGs and target points.
- `x2_head_yaw_tracker` logging computed yaw targets; no command publish.
- `x2_stereo_person_follow` printing `follow=APPROACH|STOP_BAND|ALIGN|...`
  lines roughly once per second, with `forward=...` and `angular=...`.
- `x2_raise_arms_pose` logging "Raise-arms pose node ready" and
  `auto_start=false`, with no "Starting arm pose routine." line.

Stop with `Ctrl+C` before moving to stage 2.

### 6.2 Stage 2: gantry run (robot attached, walks in place)

Goal: observe the live, physical behavior of the full pipeline with the robot
still secured to the gantry. The walking supervisor now publishes real
locomotion velocity commands, so the robot may march in place, rotate the
torso, and the arms may actually raise into the assist pose. The head-pat
reset is used to release the arm hold and resume following.

Preflight on the robot:

1. Robot attached to gantry, straps verified.
2. Area in front of the stereo cameras is clear and has an operator who can
   act as the target.
3. Emergency stop in reach.
4. Put the robot in Stable Stand first:

```bash
ros2 run py_examples set_mc_action SD
ros2 run py_examples get_current_input_source
```

Launch the integrated follow + arm pipeline with conservative speed limits:

```bash
ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py \
  device:=cpu \
  follow_enabled:=true \
  follow_dry_run:=false \
  follow_max_forward_speed:=0.15 \
  follow_min_forward_speed:=0.10 \
  follow_max_angular_speed:=0.25 \
  follow_max_forward_bearing_deg:=20.0 \
  follow_stop_min_m:=0.45 \
  follow_stop_max_m:=1.0 \
  follow_target_distance_m:=0.85 \
  depth_disparity_percentile:=75.0 \
  assist_arm_pose_enabled:=true \
  assist_head_pat_enabled:=true
```

If the run is cleared by the operator, you can instead let the launch request
Stable Stand itself by adding `follow_auto_enable_stable_stand:=true`.

In the second SSH session, verify motion actually lands on the bus:

```bash
ros2 topic hz   /aima/mc/locomotion/velocity
ros2 topic echo /aima/mc/locomotion/velocity --once

ros2 topic hz   /aima/hal/joint/head/command
ros2 topic echo /aima/hal/joint/head/command --once

# Arm command only shows traffic once the first STOP_BAND arrival fires
# the trigger; before that, expect 'No publishers on this topic'.
ros2 topic hz /aima/hal/joint/arm/command
```

Expected physical behavior:

- The operator stepping in front of the cameras: head yaws toward the
  operator; base yaws with the head; legs step in place in the direction of
  the target.
- Operator moves inside the stop band: base stops stepping, head continues
  tracking, arms smoothly lift into the assist-ready pose and hold.
- Until `follow_invert_angular:=true` is needed, the base should turn the
  same direction as the head.

Release the arm hold and resume following (second SSH session):

```bash
ros2 topic pub -1 /x2/assist/head_pat std_msgs/Bool "data: true"
```

The supervisor logs "Head-pat reset received; releasing arm hold and resuming
follow." Subsequent stops in the stop band will not re-raise the arms; this
is by design.

Runtime enable/disable of the base follow without killing the launch:

```bash
ros2 topic pub -1 /stereo_person/follow/enable std_msgs/Bool "data: false"
ros2 topic pub -1 /stereo_person/follow/enable std_msgs/Bool "data: true"
```

Stop stage 2 with `Ctrl+C`. The supervisor publishes a zero-velocity stop on
shutdown.

### 6.3 Stage 3: off-gantry deployment (no gantry)

Goal: the same pipeline, without the gantry. Use this only after stage 2 has
been reproduced reliably.

Preflight:

1. Clear area, no obstacles within 2 m of the robot.
2. An operator outside the motion envelope holds the physical E-stop.
3. A second operator is the target.
4. Speed limits start at or below the gantry numbers; lower them further for
   the first untethered run.

On-robot Stable Stand:

```bash
ros2 run py_examples set_mc_action SD
```

Launch with lower speed caps for the first untethered attempt:

```bash
ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py \
  device:=cpu \
  follow_enabled:=true \
  follow_dry_run:=false \
  follow_max_forward_speed:=0.10 \
  follow_min_forward_speed:=0.08 \
  follow_max_angular_speed:=0.20 \
  follow_max_forward_bearing_deg:=15.0 \
  follow_stop_min_m:=0.60 \
  follow_stop_max_m:=1.10 \
  follow_target_distance_m:=0.90 \
  depth_disparity_percentile:=75.0 \
  assist_arm_pose_enabled:=true \
  assist_head_pat_enabled:=true
```

Second SSH session, same inspection topics as stage 2:

```bash
ros2 topic hz   /aima/mc/locomotion/velocity
ros2 topic hz   /aima/hal/joint/head/command
ros2 topic hz   /aima/hal/joint/arm/command
ros2 topic echo /stereo_person/target_point --once
```

Head-pat release, same as stage 2:

```bash
ros2 topic pub -1 /x2/assist/head_pat std_msgs/Bool "data: true"
```

Emergency stop order: physical E-stop first, then disable follow via topic,
then `Ctrl+C` on the launch.

```bash
ros2 topic pub -1 /stereo_person/follow/enable std_msgs/Bool "data: false"
```

### 6.4 Optional: stand-alone arm-pose sanity check

Run before stage 2 if you want to verify just the arm path without any
perception, head, or base motion involved. Hold the robot stable or keep it on
the gantry while the arms move.

```bash
ros2 run py_examples set_mc_action SD

ros2 launch x2_motion_audio_tools x2_raise_arms_pose.launch.py \
  auto_start:=false \
  shoulder_pitch_deg:=10.0 \
  elbow_bend_deg:=90.0 \
  move_seconds:=3.0

# In a second terminal, start the pose:
ros2 topic pub -1 /x2/assist/raise_arms_trigger std_msgs/Bool "data: true"

# Release the hold:
ros2 topic pub -1 /x2/assist/raise_arms_trigger std_msgs/Bool "data: false"
```

### 6.5 Flags worth remembering

- `follow_dry_run:=true` — log only, no locomotion publish. Required for
  stage 1.
- `dry_run:=true` — head tracker logs only, no head-command publish.
  Required for stage 1.
- `assist_arm_pose_enabled:=true` — launch the arm node in trigger mode.
- `assist_head_pat_enabled:=true` — make the base wait for `/x2/assist/head_pat`
  before resuming follow after the first stop.
- `follow_invert_angular:=true` — flip base yaw sign if the body turns the
  wrong way.
- `follow_auto_enable_stable_stand:=true` — the supervisor requests
  STAND_DEFAULT itself instead of requiring a manual `set_mc_action SD`.
- `/stereo_person/follow/enable` — runtime enable/disable of the walking
  supervisor without killing the launch.
- `/x2/assist/head_pat` — Bool `true` releases the arm hold and resumes
  follow; only effective on the first arrival.
