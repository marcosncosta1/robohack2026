# X2 Motion and Audio Tools

ROS 2 helper nodes for AgiBot X2 audio, speech, open-loop coordinate motion,
forward/backward movement, head and torso person tracking, stereo target
following, and cautious arm raising.

This package is intentionally separate from `yolo_person_detector`. The
legacy `x2_person_follow` and `x2_person_track_torso` nodes use the local
`x2_yolo_wrapper.py` implementation so changes to the shared YOLO package
cannot alter their detection path. The newer stereo pipeline flows the other
way: `yolo_person_detector` runs stereo YOLO + depth and publishes
`/stereo_person/target_point`, which this package consumes for head
tracking, optional torso tracking, and conservative high-level locomotion
following.

## Build

From the workspace root:

```bash
source /opt/ros/humble/setup.bash
source ~/aimdk/install/setup.bash
colcon build --packages-up-to x2_motion_audio_tools --symlink-install
source install/setup.bash
```

For the AWS voice assistant and transcription logger:

```bash
python3 -m pip install -r src/x2_motion_audio_tools/requirements-voice.txt
```

For person detection and torso tracking, install this package's local YOLO
dependencies:

```bash
python3 -m pip install -r src/x2_motion_audio_tools/requirements-person.txt
sudo apt install ros-humble-sensor-msgs-py ros-humble-visualization-msgs
```

The person nodes convert the forced raw stereo `Image` messages directly and do
not import `cv_bridge`, avoiding the ROS Humble/NumPy 2 `_ARRAY_API` crash. If
the robot already has NumPy 2 installed from pip, rerun the command above so
`requirements-person.txt` downgrades it to the ROS-compatible NumPy 1.x line.

## Audio and Voice

Log raw sound level without AWS:

```bash
ros2 run x2_motion_audio_tools x2_sound_logger --stream-id 2
```

Log VAD activity and optionally transcribe utterances:

```bash
ros2 run x2_motion_audio_tools x2_mic_logger --stream-id 2 --verbose-levels
ros2 run x2_motion_audio_tools x2_mic_logger --stream-id 2 --transcribe
```

Run the Bedrock voice assistant:

```bash
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0
export TRANSCRIBE_LANGUAGE_CODE=en-US
ros2 run x2_motion_audio_tools x2_bedrock_voice_assistant --stream-id 2
```

`--stream-id 1` uses the onboard mic, `--stream-id 2` uses the external mic,
and `--stream-id 0` accepts either stream.

## Coordinate Offset Motion

`x2_go_to_offset_raise_arms` accepts a target distance and bearing, computes a
point 50 cm to the left or right of that coordinate, turns to the direct line
from the robot to that offset point, walks that one straight line, then raises
both arms. Add `--face-target` if you want a final in-place turn toward the
original target after arriving.

Angle convention:

- `0` degrees means straight ahead
- positive degrees are to the robot's left
- negative degrees are to the robot's right

Dry-run the math without moving:

```bash
ros2 run x2_motion_audio_tools x2_go_to_offset_raise_arms --dry-run
```

Run movement only first:

```bash
ros2 run py_examples set_mc_action SD
ros2 run x2_motion_audio_tools x2_go_to_offset_raise_arms --skip-arms
```

Run with known values:

```bash
ros2 run x2_motion_audio_tools x2_go_to_offset_raise_arms \
  --distance-m 2.0 \
  --angle-deg 15 \
  --side left
```

## Turn Toward Person With TTS

`x2_turn_to_person_tts` says "On my way" through the X2 TTS service, then
rotates the torso/waist toward an input bearing. It uses the HAL waist joint
topic, not locomotion velocity, so it should not step the legs. It accepts
distance too so face/person recognition can pass the same estimate shape later,
but the torso turn only uses the angle.

Dry-run the turn without moving:

```bash
ros2 run x2_motion_audio_tools x2_turn_to_person_tts --dry-run --angle-deg 25
```

Run it on the robot:

```bash
ros2 run x2_motion_audio_tools x2_turn_to_person_tts --angle-deg 25
```

If the torso rotates the wrong direction, add `--invert-waist-direction`.

Future face-recognition handoff:

```bash
ros2 run x2_motion_audio_tools x2_turn_to_person_tts \
  --distance-m FACE_DISTANCE \
  --angle-deg FACE_ANGLE
```

## Stereo Head and Walking Follow

The current stereo interaction stack is launched from
`x2_stereo_head_track.launch.py`. It starts the stereo vision pipeline, head yaw
tracker, and the stereo walking supervisor. It does not start torso control:
`SD` owns the torso/body joints for balance, and commanding waist HAL joints at
the same time can cause vibration.

Default behavior is safe:

- Head tracking is enabled.
- Walking follow is disabled unless `follow_enabled:=true`.
- Walking follow is dry-run unless `follow_dry_run:=false`.

Head tracking only:

```bash
ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py device:=cpu
```

Gantry dry-run walking follow:

```bash
ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py \
  device:=cpu \
  follow_enabled:=true \
  follow_dry_run:=true
```

Add `assist_arm_pose_enabled:=true` to the dry-run command when validating the
integrated assist state. In dry-run, the supervisor logs the timed arm trigger
and `ASSIST_WAIT` state without publishing an arm trigger.

Active gantry follow:

```bash
ros2 run py_examples set_mc_action SD

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
  assist_wait_seconds:=7.0 \
  assist_arm_move_seconds:=3.0 \
  assist_arm_hold_seconds:=3.0 \
  assist_arm_release_seconds:=0.5 \
  assist_arm_hold_indefinitely:=false
```

Once the manual `SD` sequence is trusted, the launch can request Stable Stand
for you:

```bash
ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py \
  device:=cpu \
  follow_enabled:=true \
  follow_dry_run:=false \
  follow_auto_enable_stable_stand:=true \
  follow_max_forward_speed:=0.15 \
  follow_min_forward_speed:=0.10 \
  follow_max_angular_speed:=0.25 \
  follow_max_forward_bearing_deg:=20.0 \
  follow_stop_min_m:=0.45 \
  follow_stop_max_m:=1.0 \
  follow_target_distance_m:=0.85 \
  depth_disparity_percentile:=75.0 \
  assist_arm_pose_enabled:=true \
  assist_wait_seconds:=7.0 \
  assist_arm_move_seconds:=3.0 \
  assist_arm_hold_seconds:=3.0 \
  assist_arm_release_seconds:=0.5 \
  assist_arm_hold_indefinitely:=false
```

The stereo walking supervisor publishes high-level
`aimdk_msgs/McLocomotionVelocity` commands on `/aima/mc/locomotion/velocity`.
It never commands leg joints directly. It consumes
`/stereo_person/target_point`, rotates the base toward the target, walks forward
only when the person is centered and farther than the stop band, and stops in
the `0.45-1.0 m` range by default. The supervisor checks close-range distance
before yaw alignment, so once the target is inside `1.0 m` the base fully stops;
the head can keep tracking without stepping in place.

When `assist_arm_pose_enabled:=true`, the launch also starts
`x2_raise_arms_pose` in trigger mode. The follow supervisor publishes one
arm-pose trigger the first time it reaches `STOP_BAND` or `TOO_CLOSE`, then
holds the base stationary for `assist_wait_seconds` before normal following
resumes. The integrated arm pose is time-based for now: move for
`assist_arm_move_seconds`, hold for `assist_arm_hold_seconds`, release for
`assist_arm_release_seconds`, and do not depend on a head-touch message. Later
stops do not re-trigger the arm pose.

The old waist tracking tools are still available as proof-of-concept utilities,
but do not run them during the `SD` walking demo.

Runtime enable and disable:

```bash
ros2 topic pub -1 /stereo_person/follow/enable std_msgs/Bool "data: true"
ros2 topic pub -1 /stereo_person/follow/enable std_msgs/Bool "data: false"
```

If the base turns away from the person, relaunch with
`follow_invert_angular:=true`.

## Person Body Following

`x2_person_follow` runs the package-local YOLO person detector on the forced
left front stereo camera, says "Hello" once per visible encounter, reads the
chest `PointCloud2` LiDAR, logs camera/YOLO/LiDAR details, registers a
locomotion input source, turns with the legs, walks toward the selected person,
and stops about one meter away. It does not command the waist/torso joints by
default. If LiDAR distance is unavailable, it can still turn toward the
selected person and uses a slow visual fallback capped at
`visual_fallback_max_forward_speed:=0.12` while the target is centered.

Before running, put the robot in Stable Stand and release the remote-controller
channel if your robot image exposes one. On images without an `rc` app, skip
the `aima em stop-app rc` command and check input-source arbitration instead.

```bash
ros2 run py_examples set_mc_action SD
ros2 run py_examples get_current_input_source
```

Run body following:

```bash
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py
```

By default, the node uses the same camera and LiDAR defaults as
`x2_person_track_torso`: `/aima/hal/sensor/stereo_head_front_left/rgb_image`,
intrinsics from `/aima/hal/sensor/stereo_head_front_left/camera_info`, and
`/aima/hal/sensor/lidar_chest_front/lidar_pointcloud`. It walks while the
selected person remains visible, stops when the target is lost, and stops
approaching at `stop_distance_m:=1.0` with a small `stop_deadband_m:=0.12`.
The follow and torso-tracking nodes do not expose camera or LiDAR launch
arguments; both are forced onto this sensor path in code.

The startup log should show:

```text
x2_person_follow stereo-local-yolo direct-image v0.1.2
camera_topic=/aima/hal/sensor/stereo_head_front_left/rgb_image
lidar_topic=/aima/hal/sensor/lidar_chest_front/lidar_pointcloud
```

If the console traceback still says `x2-motion-audio-tools==0.1.0`, the robot is
running an old installed overlay. Rebuild and source the workspace again before
launching.

If you want detection logs only:

```bash
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py \
  follow_enabled:=false
```

Expected detection logs include person count, selected bbox/confidence, camera
bearing, base bearing, camera FPS, LiDAR FPS, valid point count, sector point
count, estimated distance, and current motion reason.

Debug topics are published by default:

- `/x2/person_follow/debug_image` shows all person boxes, the selected `TRACK`
  target, center/deadzone lines, distance, bearing, confidence, and current
  velocity command.
- `/x2/person_follow/debug_markers` shows the selected target ray, LiDAR sector,
  target point, and motion text in RViz.
- `/x2/person_follow/status` publishes JSON status for non-GUI checks.

Open the camera and LiDAR views from launch on a GUI-capable session:

```bash
sudo apt install ros-humble-rqt-image-view ros-humble-rviz2
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py \
  start_image_view:=true \
  start_rviz:=true
```

Or open the debug streams manually:

```bash
ros2 run rqt_image_view rqt_image_view /x2/person_follow/debug_image
rviz2 -d ~/ros2_ws/install/x2_motion_audio_tools/share/x2_motion_audio_tools/launch/x2_person_follow_debug.rviz
ros2 topic echo /x2/person_follow/status
```

If the body turns away from the person, flip the angular sign by using a
negative angular gain:

```bash
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py \
  angular_gain:=-1.0
```

If the LiDAR distance samples the wrong direction, tune the sector:

```bash
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py \
  lidar_angle_offset_deg:=180 \
  lidar_window_deg:=12
```

The follower only walks forward when the target is mostly centered. It turns in
place when the target bearing is outside `max_forward_bearing_deg:=25.0`.
Default speed limits are conservative:
`max_forward_speed:=0.25`, `max_angular_speed:=0.45`, and
`forward_gain:=0.28`. Visual fallback uses
`visual_target_bbox_height_ratio:=0.55`, `visual_stop_deadband_ratio:=0.04`,
and `visual_fallback_max_forward_speed:=0.12`.

```bash
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py \
  stop_distance_m:=1.0 \
  max_forward_speed:=0.18 \
  max_angular_speed:=0.30
```

To disable the visual fallback and require LiDAR before walking:

```bash
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py \
  visual_fallback_enabled:=false
```

The previous torso-only tracker is preserved as `x2_person_track_torso`:

```bash
ros2 launch x2_motion_audio_tools x2_person_track_torso.launch.py follow_enabled:=false
```

## Control Standards

Use high-level locomotion velocity for walking. Do not command walking through
raw leg joint targets.

Use Ruckig-shaped HAL joint commands for head, waist, arms, wrists, and hands.
Initialize from live joint state, publish the full command array for the joint
group, and explicitly hold joints that are not being moved.

The detailed architecture, findings, and gantry/off-gantry workflow are in:

```bash
docs/HUMAN_FOLLOWING_AND_CONTROL.md
```

## Forward, Backward, and Arms

Forward/backward movement only:

```bash
ros2 run py_examples set_mc_action SD
ros2 run x2_motion_audio_tools x2_forward_backward_steps
```

Forward/backward movement followed by a cautious arm raise:

```bash
ros2 run py_examples set_mc_action SD
ros2 run x2_motion_audio_tools x2_forward_back_raise_arms
```

The motion nodes publish high-level locomotion velocity commands. Arm raising
uses low-level HAL joint commands, so only run arm sections when the robot is
stable and your AimDK/HAL safety procedure is satisfied.

Marcos-derived arm-only assist-ready pose test:

```bash
ros2 launch x2_motion_audio_tools x2_raise_arms_pose.launch.py \
  shoulder_pitch_deg:=10.0 \
  elbow_bend_deg:=90.0 \
  move_seconds:=3.0 \
  hold_indefinitely:=false \
  hold_seconds:=3.0 \
  release_seconds:=0.5
```

Manual trigger mode:

```bash
ros2 launch x2_motion_audio_tools x2_raise_arms_pose.launch.py auto_start:=false
ros2 topic pub -1 /x2/assist/raise_arms_trigger std_msgs/Bool "data: true"
```

`x2_raise_arms_pose` does not publish locomotion velocity and does not command
waist or torso joints. In finite mode, it stops publishing after the release
ramp so you can watch for post-release jitter with no walking enabled.
