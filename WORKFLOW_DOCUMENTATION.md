# Hexapod Navigation System: Internship Handover Guide

**Audience:** Incoming robotics interns using the Isaac Sim hexapod system.  
**Main controller:** `Scripts/amazing8.py`  
**Project folder on the current workstation:** `/home/user/Desktop/llmforhexapod`  
**Guide updated:** 21 October 2026

This guide follows the usual working sequence: start LM Studio, launch Isaac Sim, load `hexapod_terrain.usd`, open Script Editor and a robot-camera viewport, paste `amazing8.py`, run it, observe the robot, and debug. It also explains how to prepare another workstation, change models, use another robot or terrain, and save a session for the next person.

The controller moves the robot by updating its USD position and heading. For this workflow, the operator configures the robot pose, camera, road boundaries, and navigation behavior. Use the controller's own buttons to control movement during a run.

## Contents

1. [Quick start: the everyday workflow](#1-quick-start-the-everyday-workflow)
2. [Project files and system connections](#2-project-files-and-system-connections)
3. [First-time workstation preparation](#3-first-time-workstation-preparation)
4. [Prepare LM Studio and select the vision model](#4-prepare-lm-studio-and-select-the-vision-model)
5. [Load the scene and arrange the Isaac Sim workspace](#5-load-the-scene-and-arrange-the-isaac-sim-workspace)
6. [Run, pause, reset, and stop the controller](#6-run-pause-reset-and-stop-the-controller)
7. [Read the robot's behavior and debug a run](#7-read-the-robots-behavior-and-debug-a-run)
8. [Use another robot and mount its camera](#8-use-another-robot-and-mount-its-camera)
9. [Set up the camera's ROS 2 graph](#9-set-up-the-cameras-ros-2-graph)
10. [Use another terrain and configure its boundaries](#10-use-another-terrain-and-configure-its-boundaries)
11. [Change the detector, obstacles, or navigation settings](#11-change-the-detector-obstacles-or-navigation-settings)
12. [Save evidence and finish a session](#12-save-evidence-and-finish-a-session)
13. [Troubleshooting](#13-troubleshooting)
14. [Script section map](#14-script-section-map)
15. [Handover checklist and references](#15-handover-checklist-and-references)

## 1. Quick start: the everyday workflow

Use this sequence once the workstation and scene have been prepared. The later sections explain each step in detail.

| Step | Action | Check before proceeding |
|---|---|---|
| 1 | Open LM Studio, load the vision-capable model named in script Section 3.5, and start its local server. | The configured model identifier is available on port `1234`. |
| 2 | In a terminal, run `cd ~/isaacsim`, then `./isaac-sim.sh`. | Isaac Sim finishes loading. |
| 3 | Drag `hexapod_terrain.usd` into the workspace, as in the existing workflow, or use **File → Open** to open it as the stage. | The robot is at `/World/phantomX_base`; the road and obstacles are visible. |
| 4 | Open **Script Editor** from the window/view menus. | A Python editor and its output area are available. |
| 5 | Open another viewport and select `/World/phantomX_base/Camera`. | One viewport shows the road; the other shows the robot's forward view. |
| 6 | Open `Scripts/amazing8.py` in the text editor, copy all its contents, and paste into Script Editor. | This is the latest saved controller, with the correct local paths and model settings. |
| 7 | Click **Run**. | The floating control panel appears and the console reports controller registration. |
| 8 | Observe the run, inspect telemetry, and use **Pause/Play**, **Stop**, or **Reset** as needed. | RGB and depth frames arrive; movement and obstacle decisions are visible. |
| 9 | Save the screenshots, relevant logs, and run notes before resetting. | The session can be reproduced by another intern. |

**Current defaults:** Automatic looping and image collection are enabled. At a natural mission finish, the robot waits approximately two seconds, resets, and starts another run. For a single recorded run, set `AUTO_LOOP_ENABLED = False` in Section 3.10 **before** pasting and running the script.

## 2. Project files and system connections

### 2.1 Files to keep together

| File or folder | Role |
|---|---|
| `hexapod_terrain.usd` | Main saved simulation scene. Start here for the usual workflow. |
| `Scripts/amazing8.py` | Main controller used in this guide. |
| `usd_to_stl_conversion/phantomX_base.usd` | Robot asset referenced into the scene or spawned by the controller. |
| `dataset/runs/detect/train-2/weights/best.onnx` | Detector loaded by the main controller. |
| `dataset/yolo_dataset/custom.yaml` | Dataset configuration containing the class ID/name mapping. |
| `dataset/auto_collect/` | Default destination for encounter image bursts; created when needed. |
| `Test-Results/` | Existing screenshots and notes from previous trials. |
| `Logs/` | Existing manually saved logs. Choose a session folder when saving new evidence. |
| `documents/internship/RUN_RECORD_TEMPLATE.md` | Reusable record for a new test session. |

Copy the **whole project folder** when moving the system. USD files may reference other assets, textures, and files inside this folder. Updating a Python path does not automatically update asset references stored in a USD scene.

There are several historical controllers in `Scripts/`. Use `amazing8.py` for this workflow. The existing `Scripts/run_sim.sh` names `llmattempt2.py` as its entry script, so use the manual launch-and-paste procedure here when working with `amazing8.py`.

### 2.2 What communicates with what

```text
Isaac Sim scene and robot-mounted Camera
           |
           v
Camera render product and ROS 2 graph
           | /rgb                  | /depth
           v                       v
       RGB frames             sector distances
           |                       |
           +------ amazing8.py ----+
                       |
           +-----------+------------+
           |                        |
      YOLO ONNX                LM Studio VLM
      fast detection           image + prompt -> JSON
           |                        |
           +-----------+------------+
                       |
        category, risk, stress, memory, maneuver state
                       |
           USD translation/rotation and breadcrumbs
                       |
              viewports, UI, console
```

LM Studio supplies semantic decisions through HTTP. ROS 2 carries camera images. The Python controller combines the inputs and applies the movement.

### 2.3 Saved scene paths to recognize

These paths were read from the supplied `hexapod_terrain.usd`:

| Item | Saved stage path |
|---|---|
| Robot root | `/World/phantomX_base` |
| Robot camera | `/World/phantomX_base/Camera` |
| Camera graph | `/World/phantomX_base/Camera/Graph/ROS_Camera` |
| Example obstacles | `/World/Cone_01`, `/World/Car_01`, `/World/Cow_01` |
| Breadcrumb group | `/World/Breadcrumbs` |
| Navigation volumes | `/NavMeshVolume` and `/World/Terrain/section_00_asphalt/NavMeshVolume` |

The navigation-volume lookup currently written in controller Section 5 is `/World/NavMeshVolume`. When explicitly binding a terrain to a volume, select the intended volume in Stage and use its **actual full path**, as described in Section 10. If that lookup finds no prim, the script uses its configured fallback rectangle.

## 3. First-time workstation preparation

### 3.1 Confirm the installed software

Prepare these components before the first full run:

- An Isaac Sim installation that can open the supplied scene, with its graphics requirements satisfied.
- LM Studio and a downloaded, image-capable model.
- Isaac Sim's ROS 2 bridge enabled.
- Python imports available **inside Isaac Sim**: `numpy`, `onnxruntime`, `cv2`, `rclpy`, `sensor_msgs`, and `cv_bridge` for the full camera-to-VLM path.
- The robot asset, detector model, terrain, and their referenced files on disk.

The current workstation's `~/isaacsim/VERSION` identifies a **6.0.1 release-candidate build**. Use that file or Isaac Sim's About dialog when recording the installation. Menu names and ROS setup paths can differ between installations; preserve the working environment when transferring the project.

### 3.2 Launch in the appropriate environment

For the supplied workstation, the normal command is:

```bash
cd ~/isaacsim
./isaac-sim.sh
```

The installed launcher calls `setup_ros_env.sh` automatically. If `ROS_DISTRO` is unset, that setup selects bundled ROS libraries based on the operating system. A separate system ROS installation is therefore not a prerequisite just to follow this workstation's normal GUI workflow.

Use a terminal outside the project's `.venv` or a Conda environment for the simulator launch. If your team has configured an external ROS workspace, use its documented Isaac Sim-compatible setup. Keep ROS command-line tools in a separate terminal when appropriate; sourcing an arbitrary ROS distribution into the simulator terminal is not a general fix for an import problem. See [NVIDIA's ROS installation guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_ros.html) for release-specific instructions.

### 3.3 Enable the extensions and check imports

1. Launch Isaac Sim and open its **Extensions** window.
2. Find and enable **ROS 2 Bridge** (`isaacsim.ros2.bridge`). Its supporting extensions load with it in the installed build.
3. Enable Script Editor if its menu entry is unavailable. Its extension is `omni.kit.window.script_editor`.
4. In a separate Script Editor tab, run this import check:

```python
import importlib
import sys

print("Isaac Sim Python:", sys.version)
for name in ("numpy", "onnxruntime", "cv2", "rclpy",
             "sensor_msgs.msg", "cv_bridge"):
    try:
        importlib.import_module(name)
        print(name, "OK")
    except Exception as exc:
        print(name, "CHECK:", exc)
```

Run this inside the GUI, where Isaac Sim extensions are loaded. A successful import in the project's `.venv` does not establish that the simulator can import the same package.

If `onnxruntime` or `cv2` is missing, close Isaac Sim and install the missing package into the Isaac Sim environment using its Python launcher, then reopen and repeat the check:

```bash
cd ~/isaacsim
./python.sh -m pip install onnxruntime opencv-python
```

Install only packages that are missing; keep existing working versions. For `rclpy`, `sensor_msgs`, or `cv_bridge`, first check the ROS bridge and its matching ROS environment. Avoid installing a random ROS Python package into an unrelated interpreter.

**Why check `cv_bridge` as well as OpenCV?** The current script uses the successful `CvBridge` import to enable its image-encoding path. Passing both imports is part of preparing the full VLM workflow.

### 3.4 Update workstation-specific paths

Open `amazing8.py` and search for `/home/user/Desktop/llmforhexapod`. Review each occurrence when the project location changes.

| Setting or location | Where to find it | What to update |
|---|---|---|
| `ROBOT_USD_PATH` | Section 3.1 | Robot asset location. |
| `AddReference(...)` inside `spawn_phantomx()` | Section 5 | Actual reference used when the robot is absent. Update this as well as the configuration variable. |
| `YOLO_ONNX_PATH` | Section 3.6 | Detector file location. |
| `AUTO_COLLECT_DIR` | Section 3.9 | Writable destination for saved camera images. |
| Robot and camera asset references | USD Stage/Layer properties | Resolve any moved files. |

Before the first run, verify the files in a terminal:

```bash
cd /home/user/Desktop/llmforhexapod
ls -l hexapod_terrain.usd \
  Scripts/amazing8.py \
  usd_to_stl_conversion/phantomX_base.usd \
  dataset/runs/detect/train-2/weights/best.onnx
```

Change the project path above on another workstation.

## 4. Prepare LM Studio and select the vision model

### 4.1 Start the local server

1. Open LM Studio.
2. Load the model configured in `VLM_MODEL`. The current script uses `qwen3.5-9b`.
3. Confirm that the loaded model accepts images. A downloaded model being visible in the model list is not the same as confirming its vision capability.
4. Open LM Studio's **Developer / Local Server** page and start the server on port `1234`.
5. Confirm the API model identifier. Use the server's identifier rather than guessing from the displayed filename.
6. Keep LM Studio and the model running throughout the simulation session.

For a local server without authentication enabled, list the available model IDs with:

```bash
curl --fail --silent --show-error http://127.0.0.1:1234/v1/models
```

If server authentication is enabled, supply the token in the `Authorization: Bearer` header. LM Studio's [authentication instructions](https://lmstudio.ai/docs/developer/core/authentication) explain where to create a token.

### 4.2 The exact configuration section to edit

Find **Section 3.5 — Vision-Language Model (VLM) / LM Studio API Settings**. The settings below show the current values, with the API-key line shown as an environment-based example rather than reproducing a workstation credential:

```python
# 3.5 Vision-Language Model (VLM) / LM Studio API Settings
VLM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
VLM_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
VLM_MODEL = "qwen3.5-9b"
VLM_TIMEOUT = 360
VLM_MAX_RETRIES = 0
VLM_MAX_TOKENS = 3000
VLM_PRIORITY_MODE = True
HIGH_STRESS_VLM_COOLDOWN = 4.0
```

`VLM_API_KEY` in the actual script already reads `LM_STUDIO_API_KEY`; the example changes only its fallback text. For an authenticated server, set the environment variable in the **terminal that launches Isaac Sim**, before opening the application:

```bash
read -r -s -p 'LM Studio API token: ' LM_STUDIO_API_KEY
printf '\n'
export LM_STUDIO_API_KEY
cd ~/isaacsim
./isaac-sim.sh
```

Enter your token at the prompt. If Isaac Sim was already open before exporting the variable, relaunch it from this terminal to inherit the value.

| Setting | What it controls |
|---|---|
| `VLM_API_URL` | Full chat-completions endpoint, including `/v1/chat/completions`. |
| `VLM_MODEL` | Model identifier sent in each request. |
| `VLM_TIMEOUT` | HTTP wait limit in seconds; current value is six minutes. |
| `VLM_MAX_RETRIES` | Additional attempts after a failed request. Zero means no retry after the initial attempt. |
| `VLM_MAX_TOKENS` | Maximum response length requested. |
| `VLM_PRIORITY_MODE` | Prioritizes initial classification of a new obstacle; the robot can halt for the response. |
| `HIGH_STRESS_VLM_COOLDOWN` | Shorter query interval used in the high-stress band. |

The normal cooldown is `MIN_VLM_COOLDOWN = 15.0` in Section 3.3. A priority initial classification can bypass that cooldown.

### 4.3 Change to another model

1. Stop the current robot run.
2. Load the replacement vision model in LM Studio.
3. Confirm an image can be submitted to that model.
4. Copy the exact server model ID into `VLM_MODEL`.
5. If the server host or port changed, update `VLM_API_URL` too.
6. Keep the JSON response fields expected by the controller. The prompt is built inside `async_query_vlm_agent()` in Section 11; the callbacks that consume it are in Section 16.
7. Save the Python file, replace the entire Script Editor contents, and run again.
8. Observe one obstacle encounter and check LM Studio's request log plus the controller's `[VLM Query]` and `[VLM Callback]` messages.

The initial obstacle classification expects fields shaped like this example:

```json
{
  "category": "STATIC_BAR",
  "obstacle_class": "Traffic Cone",
  "risk_score": 3,
  "currently_stationary": true,
  "command": "standard_bypass",
  "reasoning": "A cone is visible in the robot's path.",
  "center_blocked": true,
  "left_sector_coverage_pct": 20,
  "right_sector_coverage_pct": 0
}
```

This is an illustrative response, not a forced decision for every cone image. Other phases request different command sets. When editing a prompt, review the branch for that phase and its receiving callback together. Preserve the exact category and command spellings.

For another server, confirm that it accepts this script's image payload and chat-completions response structure. See [LM Studio's chat-completions API](https://lmstudio.ai/docs/developer/openai-compat/chat-completions).

## 5. Load the scene and arrange the Isaac Sim workspace

### 5.1 Load the saved terrain

1. Launch Isaac Sim and wait for the main interface to finish loading.
2. Locate `hexapod_terrain.usd` in the project folder.
3. Drag it into the workspace using the established workflow. Alternatively, choose **File → Open** to open the USD as the main stage.
4. Check the Stage hierarchy. The controller must be able to find `/World/phantomX_base` at that exact path.
5. If dragging the file placed everything beneath an extra parent, reopen it through **File → Open** or adjust the configured paths to the composed hierarchy.
6. Wait for referenced geometry and materials to finish loading.

For a new terrain or obstacle arrangement, use **Save As** to create a working scene before editing it. Keep a recognizable baseline scene for the next session.

### 5.2 Open Script Editor

Open **Window → Script Editor** in the installed build. If your layout exposes it under a **View** menu, select the same Script Editor tool there. The local Script Editor extension registers its window under **Window**.

Keep the output area or Console visible. This is where import errors, sensor telemetry, model requests, and maneuver messages appear.

### 5.3 Set up two viewports

1. Keep the main viewport on **Perspective** and position it above the road.
2. Create a second viewport from the viewport/window menu and dock it beside the first.
3. Open that viewport's camera selector.
4. Choose the existing camera at `/World/phantomX_base/Camera`.
5. Check that the camera sees the road ahead and stays attached as the robot moves.

Use the Perspective viewport for the overall trajectory. Use the robot-camera viewport to understand what the detector and VLM can see. Select the camera for viewing rather than moving the mounted camera to get a better overview. A viewport by itself does not create the ROS publishers; the camera graph supplies those.

## 6. Run, pause, reset, and stop the controller

### 6.1 Run the latest saved script

1. Finish the LM Studio and sensor checks.
2. Open `Scripts/amazing8.py` in the IDE.
3. Check the paths, `VLM_MODEL`, and automatic-loop setting.
4. Save the file.
5. Copy **all** of its contents and replace the Script Editor tab's existing contents.
6. Clear any partial text selection, then click **Run** once. The installed editor executes selected text if a selection is active; use the whole script for controller startup.

The script starts the simulation timeline if it is not playing, obtains or spawns the robot, creates its ROS subscriptions and control panel, loads the detector when its file exists, and registers the per-frame controller.

Look for these messages:

```text
[YOLO] Model loaded: .../best.onnx
[YOLO] Providers: ['CPUExecutionProvider']
[YOLO] Model loaded and worker started.
[PhantomX] Controller successfully registered and active.
```

The floating window is titled **PhantomX Turn-Analyze-Verify Controller**. Confirm that movement, camera images, and telemetry update. Registration alone confirms that startup reached the event subscription; use sensor observations to confirm a full working run.

### 6.2 Use the control panel

| Control | When to use it | What happens |
|---|---|---|
| **Pause/Play** | Inspect the robot or camera mid-run. | Toggles the controller's manual-pause state and resumes its saved state. |
| **Stop** | End movement, interrupt a run, or prevent another automatic loop. | Sets the controller to `STOPPED` and clears the pending loop restart. Use Reset for a new run. |
| **Reset** | Start a fresh trial. | Returns to the configured spawn, clears breadcrumbs and encounter/trauma memories, resets stress and tracking state, and immediately starts moving again. |

Use these buttons for robot motion. Isaac Sim's timeline **Play/Pause/Stop** is a separate control for simulation and graph execution; it does not replace the controller's Pause or Stop button, because the controller is subscribed to Kit application updates.

A controller Stop does not stop LM Studio or necessarily cancel an HTTP request already being processed by the server. For a fully clean application session, stop the robot and then close Isaac Sim.

### 6.3 Re-run after editing

1. Click the controller's **Stop** button.
2. Save the edited `.py` file in the IDE.
3. Replace the complete Script Editor contents with that saved file.
4. Run it again in the same tab/context, so its cleanup code can see the previous controller globals.
5. Confirm one visible control panel and one active controller instance.
6. If multiple copies were started in different tabs or the sensor graph was substantially changed, close and reopen Isaac Sim, reload the scene, and run once.

An edit made only in Script Editor is not automatically saved back to the file in the IDE. Keep the `.py` file as the maintained source and copy from it for each run.

## 7. Read the robot's behavior and debug a run

### 7.1 Read the panel and console

| Readout | How to interpret it |
|---|---|
| **Target** | Configured destination in the scene's XY plane. |
| **Obstacle Dist** | Minimum retained depth value across the sampled left, center, and right sectors. |
| **VLM** | Current semantic decision or query status. A decision can also originate from a detector fast path or memory. |
| **Detour Ledger** | Lateral displacement tracked during avoidance and lane return. |
| **Stress** | Risk state from 0 to 10 and its LOW/MEDIUM/HIGH band. |
| **Potential** | The controller's displayed speed-and-urgency value. |
| **Status** | Current movement, inspection, waiting, reset, or stopped state. |

Telemetry is normally printed every `PRINT_INTERVAL = 2.0` seconds. It includes pose, heading, sector distances, category, phase, stress, and depth-threat information.

Depth sampling uses the middle vertical band of the image, from 35% to 65% of image height, split into left/center/right sectors. It retains finite values above `0.05` m. With a known obstacle in view, check that the appropriate sector reports a sensible distance. An all-clear display can mean no valid samples; check that frames are actually arriving before interpreting it as an empty road.

### 7.2 Recognize the main behaviors

| Category | Typical first behavior | What to watch |
|---|---|---|
| `BIOLOGICAL` | Yield and monitor. | Waiting, reinspection, and subsequent clearance or bypass decisions. |
| `STATIC_BAR` | Bypass a rigid obstacle. | Divergence, flank motion, and lane return. |
| `MINOR_SURFACE` | Scan/proceed behavior. | The selected maneuver and speed response around the surface feature. |
| `CLEAR` | Resume-path/false-alarm handling. | Sector/depth confirmation and return to sweeping. |

Use the console's decision source to understand an encounter. `[YOLO Fast-Path]` indicates a confident detector classification. `[Memory]` messages indicate a remembered encounter. `[VLM Query]` indicates a request to LM Studio. A known obstacle does not necessarily cause a fresh VLM request every time.

Stress can change speed, clearance behavior, and query timing. Tune these through the named parameters in Sections 3.7–3.9 after observing a baseline run.

### 7.3 Read the breadcrumbs

Breadcrumbs are created beneath `/World/Breadcrumbs`.

| Color | Usual meaning in this controller |
|---|---|
| Green / red | Sweeping in opposite Y directions. |
| Blue | Edge and lane-transition markers. |
| Cyan | Flank-forward movement. |
| Yellow | Divergence steps. |
| Purple shades | Big-skip maneuvers and transitions. |

Use the trail to trace which movement sequence occurred, alongside the console phase messages. Save a screenshot before Reset or an automatic loop clears it.

### 7.4 Use a repeatable debugging sequence

1. Run the unchanged baseline scene and controller once.
2. Pause when the behavior you want to inspect occurs.
3. Capture the global viewport, robot-camera view, and relevant console lines.
4. Identify the layer to inspect: scene/path, camera/ROS, YOLO, VLM, or movement state.
5. Change one setting or one scene feature at a time.
6. Save the changed file or scene, reset or relaunch as appropriate, and repeat.
7. Record what changed and what was observed in the run record.

For a particular encounter, keep the image seen by the robot together with its depth values, semantic response, and movement result. This makes the next debugging session much easier to follow.

## 8. Use another robot and mount its camera

### 8.1 Prepare the robot asset

1. Save a copy of the replacement robot USD and its referenced assets.
2. Open it in Isaac Sim and confirm its units and size. The supplied terrain and PhantomX asset are authored in meters with **Z up**.
3. Identify the root Xform that the controller should translate and rotate.
4. Create a Camera beneath a transform that follows the body. The existing robot uses a Camera directly under its root; a new model may use a chassis/body child.
5. Position the camera close to the chassis at a useful height, with the lens outside obstructing body geometry.
6. Orient its view down the travel direction. Select it in a viewport to check the image instead of relying on rotation numbers alone.
7. Attach or rebuild the camera's ROS graph as described in Section 9.
8. Save the robot asset and reference it into a working terrain scene.

Keep the sensor parented so the camera follows both translation and turns. If the new robot has an independent physics controller, configure the scene so the controller's USD pose updates govern the root used for this workflow.

### 8.2 Configure the controller

In Section 3.1, update:

```python
ROBOT_PATH = "/World/YourRobot"
ROBOT_USD_PATH = "/absolute/path/to/your_robot.usd"
SPAWN_POSITION = (0.0, 1.0, 0.1)   # Example: choose for your terrain
SPAWN_ROTATION = (0.0, 0.0, -90.0)
ROBOT_LENGTH = 0.55                # Replace with your robot's dimension
```

These are example values, not a new default for PhantomX. Also update the literal asset path inside `spawn_phantomx()` in Section 5 when using automatic spawning. The current spawning function contains its own `AddReference(...)` path.

If a robot already exists at `ROBOT_PATH`, the function uses that prim rather than replacing its asset. To switch models, explicitly replace the reference in the stage or remove the old robot from a saved working scene before allowing the script to spawn the new one.

After changing dimensions, review `DIVERGE_CLEARANCE_MARGINE`, `MIN_CHASSIS_CLEAR`, category flank clearances, and `ROBOT_SAFETY_MARGIN` in Section 5. Check the camera graph's target after the robot is referenced into the terrain: it must point to the composed camera path under `/World/YourRobot/...`.

## 9. Set up the camera's ROS 2 graph

The supplied PhantomX asset already contains the camera graph. Inspect and reuse it for the normal workflow; build a graph when preparing a replacement camera or model.

### 9.1 Existing graph layout

Open the graph at `/World/phantomX_base/Camera/Graph/ROS_Camera` in the Action Graph editor. The saved graph contains:

```text
OnPlaybackTick -> RunOnce -> RenderProduct
                                  |
                     +------------+------------+
                     v            v            v
                 RGBPublish   DepthPublish  CameraInfoPublish
                     ^            ^            ^
                     +------------+------------+
                                  |
                               Context
```

`RenderProduct` targets `/World/phantomX_base/Camera`. RGB and depth helpers receive the render-product path and execution output. `Context` supplies their ROS context. The saved graph publishes `/rgb`, `/depth`, and `camera_info` with frame ID `sim_camera`.

### 9.2 Build or adapt a graph

1. Enable the ROS 2 bridge and open/create an Action Graph for the camera.
2. Add the playback tick, ROS context, render-product creation node, and two camera helpers. When matching the existing asset, retain its run-once initialization node as well.
3. Set the render-product camera target to the robot-mounted Camera's **stage path**.
4. Set a practical render resolution, for example `640 × 480`. This is the camera resolution; the detector separately resizes input to `640 × 640`.
5. Configure one camera helper for `type = rgb`, topic `/rgb`.
6. Configure the other for `type = depth`, topic `/depth`.
7. Connect execution from the render-product node to the helpers' execution inputs.
8. Connect the render-product path output to both helpers' render-product inputs.
9. Connect ROS Context's output to each helper's context input.
10. Use a common frame ID and a ROS domain matching the controller. Domain `0` is the usual starting setting; align the graph's domain setting and `ROS_DOMAIN_ID` if you customize it.
11. Press timeline Play, allow the render graph to initialize, and check the streams.
12. Save the graph in the robot asset or working scene where you intended to author it.

Node names or supported settings can vary by Isaac Sim release; use the working graph in the supplied USD as the project-specific reference. NVIDIA's [ROS 2 camera tutorial](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_series/tutorial_ros2_camera.html) explains the camera-helper arrangement.

### 9.3 Match the script's subscriptions

Section 3.10 currently contains:

```python
RGB_TOPICS = ["/rgb", "/camera/rgb", "/phantomx/camera/rgb"]
DEPTH_TOPICS = ["/depth", "/camera/depth", "/phantomx/camera/depth"]
```

Use `/rgb` and `/depth` for the simplest configuration. If you rename or namespace topics, update the lists or make the graph publish one of the listed names. When multiple robots/cameras are active, choose one intended RGB/depth pair and narrow these lists to that pair before running.

The controller subscribes to `sensor_msgs/msg/Image`. Use raw color images and floating-point depth in meters (`32FC1`) for this workflow; a depth visualization or compressed topic is a different message/data format.

If ROS CLI tools are installed, run these in their configured ROS terminal while the simulation is playing:

```bash
ros2 topic list
ros2 topic info /rgb
ros2 topic info /depth
ros2 topic hz /rgb
ros2 topic hz /depth
```

Press `Ctrl+C` to stop each rate monitor. On a workstation without ROS CLI tools, the runtime check below provides a direct controller-side check.

### 9.4 Check the frames in Script Editor

After starting the controller, click its **Stop** button, leave the simulation timeline playing, and run this in the **same Script Editor context**:

```python
print("RGB:", None if _ros_node.latest_rgb is None
      else _ros_node.latest_rgb.shape)
print("Depth:", None if _ros_node.latest_depth is None
      else _ros_node.latest_depth.shape)
print("Sectors:", _ros_node.depth_left_dist,
      _ros_node.depth_center_dist, _ros_node.depth_right_dist)
```

You should see an RGB shape such as `(480, 640, 3)` and a depth shape such as `(480, 640)`, using your configured resolution. Place a known obstacle in view and check that the sector readings change. Click Reset to begin a new run when ready.

## 10. Use another terrain and configure its boundaries

### 10.1 Prepare the scene

1. Open the new terrain and save a working USD scene.
2. Use meter-scale geometry and Z-up orientation, or convert the asset consistently before integrating it.
3. Add the robot and its mounted-camera graph.
4. Create a navigation volume over the road/work area using the installed navigation tools. Enable the navigation extension if the tools are hidden.
5. Position and size the volume to match the intended work area. Use the navigation tools' bake/build action where required for the scene's navigation mesh.
6. Check the volume's full path and dimensions in Stage/Properties.
7. Place the obstacles and choose a clear spawn location.

The current controller reads the selected volume's **world-aligned bounding box** to define a rectangular movement area. Configure that rectangle to lie on the intended road surface. For irregular roads, choose the work rectangle deliberately; creating a baked mesh does not change which geometry this controller reads for its bounds.

### 10.2 Point Section 5 to the chosen volume

The lookup is written directly in Section 5:

```python
bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
)
navmesh_prim = stage.GetPrimAtPath("/World/NavMeshVolume")
```

Replace the string with the selected volume's full path, or create the intended volume at `/World/NavMeshVolume` in your working scene. For example, the supplied scene has a root-level volume at `/NavMeshVolume`; selecting that one means using:

```python
navmesh_prim = stage.GetPrimAtPath("/NavMeshVolume")
```

Choose the volume whose position and size define the work area you want. The scene also has a nested terrain volume, so do not select solely by the name `NavMeshVolume`.

The current fallback rectangle is configured in the same section:

```python
min_x, max_x = -10.0, 12.0
min_y, max_y = -2.0, 2.0
ROBOT_SAFETY_MARGIN = 0.35
```

When a valid selected volume is found, its computed bounds replace the rectangle. The script then shrinks each side by the safety margin. After running and stopping the controller, inspect the **actual** values in its Script Editor context:

```python
print("Selected volume exists:", navmesh_prim.IsValid())
print("Raw bounds:", min_x, max_x, min_y, max_y)
print("Safe X:", SAFE_MIN_X, SAFE_MAX_X)
print("Safe Y:", SAFE_MIN_Y, SAFE_MAX_Y)
```

Check that the ranges are finite, ordered, and cover the intended road rectangle. Use the supplied Isaac Sim environment for this check, where navigation schemas/extensions are loaded.

### 10.3 Set the spawn and destination together

Edit Section 3.1 for the spawn and Section 3.2 for the destination. The existing PhantomX settings are:

```python
SPAWN_POSITION = (0.11952272590602997, 1.8697408479533721, 0.1)
SPAWN_ROTATION = (0.0, 0.0, -90.0)
TARGET_DESTINATION = (9, -1.8266811603307502)
TARGET_RADIUS = 0.3
```

For a new terrain:

1. Choose a spawn inside the configured safe rectangle and set Z so the robot sits at the desired height.
2. Choose a destination and progress direction compatible with that rectangle.
3. Use heading `-90` for the existing initial sweep toward negative Y; the controller also explicitly recognizes `90` for the opposite initial sweep.
4. Keep the road's main progress direction aligned with world X for the existing sweep workflow. Heading `0` points along positive X; `90` points along positive Y.
5. Review `LANE_SHIFT`, `STEP_DISTANCE`, speed, and clearances for the new road width and robot size.
6. Save the controller and scene, run one trial, then confirm the path in the overhead viewport.

Mission completion can occur within `TARGET_RADIUS` or when the robot reaches/passes the destination's X coordinate in its progress direction. Choose the target X accordingly and keep it reachable within the configured bounds. Set `AUTO_LOOP_ENABLED = False` when checking a single finish point.

## 11. Change the detector, obstacles, or navigation settings

### 11.1 Use another YOLO model

The current model is loaded on the CPU through ONNX Runtime. Its inspected input is float32 `[1, 3, 640, 640]`; its output is `[1, 300, 6]`, containing detection rows interpreted as `x1, y1, x2, y2, confidence, class_id`.

1. Export a compatible ONNX detector and keep its class names with it.
2. Set `YOLO_ONNX_PATH` in Section 3.6 to that model.
3. Match `YOLO_INPUT_SIZE` to the export.
4. Match `YOLO_CLASS_NAMES` to the model's exact class ID order.
5. Update `YOLO_CLASS_TO_CATEGORY` for the desired behaviors.
6. Confirm the exported output layout matches `yolo_worker()` in Section 8. For the current parser, preserve the six-value detection-row layout, or adapt the parser alongside the export.
7. Restart the controller and inspect `[YOLO Debug]` output against the camera view.

The supplied model's metadata and dataset YAML list these classes:

| ID | Name | Controller category |
|---|---|---|
| 0 | `cow` | `BIOLOGICAL` |
| 1 | `person` | `BIOLOGICAL` |
| 2 | `cone` | `STATIC_BAR` |
| 3 | `car` | `STATIC_BAR` |
| 4 | `pothole` | `MINOR_SURFACE` |

`YOLO_CONF_THRESHOLD` is currently `0.45`. The fast-path decision adds its own checks: five consistent classification hits, confidence at least `0.60`, and distance agreement within `0.6` m. A detection appearing in debug output therefore does not always result in an immediate fast-path maneuver.

### 11.2 Add or rename obstacles

1. Add the obstacle to the saved working scene.
2. Position it in the camera-visible road area for the encounter you want to observe.
3. If using semantic labels for synthetic data, update the `LABELS` dictionary in Section 4 with the actual stage prim path and intended class.
4. Keep the model's class mapping, dataset labels, and controller category mapping consistent when training or replacing the detector.
5. Test the new arrangement and save its scene name in the run record.

`LABELS` applies USD semantic tags; it does not train the ONNX detector. A `SKIP (not found)` label message means that optional named obstacle is absent from this scene, so check the dictionary only when you intended that object to be included.

### 11.3 Common settings to tune

| Parameter | Current value | Section | Operator use |
|---|---|---|---|
| `FORWARD_SPEED` | `0.5` m/s | 3.2 | Base translation speed. |
| `TURN_SPEED` | `180.0` deg/s | 3.2 | Turning rate. |
| `STEP_DISTANCE` | `0.4` m | 3.2 | Sweep distance between scan pauses. |
| `PAUSE_DURATION` | `0.5` s | 3.2 | Scan-pause duration. |
| `LANE_SHIFT` | `0.5` m | 3.2 | Progression between sweep lanes. |
| `OBSTACLE_THRESHOLD` | `0.8` m | 3.2 | Ordinary depth-triggered encounter distance. |
| `EMERGENCY_STOP_DIST` | `0.25` m | 3.4 | Emergency proximity threshold. |
| `MIN_VLM_COOLDOWN` | `15.0` s | 3.3 | Routine VLM cooldown. |
| `ROBOT_SAFETY_MARGIN` | `0.35` m | 5 | Inset from selected movement bounds. |
| `PRINT_INTERVAL` | `2.0` s | 3.2 | Console telemetry interval. |

Change one setting at a time, save it, re-paste the controller, and record the value used. The displayed movement speed may differ from `FORWARD_SPEED` because the stress band and maneuver apply multipliers.

## 12. Save evidence and finish a session

### 12.1 Automatic image collection

Section 3.9 currently contains:

```python
AUTO_COLLECT_ENABLED = True
AUTO_COLLECT_DIR = "/home/user/Desktop/llmforhexapod/dataset/auto_collect"
BURST_FRAMES = 5       # Defined earlier in Section 3.5
BURST_INTERVAL = 0.25  # Defined earlier in Section 3.5
BURST_MIN_GAP = 3.0    # Defined earlier in Section 3.5
```

When an initial classification resolves, the controller can save a short burst under a category folder such as `BIOLOGICAL`, `STATIC_BAR`, `MINOR_SURFACE`, or `CLEAR`. Decisions whose reasoning is marked as fallback use `UNVERIFIED`. Image filenames include a timestamp, frame index, depth reading, and command tag.

Watch for `[Data Collect]: burst of ... saved -> ...` and confirm that files appear. These are saved JPEG frames for later review/annotation. The HTTP query uses its captured image; the saved burst is a separate collection operation.

To disable collection, set `AUTO_COLLECT_ENABLED = False` and rerun. To separate sessions, set `AUTO_COLLECT_DIR` to a session-specific folder before starting. Keep space available for looping runs.

### 12.2 Save a reproducible run

1. Create a session folder, for example `Test-Results/2026-10-21_terrain1_run01/`.
2. Copy `documents/internship/RUN_RECORD_TEMPLATE.md` into it and fill it in.
3. Record the controller version/copy used, scene name, model identifier, detector path, and changed settings.
4. Capture the overhead trajectory and robot-camera view before Reset or automatic looping.
5. Save the relevant console output, especially decision source, VLM response/command, sector distances, and ending status.
6. Save the working scene under a distinguishable name if its layout changed. Record which USD layer was saved.
7. Note where automatic camera images were written.

The controller prints telemetry rather than creating a dedicated run telemetry file. Save the needed output explicitly. Isaac Sim session logs are also available on this workstation under `~/.nvidia-omniverse/logs/Kit/Isaac-Sim Full/6.0/`; capture the relevant session file if needed, with the matching version folder on other installations.

### 12.3 Shut down

1. Click the controller's **Stop** button to end movement and cancel a pending auto-loop restart.
2. Save the evidence you want to keep.
3. Save intentional scene edits in the intended working USD/layer. Breadcrumbs and runtime pose changes may be present, so decide whether to keep them in that working copy.
4. Close Isaac Sim when finished.
5. Stop LM Studio's local server or unload the model if it is no longer needed.
6. Leave the run record and latest maintained script in an obvious location for the next intern.

## 13. Troubleshooting

| Symptom | Checks and next action |
|---|---|
| `No module named 'onnxruntime'` or `cv2` | Install the missing dependency in Isaac Sim's Python environment, restart, and repeat the GUI import check in Section 3.3. |
| `rclpy`, `sensor_msgs`, or `cv_bridge` import fails | Enable the ROS bridge and verify the simulator-compatible ROS environment. Relaunch through `~/isaacsim/isaac-sim.sh` from a clean terminal on this workstation. |
| Controller or scene prim cannot be found | Open the USD as the stage and compare its actual hierarchy with `ROBOT_PATH`, camera paths, and Section 5's volume lookup. Check for an added parent from drag-and-drop. |
| Robot geometry or textures are missing after moving the project | Resolve USD references in the scene/robot asset and keep the dependent folders together. Then reload the working scene. |
| Robot-camera viewport is black or obstructed | Check lighting, the selected camera, lens position/orientation, and timeline playback. Verify the camera is not looking through the body or away from the road. |
| Camera viewport works, but RGB/depth is `None` | Check ROS graph execution, render-product target, helper types, exact topic names, and domain agreement. Use Section 9.4 and ROS topic tools where available. |
| Depth remains all-clear near an obstacle | Confirm a live `32FC1` depth stream in meters, inspect the sampled image band, and verify the obstacle is in the forward view. |
| `[Perception Error]: RGB image frame is empty or invalid` | Check both received RGB data and the `cv_bridge` import used by the encoding path. Repeat the GUI import and frame checks. |
| HTTP connection refused | Start LM Studio's server and verify the endpoint host/port and model availability. |
| HTTP 401/403 | Match the API token to LM Studio's authentication settings. Export `LM_STUDIO_API_KEY` before relaunching Isaac Sim. |
| Model not found or image input rejected | Use the exact server model ID and confirm vision support. Validate one image interaction before rerunning the controller. |
| JSON parsing error | Inspect the VLM output in LM Studio. Keep the phase's JSON-only instructions, field names, and supported commands. |
| Robot waits in `VLM_INSPECTING` | Inspect server/request activity. The current HTTP timeout is 360 s, with an initial-classification watchdog at `VLM_TIMEOUT + 10` s. Use Stop to end the wait if you want to reconfigure immediately. |
| Fallback maneuver appears instead of a VLM decision | Inspect preceding messages for cooldown, depth-based skip, HTTP failure, or parsing error. Restore the desired model connection or tune the relevant settings, then reset and retry. |
| `[YOLO] Disabled — no model at ...` | Correct `YOLO_ONNX_PATH` and verify the file. Restore it when using the combined detector/VLM workflow. |
| YOLO labels are unexpected | Compare `YOLO_CLASS_NAMES` against the loaded export's class IDs and inspect its output layout. |
| No fresh VLM request for a familiar obstacle | Check for detector fast-path or spatial-memory reuse messages. Use Reset for a fresh encounter history. |
| Robot turns around or follows a boundary unexpectedly | Check actual raw/safe bounds, chosen volume, spawn, heading, destination X, and clearance settings. |
| Robot restarts after finishing | `AUTO_LOOP_ENABLED` is true. Use Stop, or set it false before running a single trial. |
| Breadcrumbs disappear | Reset and natural auto-loop restarts clear them. Save images first or disable auto-loop for a recorded run. |
| Duplicate windows or conflicting movement after re-pasting | Stop the controller, close/reopen Isaac Sim, reload the stage, and execute once in one Script Editor context. |
| Stress strip chart is absent | Read the numeric stress/potential label and the console's `[UI]` message. Check plotting compatibility for the installed Isaac Sim version. |

## 14. Script section map

`amazing8.py` is already organized into 19 numbered sections. Search by section title or function name rather than relying on line numbers after edits.

| Section | What to find there | When an intern uses it |
|---|---|---|
| 1. Imports & Dependency Checks | Isaac Sim, USD, ROS, OpenCV, ONNX imports. | Diagnose startup/import errors. |
| 2. Runtime Cleanup | Previous update subscription, node, worker flag, and window cleanup. | Understand a re-run in the same context. |
| 3. Global Configuration | Subsections 3.1–3.10. | Change model, paths, topics, speeds, thresholds, collection, and loop settings. |
| 4. Stage Initialization & Semantic Labels | Timeline startup and `LABELS`. | Match optional labels to scene obstacles. |
| 5. Robot Spawn & NavMesh Bounds | `spawn_phantomx()`, `AddReference`, volume lookup, bounds, safety margin. | Change robot or terrain. |
| 6. ROS 2 Perception | Image conversion and `PerceptionNode`. | Inspect received frames, encodings, QoS, and depth sectors. |
| 7. Spatial & Memory Helpers | Headings, geometric checks, spatial memory, breadcrumbs. | Follow pose, remembered encounters, and trail behavior. |
| 8. YOLO Detection & Tracking | ONNX loading, preprocessing, output parsing, worker. | Replace the detector or debug detections. |
| 9. Trauma Memory | Remembered high-threat encounters. | Follow context-dependent risk behavior. |
| 10. Continuous Stress | Risk update, band switching, speed helpers. | Inspect stress and speed responses. |
| 11. VLM Client & Burst Capture | `async_query_vlm_agent()`, prompts, HTTP requests, snapshot saving. | Change model interface/prompts or inspect query errors. |
| 12. Runtime State Variables | Movement and encounter state initialization. | Trace state values during debugging. |
| 13. Kinematics & Fallback | Pose updates, movement helpers, deterministic fallback. | Follow movement or non-VLM decisions. |
| 14. Depth Threat Monitor | Depth approach and bumper checks. | Inspect close-obstacle response. |
| 15. Rechecks, Reset & Mission Termination | Stress rechecks, `perform_full_reset()`, `finish_run()`. | Understand resets and looping. |
| 16. Avoidance Strategies & Callbacks | Bypass/lane-shift setup and decision handlers. | Match VLM commands to robot actions. |
| 17. Control Panel | Labels, plots, and button callbacks. | Change the operator UI. |
| 18. Main Update Loop | `zig_zag_update()` and mission completion checks. | Trace a maneuver/state transition. |
| 19. Startup & Subscription | Detector startup and Kit event subscription. | Confirm controller activation. |

For the most common changes: **VLM → 3.5; robot → 3.1 and 5; detector → 3.6 and 8; topics/looping → 3.10; terrain bounds → 5; prompts → 11.**

## 15. Handover checklist and references

### 15.1 Before the next intern starts

- [ ] Project files and USD dependencies are present on the intended workstation.
- [ ] Isaac Sim version and launch command are recorded.
- [ ] GUI import checks pass, including `cv_bridge` for the VLM path.
- [ ] LM Studio model ID, vision capability, port, and authentication setup are recorded.
- [ ] Robot root, camera, graph, topics, and chosen navigation-volume paths are recorded.
- [ ] Detector file, output layout, and class ID mapping are known.
- [ ] Spawn, destination, safe bounds, and automatic-loop setting have been checked.
- [ ] One full run has been demonstrated, including Pause/Play, Stop, and Reset.
- [ ] The latest saved controller and scene are clearly identified.
- [ ] At least one run record contains screenshots, decision messages, and configuration values.

### 15.2 Source and maintenance notes

This guide was checked against `Scripts/amazing8.py`, the saved terrain and robot USD files, the current detector's ONNX metadata, dataset class configuration, and the installed Isaac Sim launcher/extension files. The configuration and paths above describe those inspected files. Live GUI operation and LM Studio inference should be confirmed during the first operator session.

When the workflow changes, update this guide and the run-record template alongside the maintained script. Keep credentials out of the shared record.

External references for installation or UI differences:

- [NVIDIA: ROS 2 installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_ros.html)
- [NVIDIA: ROS 2 cameras](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_series/tutorial_ros2_camera.html)
- [LM Studio: Chat completions](https://lmstudio.ai/docs/developer/openai-compat/chat-completions)
- [LM Studio: Authentication](https://lmstudio.ai/docs/developer/core/authentication)

The web documentation may describe a newer release than the installed workstation. Use the installed version and saved working camera graph when choosing exact settings.
