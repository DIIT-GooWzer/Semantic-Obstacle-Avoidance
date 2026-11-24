"""
PhantomX Target-Directed Controller with Memory-Augmented VLM Avoidance & Depth Slicing
---------------------------------------------------------------------------------------
Features:
1. Depth Array Slicing: Instant Left/Center/Right obstacle clearance without LLM latency.
2. Turn-First-Then-Analyze: Rotates to inspect the flank before committing to movement.
3. Short-Term Working Memory: Tracks past avoidance decisions per obstacle encounter.
4. Look-At-Obstacle Flank Inspection: Actively peeks at the obstacle's side to ensure it has passed.
5. Displacement Odometer Ledger: Precise cross-track cancellation for exact lane return.
"""

import os
import time
import math
import json
import base64
import threading
import urllib.request
import numpy as np
import omni.usd
import omni.timeline
import omni.kit.app
import omni.ui as ui
from pxr import Usd, UsdGeom, Gf

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

try:
    import cv2
    _has_cv2 = True
except Exception:
    _has_cv2 = False

def ros_img_to_numpy(msg, desired_encoding='passthrough'):
    if _has_cv2:
        try:
            bridge = CvBridge()
            return bridge.imgmsg_to_cv2(msg, desired_encoding=desired_encoding)
        except Exception:
            pass
            
    if msg.encoding in ['32FC1', '32f']:
        return np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
    elif msg.encoding in ['rgb8', 'bgr8', '8UC3']:
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        if desired_encoding == 'bgr8' and msg.encoding == 'rgb8':
            arr = arr[:, :, ::-1]
        return arr
    elif msg.encoding in ['mono8', '8UC1']:
        return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
    else:
        return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))

stage = omni.usd.get_context().get_stage()

timeline = omni.timeline.get_timeline_interface()
if not timeline.is_playing():
    timeline.play()

# ============================================================
# 1. CONFIGURATION & STAGE SETUP
# ============================================================
ROBOT_PATH         = "/World/phantomX_base"
SPAWN_POSITION     = (0.3295789538104168, -1.681203353653679, 0.10692057425096305)
SPAWN_ROTATION     = (0.0, 0.0, 90.0)
TARGET_DESTINATION = (10.2, 2.0)
TARGET_RADIUS      = 0.3

STEP_DISTANCE      = 0.5                             # Sweep pause step size (m)
PAUSE_DURATION     = 0.5                             # Sweep pause duration (s)
FORWARD_SPEED      = 1.0                             # Base translation speed (m/s)
TURN_SPEED         = 180.0                           # Base rotation speed (deg/s)
LANE_SHIFT         = 0.5                             # Lane shift width (m)
OBSTACLE_THRESHOLD = 0.95                             # Avoidance trigger distance (m)
MIN_CHASSIS_CLEAR  = 0.40                            # Minimum lateral distance to clear chassis (m)
PRINT_INTERVAL     = 2.0                             # Telemetry print interval (s)

# Avoidance micro-step settings
AVOIDANCE_STEP_SIZE   = 0.30                         # Micro-step distance (m)
DIVERGE_ANGLE_INC     = 35.0                         # Deflection angle away from lane (deg)
MIN_VLM_COOLDOWN      = 1.5                          # Min seconds between successive VLM queries
CLEAR_PASS_SAFETY_PAD = 0.5                          # Extra margin added to flank bypass (m)

# # VLM API Configuration
# VLM_API_URL        = "https://openrouter.ai/api/v1/chat/completions"
# VLM_API_KEY        = os.getenv("OPENROUTER_API_KEY", "sk-or-v1-xxxx")
# VLM_MODEL          = "dots-studio/dots-3-note-preview:free"

# RGB_TOPICS   = ["/rgb", "/camera/rgb", "/phantomx/camera/rgb"]
# DEPTH_TOPICS = ["/depth", "/camera/depth", "/phantomx/camera/depth"]

# VLM API Configuration - Freebuff2API
VLM_API_URL = "http://localhost:8000/v1/chat/completions"
VLM_API_KEY = os.getenv("FREEBUFF_API_KEY", "231970c3-bd57-41a8-8368-1e2d81176618")
VLM_MODEL = "mimo/mimo-v2.5"

RGB_TOPICS   = ["/rgb", "/camera/rgb", "/phantomx/camera/rgb"]
DEPTH_TOPICS = ["/depth", "/camera/depth", "/phantomx/camera/depth"]

# ============================================================
# 2. SPAWN / RETRIEVE ROBOT PRIM & NAVMESH BOUNDS
# ============================================================
def spawn_phantomx(stage, prim_path=ROBOT_PATH, position=SPAWN_POSITION, rotation_euler=SPAWN_ROTATION):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(
            "/home/user/Desktop/llmforhexapod/usd_to_stl_conversion/phantomX_base.usd"
        )

    xform = UsdGeom.Xformable(prim)
    translate_op, rotate_op = None, None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() in (UsdGeom.XformOp.TypeRotateXYZ, UsdGeom.XformOp.TypeRotateZ):
            rotate_op = op

    if not translate_op:
        translate_op = xform.AddTranslateOp()
    if not rotate_op:
        rotate_op = xform.AddRotateXYZOp()

    translate_op.Set(Gf.Vec3d(*position))
    rotate_op.Set(Gf.Vec3d(*rotation_euler))
    return prim, translate_op, rotate_op

robot_prim, translate_op, rotate_op = spawn_phantomx(stage)

bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
navmesh_prim = stage.GetPrimAtPath("/World/Terrain/section_00_asphalt/NavMeshVolume")

min_x, max_x = -10.0, 12.0
min_y, max_y = -2.0, 2.0
if navmesh_prim.IsValid():
    bound = bbox_cache.ComputeWorldBound(navmesh_prim).ComputeAlignedBox()
    min_x, max_x = float(bound.GetMin()[0]), float(bound.GetMax()[0])
    min_y, max_y = float(bound.GetMin()[1]), float(bound.GetMax()[1])

# ============================================================
# 3. ROS 2 PERCEPTION NODE WITH SPATIAL DEPTH SLICING
# ============================================================
if not rclpy.ok():
    rclpy.init()

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('phantomx_isaac_perception')
        
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.latest_rgb = None
        self.latest_depth = None
        self.closest_obstacle_dist = float('inf')
        
        # Spatial Depth Slices (Left, Center, Right)
        self.depth_left_dist = float('inf')
        self.depth_center_dist = float('inf')
        self.depth_right_dist = float('inf')

        self.subs = []
        for topic in RGB_TOPICS:
            self.subs.append(self.create_subscription(Image, topic, self._rgb_callback, qos_best_effort))
            self.subs.append(self.create_subscription(Image, topic, self._rgb_callback, qos_reliable))

        for topic in DEPTH_TOPICS:
            self.subs.append(self.create_subscription(Image, topic, self._depth_callback, qos_best_effort))
            self.subs.append(self.create_subscription(Image, topic, self._depth_callback, qos_reliable))

    def _rgb_callback(self, msg):
        try:
            self.latest_rgb = ros_img_to_numpy(msg, desired_encoding='bgr8')
        except Exception:
            pass

    def _depth_callback(self, msg):
        try:
            depth_frame = ros_img_to_numpy(msg, desired_encoding='32FC1')
            self.latest_depth = depth_frame
            h, w = depth_frame.shape

            # Slice into 3 horizontal sectors (Left, Center, Right)
            row_top, row_bottom = int(h * 0.35), int(h * 0.65)
            left_col = depth_frame[row_top:row_bottom, :int(w * 0.33)]
            center_col = depth_frame[row_top:row_bottom, int(w * 0.33):int(w * 0.66)]
            right_col = depth_frame[row_top:row_bottom, int(w * 0.66):]

            def extract_min(roi):
                valid = roi[(np.isfinite(roi)) & (roi > 0.05)]
                return float(np.min(valid)) if len(valid) > 0 else float('inf')

            self.depth_left_dist = extract_min(left_col)
            self.depth_center_dist = extract_min(center_col)
            self.depth_right_dist = extract_min(right_col)
            self.closest_obstacle_dist = min(self.depth_left_dist, self.depth_center_dist, self.depth_right_dist)

        except Exception:
            pass

global _ros_node
_ros_node = PerceptionNode()

# ============================================================
# 4. DYNAMIC TARGET & DIRECTION HELPERS
# ============================================================
def get_progress_dir_x():
    return 1.0 if TARGET_DESTINATION[0] >= SPAWN_POSITION[0] else -1.0

def get_lane_shift_heading():
    return 0.0 if get_progress_dir_x() > 0 else 180.0

def get_forward_vector(yaw_deg):
    rad = math.radians(yaw_deg)
    return math.cos(rad), math.sin(rad), 0.0

def get_shortest_angle_diff(target, current):
    diff = (target - current) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff

def compute_dist_2d(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# ============================================================
# 5. MEMORY-AUGMENTED VISION LLM CLIENT
# ============================================================
global vlm_request_in_flight, last_vlm_query_time, llm_avoidance_history
vlm_request_in_flight = False
last_vlm_query_time = 0.0
llm_avoidance_history = []  # Short-term session memory for the current obstacle

def encode_frame_to_base64(rgb_array):
    if _has_cv2 and rgb_array is not None:
        _, buffer = cv2.imencode('.jpg', rgb_array)
        return base64.b64encode(buffer).decode('utf-8')
    return ""

def async_query_vlm_agent(rgb_array, obstacle_dist, current_sub_phase, detour_offset, callback_fn):
    global vlm_request_in_flight, last_vlm_query_time, llm_avoidance_history

    now = time.time()
    if vlm_request_in_flight or (now - last_vlm_query_time) < MIN_VLM_COOLDOWN:
        return

    if state in ["STOPPED", "USER_PAUSED"]:
        return

    vlm_request_in_flight = True
    last_vlm_query_time = now

    def worker():
        global vlm_request_in_flight, llm_avoidance_history
        try:
            if state in ["STOPPED", "USER_PAUSED"]:
                return

            b64_img = encode_frame_to_base64(rgb_array)

            # Build memory summary of past steps in this encounter
            history_summary = ""
            if len(llm_avoidance_history) > 0:
                history_summary = "Past avoidance decisions for this obstacle:\n" + "\n".join([f"- {h}" for h in llm_avoidance_history[-3:]]) + "\n"

            # Phase-Specific Contextual Prompts
            if current_sub_phase == "DIVERGE_SCAN":
                prompt = (
                    f"You are the navigation brain for an autonomous sweeper robot for roads. "
                    f"You are currently viewing from a camera attached ontop of the robot. "
                    f"We encountered an obstacle ahead and turned outward to clear its width.\n"
                    f"{history_summary}"
                    f"Current forward distance: {obstacle_dist:.2f}m. Lateral detour offset: {detour_offset:.2f}m.\n"
                    f"Task: Inspect the snapshot from this new heading. Is the obstacle still in the front/flank camera view, or has our line of sight fully cleared the obstacle's width?\n"
                    f"Commands:\n"
                    f"- 'diverge_more': Obstacle edge is still visible or path ahead is partially blocked; need to turn further outward.\n"
                    f"- 'flank_forward': The path ahead is fully clear of the obstacle; ready to straighten and drive along the flank.\n"
                    f"Return ONLY valid JSON: {{\"obstacle_visible\": bool, \"obstacle_location\": \"center\"|\"left\"|\"right\"|\"behind_or_passed\", \"command\": \"diverge_more\"|\"flank_forward\", \"reasoning\": string}}"
                )

            elif current_sub_phase == "FLANK_INSPECT":
                prompt = (
                    f"You are verifying the flank clearance of an obstacle.\n"
                    f"{history_summary}"
                    f"The robot drove forward along the flank and turned inward to look directly at where the obstacle should be.\n"
                    f"Current distance reading: {obstacle_dist:.2f}m. Lateral displacement: {detour_offset:.2f}m.\n"
                    f"Task: Look at the image: Is the obstacle still visible alongside/ahead of the robot, or has it passed behind?\n"
                    f"Commands:\n"
                    f"- 'flank_forward': Obstacle is still alongside or visible; must drive further forward.\n"
                    f"- 'return_to_lane': Obstacle is completely passed and no longer in front/side; ready to return to lane.\n"
                    f"Return ONLY valid JSON: {{\"obstacle_visible\": bool, \"obstacle_location\": \"center\"|\"left\"|\"right\"|\"behind_or_passed\", \"command\": \"flank_forward\"|\"return_to_lane\", \"reasoning\": string}}"
                )

            else:  # VERIFY
                prompt = (
                    f"You are verifying the return path to the original sweep line.\n"
                    f"{history_summary}"
                    f"The robot is angled directly looking down the return corridor. Lateral offset to cancel: {detour_offset:.2f}m.\n"
                    f"Task: Is the return line completely clear of obstacles?\n"
                    f"Commands:\n"
                    f"- 'return_to_lane': Return corridor is completely open; proceed to baseline.\n"
                    f"- 'flank_forward': Return corridor is obstructed; must flank forward more.\n"
                    f"Return ONLY valid JSON: {{\"obstacle_visible\": bool, \"obstacle_location\": \"center\"|\"left\"|\"right\"|\"behind_or_passed\", \"command\": \"return_to_lane\"|\"flank_forward\", \"reasoning\": string}}"
                )

            payload = {
                "model": VLM_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                        ]
                    }
                ],
                "response_format": {"type": "json_object"}
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {VLM_API_KEY}",
                "HTTP-Referer": "http://localhost",
                "User-Agent": "IsaacSim-PhantomX-Agent/1.0"
            }

            req = urllib.request.Request(
                VLM_API_URL,
                data=json.dumps(payload).encode('utf-8'),
                headers=headers
            )
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                if state in ["STOPPED", "USER_PAUSED"]:
                    return

                result = json.loads(resp.read().decode('utf-8'))
                raw_reply = result['choices'][0]['message']['content']
                parsed = json.loads(raw_reply)

                # Record decision into memory
                log_entry = f"Phase {current_sub_phase}: Decided '{parsed.get('command')}' (Offset={detour_offset:.2f}m). Reason: {parsed.get('reasoning')}"
                llm_avoidance_history.append(log_entry)

                callback_fn(parsed)

        except Exception as e:
            if state not in ["STOPPED", "USER_PAUSED"]:
                print(f"[VLM Fallback Triggered ({e})]")
                if current_sub_phase == "DIVERGE_SCAN":
                    cmd = "diverge_more" if (abs(detour_offset) < MIN_CHASSIS_CLEAR or obstacle_dist < 1.5) else "flank_forward"
                elif current_sub_phase == "FLANK_INSPECT":
                    cmd = "return_to_lane" if obstacle_dist > 1.2 else "flank_forward"
                else:
                    cmd = "return_to_lane"
                
                callback_fn({
                    "obstacle_visible": True,
                    "obstacle_location": "left",
                    "command": cmd,
                    "reasoning": "Deterministic fallback safety rule."
                })
        finally:
            vlm_request_in_flight = False

    threading.Thread(target=worker, daemon=True).start()

# ============================================================
# 6. STATE VARIABLES & DISPLACEMENT LEDGER
# ============================================================
global state, saved_pre_pause_state, current_pos, current_yaw, sweep_direction_y
global step_start_pos, target_yaw, pause_start_time, last_time, shifted_distance, last_print_time
global lane_base_yaw, lateral_detour_ledger, micro_step_progress, micro_step_target, detour_side
global avoidance_sub_phase, initial_detected_distance, required_flank_distance, flank_progress, diverge_turn_count

current_pos = list(SPAWN_POSITION)
current_yaw = SPAWN_ROTATION[2]
state = "MOVING"
saved_pre_pause_state = "MOVING"

sweep_direction_y = 1.0 if SPAWN_ROTATION[2] == 90.0 else -1.0
step_start_pos = list(current_pos)
target_yaw = current_yaw
pause_start_time = 0.0
last_time = time.time()
shifted_distance = 0.0
last_print_time = 0.0

# Displacement Ledger & Avoidance State
lane_base_yaw = current_yaw
lateral_detour_ledger = 0.0
micro_step_progress = 0.0
micro_step_target = 0.0
detour_side = 1.0
avoidance_sub_phase = "IDLE"
initial_detected_distance = 1.0
required_flank_distance = 1.4
flank_progress = 0.0
diverge_turn_count = 1

def update_usd_transform(pos, yaw_deg):
    translate_op.Set(Gf.Vec3d(*pos))
    rotate_op.Set(Gf.Vec3d(0.0, 0.0, yaw_deg))

update_usd_transform(current_pos, current_yaw)

def record_motion_to_ledger(dx, dy):
    global lateral_detour_ledger, lane_base_yaw
    rad = math.radians(lane_base_yaw)
    cos_b, sin_b = math.cos(rad), math.sin(rad)
    d_lateral = (-dx * sin_b) + (dy * cos_b)
    lateral_detour_ledger += d_lateral

def move_forward_step(distance, dt):
    global current_pos, current_yaw
    fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
    dx = fwd_x * distance
    dy = fwd_y * distance
    current_pos[0] += dx
    current_pos[1] += dy
    record_motion_to_ledger(dx, dy)

def move_forward(speed=FORWARD_SPEED, dt=0.0):
    global current_pos, current_yaw
    fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
    current_pos[0] += fwd_x * speed * dt
    current_pos[1] += fwd_y * speed * dt

def stop():
    global state, vlm_request_in_flight
    state = "STOPPED"
    vlm_request_in_flight = False
    if status_label:
        status_label.text = "Status: STOPPED"
    if vlm_label:
        vlm_label.text = "VLM: Aborted"
    print("[PhantomX]: All robot operations and VLM threads halted.")

# ============================================================
# 7. UI PANEL
# ============================================================
global _phantomx_window
if "_phantomx_window" in globals() and _phantomx_window:
    try:
        _phantomx_window.visible = False
        _phantomx_window.destroy()
    except Exception:
        pass
    _phantomx_window = None

_phantomx_window = ui.Window("PhantomX Turn-Analyze-Verify Controller", width=420, height=270, flags=ui.WINDOW_FLAGS_NO_SAVED_SETTINGS)
_phantomx_window.position_x = 80
_phantomx_window.position_y = 120
_phantomx_window.visible = True
_phantomx_window.focus()

status_label = None
sensor_label = None
ledger_label = None
vlm_label = None

with _phantomx_window.frame:
    with ui.VStack(spacing=6, style={"margin": 10}):
        ui.Label("PhantomX Memory-Augmented Sweeper", height=20, style={"font_size": 14, "color": 0xFF00FFFF})
        ui.Label(f"Target: ({TARGET_DESTINATION[0]:.1f}, {TARGET_DESTINATION[1]:.1f})", height=18, style={"font_size": 12, "color": 0xFFAAAAFF})
        sensor_label = ui.Label("Obstacle Dist: -- m", height=18, style={"font_size": 12, "color": 0xFFFFFF00})
        vlm_label = ui.Label("VLM: Idle", height=18, style={"font_size": 11, "color": 0xFFFFAA88})
        ledger_label = ui.Label("Detour Ledger: 0.00 m", height=18, style={"font_size": 11, "color": 0xFF88FF88})
        status_label = ui.Label("Status: MOVING", height=20, style={"font_size": 13})
        
        with ui.HStack(spacing=8, height=36):
            def on_emergency_stop():
                stop()

            def on_pause_resume():
                global state, saved_pre_pause_state, last_time, vlm_request_in_flight
                if state == "USER_PAUSED":
                    state = saved_pre_pause_state
                    last_time = time.time()
                    if status_label:
                        status_label.text = f"Status: RESUMED ({state})"
                elif state != "STOPPED":
                    saved_pre_pause_state = state
                    state = "USER_PAUSED"
                    vlm_request_in_flight = False
                    if status_label:
                        status_label.text = "Status: MANUALLY PAUSED"

            def on_reset():
                global state, current_pos, current_yaw, step_start_pos, last_time, shifted_distance
                global sweep_direction_y, lane_base_yaw, lateral_detour_ledger, avoidance_sub_phase
                global vlm_request_in_flight, llm_avoidance_history
                state = "MOVING"
                vlm_request_in_flight = False
                llm_avoidance_history.clear()
                current_pos = list(SPAWN_POSITION)
                current_yaw = SPAWN_ROTATION[2]
                lane_base_yaw = current_yaw
                sweep_direction_y = 1.0 if SPAWN_ROTATION[2] == 90.0 else -1.0
                lateral_detour_ledger = 0.0
                avoidance_sub_phase = "IDLE"
                step_start_pos = list(current_pos)
                shifted_distance = 0.0
                last_time = time.time()
                update_usd_transform(current_pos, current_yaw)
                if status_label:
                    status_label.text = "Status: RESET & MOVING"
                if vlm_label:
                    vlm_label.text = "VLM: Idle"

            ui.Button("Stop", clicked_fn=on_emergency_stop, style={"background_color": 0xFF2222DD})
            ui.Button("Pause/Play", clicked_fn=on_pause_resume, style={"background_color": 0xFFDD8822})
            ui.Button("Reset", clicked_fn=on_reset, style={"background_color": 0xFF22AA22})

# ============================================================
# 8. VLM DECISION HANDLER & STATE MACHINE
# ============================================================
def on_vlm_agent_decision(decision_dict):
    global state, avoidance_sub_phase, target_yaw, current_yaw, micro_step_target, micro_step_progress
    global detour_side, flank_progress, required_flank_distance, initial_detected_distance, diverge_turn_count
    
    if state in ["STOPPED", "USER_PAUSED"]:
        return

    cmd = decision_dict.get("command", "flank_forward")
    loc = decision_dict.get("obstacle_location", "center")
    reasoning = decision_dict.get("reasoning", "")
    
    print(f"\n[VLM Callback]: Phase='{avoidance_sub_phase}' | Command='{cmd}' | Location='{loc}' | Offset={lateral_detour_ledger:.2f}m | Reason: {reasoning}")
    if vlm_label:
        vlm_label.text = f"VLM: {cmd} ({loc})"

    # --- 1. DIVERGE PHASE RESOLUTION ---
    if avoidance_sub_phase == "DIVERGE_SCAN":
        # Enforce minimum chassis clearance width even if LLM thinks it cleared early
        if cmd == "diverge_more" or abs(lateral_detour_ledger) < MIN_CHASSIS_CLEAR:
            diverge_turn_count += 1
            target_yaw = (lane_base_yaw + (detour_side * (DIVERGE_ANGLE_INC + (diverge_turn_count * 10.0)))) % 360.0
            micro_step_target = AVOIDANCE_STEP_SIZE
            micro_step_progress = 0.0
            state = "AVOID_ROTATE_THEN_STEP"
        else:
            # Front clear of obstacle's full width -> Align parallel to flank
            avoidance_sub_phase = "FLANK_DRIVE"
            target_yaw = lane_base_yaw
            flank_progress = 0.0
            required_flank_distance = initial_detected_distance + CLEAR_PASS_SAFETY_PAD
            state = "AVOID_ALIGN_TO_FLANK"
            print(f"[Stage 2 Flank]: Diverge complete (Offset={lateral_detour_ledger:.2f}m). Straightening to {lane_base_yaw:.1f}° for flank drive of {required_flank_distance:.2f}m...")

    # --- 2. FLANK INWARD INSPECTION RESOLUTION ---
    elif avoidance_sub_phase == "FLANK_INSPECT":
        if cmd == "flank_forward":
            # Obstacle still visible alongside -> Extend flank by another 0.4m
            avoidance_sub_phase = "FLANK_DRIVE"
            target_yaw = lane_base_yaw
            flank_progress = 0.0
            required_flank_distance = 0.40
            state = "AVOID_ALIGN_TO_FLANK"
            print("[Flank Check]: Obstacle still beside us. Extending flank forward by 0.4m...")
        elif cmd == "return_to_lane":
            # Obstacle has passed behind chassis -> Angle to verify return corridor
            avoidance_sub_phase = "VERIFY"
            return_steer_side = -1.0 if lateral_detour_ledger > 0.0 else 1.0
            target_yaw = (lane_base_yaw + (return_steer_side * 45.0)) % 360.0
            state = "LOOK_BACK_ROTATE"
            print("[Flank Check]: Obstacle passed! Angling to verify return corridor...")

    # --- 3. RETURN VERIFICATION RESOLUTION ---
    elif avoidance_sub_phase == "VERIFY":
        if cmd == "return_to_lane":
            avoidance_sub_phase = "RETURNING"
            state = "RETURN_DRIVE_TO_LANE"
            print(f"[Verified]: Return corridor clear! Returning across lateral offset {lateral_detour_ledger:.3f}m...")
        elif cmd == "flank_forward":
            # Return line is still blocked -> Flank forward an extra 0.4m
            avoidance_sub_phase = "FLANK_DRIVE"
            target_yaw = lane_base_yaw
            flank_progress = 0.0
            required_flank_distance = 0.40
            state = "AVOID_ALIGN_TO_FLANK"

def zig_zag_update(event):
    global state, saved_pre_pause_state, current_pos, current_yaw, sweep_direction_y
    global step_start_pos, target_yaw, pause_start_time, last_time, shifted_distance, last_print_time
    global lane_base_yaw, lateral_detour_ledger, micro_step_progress, micro_step_target, detour_side
    global avoidance_sub_phase, initial_detected_distance, required_flank_distance, flank_progress, diverge_turn_count
    global llm_avoidance_history

    rclpy.spin_once(_ros_node, timeout_sec=0.0)

    if state in ["STOPPED", "USER_PAUSED"]:
        return

    now = time.time()
    dt = min(now - last_time, 0.1)
    last_time = now

    min_dist_ahead = _ros_node.closest_obstacle_dist
    rgb_frame = _ros_node.latest_rgb

    if sensor_label:
        sensor_label.text = f"Obstacle Dist: {min_dist_ahead:.2f} m" if min_dist_ahead != float('inf') else "Obstacle Dist: Clear (>10m)"

    if ledger_label:
        ledger_label.text = f"Detour Ledger: {lateral_detour_ledger:.2f} m"

    if (now - last_print_time) >= PRINT_INTERVAL:
        last_print_time = now
        print("-" * 50)
        print(f"[Telemetry] Time: {now:.1f}s | State: {state} | Phase: {avoidance_sub_phase} | Obstacle: {min_dist_ahead:.2f}m | Ledger: {lateral_detour_ledger:.2f}m")
        print(f"Pose: ({current_pos[0]:.2f}, {current_pos[1]:.2f}) | Heading: {current_yaw:.1f}° | Left: {_ros_node.depth_left_dist:.2f}m, Ctr: {_ros_node.depth_center_dist:.2f}m, Right: {_ros_node.depth_right_dist:.2f}m")
        print("-" * 50)

    # ============================================================
    # 1. NORMAL SWEEP FORWARD & BOUNDARY HANDLING
    # ============================================================
    if state == "MOVING":
        lane_base_yaw = current_yaw

        if compute_dist_2d(current_pos, TARGET_DESTINATION) <= TARGET_RADIUS:
            stop()
            if status_label:
                status_label.text = "Status: DESTINATION REACHED"
            print(f"[Target]: Destination {TARGET_DESTINATION} reached!")
            return

        # Obstacle detected -> INSTANT SPATIAL SECTOR ANALYSIS
        if min_dist_ahead < OBSTACLE_THRESHOLD:
            initial_detected_distance = min_dist_ahead
            lateral_detour_ledger = 0.0
            diverge_turn_count = 1
            llm_avoidance_history.clear()

            # Automatic Depth Slicing: Choose the sector with greater clearance
            if _ros_node.depth_left_dist >= _ros_node.depth_right_dist:
                detour_side = 1.0   # Left is clearer
                print(f"[Spatial Depth Analysis]: Left sector clearer ({_ros_node.depth_left_dist:.2f}m vs {_ros_node.depth_right_dist:.2f}m). Turning Left.")
            else:
                detour_side = -1.0  # Right is clearer
                print(f"[Spatial Depth Analysis]: Right sector clearer ({_ros_node.depth_right_dist:.2f}m vs {_ros_node.depth_left_dist:.2f}m). Turning Right.")

            # Turn first to outward angle
            avoidance_sub_phase = "DIVERGE_SCAN"
            target_yaw = (lane_base_yaw + (detour_side * DIVERGE_ANGLE_INC)) % 360.0
            state = "AVOID_INITIAL_TURN"
            if status_label:
                status_label.text = "Status: Turning Outward to Clear Width..."
            return

        move_forward(speed=FORWARD_SPEED, dt=dt)

        edge_hit = False
        margin_y = 0.3
        
        if current_pos[1] > max_y - margin_y and sweep_direction_y > 0:
            current_pos[1] = max_y - margin_y
            edge_hit = True
        elif current_pos[1] < min_y + margin_y and sweep_direction_y < 0:
            current_pos[1] = min_y + margin_y
            edge_hit = True

        progress_dir = get_progress_dir_x()
        target_x_reached = (progress_dir > 0 and current_pos[0] >= TARGET_DESTINATION[0]) or \
                        (progress_dir < 0 and current_pos[0] <= TARGET_DESTINATION[0])
        hard_limit_hit = (current_pos[0] > max_x) or (current_pos[0] < min_x)

        if target_x_reached or hard_limit_hit:
            if compute_dist_2d(current_pos, TARGET_DESTINATION) <= (TARGET_RADIUS + 0.5):
                stop()
                if status_label:
                    status_label.text = "Status: DESTINATION REACHED"
                print(f"[Target]: Destination {TARGET_DESTINATION} reached!")
                return

        dist_in_step = compute_dist_2d(current_pos, step_start_pos)

        if edge_hit:
            target_yaw = get_lane_shift_heading()
            state = "TURNING_1"
            if status_label:
                status_label.text = "Status: Edge Hit -> Shifting Lane"
        elif dist_in_step >= STEP_DISTANCE:
            state = "PAUSED"
            pause_start_time = now
            if status_label:
                status_label.text = f"Status: Step Paused ({PAUSE_DURATION}s)"

        update_usd_transform(current_pos, current_yaw)

    # ============================================================
    # 2. STEP INTERVAL PAUSE
    # ============================================================
    elif state == "PAUSED":
        if (now - pause_start_time) >= PAUSE_DURATION:
            state = "MOVING"
            step_start_pos = list(current_pos)
            if status_label:
                status_label.text = "Status: MOVING (Forward)"

    # ============================================================
    # 3. STAGE 1: TURN FIRST, THEN SCAN WITH VLM
    # ============================================================
    elif state == "AVOID_INITIAL_TURN":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            # Turn completed -> Stop & trigger VLM scan from this new heading
            state = "VLM_INSPECTING"
            if status_label:
                status_label.text = "Status: VLM Scanning Outward Heading..."
            print(f"[Stage 1 Turn Complete]: Angled at {current_yaw:.1f}°. Snapping snapshot to check clearance...")
            async_query_vlm_agent(rgb_frame, min_dist_ahead, avoidance_sub_phase, lateral_detour_ledger, on_vlm_agent_decision)

        update_usd_transform(current_pos, current_yaw)

    elif state == "AVOID_ROTATE_THEN_STEP":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            state = "AVOID_DIVERGE_STEP"
            micro_step_progress = 0.0

        update_usd_transform(current_pos, current_yaw)

    elif state == "AVOID_DIVERGE_STEP":
        step_dist = FORWARD_SPEED * 0.6 * dt
        move_forward_step(step_dist, dt)
        micro_step_progress += step_dist

        if micro_step_progress >= micro_step_target:
            # Step done -> Halt and verify clearance again
            state = "VLM_INSPECTING"
            if status_label:
                status_label.text = "Status: VLM Verifying Width Clearance..."
            print(f"[Diverge Step Done]: Offset = {lateral_detour_ledger:.2f}m. Asking VLM if width is clear...")
            async_query_vlm_agent(rgb_frame, min_dist_ahead, avoidance_sub_phase, lateral_detour_ledger, on_vlm_agent_decision)

        update_usd_transform(current_pos, current_yaw)

    # ============================================================
    # 4. STAGE 2: MANDATORY PARALLEL FLANK & PEEK-IN AT OBSTACLE
    # ============================================================
    elif state == "AVOID_ALIGN_TO_FLANK":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            state = "AVOID_FLANK_DRIVE"

        update_usd_transform(current_pos, current_yaw)

    elif state == "AVOID_FLANK_DRIVE":
        dist_increment = FORWARD_SPEED * 0.8 * dt
        move_forward_step(dist_increment, dt)
        flank_progress += dist_increment

        if flank_progress >= required_flank_distance:
            # Forward drive done -> Turn INWARD (-35°) to look directly at the obstacle's flank
            avoidance_sub_phase = "FLANK_INSPECT"
            inspect_angle_side = -1.0 if detour_side > 0 else 1.0
            target_yaw = (lane_base_yaw + (inspect_angle_side * 35.0)) % 360.0
            state = "AVOID_ROTATE_TO_INSPECT_FLANK"
            if status_label:
                status_label.text = "Status: Peeking Inward at Obstacle..."
            print(f"[Flank Distance Met ({flank_progress:.2f}m)]: Angling to {target_yaw:.1f}° to inspect obstacle flank...")

        update_usd_transform(current_pos, current_yaw)

    elif state == "AVOID_ROTATE_TO_INSPECT_FLANK":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            # Stop and ask VLM: Is the obstacle still beside us or completely passed?
            state = "VLM_INSPECTING"
            if status_label:
                status_label.text = "Status: VLM Inspecting Obstacle Flank..."
            print("[Flank Inward Inspection]: Snapping image to verify if obstacle is cleared...")
            async_query_vlm_agent(rgb_frame, min_dist_ahead, avoidance_sub_phase, lateral_detour_ledger, on_vlm_agent_decision)

        update_usd_transform(current_pos, current_yaw)

    # ============================================================
    # 5. STAGE 3: LOOK-BACK & VERIFY RETURN CORRIDOR
    # ============================================================
    elif state == "LOOK_BACK_ROTATE":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            state = "VLM_INSPECTING"
            if status_label:
                status_label.text = "Status: VLM Verifying Return Path..."
            print("[Return Verify]: Angled toward lane. Asking VLM to confirm return corridor is open...")
            async_query_vlm_agent(rgb_frame, min_dist_ahead, avoidance_sub_phase, lateral_detour_ledger, on_vlm_agent_decision)

        update_usd_transform(current_pos, current_yaw)

    # ============================================================
    # 6. RETURN DRIVE & ZEROING THE LEDGER
    # ============================================================
    elif state == "RETURN_DRIVE_TO_LANE":
        step_dist = FORWARD_SPEED * 0.8 * dt
        move_forward_step(step_dist, dt)

        # Drive inward until lateral displacement returns to zero
        if abs(lateral_detour_ledger) <= 0.05 or (lateral_detour_ledger * detour_side <= 0.0):
            lateral_detour_ledger = 0.0
            target_yaw = lane_base_yaw
            state = "RETURN_FINAL_ALIGN"
            if status_label:
                status_label.text = "Status: Re-aligning to Lane Heading..."

        update_usd_transform(current_pos, current_yaw)

    elif state == "RETURN_FINAL_ALIGN":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            state = "MOVING"
            avoidance_sub_phase = "IDLE"
            llm_avoidance_history.clear()  # Reset working memory for future obstacles
            step_start_pos = list(current_pos)
            print("[Avoidance Complete]: Lane baseline fully restored. Memory cleared. Resuming sweep.")
            if status_label:
                status_label.text = "Status: MOVING"

        update_usd_transform(current_pos, current_yaw)

    # ============================================================
    # 7. MONOTONIC ZIG-ZAG U-TURNS (PROGRESS TOWARD TARGET)
    # ============================================================
    elif state == "TURNING_1":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            state = "LANE_SHIFTING"
            shifted_distance = 0.0
            if status_label:
                status_label.text = "Status: Shifting Lane"

        update_usd_transform(current_pos, current_yaw)

    elif state == "LANE_SHIFTING":
        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        dist = FORWARD_SPEED * dt
        current_pos[0] += fwd_x * dist
        current_pos[1] += fwd_y * dist
        shifted_distance += dist

        if shifted_distance >= LANE_SHIFT:
            state = "TURNING_2"
            sweep_direction_y = -sweep_direction_y
            target_yaw = 90.0 if sweep_direction_y > 0 else 270.0
            if status_label:
                status_label.text = "Status: Re-aligning to Sweep Lane"

        update_usd_transform(current_pos, current_yaw)

    elif state == "TURNING_2":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            lane_base_yaw = current_yaw
            state = "MOVING"
            step_start_pos = list(current_pos)
            if status_label:
                status_label.text = "Status: MOVING (New Lane)"

        update_usd_transform(current_pos, current_yaw)

# Register update loop
global _phantomx_sub
_phantomx_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(zig_zag_update)