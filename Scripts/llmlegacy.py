"""
PhantomX Target-Directed Zig-Zag Controller with Closed-Loop Agentic VLM Avoidance & Odometer Ledger
------------------------------------------------------------------------------------------------------
Features:
  1. Dynamic Target Progression Vector: Correct monotonic lane progression for positive or negative X goals.
  2. Live ROS 2 RGB & Depth camera stream (/camera/rgb, /camera/depth).
  3. Closed-Loop Agentic VLM Avoidance: Micro-step visual perception loop.
  4. Odometer Ledger: Tracks all lateral (cross-track) offsets to guarantee exact lane return.
  5. Step Interval Pausing & On-Screen Control Panel.
"""

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
TARGET_DESTINATION = (-10.2, 2.0)                     # Supports positive or negative X targets
TARGET_RADIUS      = 0.3

STEP_DISTANCE      = 0.5                             # Sweep pause step size (m)
PAUSE_DURATION     = 0.5                             # Sweep pause duration (s)
FORWARD_SPEED      = 1.0                             # Base speed (m/s)
TURN_SPEED         = 180.0                           # Turn speed (deg/s)
LANE_SHIFT         = 0.5                             # Lane shift width (m)
OBSTACLE_THRESHOLD = 1.3                             # Avoidance trigger threshold (m)
PRINT_INTERVAL     = 2.0                             # Telemetry print interval (s)

# Avoidance micro-step settings
AVOIDANCE_STEP_SIZE   = 0.25                         # Micro-step distance per VLM query (m)
DIVERGE_ANGLE_INC     = 30.0                         # Heading deflection angle (deg)

# VLM API Configuration (Ollama / Local / OpenAI endpoint compatible)
VLM_API_URL        = "https://openrouter.ai/api/v1/chat/completions"
VLM_API_KEY        = "sk-or-v1-xxxx"                                    # Set key if using cloud APIs
VLM_MODEL          = "~openai/gpt-latest"                           # e.g., gpt-4o, llama3.2-vision, qwen2-vl

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
# 3. ROS 2 PERCEPTION NODE
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
            roi = depth_frame[int(h * 0.35):int(h * 0.65), int(w * 0.35):int(w * 0.65)]
            valid = roi[(np.isfinite(roi)) & (roi > 0.05)]
            
            if len(valid) > 0:
                self.closest_obstacle_dist = float(np.min(valid))
            else:
                self.closest_obstacle_dist = float('inf')
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
# 5. CLOSED-LOOP VISION LLM INTEGRATION
# ============================================================
def encode_frame_to_base64(rgb_array):
    if _has_cv2 and rgb_array is not None:
        _, buffer = cv2.imencode('.jpg', rgb_array)
        return base64.b64encode(buffer).decode('utf-8')
    return ""

def async_query_vlm_agent(rgb_array, obstacle_dist, current_sub_phase, detour_offset, callback_fn):
    """
    Non-blocking background thread: Queries Vision LLM with image and state payload.
    """
    def worker():
        b64_img = encode_frame_to_base64(rgb_array)
        prompt = (
            f"You are the visual navigation system for an autonomous road sweeper. "
            f"Current obstacle distance: {obstacle_dist:.2f}m. "
            f"Current maneuver sub-phase: '{current_sub_phase}'. "
            f"Current accumulated lateral detour offset: {detour_offset:.2f}m. "
            f"Inspect the attached camera snapshot carefully: "
            f"1. Is the path directly ahead still blocked by the obstacle? "
            f"2. Where is the obstacle relative to the frame ('center', 'left', 'right', 'behind_or_passed')? "
            f"3. What tactical command should be taken next? "
            f"Available commands: "
            f"- 'diverge_more': The front is still blocked; turn away further and step outward. "
            f"- 'flank_forward': The path directly ahead is clear; move forward along the flank parallel to the lane. "
            f"- 'return_to_lane': The obstacle is completely behind or outside the field of view; ready to return to original lane line. "
            f"Return STRICT JSON only: "
            f"{{\"obstacle_visible\": true/false, \"obstacle_location\": \"center\"|\"left\"|\"right\"|\"behind_or_passed\", \"command\": \"diverge_more\"|\"flank_forward\"|\"return_to_lane\", \"reasoning\": \"short visual explanation\"}}"
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
        
        try:
            req = urllib.request.Request(
                VLM_API_URL,
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {VLM_API_KEY}"}
            )
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                content = result['choices'][0]['message']['content']
                parsed = json.loads(content)
                callback_fn(parsed)
        except Exception as e:
            # Fallback heuristic if API server is offline
            print(f"[VLM Request Failed ({e})]: Using local fallback heuristic.")
            if current_sub_phase == "DIVERGE" and obstacle_dist < (OBSTACLE_THRESHOLD + 0.3):
                cmd = "diverge_more"
            elif current_sub_phase == "FLANK" and obstacle_dist >= (OBSTACLE_THRESHOLD + 0.5):
                cmd = "return_to_lane"
            else:
                cmd = "flank_forward"
            callback_fn({"obstacle_visible": True, "obstacle_location": "left", "command": cmd, "reasoning": "Local geometric fallback."})

    threading.Thread(target=worker, daemon=True).start()

# ============================================================
# 6. STATE VARIABLES & DISPLACEMENT LEDGER
# ============================================================
global state, saved_pre_pause_state, current_pos, current_yaw, sweep_direction_y
global step_start_pos, target_yaw, pause_start_time, last_time, shifted_distance, last_print_time
global lane_base_yaw, lateral_detour_ledger, micro_step_progress, micro_step_target, detour_side, avoidance_sub_phase

current_pos = list(SPAWN_POSITION)
current_yaw = SPAWN_ROTATION[2]
state = "MOVING"
saved_pre_pause_state = "MOVING"

# Initial Y sweep direction (+1 for 90 deg / +Y, -1 for 270 deg / -Y)
sweep_direction_y = 1.0 if SPAWN_ROTATION[2] == 90.0 else -1.0
step_start_pos = list(current_pos)
target_yaw = current_yaw
pause_start_time = 0.0
last_time = time.time()
shifted_distance = 0.0
last_print_time = 0.0

# Displacement Ledger & Avoidance State
lane_base_yaw = current_yaw
lateral_detour_ledger = 0.0      # Cross-track offset (+ for Left, - for Right)
micro_step_progress = 0.0
micro_step_target = 0.0
detour_side = 1.0                # +1.0 for Left deflection, -1.0 for Right
avoidance_sub_phase = "IDLE"     # "DIVERGE", "FLANK", "RETURNING"

def update_usd_transform(pos, yaw_deg):
    translate_op.Set(Gf.Vec3d(*pos))
    rotate_op.Set(Gf.Vec3d(0.0, 0.0, yaw_deg))

update_usd_transform(current_pos, current_yaw)

def record_motion_to_ledger(dx, dy):
    """Projects world displacement into the active lane coordinate frame."""
    global lateral_detour_ledger, lane_base_yaw
    rad = math.radians(lane_base_yaw)
    cos_b, sin_b = math.cos(rad), math.sin(rad)
    # Cross-track lateral displacement (+Left, -Right)
    d_lateral = (-dx * sin_b) + (dy * cos_b)
    lateral_detour_ledger += d_lateral

def move_forward_step(distance, dt):
    """Moves forward and records lateral drift to ledger."""
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
    global state
    state = "STOPPED"
    if status_label:
        status_label.text = "Status: STOPPED"
    print("[PhantomX]: Robot Movement Halted.")

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

_phantomx_window = ui.Window("PhantomX Motion & Avoidance Control", width=420, height=270, flags=ui.WINDOW_FLAGS_NO_SAVED_SETTINGS)
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
        ui.Label("PhantomX Target-Directed Sweeper", height=20, style={"font_size": 14, "color": 0xFF00FFFF})
        ui.Label(f"Target: ({TARGET_DESTINATION[0]:.1f}, {TARGET_DESTINATION[1]:.1f})", height=18, style={"font_size": 12, "color": 0xFFAAAAFF})
        sensor_label = ui.Label("Obstacle Dist: -- m", height=18, style={"font_size": 12, "color": 0xFFFFFF00})
        vlm_label = ui.Label("VLM: Idle", height=18, style={"font_size": 11, "color": 0xFFFFAA88})
        ledger_label = ui.Label("Detour Ledger: 0.00 m", height=18, style={"font_size": 11, "color": 0xFF88FF88})
        status_label = ui.Label("Status: MOVING", height=20, style={"font_size": 13})
        
        with ui.HStack(spacing=8, height=36):
            def on_emergency_stop():
                stop()

            def on_pause_resume():
                global state, saved_pre_pause_state, last_time
                if state == "USER_PAUSED":
                    state = saved_pre_pause_state
                    last_time = time.time()
                    if status_label:
                        status_label.text = f"Status: RESUMED ({state})"
                elif state != "STOPPED":
                    saved_pre_pause_state = state
                    state = "USER_PAUSED"
                    if status_label:
                        status_label.text = "Status: MANUALLY PAUSED"

            def on_reset():
                global state, current_pos, current_yaw, step_start_pos, last_time, shifted_distance
                global sweep_direction_y, lane_base_yaw, lateral_detour_ledger, avoidance_sub_phase
                state = "MOVING"
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

            ui.Button("Stop", clicked_fn=on_emergency_stop, style={"background_color": 0xFF2222DD})
            ui.Button("Pause/Play", clicked_fn=on_pause_resume, style={"background_color": 0xFFDD8822})
            ui.Button("Reset", clicked_fn=on_reset, style={"background_color": 0xFF22AA22})

# ============================================================
# 8. VLM DECISION HANDLER & STATE MACHINE
# ============================================================
def on_vlm_agent_decision(decision_dict):
    """
    Receives tactical decision from the VLM and sets up the next micro-step.
    """
    global state, avoidance_sub_phase, target_yaw, current_yaw, micro_step_target, micro_step_progress, detour_side
    cmd = decision_dict.get("command", "flank_forward")
    loc = decision_dict.get("obstacle_location", "center")
    reasoning = decision_dict.get("reasoning", "")
    
    print(f"\n[VLM Callback]: Command='{cmd}' | Location='{loc}' | Offset={lateral_detour_ledger:.2f}m | Reason: {reasoning}")
    if vlm_label:
        vlm_label.text = f"VLM: {cmd} ({loc})"

    # 1. COMMAND: DIVERGE MORE (Path still blocked ahead)
    if cmd == "diverge_more":
        avoidance_sub_phase = "DIVERGE"
        target_yaw = (lane_base_yaw + (detour_side * DIVERGE_ANGLE_INC)) % 360.0
        micro_step_target = AVOIDANCE_STEP_SIZE
        micro_step_progress = 0.0
        state = "AVOID_ROTATE"

    # 2. COMMAND: FLANK FORWARD (Path ahead is clear, moving parallel along flank)
    elif cmd == "flank_forward":
        avoidance_sub_phase = "FLANK"
        target_yaw = lane_base_yaw
        micro_step_target = AVOIDANCE_STEP_SIZE
        micro_step_progress = 0.0
        state = "AVOID_ROTATE"

    # 3. COMMAND: RETURN TO LANE (Obstacle passed, initiate ledger zeroing)
    elif cmd == "return_to_lane":
        avoidance_sub_phase = "RETURNING"
        return_steer_side = -1.0 if lateral_detour_ledger > 0.0 else 1.0
        target_yaw = (lane_base_yaw + (return_steer_side * 45.0)) % 360.0
        state = "RETURN_TURN_TO_LANE"
        print(f"[Ledger Recovery]: Total Lateral Detour = {lateral_detour_ledger:.3f}m. Returning...")

def zig_zag_update(event):
    global state, saved_pre_pause_state, current_pos, current_yaw, sweep_direction_y
    global step_start_pos, target_yaw, pause_start_time, last_time, shifted_distance, last_print_time
    global lane_base_yaw, lateral_detour_ledger, micro_step_progress, micro_step_target, detour_side, avoidance_sub_phase

    rclpy.spin_once(_ros_node, timeout_sec=0.0)

    if state == "STOPPED" or state == "USER_PAUSED":
        return

    now = time.time()
    dt = min(now - last_time, 0.1)
    last_time = now

    min_dist_ahead = _ros_node.closest_obstacle_dist
    rgb_frame = _ros_node.latest_rgb

    if sensor_label:
        if min_dist_ahead != float('inf'):
            sensor_label.text = f"Obstacle Dist: {min_dist_ahead:.2f} m"
        else:
            sensor_label.text = "Obstacle Dist: Clear (>10m)"

    if ledger_label:
        ledger_label.text = f"Detour Ledger: {lateral_detour_ledger:.2f} m"

    # Periodic Diagnostic Print
    if (now - last_print_time) >= PRINT_INTERVAL:
        last_print_time = now
        print("-" * 50)
        print(f"[Telemetry] Time: {now:.1f}s | State: {state} | Obstacle: {min_dist_ahead:.2f}m | Ledger: {lateral_detour_ledger:.2f}m")
        print(f"Pose: ({current_pos[0]:.2f}, {current_pos[1]:.2f}) | Heading: {current_yaw:.1f}°")
        print("-" * 50)

    # ============================================================
    # 1. NORMAL SWEEP FORWARD & BOUNDARY HANDLING
    # ============================================================
    if state == "MOVING":
        lane_base_yaw = current_yaw

        # Destination arrival check
        if compute_dist_2d(current_pos, TARGET_DESTINATION) <= TARGET_RADIUS:
            stop()
            if status_label:
                status_label.text = "Status: DESTINATION REACHED"
            print(f"[Target]: Destination {TARGET_DESTINATION} reached!")
            return

        # Obstacle detected -> TRIGGER INITIAL VLM INSPECTION
        if min_dist_ahead < OBSTACLE_THRESHOLD:
            state = "VLM_INSPECTING"
            avoidance_sub_phase = "DIVERGE"
            lateral_detour_ledger = 0.0
            detour_side = 1.0  # Diverge left by default
            if status_label:
                status_label.text = "Status: VLM INSPECTING..."
            print(f"\n[Obstacle Detected ({min_dist_ahead:.2f}m)]: Triggering initial VLM inspection...")
            async_query_vlm_agent(rgb_frame, min_dist_ahead, avoidance_sub_phase, lateral_detour_ledger, on_vlm_agent_decision)
            return

        move_forward(speed=FORWARD_SPEED, dt=dt)

        # Boundary checks along Y-axis (top and bottom of sweep lanes)
        edge_hit = False
        margin_y = 0.3
        
        if current_pos[1] > max_y - margin_y and sweep_direction_y > 0:
            current_pos[1] = max_y - margin_y
            edge_hit = True
        elif current_pos[1] < min_y + margin_y and sweep_direction_y < 0:
            current_pos[1] = min_y + margin_y
            edge_hit = True

        # Check if robot has swept past the Target X coordinate or hard stage limits
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
            # Turn 1: Face progression heading toward target
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
    # 3. CLOSED-LOOP AVOIDANCE: ROTATE TO MICRO-STEP HEADING
    # ============================================================
    elif state == "AVOID_ROTATE":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            state = "AVOID_STEP_FORWARD"
            micro_step_progress = 0.0

        update_usd_transform(current_pos, current_yaw)

    # ============================================================
    # 4. CLOSED-LOOP AVOIDANCE: TRANSLATE MICRO-STEP & RE-QUERY VLM
    # ============================================================
    elif state == "AVOID_STEP_FORWARD":
        step_dist = FORWARD_SPEED * 0.6 * dt
        move_forward_step(step_dist, dt)
        micro_step_progress += step_dist

        if micro_step_progress >= micro_step_target:
            # Micro-step done -> HALT & TRIGGER NEXT VLM INSPECTION
            state = "VLM_INSPECTING"
            if status_label:
                status_label.text = f"Status: VLM Inspecting ({avoidance_sub_phase})"
            print(f"[Micro-Step Done]: Triggering VLM query at lateral offset = {lateral_detour_ledger:.2f}m...")
            async_query_vlm_agent(rgb_frame, min_dist_ahead, avoidance_sub_phase, lateral_detour_ledger, on_vlm_agent_decision)

        update_usd_transform(current_pos, current_yaw)

    # ============================================================
    # 5. CLOSED-LOOP AVOIDANCE: ZEROING THE LEDGER & RETURN
    # ============================================================
    elif state == "RETURN_TURN_TO_LANE":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            state = "RETURN_DRIVE_TO_LANE"

        update_usd_transform(current_pos, current_yaw)

    elif state == "RETURN_DRIVE_TO_LANE":
        step_dist = FORWARD_SPEED * 0.8 * dt
        move_forward_step(step_dist, dt)

        # Stop driving inward when cross-track ledger reaches zero
        if abs(lateral_detour_ledger) <= 0.05 or (lateral_detour_ledger * detour_side <= 0.0):
            lateral_detour_ledger = 0.0
            target_yaw = lane_base_yaw
            state = "RETURN_FINAL_ALIGN"
            if status_label:
                status_label.text = "Status: Re-aligning to Lane..."

        update_usd_transform(current_pos, current_yaw)

    elif state == "RETURN_FINAL_ALIGN":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            state = "MOVING"
            step_start_pos = list(current_pos)
            print("[Avoidance]: Lane baseline fully restored. Resuming sweep.")
            if status_label:
                status_label.text = "Status: MOVING"

        update_usd_transform(current_pos, current_yaw)

    # ============================================================
    # 6. MONOTONIC ZIG-ZAG U-TURNS (PROGRESS TOWARD TARGET)
    # ============================================================
    # Turn 1: Align heading toward progression axis (0 deg for +X, 180 deg for -X)
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

    # Lane Shift: Drive forward along progression axis (moves toward target X)
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

    # Turn 2: Re-align heading to the new sweep direction (90 deg or 270 deg)
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