"""
PhantomX Target-Directed Zig-Zag Controller with Live ROS 2 Camera Bridge
-------------------------------------------------------------------------
- 100% Preserved Orientation-Based Movement Logic from User Script.
- Modified ONLY ROS 2 Camera perception setup to fix topic & QoS matching.
"""

import time
import math
import numpy as np
import omni.usd
import omni.timeline
import omni.kit.app
import omni.ui as ui
from pxr import Usd, UsdGeom, Gf

# ROS 2 & OpenCV imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

try:
    from cv_bridge import CvBridge
    _has_cv_bridge = True
except Exception:
    _has_cv_bridge = False

def ros_img_to_numpy(msg, desired_encoding='passthrough'):
    """Converts ROS Image to numpy array safely with or without cv_bridge"""
    if _has_cv_bridge:
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

# ============================================================
# 1. CONFIGURATION & STAGE SETUP
# ============================================================
stage = omni.usd.get_context().get_stage()

# Auto-start simulation timeline so OmniGraph OnPlaybackTick node fires!
timeline = omni.timeline.get_timeline_interface()
if not timeline.is_playing():
    print("▶️ Starting Simulation Timeline so ROS 2 Action Graph ticks...")
    timeline.play()

ROBOT_PATH         = "/World/phantomX_base"
SPAWN_POSITION     = (0.3295789538104168, -1.681203353653679, 0.10692057425096305)
SPAWN_ROTATION     = (0.0, 0.0, 90.0)
TARGET_DESTINATION = (10.0, 1.5)                     # (X, Y) Destination goal
TARGET_RADIUS      = 0.3                             # Arrival threshold in meters

FORWARD_SPEED      = 1.0                             # m/s
TURN_SPEED         = 180.0                           # deg/s
LANE_SHIFT         = 0.5                             # Lane shift width (m)
OBSTACLE_THRESHOLD = 0.8                             # Stop threshold (m)
PRINT_INTERVAL     = 2.0                             # Telemetry print interval (s)

# ROS 2 Topic Candidates (matching ActionGraph topics: /rgb & /depth)
RGB_TOPICS   = ["/rgb", "/camera/rgb", "/phantomx/camera/rgb"]
DEPTH_TOPICS = ["/depth", "/camera/depth", "/phantomx/camera/depth"]

# ============================================================
# 2. SPAWN / RETRIEVE ROBOT PRIM (NON-DESTRUCTIVE)
# ============================================================
def spawn_phantomx(stage, prim_path=ROBOT_PATH, position=SPAWN_POSITION, rotation_euler=SPAWN_ROTATION):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(
            "/home/user/Desktop/llmforhexapod/usd_to_stl_conversion/phantomX_base.usd"
        )

    xform = UsdGeom.Xformable(prim)
    
    # Non-destructive transform update: reuse existing ops if present
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

# Bounding box bounds
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
navmesh_prim = stage.GetPrimAtPath("/World/Terrain/section_00_asphalt/NavMeshVolume")

min_x, max_x = 0.0, 12.0
min_y, max_y = -2.0, 2.0
if navmesh_prim.IsValid():
    bound = bbox_cache.ComputeWorldBound(navmesh_prim).ComputeAlignedBox()
    min_x, max_x = float(bound.GetMin()[0]), float(bound.GetMax()[0])
    min_y, max_y = float(bound.GetMin()[1]), float(bound.GetMax()[1])

# ============================================================
# 3. ROS 2 PERCEPTION NODE INITIALIZATION
# ============================================================
if not rclpy.ok():
    rclpy.init()

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('phantomx_isaac_perception')
        
        # SensorData / BestEffort QoS profile for high-rate video streams from Isaac Sim
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
        except Exception as e:
            pass

    def _depth_callback(self, msg):
        try:
            depth_frame = ros_img_to_numpy(msg, desired_encoding='32FC1')
            self.latest_depth = depth_frame
            
            # Compute center ROI distance
            h, w = depth_frame.shape
            roi = depth_frame[int(h * 0.35):int(h * 0.65), int(w * 0.35):int(w * 0.65)]
            valid = roi[(np.isfinite(roi)) & (roi > 0.05)]
            
            if len(valid) > 0:
                self.closest_obstacle_dist = float(np.min(valid))
            else:
                self.closest_obstacle_dist = float('inf')
        except Exception as e:
            pass

global _ros_node
_ros_node = PerceptionNode()

# ============================================================
# 4. MATH & ORIENTATION HELPERS (EXACT USER CODE)
# ============================================================
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

def calculate_turn_toward_target(current_p, current_yaw_deg, target_p):
    fwd_x, fwd_y, _ = get_forward_vector(current_yaw_deg)
    to_target_x = target_p[0] - current_p[0]
    to_target_y = target_p[1] - current_p[1]
    cross_z = (fwd_x * to_target_y) - (fwd_y * to_target_x)
    return 1.0 if cross_z >= 0.0 else -1.0

# ============================================================
# 5. STATE VARIABLES (EXACT USER CODE)
# ============================================================
global state, saved_pre_pause_state, current_pos, current_yaw, turn_sign, initial_turn_calculated
global target_yaw, last_time, shifted_distance, last_print_time

current_pos = list(SPAWN_POSITION)
current_yaw = SPAWN_ROTATION[2]
state = "MOVING"
saved_pre_pause_state = "MOVING"

turn_sign = 1.0
initial_turn_calculated = False
target_yaw = current_yaw
last_time = time.time()
shifted_distance = 0.0
last_print_time = 0.0

def update_usd_transform(pos, yaw_deg):
    translate_op.Set(Gf.Vec3d(*pos))
    rotate_op.Set(Gf.Vec3d(0.0, 0.0, yaw_deg))

update_usd_transform(current_pos, current_yaw)

# ============================================================
# 6. UI PANEL
# ============================================================
global _phantomx_window
if "_phantomx_window" in globals() and _phantomx_window:
    try:
        _phantomx_window.visible = False
        _phantomx_window.destroy()
    except Exception:
        pass
    _phantomx_window = None

_phantomx_window = ui.Window("PhantomX ROS 2 Perception Control", width=380, height=220, flags=ui.WINDOW_FLAGS_NO_SAVED_SETTINGS)
_phantomx_window.position_x = 80
_phantomx_window.position_y = 120
_phantomx_window.visible = True
_phantomx_window.focus()

try:
    ui.Workspace.show_window("PhantomX ROS 2 Perception Control", True)
except Exception:
    pass

status_label = None
sensor_label = None

with _phantomx_window.frame:
    with ui.VStack(spacing=8, style={"margin": 12}):
        ui.Label("🤖 PhantomX + ROS 2 OmniGraph Tracking", height=22, style={"font_size": 14, "color": 0xFF00FFFF})
        sensor_label = ui.Label("Obstacle Dist: -- m", height=18, style={"font_size": 12, "color": 0xFFFFFF00})
        status_label = ui.Label("Status: ▶️ MOVING", height=22, style={"font_size": 13})
        
        with ui.HStack(spacing=10, height=40):
            def on_emergency_stop():
                global state
                state = "STOPPED"
                if status_label:
                    status_label.text = "Status: 🛑 EMERGENCY STOPPED"

            def on_pause_resume():
                global state, saved_pre_pause_state, last_time
                if state == "USER_PAUSED":
                    state = saved_pre_pause_state
                    last_time = time.time()
                    if status_label:
                        status_label.text = f"Status: ▶️ RESUMED ({state})"
                elif state != "STOPPED":
                    saved_pre_pause_state = state
                    state = "USER_PAUSED"
                    if status_label:
                        status_label.text = "Status: ⏸️ MANUALLY PAUSED"

            def on_reset():
                global state, current_pos, current_yaw, last_time, shifted_distance, initial_turn_calculated
                state = "MOVING"
                current_pos = list(SPAWN_POSITION)
                current_yaw = SPAWN_ROTATION[2]
                initial_turn_calculated = False
                shifted_distance = 0.0
                last_time = time.time()
                update_usd_transform(current_pos, current_yaw)
                if status_label:
                    status_label.text = "Status: 🔄 RESET & MOVING"

            ui.Button("🛑 Stop", clicked_fn=on_emergency_stop, style={"background_color": 0xFF2222DD})
            ui.Button("⏯️ Pause/Play", clicked_fn=on_pause_resume, style={"background_color": 0xFFDD8822})
            ui.Button("🔄 Reset", clicked_fn=on_reset, style={"background_color": 0xFF22AA22})

# ============================================================
# 7. MAIN CONTROLLER & SENSOR LOOP (EXACT USER CODE)
# ============================================================
def zig_zag_update(event):
    global state, saved_pre_pause_state, current_pos, current_yaw, turn_sign, initial_turn_calculated
    global target_yaw, last_time, shifted_distance, last_print_time

    # 1. Spin ROS 2 events non-blockingly
    rclpy.spin_once(_ros_node, timeout_sec=0.0)

    if state == "STOPPED" or state == "USER_PAUSED":
        return

    now = time.time()
    dt = min(now - last_time, 0.1)
    last_time = now

    min_dist_ahead = _ros_node.closest_obstacle_dist
    rgb_frame = _ros_node.latest_rgb
    depth_frame = _ros_node.latest_depth

    # Update sensor UI label
    if sensor_label:
        if min_dist_ahead != float('inf'):
            sensor_label.text = f"Obstacle Dist: {min_dist_ahead:.2f} m"
        else:
            sensor_label.text = "Obstacle Dist: Clear (>10m)"

    # --- PERIODIC DIAGNOSTIC PRINT (EVERY 2 SECONDS) ---
    if (now - last_print_time) >= PRINT_INTERVAL:
        last_print_time = now
        print("\n" + "-" * 50)
        print(f"⏱️ [ROS 2 Camera Telemetry] Time: {now:.1f}s | State: {state}")
        
        if depth_frame is not None:
            h, w = depth_frame.shape
            print(f"📊 Depth Buffer : Shape {depth_frame.shape} | Closest ROI Dist: {min_dist_ahead:.3f} m")
        else:
            print(f"📊 Depth Buffer : [Waiting for topic '/depth']")

        if rgb_frame is not None:
            h, w, c = rgb_frame.shape
            print(f"🖼️ RGB Buffer   : Shape {rgb_frame.shape} | Center Pixel BGR: {list(rgb_frame[h//2, w//2])}")
        else:
            print(f"🖼️ RGB Buffer   : [Waiting for topic '/rgb']")
            
        print(f"📍 Robot Pose    : Pos ({current_pos[0]:.2f}, {current_pos[1]:.2f}) | Heading: {current_yaw:.1f}°")
        print("-" * 50)

    # 1. SWEEP FORWARD (EXACT USER ORIENTATION-BASED CODE)
    if state == "MOVING":
        # Target destination reached
        if compute_dist_2d(current_pos, TARGET_DESTINATION) <= TARGET_RADIUS:
            state = "STOPPED"
            if status_label:
                status_label.text = "Status: 🎯 DESTINATION REACHED"
            print(f"🎯 Target {TARGET_DESTINATION} reached!")
            return

        # Obstacle detected ahead: halt to avoid collision
        if min_dist_ahead < OBSTACLE_THRESHOLD:
            if status_label:
                status_label.text = f"Status: ⚠️ OBSTACLE DETECTED ({min_dist_ahead:.2f}m)"
            return

        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] += fwd_x * FORWARD_SPEED * dt
        current_pos[1] += fwd_y * FORWARD_SPEED * dt

        # Boundary checks
        edge_hit = False
        margin = 0.3
        if current_pos[0] < min_x + margin:
            current_pos[0] = min_x + margin
            edge_hit = True
        elif current_pos[0] > max_x - margin:
            current_pos[0] = max_x - margin
            edge_hit = True
            
        if current_pos[1] < min_y + margin:
            current_pos[1] = min_y + margin
            edge_hit = True
        elif current_pos[1] > max_y - margin:
            current_pos[1] = max_y - margin
            edge_hit = True

        if edge_hit:
            if not initial_turn_calculated:
                turn_sign = calculate_turn_toward_target(current_pos, current_yaw, TARGET_DESTINATION)
                initial_turn_calculated = True

            target_yaw = (current_yaw + turn_sign * 90.0) % 360.0
            state = "TURNING_1"
            turn_name = "Left" if turn_sign > 0 else "Right"
            if status_label:
                status_label.text = f"Status: 🔄 Edge hit -> Turn 1 ({turn_name})"

        update_usd_transform(current_pos, current_yaw)

    # 2. FIRST 90° TURN
    elif state == "TURNING_1":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            state = "LANE_SHIFTING"
            shifted_distance = 0.0
            if status_label:
                status_label.text = "Status: ➡️ Shifting Lane"

        update_usd_transform(current_pos, current_yaw)

    # 3. LANE SHIFT
    elif state == "LANE_SHIFTING":
        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        dist = FORWARD_SPEED * dt
        current_pos[0] += fwd_x * dist
        current_pos[1] += fwd_y * dist
        shifted_distance += dist

        if shifted_distance >= LANE_SHIFT:
            state = "TURNING_2"
            target_yaw = (current_yaw + turn_sign * 90.0) % 360.0
            turn_name = "Left" if turn_sign > 0 else "Right"
            if status_label:
                status_label.text = f"Status: 🔄 Turn 2 ({turn_name})"

        update_usd_transform(current_pos, current_yaw)

    # 4. SECOND 90° TURN (RE-ALIGNING)
    elif state == "TURNING_2":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            turn_sign = -turn_sign
            state = "MOVING"
            if status_label:
                status_label.text = "Status: ▶️ MOVING (New Lane)"

        update_usd_transform(current_pos, current_yaw)

# Register update loop
global _phantomx_sub
_phantomx_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(zig_zag_update)