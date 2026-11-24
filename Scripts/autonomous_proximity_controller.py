"""
PhantomX Target-Directed Controller: Autonomous Proximity-Based Obstacle Avoidance
----------------------------------------------------------------------------------
Fully Autonomous Controller for Isaac Sim.
Zero external VLM / LLM dependencies. Zero YOLO / ONNX dependencies.
Operates exclusively on real-time spatial depth perception, geometric projection,
target-directed divergence, and an odometer displacement ledger.

Key Features:
1. Target-Directed Zig-Zag Sweeper:
   - Sweeps area along target direction with configurable step & pause pacing.
   - Dynamic lane shifting along target progression vector.
2. 3-Sector Spatial Depth Slicing:
   - High-rate ROS 2 depth streaming (/camera/depth).
   - Real-time Left, Center, and Right sector proximity monitoring.
3. Target-First Geometric Avoidance:
   - Selects detour side (Left/Right) favoring the global target bearing.
   - Verifies lateral clearance using sector depths; flips side if target is blocked.
   - NavMesh boundary projection to prevent pathing off road edges.
4. Closed-Loop Ledger Detour & Return:
   - Accumulates cross-track displacement in lateral_detour_ledger.
   - Stepwise divergence -> flank pre-check -> flank bypass -> inward verification.
   - Exact zero-ledger return trajectory to resume original sweep lane.
5. Autonomous Deadlock & Edge Escape:
   - Boundary-aware early lane shift when obstacles block road limits.
   - Big-lane-skip across road on repeated edge deadlocks.
   - Emergency reverse with escape turns if trapped against boundary.
6. Breadcrumb Trail Visualization:
   - Ground-projected colored plates illustrating robot trajectory and states.
7. Interactive Isaac Sim UI:
   - Real-time telemetry (pose, sector depths, detour ledger, state).
   - Stop, Pause/Play, and Reset controls.
"""

import time
import math
import numpy as np
import omni.usd
import omni.timeline
import omni.kit.app
import omni.ui as ui
from pxr import Usd, UsdGeom, Gf, Sdf

# ============================================================
# 0. PREVIOUS INSTANCE CLEANUP (Safe Script Reload)
# ============================================================
if globals().get("_phantomx_sub"):
    try:
        globals()["_phantomx_sub"].unsubscribe()
    except Exception:
        pass
    globals()["_phantomx_sub"] = None

if globals().get("_ros_node"):
    try:
        globals()["_ros_node"].destroy_node()
    except Exception:
        pass
    globals()["_ros_node"] = None

if globals().get("_phantomx_window"):
    try:
        globals()["_phantomx_window"].visible = False
        globals()["_phantomx_window"].destroy()
    except Exception:
        pass
    globals()["_phantomx_window"] = None

# ============================================================
# 1. ROS 2 IMPORTS & IMAGE BRIDGE
# ============================================================
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

try:
    import cv2
    from cv_bridge import CvBridge
    _has_cv_bridge = True
except Exception:
    _has_cv_bridge = False

def ros_img_to_numpy(msg, desired_encoding='passthrough'):
    """Convert ROS 2 sensor_msgs/Image to numpy array without external dependency hard-fails."""
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

# Ensure Isaac Sim timeline is running
stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()
if not timeline.is_playing():
    timeline.play()

# ============================================================
# 2. CONFIGURATION & STAGE PARAMETERS
# ============================================================
ROBOT_PATH         = "/World/phantomX_base"
ROBOT_USD_PATH     = "/home/user/Desktop/llmforhexapod/usd_to_stl_conversion/phantomX_base.usd"
SPAWN_POSITION     = (0.11952272590602997, 1.8697408479533721, 0.1)
SPAWN_ROTATION     = (0.0, 0.0, -90.0)
TARGET_DESTINATION = (9.0, -1.8266811603307502)
TARGET_RADIUS      = 0.3

# Robot Dimensions
ROBOT_LENGTH               = 0.55
DIVERGE_CLEARANCE_MARGIN   = 0.20
MIN_DIVERGE_STEP           = ROBOT_LENGTH + DIVERGE_CLEARANCE_MARGIN  # 0.75m
MIN_CHASSIS_CLEAR          = 0.40                                    # Min lateral offset to clear chassis (m)

# Motion Parameters
STEP_DISTANCE      = 0.4      # Sweep step distance (m)
PAUSE_DURATION     = 0.5      # Sweep pause interval (s)
FORWARD_SPEED      = 0.5      # Base translational speed (m/s)
TURN_SPEED         = 180.0    # Angular rotational speed (deg/s)
LANE_SHIFT         = 0.5      # Lane width increment (m)
MANEUVER_SPEED_RATIO = 0.60   # Cautious speed factor during detour maneuvers

# Proximity Detection Thresholds
OBSTACLE_THRESHOLD = 0.80     # Proximity avoidance trigger threshold (m)
EMERGENCY_STOP_DIST = 0.25    # Immediate emergency stop/backup threshold (m)
EMERGENCY_BACKUP_CLEAR_DIST = MIN_DIVERGE_STEP  # Clear distance to exit backup (0.75m)
EMERGENCY_BACKUP_MAX_DIST = 2.50                # Max backup safety limit (m)
EMERGENCY_BACKUP_SPEED    = 0.45                # Reverse speed (m/s)
EMERGENCY_BACKUP_STUCK_TIME = 4.5               # Seconds stuck backing before escape turn
EMERGENCY_ESCAPE_DIST     = 0.60                # Distance to advance along escape yaw (m)
EMERGENCY_COOLDOWN_DIST   = 1.30                # Forward progress required to reset emergency
EMERGENCY_COOLDOWN_TIME   = 4.0                 # Minimum seconds between emergency triggers

# Divergence & Flank Settings
DIVERGE_ANGLE_INC         = 31.0   # Deflection angle increment (deg)
DIVERGE_ESCALATION_LIMIT  = 3      # Max micro-diverge attempts before wide detour
WIDE_DIVERGE_ANGLE        = 65.0   # Wide evasive angle (deg)
WIDE_DIVERGE_STEP         = 0.75   # Wide evasive step size (m)
DIVERGE_GRACE_PERIOD      = 0.60   # Seconds protecting deliberate diverge from false reverse
CRITICAL_TOUCH_DIST       = 0.18   # Hard touch barrier (m)

# Flank Distance Limits
FLANK_SAFETY_PAD          = 0.30
FLANK_MIN_DIST            = 0.80
FLANK_MAX_DIST            = 2.20
DEFAULT_FLANK_EXTRA       = 0.45

# Lane Shift & Deadlock Parameters
LANE_STUCK_THRESHOLD      = 2      # Consecutive blocked shifts before big skip
PRINT_INTERVAL            = 2.0    # Telemetry logging rate (s)
AUTO_LOOP_ENABLED         = True   # Automatically restart from spawn when destination reached
AUTO_LOOP_DELAY           = 2.5    # Pause before restarting

RGB_TOPICS   = ["/rgb", "/camera/rgb", "/phantomx/camera/rgb"]
DEPTH_TOPICS = ["/depth", "/camera/depth", "/phantomx/camera/depth"]

# ============================================================
# 3. ROBOT SPAWN & NAVMESH BOUNDARIES
# ============================================================
def spawn_phantomx(stage, prim_path=ROBOT_PATH, position=SPAWN_POSITION, rotation_euler=SPAWN_ROTATION):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(ROBOT_USD_PATH)

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

# Extract NavMesh Bounds
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
navmesh_prim = stage.GetPrimAtPath("/World/NavMeshVolume")
if not navmesh_prim.IsValid():
    navmesh_prim = stage.GetPrimAtPath("/World/Terrain/section_00_asphalt/NavMeshVolume")

min_x, max_x = -10.0, 12.0
min_y, max_y = -2.0, 2.0
if navmesh_prim.IsValid():
    bound = bbox_cache.ComputeWorldBound(navmesh_prim).ComputeAlignedBox()
    min_x, max_x = float(bound.GetMin()[0]), float(bound.GetMax()[0])
    min_y, max_y = float(bound.GetMin()[1]), float(bound.GetMax()[1])

ROBOT_SAFETY_MARGIN = 0.35
SAFE_MIN_Y = min_y + ROBOT_SAFETY_MARGIN
SAFE_MAX_Y = max_y - ROBOT_SAFETY_MARGIN
SAFE_MIN_X = min_x + ROBOT_SAFETY_MARGIN
SAFE_MAX_X = max_x - ROBOT_SAFETY_MARGIN

print(f"[NavMesh Setup]: X bounds=[{SAFE_MIN_X:.2f}, {SAFE_MAX_X:.2f}], Y bounds=[{SAFE_MIN_Y:.2f}, {SAFE_MAX_Y:.2f}]")

# ============================================================
# 4. ROS 2 PERCEPTION NODE (SPATIAL DEPTH SLICING)
# ============================================================
if not rclpy.ok():
    rclpy.init()

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('phantomx_autonomous_perception')
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

            # Extract 3 horizontal spatial sectors across middle 30% vertical view
            row_top, row_bottom = int(h * 0.35), int(h * 0.65)
            left_col = depth_frame[row_top:row_bottom, :int(w * 0.33)]
            center_col = depth_frame[row_top:row_bottom, int(w * 0.33):int(w * 0.66)]
            right_col = depth_frame[row_top:row_bottom, int(w * 0.66):]

            def extract_valid_min(roi):
                valid = roi[(np.isfinite(roi)) & (roi > 0.05)]
                return float(np.min(valid)) if len(valid) > 0 else float('inf')

            self.depth_left_dist = extract_valid_min(left_col)
            self.depth_center_dist = extract_valid_min(center_col)
            self.depth_right_dist = extract_valid_min(right_col)
            self.closest_obstacle_dist = min(self.depth_left_dist, self.depth_center_dist, self.depth_right_dist)
        except Exception:
            pass

global _ros_node
_ros_node = PerceptionNode()

# ============================================================
# 5. VECTOR & GEOMETRIC UTILITIES
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
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_target_angle_diff(current_pose, current_heading_deg, target_pose):
    """
    Computes relative angle between robot heading and target destination.
    Positive -> Target is to the LEFT.
    Negative -> Target is to the RIGHT.
    """
    dx = target_pose[0] - current_pose[0]
    dy = target_pose[1] - current_pose[1]
    target_bearing_deg = math.degrees(math.atan2(dy, dx)) % 360.0
    return (target_bearing_deg - current_heading_deg + 180.0) % 360.0 - 180.0

def is_pushing_into_edge(pos, side, margin=0.10):
    """True if lateral step along side would violate NavMesh boundaries."""
    if side > 0 and pos[1] >= SAFE_MAX_Y - margin:
        return True
    if side < 0 and pos[1] <= SAFE_MIN_Y + margin:
        return True
    return False

def pick_escape_yaw():
    """Pick 90 deg left or right whose rear direction has more road clearance."""
    best_yaw, best_margin = None, -1e9
    for sign in (1.0, -1.0):
        cand = (current_yaw + sign * 90.0) % 360.0
        fx, fy, _ = get_forward_vector(cand)
        px = current_pos[0] - fx * EMERGENCY_ESCAPE_DIST * 1.5
        py = current_pos[1] - fy * EMERGENCY_ESCAPE_DIST * 1.5
        margin = min(px - SAFE_MIN_X, SAFE_MAX_X - px, py - SAFE_MIN_Y, SAFE_MAX_Y - py)
        if margin > best_margin:
            best_margin, best_yaw = margin, cand
    return best_yaw if best_margin > 0.05 else None

# ============================================================
# 6. BREADCRUMB VISUALIZATION SYSTEM
# ============================================================
breadcrumb_counter = 0
BREADCRUMB_SIZE = 0.15
BREADCRUMB_THICKNESS = 0.01

def add_breadcrumb(pos, color=(0.0, 1.0, 0.0)):
    global breadcrumb_counter
    stage = omni.usd.get_context().get_stage()
    prim_path = f"/World/Breadcrumbs/plate_{breadcrumb_counter}"
    breadcrumb_counter += 1

    existing = stage.GetPrimAtPath(prim_path)
    if existing.IsValid():
        stage.RemovePrim(prim_path)

    plate = UsdGeom.Cube.Define(stage, prim_path)
    plate.CreateSizeAttr(2.5)

    xform = UsdGeom.Xformable(plate)
    translate_op, scale_op = None, None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op

    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    if scale_op is None:
        scale_op = xform.AddScaleOp()

    translate_op.Set(Gf.Vec3d(pos[0], pos[1], BREADCRUMB_THICKNESS / 2))
    scale_op.Set(Gf.Vec3d(BREADCRUMB_SIZE, BREADCRUMB_SIZE, BREADCRUMB_THICKNESS))
    plate.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    prim = plate.GetPrim()
    prim.CreateAttribute("purpose", Sdf.ValueTypeNames.Token).Set(UsdGeom.Tokens.guide)
    prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)

# ============================================================
# 7. STATE VARIABLES & MOTION LEDGER
# ============================================================
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

# Odometer Displacement Ledger & Avoidance State
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
wide_diverge_count = 0
diverge_grace_until = 0.0

flank_step_start_pos = list(current_pos)
flank_step_progress = 0.0
flank_pause_start_time = 0.0
pre_check_settle_start = 0.0

# Emergency & Deadlock State
emergency_triggered = False
emergency_cooldown_start = 0.0
emergency_trigger_pos = [0.0, 0.0]
emergency_backup_start_time = 0.0
emergency_escape_turns = 0
emergency_original_yaw = 0.0
backup_progress = 0.0
escape_progress = 0.0

repeated_backup_pos = None
repeated_backup_count = 0
deadlock_retreat_count = 0
force_retreat_target = 1.5

consecutive_blocked_shifts = 0
lane_stuck_pos = None
edge_stuck_time = 0.0

# Big Skip State
big_skip_step_progress = 0.0
big_skip_advance_progress = 0.0
big_skip_pause_start = 0.0
big_skip_target_y = 0.0

# Loop Management
_loop_restart_pending = False
_loop_restart_time = 0.0

def update_usd_transform(pos, yaw_deg):
    if translate_op is None or rotate_op is None:
        return
    pos[0] = max(SAFE_MIN_X, min(SAFE_MAX_X, pos[0]))
    pos[1] = max(SAFE_MIN_Y, min(SAFE_MAX_Y, pos[1]))
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

def get_adaptive_speed():
    """Modulates forward speed according to obstacle proximity."""
    dist = _ros_node.closest_obstacle_dist
    if avoidance_sub_phase != "IDLE":
        return FORWARD_SPEED * MANEUVER_SPEED_RATIO
    if dist < OBSTACLE_THRESHOLD:
        scale = max(0.40, min(1.0, dist / OBSTACLE_THRESHOLD))
        return FORWARD_SPEED * scale
    return FORWARD_SPEED

def check_emergency_proximity(dist_ahead):
    if state in ["STOPPED", "USER_PAUSED", "EMERGENCY_BACKUP", "EMERGENCY_ESCAPE_TURN",
                 "EMERGENCY_ESCAPE_BACK", "EMERGENCY_ESCAPE_REALIGN"]:
        return False
    if time.time() < diverge_grace_until:
        return dist_ahead < CRITICAL_TOUCH_DIST
    if dist_ahead >= EMERGENCY_STOP_DIST:
        return False
    dist_from_trigger = compute_dist_2d(current_pos, emergency_trigger_pos)
    time_since_trigger = time.time() - emergency_cooldown_start
    if emergency_triggered and (dist_from_trigger < EMERGENCY_COOLDOWN_DIST or time_since_trigger < EMERGENCY_COOLDOWN_TIME):
        return False
    return True

def initiate_emergency_backup():
    global state, emergency_triggered, emergency_cooldown_start, emergency_trigger_pos
    global backup_progress, avoidance_sub_phase, lateral_detour_ledger
    global repeated_backup_pos, repeated_backup_count
    global emergency_backup_start_time, emergency_escape_turns, emergency_original_yaw

    emergency_backup_start_time = time.time()
    emergency_escape_turns = 0
    emergency_original_yaw = current_yaw

    if repeated_backup_pos is not None and compute_dist_2d(current_pos, repeated_backup_pos) < 0.5:
        repeated_backup_count += 1
    else:
        repeated_backup_count = 1
        repeated_backup_pos = list(current_pos)

    print(f"[EMERGENCY BACKUP]: Distance critical ({_ros_node.closest_obstacle_dist:.2f}m). Backing up...")
    emergency_trigger_pos = list(current_pos)
    emergency_triggered = True
    emergency_cooldown_start = time.time()
    backup_progress = 0.0
    avoidance_sub_phase = "IDLE"
    lateral_detour_ledger = 0.0
    state = "EMERGENCY_BACKUP"

    if status_label:
        status_label.text = "Status: EMERGENCY BACKUP"

def trigger_early_lane_shift():
    global state, target_yaw, avoidance_sub_phase, step_start_pos, shifted_distance, sweep_direction_y
    print("[Edge Obstacle Guard]: Boundary obstacle detected. Executing early lane shift.")
    target_yaw = get_lane_shift_heading()
    avoidance_sub_phase = "IDLE"

    if abs(get_shortest_angle_diff(target_yaw, current_yaw)) < 1.0:
        state = "TURNING_2"
        sweep_direction_y = -sweep_direction_y
        target_yaw = 90.0 if sweep_direction_y > 0 else 270.0
        shifted_distance = 0.0
        step_start_pos = list(current_pos)
    else:
        state = "TURNING_1"

def trigger_big_lane_skip():
    global state, target_yaw, sweep_direction_y, consecutive_blocked_shifts, lane_stuck_pos
    global avoidance_sub_phase, big_skip_target_y, big_skip_step_progress, big_skip_advance_progress
    print("[BIG LANE SKIP]: Repeated shift deadlock at edge. Crossing to opposite side of road.")

    dist_to_top = SAFE_MAX_Y - current_pos[1]
    dist_to_bot = current_pos[1] - SAFE_MIN_Y

    if dist_to_top < dist_to_bot:
        big_skip_target_y = SAFE_MIN_Y
        sweep_direction_y = 1.0
        target_yaw = 270.0
    else:
        big_skip_target_y = SAFE_MAX_Y
        sweep_direction_y = -1.0
        target_yaw = 90.0

    avoidance_sub_phase = "IDLE"
    consecutive_blocked_shifts = 0
    lane_stuck_pos = None
    big_skip_step_progress = 0.0
    big_skip_advance_progress = 0.0
    state = "BIG_SKIP_TURN"

    if status_label:
        status_label.text = "Status: Big Skip Across Road"

def enter_wide_obstacle_diverge(angle=WIDE_DIVERGE_ANGLE, step=WIDE_DIVERGE_STEP, reason=""):
    global avoidance_sub_phase, target_yaw, micro_step_target, micro_step_progress
    global state, wide_diverge_count, diverge_grace_until

    if wide_diverge_count >= 1:
        print(f"[Wide Diverge Limit]: Max wide detour attempts reached ({reason}). Skipping lane.")
        trigger_early_lane_shift()
        return

    wide_diverge_count += 1
    print(f"[Wide Diverge]: Escalating ({reason}). Detour angle={angle} deg.")
    avoidance_sub_phase = "WIDE_OBSTACLE_DIVERGE"
    target_yaw = (lane_base_yaw + (detour_side * angle)) % 360.0
    micro_step_target = step
    micro_step_progress = 0.0
    diverge_grace_until = time.time() + DIVERGE_GRACE_PERIOD
    state = "AVOID_ROTATE_THEN_STEP"

def initiate_bypass_sequence():
    """Deterministically plans divergence direction based on target bearing & sector depths."""
    global state, avoidance_sub_phase, target_yaw, detour_side, diverge_turn_count
    global wide_diverge_count, lane_base_yaw, initial_detected_distance, lateral_detour_ledger
    global diverge_grace_until

    diverge_turn_count = 1
    wide_diverge_count = 0
    lane_base_yaw = current_yaw
    initial_detected_distance = _ros_node.closest_obstacle_dist
    lateral_detour_ledger = 0.0

    # Road edge proximity check: don't attempt detour if pressed against edge
    dist_to_top = SAFE_MAX_Y - current_pos[1]
    dist_to_bot = current_pos[1] - SAFE_MIN_Y
    if (sweep_direction_y > 0 and dist_to_top < 0.60) or (sweep_direction_y < 0 and dist_to_bot < 0.60):
        trigger_early_lane_shift()
        return

    left_d = _ros_node.depth_left_dist
    right_d = _ros_node.depth_right_dist

    # Target-First Preference with Proximity Safety Check
    target_angle_diff = get_target_angle_diff(current_pos, current_yaw, TARGET_DESTINATION)
    preferred_side = 1.0 if target_angle_diff > 0 else -1.0
    preferred_depth = left_d if preferred_side > 0 else right_d
    opposite_depth  = right_d if preferred_side > 0 else left_d

    MIN_VIABLE_CLEAR = MIN_CHASSIS_CLEAR + 0.20  # 0.60m

    if preferred_depth < MIN_VIABLE_CLEAR and opposite_depth > (preferred_depth + 0.25):
        detour_side = -preferred_side
        print(f"[Target-First Diverge]: Target side blocked ({preferred_depth:.2f}m). Rerouting to open side: {'LEFT' if detour_side > 0 else 'RIGHT'}")
    else:
        detour_side = preferred_side
        print(f"[Target-First Diverge]: Steering toward target -> {'LEFT' if detour_side > 0 else 'RIGHT'} (angle diff={target_angle_diff:.1f} deg)")

    # NavMesh Boundary Clearance Check
    test_yaw = (lane_base_yaw + (detour_side * DIVERGE_ANGLE_INC)) % 360.0
    fwd_x, fwd_y, _ = get_forward_vector(test_yaw)
    pred_x = current_pos[0] + (fwd_x * 0.60)
    pred_y = current_pos[1] + (fwd_y * 0.60)

    if pred_x < SAFE_MIN_X or pred_x > SAFE_MAX_X or pred_y < SAFE_MIN_Y or pred_y > SAFE_MAX_Y:
        detour_side = -detour_side
        print(f"[Boundary Safety]: Road boundary reached. Inverting detour side to: {'LEFT' if detour_side > 0 else 'RIGHT'}")

    avoidance_sub_phase = "DIVERGE_SCAN"
    target_yaw = (lane_base_yaw + (detour_side * DIVERGE_ANGLE_INC)) % 360.0
    diverge_grace_until = time.time() + DIVERGE_GRACE_PERIOD
    state = "AVOID_INITIAL_TURN"
    if status_label:
        status_label.text = "Status: Executing Bypass Diverge..."

def is_target_x_reached():
    progress_dir = get_progress_dir_x()
    if progress_dir > 0 and current_pos[0] >= TARGET_DESTINATION[0]:
        return True
    if progress_dir < 0 and current_pos[0] <= TARGET_DESTINATION[0]:
        return True
    return False

def stop():
    global state
    state = "STOPPED"
    if status_label:
        status_label.text = "Status: STOPPED"
    print("[PhantomX]: Controller stopped.")

def finish_run(reason="Destination Reached"):
    global _loop_restart_pending, _loop_restart_time
    stop()
    if AUTO_LOOP_ENABLED:
        print(f"[Auto Loop]: Run finished ({reason}). Restarting in {AUTO_LOOP_DELAY:.1f}s...")
        if status_label:
            status_label.text = f"Status: Finished ({reason}) - Looping..."
        _loop_restart_pending = True
        _loop_restart_time = time.time() + AUTO_LOOP_DELAY
    else:
        if status_label:
            status_label.text = f"Status: {reason}"

def perform_full_reset():
    global state, current_pos, current_yaw, step_start_pos, last_time, shifted_distance
    global sweep_direction_y, lane_base_yaw, lateral_detour_ledger, avoidance_sub_phase
    global emergency_triggered, emergency_cooldown_start, emergency_trigger_pos
    global wide_diverge_count, consecutive_blocked_shifts

    wide_diverge_count = 0
    state = "MOVING"
    emergency_triggered = False
    emergency_cooldown_start = 0.0
    emergency_trigger_pos = list(SPAWN_POSITION)
    current_pos = list(SPAWN_POSITION)
    current_yaw = SPAWN_ROTATION[2]
    lane_base_yaw = current_yaw
    sweep_direction_y = 1.0 if SPAWN_ROTATION[2] == 90.0 else -1.0
    lateral_detour_ledger = 0.0
    avoidance_sub_phase = "IDLE"
    step_start_pos = list(current_pos)
    shifted_distance = 0.0
    last_time = time.time()
    consecutive_blocked_shifts = 0

    stage = omni.usd.get_context().get_stage()
    trail_prim = stage.GetPrimAtPath("/World/Breadcrumbs")
    if trail_prim.IsValid():
        stage.RemovePrim("/World/Breadcrumbs")

    update_usd_transform(current_pos, current_yaw)
    if status_label:
        status_label.text = "Status: RESET & MOVING"
    print("[PhantomX]: Reset complete. Robot returned to spawn.")

# ============================================================
# 8. OMNI UI CONTROLLER PANEL
# ============================================================
_phantomx_window = ui.Window(
    "PhantomX Autonomous Proximity Controller",
    width=400,
    height=260,
    flags=ui.WINDOW_FLAGS_NO_SAVED_SETTINGS
)
_phantomx_window.position_x = 80
_phantomx_window.position_y = 120
_phantomx_window.visible = True
_phantomx_window.focus()

status_label = None
sensor_label = None
sectors_label = None
ledger_label = None
pose_label = None

with _phantomx_window.frame:
    with ui.VStack(spacing=6, style={"margin": 10}):
        ui.Label("Autonomous Proximity Controller (No VLM / No YOLO)", height=20, style={"font_size": 13, "color": 0xFF00E5FF})
        ui.Label(f"Target Destination: ({TARGET_DESTINATION[0]:.1f}, {TARGET_DESTINATION[1]:.1f})", height=16, style={"font_size": 11, "color": 0xFFAAAAFF})
        sensor_label = ui.Label("Obstacle Dist: -- m", height=16, style={"font_size": 12, "color": 0xFFFFFF00})
        sectors_label = ui.Label("Sectors: L: -- | C: -- | R: --", height=16, style={"font_size": 11, "color": 0xFFCCCCCC})
        ledger_label = ui.Label("Detour Ledger: 0.00 m", height=16, style={"font_size": 11, "color": 0xFF88FF88})
        pose_label = ui.Label("Pose: (0.00, 0.00) @ 0.0°", height=16, style={"font_size": 11, "color": 0xFFB0BEC5})
        status_label = ui.Label("Status: MOVING", height=20, style={"font_size": 13})

        with ui.HStack(spacing=8, height=36):
            def on_stop():
                global _loop_restart_pending
                _loop_restart_pending = False
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

            def on_reset_btn():
                perform_full_reset()

            ui.Button("Stop", clicked_fn=on_stop, style={"background_color": 0xFF2222DD})
            ui.Button("Pause/Play", clicked_fn=on_pause_resume, style={"background_color": 0xFFDD8822})
            ui.Button("Reset", clicked_fn=on_reset_btn, style={"background_color": 0xFF22AA22})

# ============================================================
# 9. MAIN AUTONOMOUS UPDATE LOOP
# ============================================================
def autonomous_update(event):
    global state, saved_pre_pause_state, current_pos, current_yaw, sweep_direction_y
    global step_start_pos, target_yaw, pause_start_time, last_time, shifted_distance, last_print_time
    global lane_base_yaw, lateral_detour_ledger, micro_step_progress, micro_step_target, detour_side
    global avoidance_sub_phase, initial_detected_distance, required_flank_distance, flank_progress
    global diverge_turn_count, wide_diverge_count, edge_stuck_time
    global flank_step_start_pos, flank_step_progress, flank_pause_start_time, pre_check_settle_start
    global emergency_triggered, emergency_cooldown_start, emergency_trigger_pos, backup_progress
    global repeated_backup_count, repeated_backup_pos, emergency_backup_start_time
    global emergency_escape_turns, emergency_original_yaw, escape_progress
    global consecutive_blocked_shifts, lane_stuck_pos, deadlock_retreat_count
    global big_skip_step_progress, big_skip_advance_progress, big_skip_pause_start, big_skip_target_y
    global _loop_restart_pending, _loop_restart_time

    # Spin ROS 2 perception callbacks
    rclpy.spin_once(_ros_node, timeout_sec=0.0)

    if _loop_restart_pending:
        if time.time() >= _loop_restart_time:
            _loop_restart_pending = False
            perform_full_reset()
        else:
            return

    if state in ["STOPPED", "USER_PAUSED"]:
        return

    now = time.time()
    dt = min(now - last_time, 0.1)
    last_time = now

    min_dist_ahead = _ros_node.closest_obstacle_dist
    speed = get_adaptive_speed()

    # Global Proximity Emergency Guard (except in reverse/escape states)
    if state not in ["MOVING", "STOPPED", "USER_PAUSED", "EMERGENCY_BACKUP",
                     "EMERGENCY_ESCAPE_TURN", "EMERGENCY_ESCAPE_BACK", "EMERGENCY_ESCAPE_REALIGN"]:
        if check_emergency_proximity(min_dist_ahead):
            print(f"[EMERGENCY GUARD]: {min_dist_ahead:.2f}m in state '{state}'. Backing up.")
            initiate_emergency_backup()
            return

    # Update UI telemetry
    if sensor_label:
        sensor_label.text = f"Obstacle Dist: {min_dist_ahead:.2f} m" if min_dist_ahead != float('inf') else "Obstacle Dist: Clear (>10m)"
    if sectors_label:
        sectors_label.text = f"Sectors: L: {_ros_node.depth_left_dist:.2f}m | C: {_ros_node.depth_center_dist:.2f}m | R: {_ros_node.depth_right_dist:.2f}m"
    if ledger_label:
        ledger_label.text = f"Detour Ledger: {lateral_detour_ledger:.2f} m"
    if pose_label:
        pose_label.text = f"Pose: ({current_pos[0]:.2f}, {current_pos[1]:.2f}) @ {current_yaw:.1f}°"

    # Periodic Telemetry Log
    if (now - last_print_time) >= PRINT_INTERVAL:
        last_print_time = now
        print(f"[Telemetry] State: {state} | Phase: {avoidance_sub_phase} | Ledger: {lateral_detour_ledger:.2f}m | Dist: {min_dist_ahead:.2f}m | Sectors: L={_ros_node.depth_left_dist:.2f} C={_ros_node.depth_center_dist:.2f} R={_ros_node.depth_right_dist:.2f}")

    # ============================================================
    # FSM STATE HANDLING
    # ============================================================

    # --- 1. NORMAL SWEEP FORWARD ---
    if state == "MOVING":
        lane_base_yaw = current_yaw

        # Goal Completion Check
        if compute_dist_2d(current_pos, TARGET_DESTINATION) <= TARGET_RADIUS or is_target_x_reached():
            finish_run("Destination Reached")
            return

        # NavMesh Boundary Hit Check
        edge_hit = False
        margin_y = ROBOT_SAFETY_MARGIN
        eps = 1e-3

        if current_pos[1] + eps >= max_y - margin_y and sweep_direction_y > 0:
            current_pos[1] = max_y - margin_y
            edge_hit = True
            edge_stuck_time = now
        elif current_pos[1] - eps <= min_y + margin_y and sweep_direction_y < 0:
            current_pos[1] = min_y + margin_y
            edge_hit = True
            edge_stuck_time = now

        # Timeout stuck at road edge guard
        if (now - edge_stuck_time) > 3.0:
            if (sweep_direction_y > 0 and abs(current_pos[1] - (max_y - margin_y)) < 0.05) or \
               (sweep_direction_y < 0 and abs(current_pos[1] - (min_y + margin_y)) < 0.05):
                target_yaw = get_lane_shift_heading()
                state = "TURNING_1"
                edge_stuck_time = now
                add_breadcrumb(current_pos, color=(0.0, 0.0, 1.0))
                update_usd_transform(current_pos, current_yaw)
                return

        if edge_hit:
            target_yaw = get_lane_shift_heading()
            state = "TURNING_1"
            if status_label:
                status_label.text = "Status: Edge Hit -> Shifting Lane"
            add_breadcrumb(current_pos, color=(0.0, 0.0, 1.0))
            update_usd_transform(current_pos, current_yaw)
            return

        # Proximity Avoidance Trigger
        if min_dist_ahead < OBSTACLE_THRESHOLD:
            if min_dist_ahead < EMERGENCY_STOP_DIST:
                # Check cooldown to avoid rapid chattering
                dist_from_trigger = compute_dist_2d(current_pos, emergency_trigger_pos)
                time_since_trigger = now - emergency_cooldown_start
                if emergency_triggered and (dist_from_trigger < EMERGENCY_COOLDOWN_DIST or time_since_trigger < EMERGENCY_COOLDOWN_TIME):
                    # In cooldown, bypass instead
                    initiate_bypass_sequence()
                    return
                initiate_emergency_backup()
                return

            # Trigger deterministic proximity bypass
            initiate_bypass_sequence()
            return

        # Drive forward along current sweep line
        move_forward(speed=speed, dt=dt)

        # Step Pacing & Breadcrumb Drop
        dist_in_step = compute_dist_2d(current_pos, step_start_pos)
        if dist_in_step >= STEP_DISTANCE:
            state = "PAUSED"
            pause_start_time = now
            consecutive_blocked_shifts = 0
            if status_label:
                status_label.text = f"Status: Step Paused ({PAUSE_DURATION}s)"
            color = (0.0, 1.0, 0.0) if sweep_direction_y > 0 else (1.0, 0.0, 0.0)
            add_breadcrumb(current_pos, color=color)

        update_usd_transform(current_pos, current_yaw)

    # --- 2. STEP PAUSE ---
    elif state == "PAUSED":
        if (now - pause_start_time) >= PAUSE_DURATION:
            state = "MOVING"
            step_start_pos = list(current_pos)
            if status_label:
                status_label.text = "Status: MOVING"

    # --- 3. GEOMETRIC DIVERGENCE MANEUVERS ---
    elif state == "AVOID_INITIAL_TURN":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            micro_step_target = MIN_DIVERGE_STEP
            micro_step_progress = 0.0
            state = "AVOID_DIVERGE_STEP"

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
        step_dist = speed * dt
        move_forward_step(step_dist, dt)
        micro_step_progress += step_dist

        if check_emergency_proximity(min_dist_ahead):
            initiate_emergency_backup()
            return

        if is_pushing_into_edge(current_pos, detour_side):
            print("[Edge Blocked during Diverge]: Skipping lane instead of colliding with road boundary.")
            trigger_early_lane_shift()
            return

        if micro_step_progress >= micro_step_target:
            add_breadcrumb(current_pos, color=(1.0, 1.0, 0.0))  # Yellow plate for detour
            critical_side_depth = _ros_node.depth_left_dist if detour_side > 0 else _ros_node.depth_right_dist
            lateral_ok = abs(lateral_detour_ledger) >= (MIN_CHASSIS_CLEAR + 0.25)

            if critical_side_depth < 0.60 or not lateral_ok:
                diverge_turn_count += 1
                if diverge_turn_count > DIVERGE_ESCALATION_LIMIT:
                    enter_wide_obstacle_diverge(reason="Insufficient lateral clearance")
                    return
                max_angle = min(45.0, DIVERGE_ANGLE_INC + diverge_turn_count * 5.0)
                target_yaw = (lane_base_yaw + detour_side * max_angle) % 360.0
                micro_step_target = MIN_DIVERGE_STEP
                micro_step_progress = 0.0
                state = "AVOID_ROTATE_THEN_STEP"
            else:
                target_yaw = lane_base_yaw
                state = "AVOID_ALIGN_TO_FLANK"

        update_usd_transform(current_pos, current_yaw)

    # --- 4. FLANK PRE-CHECK & DRIVE ---
    elif state == "AVOID_ALIGN_TO_FLANK":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            state = "AVOID_FLANK_PRE_CHECK_SETTLE"
            pre_check_settle_start = now
            if status_label:
                status_label.text = "Status: Aligned to Flank. Checking corridor..."

        update_usd_transform(current_pos, current_yaw)

    elif state == "AVOID_FLANK_PRE_CHECK_SETTLE":
        if check_emergency_proximity(min_dist_ahead):
            initiate_emergency_backup()
            return

        if (now - pre_check_settle_start) >= 0.35:
            center_clear = _ros_node.depth_center_dist > 0.60
            side_clear = (_ros_node.depth_right_dist > 0.55) if detour_side > 0 else (_ros_node.depth_left_dist > 0.55)

            if center_clear and side_clear:
                avoidance_sub_phase = "FLANK_DRIVE"
                flank_progress = 0.0
                flank_step_progress = 0.0
                flank_step_start_pos = list(current_pos)
                required_flank_distance = max(
                    FLANK_MIN_DIST,
                    min(FLANK_MAX_DIST, initial_detected_distance + FLANK_SAFETY_PAD + DEFAULT_FLANK_EXTRA)
                )
                state = "FLANK_STEP"
                if status_label:
                    status_label.text = f"Status: Driving Flank ({required_flank_distance:.2f}m)..."
            else:
                enter_wide_obstacle_diverge(angle=WIDE_DIVERGE_ANGLE, step=WIDE_DIVERGE_STEP, reason="Flank corridor obstructed")

        update_usd_transform(current_pos, current_yaw)

    elif state == "FLANK_STEP":
        step_dist = speed * dt
        move_forward_step(step_dist, dt)
        flank_step_progress += step_dist
        flank_progress += step_dist

        if flank_step_progress >= STEP_DISTANCE:
            flank_step_progress = 0.0
            add_breadcrumb(current_pos, color=(0.0, 1.0, 1.0))  # Cyan for flank
            state = "FLANK_PAUSED"
            flank_pause_start_time = now

        if (detour_side > 0 and current_pos[1] >= SAFE_MAX_Y - 0.05) or \
           (detour_side < 0 and current_pos[1] <= SAFE_MIN_Y + 0.05):
            print("[Edge Blocked during Flank]: Skipping lane.")
            trigger_early_lane_shift()
            return

        update_usd_transform(current_pos, current_yaw)

    elif state == "FLANK_PAUSED":
        if (now - flank_pause_start_time) >= 0.30:
            if flank_progress >= required_flank_distance:
                avoidance_sub_phase = "FLANK_INSPECT"
                inspect_angle_side = -1.0 if detour_side > 0 else 1.0
                target_yaw = (lane_base_yaw + (inspect_angle_side * 35.0)) % 360.0
                state = "AVOID_ROTATE_TO_INSPECT_FLANK"
                if status_label:
                    status_label.text = "Status: Flank Complete - Inspecting"
            else:
                state = "FLANK_STEP"
                flank_step_start_pos = list(current_pos)
                flank_step_progress = 0.0

        update_usd_transform(current_pos, current_yaw)

    # --- 5. INWARD INSPECTION & RETURN TO LANE ---
    elif state == "AVOID_ROTATE_TO_INSPECT_FLANK":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            if min_dist_ahead < 0.75:
                # Obstacle still alongside, extend flank
                avoidance_sub_phase = "FLANK_DRIVE"
                target_yaw = lane_base_yaw
                flank_progress = 0.0
                required_flank_distance = 0.45
                state = "AVOID_ALIGN_TO_FLANK"
            else:
                # Obstacle cleared, look back toward lane
                avoidance_sub_phase = "VERIFY"
                return_steer_side = -1.0 if lateral_detour_ledger > 0.0 else 1.0
                target_yaw = (lane_base_yaw + (return_steer_side * 45.0)) % 360.0
                state = "LOOK_BACK_ROTATE"

        update_usd_transform(current_pos, current_yaw)

    elif state == "LOOK_BACK_ROTATE":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            avoidance_sub_phase = "RETURNING"
            state = "RETURN_DRIVE_TO_LANE"

        update_usd_transform(current_pos, current_yaw)

    elif state == "RETURN_DRIVE_TO_LANE":
        if is_target_x_reached():
            finish_run("Target X Reached during Return")
            return

        step_dist = speed * dt
        move_forward_step(step_dist, dt)

        if check_emergency_proximity(min_dist_ahead):
            initiate_emergency_backup()
            return

        # Ledger reaches zero -> Robot has fully returned to original lane track
        if abs(lateral_detour_ledger) <= 0.05 or (lateral_detour_ledger * detour_side <= 0.0):
            lateral_detour_ledger = 0.0
            target_yaw = lane_base_yaw
            state = "RETURN_FINAL_ALIGN"

        update_usd_transform(current_pos, current_yaw)

    elif state == "RETURN_FINAL_ALIGN":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            state = "MOVING"
            avoidance_sub_phase = "IDLE"
            step_start_pos = list(current_pos)
            if status_label:
                status_label.text = "Status: Returned to Lane -> Resuming Sweep"

        update_usd_transform(current_pos, current_yaw)

    # --- 6. EMERGENCY REVERSE & ESCAPE ---
    elif state == "EMERGENCY_BACKUP":
        step_dist = EMERGENCY_BACKUP_SPEED * dt
        backup_progress += step_dist

        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] -= fwd_x * step_dist
        current_pos[1] -= fwd_y * step_dist

        min_dist_ahead = _ros_node.closest_obstacle_dist

        # Check if backing up against an edge without clearing
        if (now - emergency_backup_start_time) >= EMERGENCY_BACKUP_STUCK_TIME and \
           emergency_escape_turns < EMERGENCY_MAX_ESCAPE_TURNS:
            esc_yaw = pick_escape_yaw()
            if esc_yaw is not None:
                print(f"[Emergency Backup]: Stuck near edge. Executing escape turn to {esc_yaw:.0f} deg.")
                emergency_escape_turns += 1
                emergency_original_yaw = current_yaw
                target_yaw = esc_yaw
                escape_progress = 0.0
                state = "EMERGENCY_ESCAPE_TURN"
                update_usd_transform(current_pos, current_yaw)
                return

        if min_dist_ahead >= EMERGENCY_BACKUP_CLEAR_DIST:
            print(f"[Emergency Backup]: Cleared ({backup_progress:.2f}m backed). Resuming sweep.")
            emergency_triggered = False
            backup_progress = 0.0
            if repeated_backup_count >= 2:
                repeated_backup_count = 0
                repeated_backup_pos = None
                trigger_early_lane_shift()
                return
            state = "MOVING"
            step_start_pos = list(current_pos)

        elif backup_progress >= EMERGENCY_BACKUP_MAX_DIST:
            print("[Emergency Backup]: Max backup reached without clearing. Retreating.")
            emergency_triggered = False
            backup_progress = 0.0
            deadlock_retreat_count += 1
            if deadlock_retreat_count >= 3:
                deadlock_retreat_count = 0
                avoidance_sub_phase = "IDLE"
                lane_base_yaw = 0.0 if get_progress_dir_x() > 0 else 180.0
                target_yaw = lane_base_yaw
                state = "DEADLOCK_RESET_TURN"
                return
            else:
                state = "FORCE_RETREAT_BACK"

        update_usd_transform(current_pos, current_yaw)

    elif state == "EMERGENCY_ESCAPE_TURN":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        current_yaw += math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            escape_progress = 0.0
            state = "EMERGENCY_ESCAPE_BACK"
        update_usd_transform(current_pos, current_yaw)

    elif state == "EMERGENCY_ESCAPE_BACK":
        step_dist = EMERGENCY_BACKUP_SPEED * dt
        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] -= fwd_x * step_dist
        current_pos[1] -= fwd_y * step_dist
        escape_progress += step_dist

        if escape_progress >= EMERGENCY_ESCAPE_DIST:
            target_yaw = emergency_original_yaw
            state = "EMERGENCY_ESCAPE_REALIGN"

        update_usd_transform(current_pos, current_yaw)

    elif state == "EMERGENCY_ESCAPE_REALIGN":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        current_yaw += math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            emergency_triggered = False
            backup_progress = 0.0
            avoidance_sub_phase = "IDLE"
            state = "MOVING"
            step_start_pos = list(current_pos)

        update_usd_transform(current_pos, current_yaw)

    elif state == "FORCE_RETREAT_BACK":
        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] -= fwd_x * FORWARD_SPEED * dt
        current_pos[1] -= fwd_y * FORWARD_SPEED * dt
        backup_progress += FORWARD_SPEED * dt

        if backup_progress >= force_retreat_target:
            emergency_triggered = False
            backup_progress = 0.0
            avoidance_sub_phase = "IDLE"
            target_yaw = 0.0 if get_progress_dir_x() > 0 else 180.0
            current_yaw = target_yaw
            state = "MOVING"
            step_start_pos = list(current_pos)
            update_usd_transform(current_pos, current_yaw)
            return

        update_usd_transform(current_pos, current_yaw)

    # --- 7. STANDARD SWEEP LANE SHIFTING ---
    elif state == "TURNING_1":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            state = "LANE_SHIFTING"
            shifted_distance = 0.0

        update_usd_transform(current_pos, current_yaw)

    elif state == "LANE_SHIFTING":
        if is_target_x_reached():
            finish_run("Target X Reached during Lane Shift")
            return

        if _ros_node.closest_obstacle_dist < 0.60:
            consecutive_blocked_shifts += 1
            lane_stuck_pos = list(current_pos)
            print(f"[Lane Shift Blocked]: Blocked ({consecutive_blocked_shifts}/{LANE_STUCK_THRESHOLD}).")

            if consecutive_blocked_shifts >= LANE_STUCK_THRESHOLD:
                consecutive_blocked_shifts = 0
                trigger_big_lane_skip()
                return

            state = "TURNING_2"
            sweep_direction_y = -sweep_direction_y
            target_yaw = 90.0 if sweep_direction_y > 0 else 270.0
            shifted_distance = 0.0
            step_start_pos = list(current_pos)
            return

        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        dist = speed * dt
        current_pos[0] += fwd_x * dist
        current_pos[1] += fwd_y * dist
        shifted_distance += dist

        if shifted_distance >= LANE_SHIFT:
            state = "TURNING_2"
            sweep_direction_y = -sweep_direction_y
            target_yaw = 90.0 if sweep_direction_y > 0 else 270.0
            consecutive_blocked_shifts = 0

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
            edge_stuck_time = now
            add_breadcrumb(current_pos, color=(0.0, 0.0, 1.0))

        update_usd_transform(current_pos, current_yaw)

    # --- 8. BIG LANE SKIP (CROSS-ROAD TRAVERSAL) ---
    elif state == "BIG_SKIP_TURN":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            state = "BIG_SKIP_DRIVE"

        update_usd_transform(current_pos, current_yaw)

    elif state == "BIG_SKIP_DRIVE":
        step_dist = speed * dt
        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] += fwd_x * step_dist
        current_pos[1] += fwd_y * step_dist
        big_skip_step_progress += step_dist

        if check_emergency_proximity(min_dist_ahead):
            initiate_emergency_backup()
            return

        reached = (
            (target_yaw == 90.0 and current_pos[1] >= big_skip_target_y - 0.05) or
            (target_yaw == 270.0 and current_pos[1] <= big_skip_target_y + 0.05)
        )

        if reached:
            current_pos[1] = big_skip_target_y
            big_skip_step_progress = 0.0
            big_skip_advance_progress = 0.0
            add_breadcrumb(current_pos, color=(0.6, 0.0, 1.0))  # Purple plate
            target_yaw = get_lane_shift_heading()
            state = "BIG_SKIP_ADVANCE_TURN"
        elif big_skip_step_progress >= STEP_DISTANCE:
            big_skip_step_progress = 0.0
            add_breadcrumb(current_pos, color=(0.4, 0.0, 0.8))
            state = "BIG_SKIP_PAUSE"
            big_skip_pause_start = now

        update_usd_transform(current_pos, current_yaw)

    elif state == "BIG_SKIP_PAUSE":
        if (now - big_skip_pause_start) >= PAUSE_DURATION:
            state = "BIG_SKIP_DRIVE"
        update_usd_transform(current_pos, current_yaw)

    elif state == "BIG_SKIP_ADVANCE_TURN":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            big_skip_step_progress = 0.0
            state = "BIG_SKIP_ADVANCE_DRIVE"

        update_usd_transform(current_pos, current_yaw)

    elif state == "BIG_SKIP_ADVANCE_DRIVE":
        step_dist = speed * dt
        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] += fwd_x * step_dist
        current_pos[1] += fwd_y * step_dist
        big_skip_step_progress += step_dist
        big_skip_advance_progress += step_dist

        if check_emergency_proximity(min_dist_ahead):
            initiate_emergency_backup()
            return

        if big_skip_advance_progress >= LANE_SHIFT:
            add_breadcrumb(current_pos, color=(0.6, 0.0, 1.0))
            target_yaw = 90.0 if sweep_direction_y > 0 else 270.0
            state = "BIG_SKIP_FINAL_TURN"
        elif big_skip_step_progress >= STEP_DISTANCE:
            big_skip_step_progress = 0.0
            add_breadcrumb(current_pos, color=(0.6, 0.0, 1.0))
            state = "BIG_SKIP_ADVANCE_PAUSE"
            big_skip_pause_start = now

        update_usd_transform(current_pos, current_yaw)

    elif state == "BIG_SKIP_ADVANCE_PAUSE":
        if (now - big_skip_pause_start) >= PAUSE_DURATION:
            state = "BIG_SKIP_ADVANCE_DRIVE"
        update_usd_transform(current_pos, current_yaw)

    elif state == "BIG_SKIP_FINAL_TURN":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            lane_base_yaw = current_yaw
            state = "MOVING"
            step_start_pos = list(current_pos)
            if status_label:
                status_label.text = "Status: Big Skip Complete - Resuming Sweep"

        update_usd_transform(current_pos, current_yaw)

    # --- 9. DEADLOCK RESET ESCAPES ---
    elif state == "DEADLOCK_RESET_TURN":
        yaw_diff = get_shortest_angle_diff(lane_base_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = lane_base_yaw
            state = "DEADLOCK_RESET_DRIVE"
            deadlock_retreat_count = 0

        update_usd_transform(current_pos, current_yaw)

    elif state == "DEADLOCK_RESET_DRIVE":
        step_dist = speed * dt
        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] += fwd_x * step_dist
        current_pos[1] += fwd_y * step_dist
        backup_progress += step_dist

        if backup_progress >= 0.50:
            backup_progress = 0.0
            trigger_early_lane_shift()
            return

        update_usd_transform(current_pos, current_yaw)

# ============================================================
# 10. REGISTER UPDATE SUBSCRIPTION
# ============================================================
global _phantomx_sub
_phantomx_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(autonomous_update)
print("[PhantomX Autonomous Proximity Controller]: Loaded and active.")
