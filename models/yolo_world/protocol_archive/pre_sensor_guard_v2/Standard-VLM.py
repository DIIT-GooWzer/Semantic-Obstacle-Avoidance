"""PhantomX VLM-only obstacle decisions with kinematic navigation.

Only VLM supplies obstacle categories. Depth guards and maneuvers are shared;
there is no other classifier, model fallback, stress override, or category cache.
Inference failure is logged as abstention and stops the run.
Run in Isaac Sim Script Editor with RGB/depth ROS 2 topics available.
"""

import os
import time
import math
import json
import threading
import queue
import uuid
import numpy as np
import cv2
import omni.usd
import omni.timeline
import omni.kit.app
import omni.ui as ui
from pxr import Usd, UsdGeom, Gf, Sdf
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

try:
    from cv_bridge import CvBridge

    _has_cv2 = True
except ImportError:
    _has_cv2 = False
import base64
import re
import urllib.request

_previous_stop = globals().get("_worker_stop")
if _previous_stop is not None:
    _previous_stop.set()
_previous_thread = globals().get("_inference_thread")
if _previous_thread is not None:
    _previous_thread.join(timeout=1.0)
# Migration cleanup only: stop any worker left by the old hybrid script.
globals()["yolo_thread_running"] = False
for key, method in [("_phantomx_sub", "unsubscribe"), ("_ros_node", "destroy_node")]:
    previous = globals().get(key)
    if previous:
        try:
            getattr(previous, method)()
        except Exception:
            pass
previous_window = globals().get("_phantomx_window")
if previous_window:
    previous_window.visible = False
# Model-specific configuration
PIPELINE = "VLM"
ROBOT_PATH = "/World/phantomX_base"
SPAWN_POSITION = (0.11952272590602997, 1.8697408479533721, 0.1)
SPAWN_ROTATION = (0.0, 0.0, -90.0)
ROBOT_LENGTH = 0.55
DIVERGE_CLEARANCE_MARGINE = 0.2
MIN_DIVERGE_STEP = ROBOT_LENGTH + DIVERGE_CLEARANCE_MARGINE
TARGET_DESTINATION = (9, -1.8266811603307502)
TARGET_RADIUS = 0.3
STEP_DISTANCE = 0.4
PAUSE_DURATION = 0.5
FORWARD_SPEED = 0.5
TURN_SPEED = 180.0
LANE_SHIFT = 0.5
OBSTACLE_THRESHOLD = 0.8
MIN_CHASSIS_CLEAR = 0.4
PRINT_INTERVAL = 2.0
YIELD_WAIT_DURATION = 3.0
MAX_YIELD_WAIT = 10.0
AVOIDANCE_STEP_SIZE = MIN_DIVERGE_STEP
DIVERGE_ANGLE_INC = 31.0
DIVERGE_ESCALATION_LIMIT = 3
LANE_STUCK_THRESHOLD = 2
DIVERGE_GRACE_PERIOD = 0.6
CRITICAL_TOUCH_DIST = 0.18
EMERGENCY_STOP_DIST = 0.25
EMERGENCY_BACKUP_DIST = 0.3
EMERGENCY_BACKUP_SPEED = 0.5
EMERGENCY_BACKUP_CLEAR_DIST = MIN_DIVERGE_STEP
EMERGENCY_BACKUP_MAX_DIST = 2.5
EMERGENCY_BACKUP_STUCK_TIME = 5.0
EMERGENCY_ESCAPE_DIST = 0.6
EMERGENCY_MAX_ESCAPE_TURNS = 1
EMERGENCY_ESCAPE_MIN_ROOM = 0.3
EMERGENCY_COOLDOWN_DIST = 1.3
EMERGENCY_COOLDOWN_TIME = 5.0
REPEATED_BACKUP_RADIUS = 0.5
REPEATED_BACKUP_LIMIT = 2
VLM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
VLM_API_KEY = os.getenv("LM_STUDIO_API_KEY", "")
VLM_MODEL = "qwen3.5-9b"
VLM_TIMEOUT = 360
VLM_MAX_TOKENS = 3000
BREADCRUMB_SIZE = 0.15
BREADCRUMB_THICKNESS = 0.01
RGB_TOPICS = ["/rgb", "/camera/rgb", "/phantomx/camera/rgb"]
DEPTH_TOPICS = ["/depth", "/camera/depth", "/phantomx/camera/depth"]
AUTO_LOOP_ENABLED = False
AUTO_LOOP_DELAY = 2.0
ROBOT_SAFETY_MARGIN = 0.35
# Isaac scene and ROS camera input
stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()
if not timeline.is_playing():
    timeline.play()


def navigation_time():
    """Simulation seconds for motion/policy; inference and sensors use monotonic time."""
    return float(timeline.get_current_time())


def spawn_phantomx(
    stage, prim_path=ROBOT_PATH, position=SPAWN_POSITION, rotation_euler=SPAWN_ROTATION
):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(
            "/home/user/Desktop/llmforhexapod/usd_to_stl_conversion/phantomX_base.usd"
        )
    xform = UsdGeom.Xformable(prim)
    translate_op, rotate_op = (None, None)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() in (
            UsdGeom.XformOp.TypeRotateXYZ,
            UsdGeom.XformOp.TypeRotateZ,
        ):
            rotate_op = op
    if not translate_op:
        translate_op = xform.AddTranslateOp()
    if not rotate_op:
        rotate_op = xform.AddRotateXYZOp()
    translate_op.Set(Gf.Vec3d(*position))
    rotate_op.Set(Gf.Vec3d(*rotation_euler))
    return (prim, translate_op, rotate_op)


robot_prim, translate_op, rotate_op = spawn_phantomx(stage)
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
# Same ordered resolution in both pipelines; missing bounds is a setup error.
NAVMESH_CANDIDATE_PATHS = (
    "/World/Terrain/section_00_asphalt/NavMeshVolume",
    "/World/NavMeshVolume",
)


def resolve_navmesh(scene):
    for path in NAVMESH_CANDIDATE_PATHS:
        prim = scene.GetPrimAtPath(path)
        if prim.IsValid():
            return path, prim
    raise RuntimeError(
        "No NavMeshVolume found. Checked: " + ", ".join(NAVMESH_CANDIDATE_PATHS)
    )


NAVMESH_PATH, navmesh_prim = resolve_navmesh(stage)
bound = bbox_cache.ComputeWorldBound(navmesh_prim).ComputeAlignedBox()
min_x, max_x = float(bound.GetMin()[0]), float(bound.GetMax()[0])
min_y, max_y = float(bound.GetMin()[1]), float(bound.GetMax()[1])
if not all(math.isfinite(value) for value in (min_x, max_x, min_y, max_y)) or (
    max_x - min_x <= 2 * ROBOT_SAFETY_MARGIN or max_y - min_y <= 2 * ROBOT_SAFETY_MARGIN
):
    raise RuntimeError(f"Invalid NavMesh bounds for {NAVMESH_PATH}")
print(
    f"[{PIPELINE}] NavMesh: {NAVMESH_PATH}; X=({min_x}, {max_x}), Y=({min_y}, {max_y})"
)

SAFE_MIN_Y, SAFE_MAX_Y = (min_y + ROBOT_SAFETY_MARGIN, max_y - ROBOT_SAFETY_MARGIN)
SAFE_MIN_X, SAFE_MAX_X = (min_x + ROBOT_SAFETY_MARGIN, max_x - ROBOT_SAFETY_MARGIN)
if not rclpy.ok():
    rclpy.init()


def ros_img_to_numpy(msg, desired_encoding="passthrough"):
    if _has_cv2:
        try:
            bridge = CvBridge()
            return bridge.imgmsg_to_cv2(msg, desired_encoding=desired_encoding)
        except Exception:
            pass
    if msg.encoding in ["32FC1", "32f"]:
        return np.frombuffer(msg.data, dtype=np.float32).reshape(
            (msg.height, msg.width)
        )
    elif msg.encoding in ["rgb8", "bgr8", "8UC3"]:
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, 3)
        )
        if desired_encoding == "bgr8" and msg.encoding == "rgb8":
            arr = arr[:, :, ::-1]
        return arr
    elif msg.encoding in ["mono8", "8UC1"]:
        return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
    else:
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, -1)
        )


class PerceptionNode(Node):

    def __init__(self):
        super().__init__("phantomx_isaac_perception")
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._lock = threading.Lock()
        self.rgb_sequence = 0
        self.rgb_received_at = self.depth_received_at = 0.0
        self._rgb_stamp = None
        self._depth_stamp = None
        self.latest_rgb = None
        self.latest_depth = None
        self.closest_obstacle_dist = float("inf")
        self.depth_left_dist = float("inf")
        self.depth_center_dist = float("inf")
        self.depth_right_dist = float("inf")
        self.subs = []
        for topic in RGB_TOPICS:
            self.subs.append(
                self.create_subscription(
                    Image, topic, self._rgb_callback, qos_best_effort
                )
            )
        for topic in DEPTH_TOPICS:
            self.subs.append(
                self.create_subscription(
                    Image, topic, self._depth_callback, qos_best_effort
                )
            )

    def _rgb_callback(self, msg):
        try:
            stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            if stamp != (0, 0) and stamp == self._rgb_stamp:
                return
            image = ros_img_to_numpy(msg, desired_encoding="bgr8")
            with self._lock:
                self.latest_rgb = image
                self.rgb_sequence += 1
                self.rgb_received_at = time.monotonic()
                self._rgb_stamp = stamp
        except Exception:
            pass

    def _depth_callback(self, msg):
        try:
            depth_frame = ros_img_to_numpy(msg, desired_encoding="32FC1")
            self.latest_depth = depth_frame
            h, w = depth_frame.shape
            row_top, row_bottom = (int(h * 0.35), int(h * 0.65))
            left_col = depth_frame[row_top:row_bottom, : int(w * 0.33)]
            center_col = depth_frame[row_top:row_bottom, int(w * 0.33) : int(w * 0.66)]
            right_col = depth_frame[row_top:row_bottom, int(w * 0.66) :]

            def extract_min(roi):
                valid = roi[np.isfinite(roi) & (roi > 0.05)]
                return float(np.min(valid)) if len(valid) > 0 else float("inf")

            with self._lock:
                self.depth_left_dist = extract_min(left_col)
                self.depth_center_dist = extract_min(center_col)
                self.depth_right_dist = extract_min(right_col)
                self.closest_obstacle_dist = min(
                    self.depth_left_dist, self.depth_center_dist, self.depth_right_dist
                )
                self.depth_received_at = time.monotonic()
                self._depth_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        except Exception:
            pass

    def snapshot(self):
        with self._lock:
            return (
                self.latest_rgb,
                self.rgb_sequence,
                self.rgb_received_at,
                (self.depth_left_dist, self.depth_center_dist, self.depth_right_dist),
                self.depth_received_at,
                self.closest_obstacle_dist,
            )

    def clear_frames(self):
        with self._lock:
            self.latest_rgb = self.latest_depth = None
            self.rgb_received_at = self.depth_received_at = 0.0
            self._rgb_stamp = self._depth_stamp = None
            self.depth_left_dist = self.depth_center_dist = self.depth_right_dist = (
                float("inf")
            )
            self.closest_obstacle_dist = float("inf")


_ros_node = PerceptionNode()
state = "MOVING"
saved_pre_pause_state = "MOVING"
current_pos = list(SPAWN_POSITION)
current_yaw = SPAWN_ROTATION[2]
sweep_direction_y = 1.0 if SPAWN_ROTATION[2] == 90.0 else -1.0
step_start_pos = list(current_pos)
target_yaw = current_yaw
pause_start_time = 0.0
last_time = navigation_time()
shifted_distance = 0.0
last_print_time = 0.0
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
active_obstacle_category = "UNKNOWN"
yield_wait_start_time = 0.0
yield_wait_total_start_time = 0.0
pre_check_settle_start = 0.0
push_progress = 0.0
push_target = 0.0
flank_step_start_pos = list(current_pos)
flank_step_progress = 0.0
flank_pause_start_time = 0.0
emergency_triggered = False
emergency_cooldown_start = 0.0
emergency_trigger_pos = [0.0, 0.0]
backup_progress = 0.0
deadlock_retreat_count = 0
force_retreat_target = 1.5
wide_diverge_count = 0
edge_stuck_time = 0.0
emergency_pin_time = 0.0
escape_target_dist = EMERGENCY_ESCAPE_DIST
emergency_backup_start_time = 0.0
emergency_escape_turns = 0
emergency_original_yaw = 0.0
escape_progress = 0.0
diverge_grace_until = 0.0
repeated_backup_pos = None
repeated_backup_count = 0
lane_stuck_count = 0
lane_stuck_pos = None
big_skip_target_y = 0.0
big_skip_step_progress = 0.0
big_skip_advance_progress = 0.0
big_skip_pause_start = 0.0
consecutive_blocked_shifts = 0
current_encounter_id = 0
_loop_restart_pending = False
_loop_restart_time = 0.0
breadcrumb_counter = 0


# Shared kinematic navigation helpers
def get_progress_dir_x():
    return 1.0 if TARGET_DESTINATION[0] >= SPAWN_POSITION[0] else -1.0


def get_lane_shift_heading():
    return 0.0 if get_progress_dir_x() > 0 else 180.0


def get_forward_vector(yaw_deg):
    rad = math.radians(yaw_deg)
    return (math.cos(rad), math.sin(rad), 0.0)


def get_shortest_angle_diff(target, current):
    diff = (target - current) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


def is_pushing_into_edge(pos, side, margin=0.1):
    if side > 0 and pos[1] >= SAFE_MAX_Y - margin:
        return True
    if side < 0 and pos[1] <= SAFE_MIN_Y + margin:
        return True
    return False


def room_along(pos, yaw_deg, travel_dir=-1.0):
    fx, fy, _ = get_forward_vector(yaw_deg)
    vx, vy = (fx * travel_dir, fy * travel_dir)
    room = float("inf")
    if vx > 1e-06:
        room = min(room, (SAFE_MAX_X - pos[0]) / vx)
    elif vx < -1e-06:
        room = min(room, (SAFE_MIN_X - pos[0]) / vx)
    if vy > 1e-06:
        room = min(room, (SAFE_MAX_Y - pos[1]) / vy)
    elif vy < -1e-06:
        room = min(room, (SAFE_MIN_Y - pos[1]) / vy)
    return max(0.0, room)


def pick_escape_yaw():
    best_yaw, best_room = (None, 0.0)
    for offset in (90.0, -90.0, 45.0, -45.0):
        cand = (current_yaw + offset) % 360.0
        room = min(room_along(current_pos, cand), EMERGENCY_ESCAPE_DIST * 1.5)
        if room > best_room + 1e-06:
            best_room, best_yaw = (room, cand)
    if best_room < EMERGENCY_ESCAPE_MIN_ROOM:
        return None
    return best_yaw


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
    translate_op = None
    scale_op = None
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


def compute_dist_2d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_target_angle_diff(current_pose, current_heading_deg, target_pose):
    dx = target_pose[0] - current_pose[0]
    dy = target_pose[1] - current_pose[1]
    target_bearing_deg = math.degrees(math.atan2(dy, dx)) % 360.0
    diff = (target_bearing_deg - current_heading_deg + 180.0) % 360.0 - 180.0
    return diff


def update_usd_transform(pos, yaw_deg):
    if translate_op is None or rotate_op is None:
        return
    pos[0] = max(SAFE_MIN_X, min(SAFE_MAX_X, pos[0]))
    pos[1] = max(SAFE_MIN_Y, min(SAFE_MAX_Y, pos[1]))
    translate_op.Set(Gf.Vec3d(*pos))
    rotate_op.Set(Gf.Vec3d(0.0, 0.0, yaw_deg))


def record_motion_to_ledger(dx, dy):
    global lateral_detour_ledger
    rad = math.radians(lane_base_yaw)
    cos_b, sin_b = (math.cos(rad), math.sin(rad))
    d_lateral = -dx * sin_b + dy * cos_b
    lateral_detour_ledger += d_lateral


def move_forward_step(distance, dt):
    fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
    dx = fwd_x * distance
    dy = fwd_y * distance
    current_pos[0] += dx
    current_pos[1] += dy
    record_motion_to_ledger(dx, dy)


def move_forward(speed=FORWARD_SPEED, dt=0.0):
    fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
    current_pos[0] += fwd_x * speed * dt
    current_pos[1] += fwd_y * speed * dt


def initiate_emergency_backup():
    global state, emergency_triggered, emergency_cooldown_start, emergency_trigger_pos
    global backup_progress, avoidance_sub_phase, lateral_detour_ledger
    global repeated_backup_pos, repeated_backup_count
    global emergency_backup_start_time, emergency_escape_turns, emergency_original_yaw
    emergency_backup_start_time = navigation_time()
    emergency_escape_turns = 0
    emergency_original_yaw = current_yaw
    global emergency_pin_time
    emergency_pin_time = 0.0
    if (
        repeated_backup_pos is not None
        and compute_dist_2d(current_pos, repeated_backup_pos) < REPEATED_BACKUP_RADIUS
    ):
        repeated_backup_count += 1
    else:
        repeated_backup_count = 1
        repeated_backup_pos = list(current_pos)
    print(
        f"[Emergency Backup]: Obstacle too close! Backing up {EMERGENCY_BACKUP_DIST}m..."
    )
    emergency_trigger_pos = list(current_pos)
    emergency_triggered = True
    emergency_cooldown_start = navigation_time()
    backup_progress = 0.0
    avoidance_sub_phase = "IDLE"
    lateral_detour_ledger = 0.0
    state = "EMERGENCY_BACKUP"
    if status_label:
        status_label.text = f"Status: EMERGENCY BACKUP ({EMERGENCY_BACKUP_DIST:.1f}m)"


def check_emergency_proximity(dist_ahead):
    if state in [
        "STOPPED",
        "USER_PAUSED",
        "EMERGENCY_BACKUP",
        "CLASSIFYING",
        "EMERGENCY_ESCAPE_TURN",
        "EMERGENCY_ESCAPE_BACK",
        "EMERGENCY_ESCAPE_REALIGN",
    ]:
        return False
    if navigation_time() < diverge_grace_until:
        return dist_ahead < CRITICAL_TOUCH_DIST
    if dist_ahead >= EMERGENCY_STOP_DIST:
        return False
    distance_from_trigger = compute_dist_2d(current_pos, emergency_trigger_pos)
    time_since_trigger = navigation_time() - emergency_cooldown_start
    if emergency_triggered and (
        distance_from_trigger < EMERGENCY_COOLDOWN_DIST
        or time_since_trigger < EMERGENCY_COOLDOWN_TIME
    ):
        return False
    return True


def stop_blocked_lane():
    """End an unsuccessful maneuver instead of abandoning this lane for another."""
    update_usd_transform(current_pos, current_yaw)
    finish_run("Lane blocked")


def lane_end_reached():
    """The current lane ends at the safe Y boundary in the sweep direction."""
    if sweep_direction_y > 0:
        return current_pos[1] >= SAFE_MAX_Y - 0.001
    return current_pos[1] <= SAFE_MIN_Y + 0.001


def enter_wide_obstacle_diverge(angle=65.0, step=0.75, reason=""):
    if step is None:
        step = MIN_DIVERGE_STEP * 1.3
    global avoidance_sub_phase, target_yaw, micro_step_target, micro_step_progress
    global state, wide_diverge_count
    global diverge_grace_until
    if wide_diverge_count >= 1:
        print(
            f"[Wide Diverge Limit]: Already attempted wide diverge once for this obstacle ({reason}). Giving up — stopping in this lane."
        )
        stop_blocked_lane()
        return
    wide_diverge_count += 1
    print(f"[Wide Diverge]: Escalating (attempt {wide_diverge_count}). {reason}")
    avoidance_sub_phase = "WIDE_OBSTACLE_DIVERGE"
    target_yaw = (lane_base_yaw + detour_side * angle) % 360.0
    micro_step_target = step
    micro_step_progress = 0.0
    diverge_grace_until = navigation_time() + DIVERGE_GRACE_PERIOD
    state = "AVOID_ROTATE_THEN_STEP"


def initiate_bypass_sequence():
    global state, avoidance_sub_phase, target_yaw, detour_side, diverge_turn_count
    global wide_diverge_count, lane_base_yaw
    diverge_turn_count = 1
    wide_diverge_count = 0
    lane_base_yaw = current_yaw
    dist_to_top = SAFE_MAX_Y - current_pos[1]
    dist_to_bot = current_pos[1] - SAFE_MIN_Y
    if (
        sweep_direction_y > 0
        and dist_to_top < 0.6
        or (sweep_direction_y < 0 and dist_to_bot < 0.6)
    ):
        stop_blocked_lane()
        return
    left_d = _ros_node.depth_left_dist
    right_d = _ros_node.depth_right_dist
    target_angle_diff = get_target_angle_diff(
        current_pos, current_yaw, TARGET_DESTINATION
    )
    preferred_side = 1.0 if target_angle_diff > 0 else -1.0
    preferred_depth = right_d if preferred_side < 0 else left_d
    opposite_depth = left_d if preferred_side < 0 else right_d
    MIN_VIABLE_CLEAR = MIN_CHASSIS_CLEAR + 0.2
    if preferred_depth < MIN_VIABLE_CLEAR and opposite_depth > preferred_depth + 0.3:
        detour_side = -preferred_side
        print(
            f"[Target-First Diverge]: Target side blocked ({preferred_depth:.2f}m < {MIN_VIABLE_CLEAR:.2f}m). Rerouting to open side: {('LEFT' if detour_side > 0 else 'RIGHT')}"
        )
    else:
        detour_side = preferred_side
        print(
            f"[Target-First Diverge]: Turning toward target -> {('LEFT' if detour_side > 0 else 'RIGHT')} (angle_diff={target_angle_diff:.1f}°, depth={preferred_depth:.2f}m)"
        )
    test_yaw = (lane_base_yaw + detour_side * DIVERGE_ANGLE_INC) % 360.0
    fwd_x, fwd_y, _ = get_forward_vector(test_yaw)
    pred_x = current_pos[0] + fwd_x * 0.6
    pred_y = current_pos[1] + fwd_y * 0.6
    if (
        pred_x < SAFE_MIN_X
        or pred_x > SAFE_MAX_X
        or pred_y < SAFE_MIN_Y
        or (pred_y > SAFE_MAX_Y)
    ):
        detour_side = -detour_side
        print(
            f"[NavMesh Safety]: Boundary blocked on chosen side. Re-routing detour to: {('LEFT' if detour_side > 0 else 'RIGHT')}"
        )
    avoidance_sub_phase = "DIVERGE_SCAN"
    target_yaw = (lane_base_yaw + detour_side * DIVERGE_ANGLE_INC) % 360.0
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


def current_forward_speed():
    return FORWARD_SPEED


def maneuver_speed_factor():
    return 0.6


def reset_navigation_state():
    global _loop_restart_pending, _loop_restart_time, active_obstacle_category, avoidance_sub_phase
    global backup_progress, big_skip_advance_progress, big_skip_pause_start, big_skip_step_progress
    global big_skip_target_y, breadcrumb_counter, consecutive_blocked_shifts
    global current_pos, current_yaw, deadlock_retreat_count, detour_side
    global diverge_grace_until, diverge_turn_count, edge_stuck_time, emergency_backup_start_time
    global emergency_cooldown_start, emergency_escape_turns, emergency_original_yaw, emergency_pin_time
    global emergency_trigger_pos, emergency_triggered, escape_progress, escape_target_dist
    global flank_pause_start_time, flank_progress, flank_step_progress, flank_step_start_pos
    global force_retreat_target, initial_detected_distance, lane_base_yaw, lane_stuck_count
    global lane_stuck_pos, last_print_time, last_time, lateral_detour_ledger
    global micro_step_progress, micro_step_target, pause_start_time, pre_check_settle_start
    global push_progress, push_target, repeated_backup_count, repeated_backup_pos
    global required_flank_distance, saved_pre_pause_state, shifted_distance, state
    global step_start_pos, sweep_direction_y, target_yaw, wide_diverge_count
    global yield_wait_start_time, yield_wait_total_start_time
    state = "MOVING"
    saved_pre_pause_state = "MOVING"
    current_pos = list(SPAWN_POSITION)
    current_yaw = SPAWN_ROTATION[2]
    sweep_direction_y = 1.0 if SPAWN_ROTATION[2] == 90.0 else -1.0
    step_start_pos = list(current_pos)
    target_yaw = current_yaw
    pause_start_time = 0.0
    last_time = navigation_time()
    shifted_distance = 0.0
    last_print_time = 0.0
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
    active_obstacle_category = "UNKNOWN"
    yield_wait_start_time = 0.0
    yield_wait_total_start_time = 0.0
    pre_check_settle_start = 0.0
    push_progress = 0.0
    push_target = 0.0
    flank_step_start_pos = list(current_pos)
    flank_step_progress = 0.0
    flank_pause_start_time = 0.0
    emergency_triggered = False
    emergency_cooldown_start = 0.0
    emergency_trigger_pos = [0.0, 0.0]
    backup_progress = 0.0
    deadlock_retreat_count = 0
    force_retreat_target = 1.5
    wide_diverge_count = 0
    edge_stuck_time = 0.0
    emergency_pin_time = 0.0
    escape_target_dist = EMERGENCY_ESCAPE_DIST
    emergency_backup_start_time = 0.0
    emergency_escape_turns = 0
    emergency_original_yaw = 0.0
    escape_progress = 0.0
    diverge_grace_until = 0.0
    repeated_backup_pos = None
    repeated_backup_count = 0
    lane_stuck_count = 0
    lane_stuck_pos = None
    big_skip_target_y = 0.0
    big_skip_step_progress = 0.0
    big_skip_advance_progress = 0.0
    big_skip_pause_start = 0.0
    consecutive_blocked_shifts = 0
    _loop_restart_pending = False
    _loop_restart_time = 0.0
    breadcrumb_counter = 0


# Shared category policy, request lifecycle, and decision logging
COMMAND_BY_CATEGORY = {
    "BIOLOGICAL": "yield_and_wait",
    "STATIC_BAR": "standard_bypass",
    "MINOR_SURFACE": "scan_and_proceed",
    "CLEAR": "resume_path",
}
SENSOR_MAX_AGE = 1.0
CLASSIFICATION_WAIT_SECONDS = 5.0
RESULT_LOG_PATH = os.path.join(
    os.path.expanduser("~"), "hexapod_results", PIPELINE + "_decisions.jsonl"
)
_result_queue = queue.Queue()
_model_token = object()
_request_serial = 0
_active_request = None
classify_started_at = 0.0


LOG_SCHEMA_VERSION = 1
_session_id = uuid.uuid4().hex
_trial_id = uuid.uuid4().hex
_event_sequence = 0
_trial_finished = False
_trial_context = {
    "scenario_id": None,
    "scenario_config_id": None,
    "scenario_instance_id": None,
    "expected_initial_command": None,
}
_classification_observation = None
_classification_rgb = None


def json_safe(value):
    """Write strict JSON: unavailable/nonfinite measurements are null, never Infinity."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported log value: {type(value).__name__}")


def capture_observation(sample=None):
    """Snapshot observation/pose on the simulation thread, independent of later replies."""
    if sample is None:
        sample = _ros_node.snapshot()
    _, sequence, rgb_at, sectors, depth_at, distance = sample
    now = time.monotonic()
    return {
        "robot_pos": list(current_pos),
        "robot_yaw": float(current_yaw),
        "obstacle_distance_m": distance,
        "sector_depths_m": dict(zip(("left", "center", "right"), sectors)),
        "frame_sequence": int(sequence),
        "rgb_stamp_s": (
            None
            if _ros_node._rgb_stamp is None
            else _ros_node._rgb_stamp[0] + _ros_node._rgb_stamp[1] * 1e-9
        ),
        "depth_stamp_s": (
            None
            if _ros_node._depth_stamp is None
            else _ros_node._depth_stamp[0] + _ros_node._depth_stamp[1] * 1e-9
        ),
        "frame_age_s": max(0.0, now - rgb_at) if rgb_at > 0 else None,
        "depth_age_s": max(0.0, now - depth_at) if depth_at > 0 else None,
        "wall_time": time.time(),
        "monotonic_time": now,
        "sim_time": float(timeline.get_current_time()),
    }


def log_decision(status, *, model_metadata=None, **fields):
    global _event_sequence, state, _active_request
    _event_sequence += 1
    record = {
        "schema_version": LOG_SCHEMA_VERSION,
        "event_sequence": _event_sequence,
        "session_id": _session_id,
        "trial_id": _trial_id,
        **_trial_context,
        "pipeline": PIPELINE,
        "status": status,
        "encounter_id": (
            current_encounter_id
            if _classification_observation is not None and current_encounter_id > 0
            else None
        ),
        "request_id": _active_request,
        **capture_observation(),
        "state": state,
        "phase": avoidance_sub_phase,
        "navigation_clock": "simulation_time",
        "inference_clock": "monotonic_wall_time",
        "navmesh_path": NAVMESH_PATH,
        "navmesh_bounds_m": {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
        },
        "classification_start": _classification_observation,
        "category": fields.pop("category", None),
        "command": fields.pop("command", None),
        "decision_latency_s": fields.pop("decision_latency_s", None),
        "inference_s": fields.pop("inference_s", None),
        "reason": fields.pop("reason", None),
        "outcome": fields.pop("outcome", None),
        "model_metadata": {} if model_metadata is None else model_metadata,
        "event_metadata": fields,
    }
    line = json.dumps(json_safe(record), allow_nan=False)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(RESULT_LOG_PATH)), exist_ok=True)
        with open(RESULT_LOG_PATH, "a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
    except OSError as error:
        # Continuing an experiment with missing evidence would hide failed trials.
        state = "STOPPED"
        _active_request = None
        raise RuntimeError(f"Decision logging failed: {error}") from error
    print("[Decision] " + line)
    return json_safe(record)


def begin_classification(distance):
    global state, avoidance_sub_phase, initial_detected_distance
    global lateral_detour_ledger, current_encounter_id, classify_started_at
    global _request_serial, _active_request, _classification_observation, _classification_rgb
    initial_detected_distance = distance
    lateral_detour_ledger = 0.0
    current_encounter_id += 1
    _request_serial += 1
    _active_request = _request_serial
    classify_started_at = time.monotonic()
    avoidance_sub_phase = "CLASSIFY_OBSTACLE"
    state = "CLASSIFYING"
    sample = _ros_node.snapshot()
    _classification_observation = capture_observation(sample)
    _classification_observation["obstacle_distance_m"] = distance
    _classification_rgb = None if sample[0] is None else sample[0].copy()
    log_decision("detected")
    start_selected_model()


def fail_classification(reason):
    global _active_request
    if _active_request is None:
        return
    log_decision(
        "abstain",
        reason=reason,
        decision_latency_s=time.monotonic() - classify_started_at,
    )
    _active_request = None
    stop(reason="classification_failed")
    status_label.text = "Status: Classification failed - " + reason


def apply_category(category, **metadata):
    global state, avoidance_sub_phase, active_obstacle_category, _active_request
    global yield_wait_start_time, yield_wait_total_start_time, push_progress, push_target
    global step_start_pos
    if state != "CLASSIFYING" or _active_request is None:
        return
    if category not in COMMAND_BY_CATEGORY:
        fail_classification("invalid category")
        return
    command = COMMAND_BY_CATEGORY[category]
    log_decision(
        "predicted",
        category=category,
        command=command,
        decision_latency_s=time.monotonic() - classify_started_at,
        inference_s=metadata.pop("inference_s", None),
        model_metadata=metadata,
    )
    _active_request = None
    active_obstacle_category = category
    decision_label.text = f"{PIPELINE}: {category} -> {command}"
    if category == "CLEAR":
        state, avoidance_sub_phase = ("MOVING", "IDLE")
        step_start_pos = list(current_pos)
    elif category == "BIOLOGICAL":
        state = "YIELD_WAITING"
        yield_wait_start_time = yield_wait_total_start_time = navigation_time()
    elif category == "MINOR_SURFACE":
        state = "SCAN_AND_PROCEED"
        push_progress, push_target = (0.0, 1.0)
    else:
        initiate_bypass_sequence()
    if state != "STOPPED":
        status_label.text = "Status: " + command


def stop(reason="manual_stop"):
    global state, _active_request, _trial_finished
    if _active_request is not None:
        log_decision(
            "abstain",
            reason=reason,
            decision_latency_s=time.monotonic() - classify_started_at,
        )
    _active_request = None
    state = "STOPPED"
    if not _trial_finished:
        log_decision("run_finished", reason=reason, outcome="aborted")
        _trial_finished = True
    if status_label:
        status_label.text = "Status: STOPPED"


def finish_run(reason="Destination Reached"):
    global _loop_restart_pending, _loop_restart_time, _trial_finished
    if not _trial_finished:
        outcome = (
            "completed"
            if reason in ("Destination Reached", "Target X Reached", "Lane End Reached")
            else "aborted"
        )
        log_decision("run_finished", reason=reason, outcome=outcome)
        _trial_finished = True
    stop(reason=reason)
    if AUTO_LOOP_ENABLED:
        _loop_restart_pending = True
        _loop_restart_time = navigation_time() + AUTO_LOOP_DELAY
    status_label.text = "Status: " + reason


def perform_full_reset(trial_context=None):
    """Start a fresh execution; a future runner can pass explicit scenario identifiers.

    scenario_instance_id is the shared pairing key for executions of the same
    scenario/seed by different pipelines. trial_id identifies this execution only.
    """
    global _active_request, _request_serial, _trial_id, _trial_context, _trial_finished
    global _classification_observation, _classification_rgb
    context = dict(_trial_context)
    context["scenario_instance_id"] = None
    if trial_context is not None:
        unknown = set(trial_context) - set(context)
        if unknown:
            raise ValueError(f"Unknown trial context keys: {sorted(unknown)}")
        context.update(trial_context)
    if any(
        value is not None and not isinstance(value, str) for value in context.values()
    ):
        raise ValueError("Trial context values must be strings or None")
    if (
        context["expected_initial_command"] is not None
        and context["expected_initial_command"] not in COMMAND_BY_CATEGORY.values()
    ):
        raise ValueError("Invalid expected_initial_command")
    if not _trial_finished:
        stop(reason="reset_requested")
    _request_serial += 1
    _active_request = None
    reset_navigation_state()
    reset_selected_model()
    _ros_node.clear_frames()
    while True:
        try:
            _result_queue.get_nowait()
        except queue.Empty:
            break
    trail = stage.GetPrimAtPath("/World/Breadcrumbs")
    if trail.IsValid():
        stage.RemovePrim("/World/Breadcrumbs")
    update_usd_transform(current_pos, current_yaw)
    _trial_id = uuid.uuid4().hex
    _trial_context = context
    _classification_observation = _classification_rgb = None
    _trial_finished = False
    log_decision("trial_started")
    status_label.text = "Status: RESET - waiting for fresh sensors"
    decision_label.text = PIPELINE + ": Idle"


# VLM-only HTTP client; results are applied on the simulation thread.
VLM_PROMPT = 'Classify the obstacle blocking this robot\'s forward corridor.\nCategories: BIOLOGICAL (person or animal), STATIC_BAR (cone, car or solid barrier),\nMINOR_SURFACE (pothole or road defect), CLEAR (no obstructing obstacle).\nReturn only a JSON object with category and a brief reason. Example:\n{"category":"STATIC_BAR","reason":"A cone blocks the corridor."}\n'


def parse_vlm_reply(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub("^```(?:json)?\\s*", "", cleaned)
        cleaned = re.sub("\\s*```$", "", cleaned)
    parsed = json.loads(cleaned)
    if (
        not isinstance(parsed, dict)
        or parsed.get("category") not in COMMAND_BY_CATEGORY
    ):
        raise ValueError("response must contain a valid category")
    return parsed


def start_selected_model():
    request_id = _active_request
    token = _model_token
    rgb = _classification_rgb
    captured_at = _classification_observation["monotonic_time"] - (
        _classification_observation["frame_age_s"] or 0.0
    )
    if rgb is None or time.monotonic() - captured_at > SENSOR_MAX_AGE:
        fail_classification("no fresh RGB frame")
        return
    frame = rgb.copy()

    def worker():
        started = time.monotonic()
        try:
            ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            )
            if not ok:
                raise ValueError("image encoding failed")
            data_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode(
                "ascii"
            )
            payload = {
                "model": VLM_MODEL,
                "temperature": 0.0,
                "max_tokens": VLM_MAX_TOKENS,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VLM_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
            }
            request = urllib.request.Request(
                VLM_API_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + VLM_API_KEY,
                },
            )
            with urllib.request.urlopen(request, timeout=VLM_TIMEOUT) as response:
                raw = json.loads(response.read().decode("utf-8"))["choices"][0][
                    "message"
                ]["content"]
            parsed = parse_vlm_reply(raw)
            _result_queue.put(
                (token, request_id, parsed, raw, time.monotonic() - started, None)
            )
        except Exception as error:
            _result_queue.put(
                (token, request_id, None, None, time.monotonic() - started, str(error))
            )

    threading.Thread(target=worker, daemon=True).start()


def drain_model_results():
    while True:
        try:
            token, request_id, parsed, raw, duration, error = _result_queue.get_nowait()
        except queue.Empty:
            return
        if (
            token is not _model_token
            or request_id != _active_request
            or state != "CLASSIFYING"
        ):
            continue
        if time.monotonic() - classify_started_at >= VLM_TIMEOUT:
            fail_classification("VLM timeout")
            continue
        if error:
            fail_classification(error)
        else:
            apply_category(
                parsed["category"],
                inference_s=duration,
                raw_response=raw,
                reason=parsed.get("reason", ""),
            )


def poll_classification():
    if time.monotonic() - classify_started_at >= VLM_TIMEOUT:
        fail_classification("VLM timeout")


def reset_selected_model():
    pass


_phantomx_window = ui.Window(
    "PhantomX " + PIPELINE + " controller", width=440, height=230
)
status_label = sensor_label = ledger_label = decision_label = None
with _phantomx_window.frame:
    with ui.VStack(spacing=6):
        ui.Label("PhantomX - " + PIPELINE + " only (kinematic)")
        sensor_label = ui.Label("Obstacle Dist: waiting for sensors")
        decision_label = ui.Label(PIPELINE + ": Idle")
        ledger_label = ui.Label("Detour Ledger: 0.00 m")
        status_label = ui.Label("Status: MOVING")
        with ui.HStack():

            def on_emergency_stop():
                global _loop_restart_pending
                _loop_restart_pending = False
                stop()

            def on_pause_resume():
                global state, saved_pre_pause_state, last_time
                if state == "USER_PAUSED":
                    state = saved_pre_pause_state
                    last_time = navigation_time()
                elif state not in ("STOPPED", "CLASSIFYING"):
                    saved_pre_pause_state, state = (state, "USER_PAUSED")
                status_label.text = "Status: " + state

            ui.Button("Stop", clicked_fn=on_emergency_stop)
            ui.Button("Pause/Play", clicked_fn=on_pause_resume)
            ui.Button("Reset", clicked_fn=perform_full_reset)


# Kinematic navigation state machine; classification delegates to this file's model.
def zig_zag_update(event):
    global _loop_restart_pending, active_obstacle_category, avoidance_sub_phase, backup_progress
    global current_yaw, deadlock_retreat_count, diverge_turn_count
    global emergency_escape_turns, emergency_original_yaw, emergency_triggered, escape_progress
    global flank_pause_start_time, flank_progress, flank_step_progress, flank_step_start_pos
    global lane_base_yaw, lane_stuck_count, lane_stuck_pos, last_print_time
    global last_time, lateral_detour_ledger, micro_step_progress, micro_step_target
    global pause_start_time, pre_check_settle_start, push_progress, repeated_backup_count
    global repeated_backup_pos, required_flank_distance, state
    global step_start_pos, target_yaw, yield_wait_start_time
    global yield_wait_total_start_time
    rclpy.spin_once(_ros_node, timeout_sec=0.0)
    drain_model_results()
    if _loop_restart_pending:
        if navigation_time() >= _loop_restart_time:
            _loop_restart_pending = False
            perform_full_reset()
        else:
            return
    now = navigation_time()
    dt = now - last_time
    last_time = now
    if dt < 0.0:
        # Old maneuver timers are invalid after a rewind. Require a fresh trial.
        _loop_restart_pending = False
        stop(reason="simulation_time_rewound")
        return
    if state in ["STOPPED", "USER_PAUSED"]:
        return
    if not timeline.is_playing() or dt == 0.0:
        # Wall-time inference deadlines still apply while simulation is paused.
        if state == "CLASSIFYING":
            poll_classification()
        return
    if (
        time.monotonic() - _ros_node.depth_received_at > SENSOR_MAX_AGE
        or time.monotonic() - _ros_node.rgb_received_at > SENSOR_MAX_AGE
    ):
        status_label.text = "Status: Waiting for fresh RGB/depth"
        if state == "CLASSIFYING":
            poll_classification()
        return
    min_dist_ahead = _ros_node.closest_obstacle_dist
    if state not in [
        "MOVING",
        "STOPPED",
        "USER_PAUSED",
        "EMERGENCY_BACKUP",
        "CLASSIFYING",
        "EMERGENCY_ESCAPE_TURN",
        "EMERGENCY_ESCAPE_BACK",
        "EMERGENCY_ESCAPE_REALIGN",
    ]:
        if check_emergency_proximity(min_dist_ahead):
            print(
                f"[EMERGENCY STOP - Global Guard]: {min_dist_ahead:.2f}m while in '{state}'. Backing up."
            )
            initiate_emergency_backup()
            return
    if sensor_label:
        sensor_label.text = (
            f"Obstacle Dist: {min_dist_ahead:.2f} m"
            if min_dist_ahead != float("inf")
            else "Obstacle Dist: Clear (>10m)"
        )
    if ledger_label:
        ledger_label.text = f"Detour Ledger: {lateral_detour_ledger:.2f} m"
    if now - last_print_time >= PRINT_INTERVAL:
        last_print_time = now
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        print("-" * 65)
        print(
            f"[{timestamp_str}] [Telemetry] State: {state} | Cat: {active_obstacle_category} | Phase: {avoidance_sub_phase}"
        )
        print(
            f"            Pose: ({current_pos[0]:.2f}, {current_pos[1]:.2f}) | Heading: {current_yaw:.1f}° | Obstacle: {min_dist_ahead:.2f}m"
        )
        print(
            f"            Sectors: Left: {_ros_node.depth_left_dist:.2f}m, Ctr: {_ros_node.depth_center_dist:.2f}m, Right: {_ros_node.depth_right_dist:.2f}m"
        )
    if state == "YIELD_WAITING":
        effective_max_wait = MAX_YIELD_WAIT
        effective_check_interval = YIELD_WAIT_DURATION
        if now - yield_wait_total_start_time >= effective_max_wait:
            print(
                f"[Yield Timeout]: Waited {effective_max_wait:.1f}s, still present. Initiating bypass as if STATIC_BAR."
            )
            log_decision(
                "policy_override",
                category=active_obstacle_category,
                command="standard_bypass",
                reason="yield_timeout",
                original_command="yield_and_wait",
                elapsed_wait_s=now - yield_wait_total_start_time,
            )
            lateral_detour_ledger = 0.0
            yield_wait_start_time = 0.0
            yield_wait_total_start_time = 0.0
            initiate_bypass_sequence()
            return
        if now - yield_wait_start_time >= effective_check_interval:
            yield_wait_start_time = now
            if min_dist_ahead >= OBSTACLE_THRESHOLD + 0.4:
                print(
                    f"[Auto Resume]: Path completely clear ({min_dist_ahead:.2f}m). Resuming sweep operations."
                )
                state = "MOVING"
                avoidance_sub_phase = "IDLE"
                active_obstacle_category = "UNKNOWN"
                lateral_detour_ledger = 0.0
                target_yaw = lane_base_yaw
                current_yaw = lane_base_yaw
                step_start_pos = list(current_pos)
                if status_label:
                    status_label.text = "Status: Path Clear -> Resuming Operations"
                update_usd_transform(current_pos, current_yaw)
                return
            else:
                print("[Yield Waiting]: Obstacle still present. Continuing wait...")
    if state == "MOVING":
        lane_base_yaw = current_yaw
        if compute_dist_2d(current_pos, TARGET_DESTINATION) <= TARGET_RADIUS:
            finish_run("Destination Reached")
            return
        if lane_end_reached():
            update_usd_transform(current_pos, current_yaw)
            finish_run("Lane End Reached")
            return
        if min_dist_ahead < OBSTACLE_THRESHOLD:
            if min_dist_ahead < EMERGENCY_STOP_DIST:
                distance_from_trigger = compute_dist_2d(
                    current_pos, emergency_trigger_pos
                )
                time_since_trigger = navigation_time() - emergency_cooldown_start
                if emergency_triggered and (
                    distance_from_trigger < EMERGENCY_COOLDOWN_DIST
                    or time_since_trigger < EMERGENCY_COOLDOWN_TIME
                ):
                    print(
                        f"[Emergency Cooldown]: Ignoring re-trigger. Dist from trigger: {distance_from_trigger:.2f}m, Time: {time_since_trigger:.1f}s"
                    )
                    begin_classification(min_dist_ahead)
                    return
                print(
                    f"[EMERGENCY STOP]: Obstacle at {min_dist_ahead:.2f}m – too close! Initiating backup."
                )
                initiate_emergency_backup()
                return
            begin_classification(min_dist_ahead)
            return
        move_forward(speed=current_forward_speed(), dt=dt)
        hard_limit_hit = current_pos[0] > max_x or current_pos[0] < min_x
        if (
            compute_dist_2d(current_pos, TARGET_DESTINATION) <= TARGET_RADIUS
            or is_target_x_reached()
            or hard_limit_hit
        ):
            print(
                f"[Destination Reached]: X={current_pos[0]:.2f} >= {TARGET_DESTINATION[0]:.2f}"
            )
            finish_run(
                "Destination Reached" if not hard_limit_hit else "Hard Limit Reached"
            )
            return
        if lane_end_reached():
            update_usd_transform(current_pos, current_yaw)
            finish_run("Lane End Reached")
            return
        dist_in_step = compute_dist_2d(current_pos, step_start_pos)
        if dist_in_step >= STEP_DISTANCE:
            state = "PAUSED"
            pause_start_time = now
            lane_stuck_count = 0
            lane_stuck_pos = None
            if status_label:
                status_label.text = f"Status: Step Paused ({PAUSE_DURATION}s)"
            color = (0.0, 1.0, 0.0) if sweep_direction_y > 0 else (1.0, 0.0, 0.0)
            add_breadcrumb(current_pos, color=color)
        update_usd_transform(current_pos, current_yaw)
    elif state == "FALSE_ALARM_PUSH":
        if min_dist_ahead < EMERGENCY_STOP_DIST:
            distance_from_trigger = compute_dist_2d(current_pos, emergency_trigger_pos)
            time_since_trigger = navigation_time() - emergency_cooldown_start
            if emergency_triggered and (
                distance_from_trigger < EMERGENCY_COOLDOWN_DIST
                or time_since_trigger < EMERGENCY_COOLDOWN_TIME
            ):
                print(
                    "[Emergency Cooldown]: Continuing push despite proximity. Cooldown active."
                )
                pass
            else:
                print("[EMERGENCY STOP] False-alarm push aborted – obstacle too close.")
                initiate_emergency_backup()
                return
        step_dist = current_forward_speed() * 0.6 * dt
        move_forward_step(step_dist, dt)
        push_progress += step_dist
        if push_progress >= push_target:
            state = "MOVING"
            step_start_pos = list(current_pos)
            if status_label:
                status_label.text = "Status: Sweep Resumed"
            print("[False Alarm Recovery]: Push complete. Standard sweeping resumed.")
            emergency_triggered = False
        update_usd_transform(current_pos, current_yaw)
    elif state == "SCAN_AND_PROCEED":
        step_dist = current_forward_speed() * 0.5 * dt
        move_forward_step(step_dist, dt)
        push_progress += step_dist
        if push_progress >= push_target:
            state = "MOVING"
            avoidance_sub_phase = "IDLE"
            step_start_pos = list(current_pos)
        update_usd_transform(current_pos, current_yaw)
    elif state == "PAUSED":
        if now - pause_start_time >= PAUSE_DURATION:
            state = "MOVING"
            step_start_pos = list(current_pos)
    elif state == "FLANK_STEP":
        step_dist = current_forward_speed() * maneuver_speed_factor() * dt
        move_forward_step(step_dist, dt)
        flank_step_progress += step_dist
        flank_progress += step_dist
        if flank_step_progress >= STEP_DISTANCE:
            flank_step_progress = 0.0
            add_breadcrumb(current_pos, color=(0.0, 1.0, 1.0))
            state = "FLANK_PAUSED"
            flank_pause_start_time = now
            if status_label:
                status_label.text = "Status: Flank Paused"
        if (
            detour_side > 0
            and current_pos[1] >= SAFE_MAX_Y - 0.05
            or (detour_side < 0 and current_pos[1] <= SAFE_MIN_Y + 0.05)
        ):
            print("[Edge Blocked during Flank Step]: Stopping in this lane.")
            stop_blocked_lane()
            return
        update_usd_transform(current_pos, current_yaw)
    elif state == "FLANK_PAUSED":
        if now - flank_pause_start_time >= PAUSE_DURATION:
            if flank_progress >= required_flank_distance:
                avoidance_sub_phase = "FLANK_INSPECT"
                inspect_angle_side = -1.0 if detour_side > 0 else 1.0
                target_yaw = (lane_base_yaw + inspect_angle_side * 35.0) % 360.0
                state = "AVOID_ROTATE_TO_INSPECT_FLANK"
                if status_label:
                    status_label.text = "Status: Flank Complete – Inspecting"
            else:
                state = "FLANK_STEP"
                flank_step_start_pos = list(current_pos)
                flank_step_progress = 0.0
                if status_label:
                    status_label.text = "Status: Flank Step..."
        update_usd_transform(current_pos, current_yaw)
    elif state == "AVOID_INITIAL_TURN":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot
        if abs(yaw_diff) <= TURN_SPEED * dt:
            current_yaw = target_yaw
            micro_step_target = AVOIDANCE_STEP_SIZE
            micro_step_progress = 0.0
            state = "AVOID_DIVERGE_STEP"
        update_usd_transform(current_pos, current_yaw)
    elif state == "AVOID_ROTATE_THEN_STEP":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot
        if abs(yaw_diff) <= TURN_SPEED * dt:
            current_yaw = target_yaw
            state = "AVOID_DIVERGE_STEP"
            micro_step_progress = 0.0
        update_usd_transform(current_pos, current_yaw)
    elif state == "AVOID_DIVERGE_STEP":
        step_dist = current_forward_speed() * maneuver_speed_factor() * dt
        move_forward_step(step_dist, dt)
        micro_step_progress += step_dist
        if check_emergency_proximity(min_dist_ahead):
            print(
                f"[EMERGENCY STOP - Step Guard]: {min_dist_ahead:.2f}m in '{state}'. Backing up."
            )
            initiate_emergency_backup()
            return
        if is_pushing_into_edge(current_pos, detour_side):
            print(
                "[Edge Blocked during Diverge Step]: Stopping instead of diverging into the boundary."
            )
            stop_blocked_lane()
            return
        if micro_step_progress >= micro_step_target:
            add_breadcrumb(current_pos, color=(1.0, 1.0, 0.0))
            critical_side_depth = (
                _ros_node.depth_left_dist
                if detour_side > 0
                else _ros_node.depth_right_dist
            )
            REQUIRED_LATERAL = MIN_CHASSIS_CLEAR + 0.3
            lateral_ok = abs(lateral_detour_ledger) >= REQUIRED_LATERAL
            if critical_side_depth < 0.6 or not lateral_ok:
                diverge_turn_count += 1
                if diverge_turn_count > DIVERGE_ESCALATION_LIMIT:
                    enter_wide_obstacle_diverge(reason="Insufficient lateral clearance")
                    return
                max_angle = min(45.0, DIVERGE_ANGLE_INC + diverge_turn_count * 5.0)
                target_yaw = (lane_base_yaw + detour_side * max_angle) % 360.0
                micro_step_target = AVOIDANCE_STEP_SIZE
                micro_step_progress = 0.0
                state = "AVOID_ROTATE_THEN_STEP"
            else:
                target_yaw = lane_base_yaw
                state = "AVOID_ALIGN_TO_FLANK"
        update_usd_transform(current_pos, current_yaw)
    elif state == "AVOID_ALIGN_TO_FLANK":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot
        if abs(yaw_diff) <= TURN_SPEED * dt:
            current_yaw = target_yaw
            state = "AVOID_FLANK_PRE_CHECK_SETTLE"
            pre_check_settle_start = now
            if status_label:
                status_label.text = "Status: Aligned to Flank. Settling camera..."
        update_usd_transform(current_pos, current_yaw)
    elif state == "AVOID_FLANK_PRE_CHECK_SETTLE":
        if check_emergency_proximity(min_dist_ahead):
            print(
                f"[EMERGENCY STOP - Step Guard]: {min_dist_ahead:.2f}m in '{state}'. Backing up."
            )
            initiate_emergency_backup()
            return
        if now - pre_check_settle_start >= 0.4:
            center_clear = _ros_node.depth_center_dist > 0.6
            side_clear = (
                _ros_node.depth_right_dist > 0.6
                if detour_side > 0
                else _ros_node.depth_left_dist > 0.6
            )
            if center_clear and side_clear:
                print("[Flank Pre-Check Passed]: Corridor clear. Driving flank...")
                avoidance_sub_phase = "FLANK_DRIVE"
                flank_progress = 0.0
                flank_step_progress = 0.0
                flank_step_start_pos = list(current_pos)
                state = "FLANK_STEP"
            else:
                enter_wide_obstacle_diverge(
                    angle=65.0, step=0.75, reason="Deterministic pre-check failed."
                )
        update_usd_transform(current_pos, current_yaw)
    elif state == "AVOID_ROTATE_TO_INSPECT_FLANK":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot
        if abs(yaw_diff) <= TURN_SPEED * dt:
            current_yaw = target_yaw
            if min_dist_ahead < 0.8:
                avoidance_sub_phase = "FLANK_DRIVE"
                target_yaw = lane_base_yaw
                flank_progress = 0.0
                required_flank_distance = 0.5
                state = "AVOID_ALIGN_TO_FLANK"
            else:
                avoidance_sub_phase = "VERIFY"
                return_steer_side = -1.0 if lateral_detour_ledger > 0.0 else 1.0
                target_yaw = (lane_base_yaw + return_steer_side * 45.0) % 360.0
                state = "LOOK_BACK_ROTATE"
        update_usd_transform(current_pos, current_yaw)
    elif state == "LOOK_BACK_ROTATE":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot
        if abs(yaw_diff) <= TURN_SPEED * dt:
            current_yaw = target_yaw
            avoidance_sub_phase = "RETURNING"
            state = "RETURN_DRIVE_TO_LANE"
        update_usd_transform(current_pos, current_yaw)
    elif state == "RETURN_DRIVE_TO_LANE":
        if is_target_x_reached():
            print("[Target X Reached]: Stopping bypass.")
            finish_run("Target X Reached")
            return
        step_dist = current_forward_speed() * maneuver_speed_factor() * dt
        move_forward_step(step_dist, dt)
        if check_emergency_proximity(min_dist_ahead):
            print(
                f"[EMERGENCY STOP - Step Guard]: {min_dist_ahead:.2f}m in '{state}'. Backing up."
            )
            initiate_emergency_backup()
            return
        if (
            abs(lateral_detour_ledger) <= 0.05
            or lateral_detour_ledger * detour_side <= 0.0
        ):
            lateral_detour_ledger = 0.0
            target_yaw = lane_base_yaw
            state = "RETURN_FINAL_ALIGN"
        update_usd_transform(current_pos, current_yaw)
    elif state == "RETURN_FINAL_ALIGN":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot
        if abs(yaw_diff) <= TURN_SPEED * dt:
            current_yaw = target_yaw
            deadlock_retreat_count = 0
            state = "MOVING"
            avoidance_sub_phase = "IDLE"
            step_start_pos = list(current_pos)
        update_usd_transform(current_pos, current_yaw)
    elif state == "EMERGENCY_BACKUP":
        step_dist = EMERGENCY_BACKUP_SPEED * dt
        backup_progress += step_dist
        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] -= fwd_x * step_dist
        current_pos[1] -= fwd_y * step_dist
        min_dist_ahead = _ros_node.closest_obstacle_dist
        if (
            now - emergency_backup_start_time >= EMERGENCY_BACKUP_STUCK_TIME
            and emergency_escape_turns < EMERGENCY_MAX_ESCAPE_TURNS
        ):
            esc_yaw = pick_escape_yaw()
            if esc_yaw is not None:
                print(
                    f"[Emergency Backup]: Backing for {now - emergency_backup_start_time:.1f}s without clearing (likely against edge). Turning away to {esc_yaw:.0f}°."
                )
                emergency_escape_turns += 1
                emergency_original_yaw = current_yaw
                target_yaw = esc_yaw
                escape_progress = 0.0
                state = "EMERGENCY_ESCAPE_TURN"
                if status_label:
                    status_label.text = "Status: Emergency Backup stuck -> Turning Away"
                update_usd_transform(current_pos, current_yaw)
                return
        if min_dist_ahead >= EMERGENCY_BACKUP_CLEAR_DIST:
            print(
                f"[Emergency Backup]: Obstacle cleared after backing up {backup_progress:.2f}m. Resuming."
            )
            deadlock_retreat_count = 0
            emergency_triggered = False
            backup_progress = 0.0
            if repeated_backup_count >= REPEATED_BACKUP_LIMIT:
                print(
                    f"[Emergency Backup]: {repeated_backup_count} repeated backups near same spot. Stopping instead of retrying."
                )
                repeated_backup_count = 0
                repeated_backup_pos = None
                stop_blocked_lane()
                return
            state = "MOVING"
        elif backup_progress >= EMERGENCY_BACKUP_MAX_DIST:
            print(
                "[Emergency Backup]: Max backup distance reached, obstacle still present. Giving up on this lane."
            )
            deadlock_retreat_count += 1
            emergency_triggered = False
            backup_progress = 0.0
            if deadlock_retreat_count >= 3:
                print(
                    "[DEADLOCK RESET]: 3 backups reached. Resetting heading to forward, driving clear, then shifting lane."
                )
                deadlock_retreat_count = 0
                avoidance_sub_phase = "IDLE"
                if get_progress_dir_x() > 0:
                    lane_base_yaw = 0.0
                else:
                    lane_base_yaw = 180.0
                target_yaw = lane_base_yaw
                state = "DEADLOCK_RESET_TURN"
                if status_label:
                    status_label.text = "Status: Deadlock Reset - Turning Forward"
                return
            else:
                print(
                    "[DEADLOCK RESOLVED]: Backing off to lane and skipping current lane."
                )
                deadlock_retreat_count = 0
                state = "FORCE_RETREAT_BACK"
        update_usd_transform(current_pos, current_yaw)
    elif state == "EMERGENCY_ESCAPE_TURN":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        current_yaw += math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        if abs(yaw_diff) <= TURN_SPEED * dt:
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
        if (
            escape_progress >= escape_target_dist
            or room_along(current_pos, current_yaw) < 0.02
        ):
            target_yaw = emergency_original_yaw
            state = "EMERGENCY_ESCAPE_REALIGN"
        update_usd_transform(current_pos, current_yaw)
    elif state == "EMERGENCY_ESCAPE_REALIGN":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        current_yaw += math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        if abs(yaw_diff) <= TURN_SPEED * dt:
            current_yaw = target_yaw
            emergency_triggered = False
            backup_progress = 0.0
            deadlock_retreat_count = 0
            avoidance_sub_phase = "IDLE"
            state = "MOVING"
            step_start_pos = list(current_pos)
        update_usd_transform(current_pos, current_yaw)
    elif state == "CLASSIFYING":
        if min_dist_ahead < EMERGENCY_STOP_DIST:
            fail_classification("emergency proximity while classifying")
            return
        poll_classification()
        update_usd_transform(current_pos, current_yaw)
    elif state == "DEADLOCK_FLANK_PAST":
        yaw_diff = get_shortest_angle_diff(lane_base_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot
        if abs(yaw_diff) <= TURN_SPEED * dt:
            current_yaw = lane_base_yaw
            state = "MOVING"
            step_start_pos = list(current_pos)
            if status_label:
                status_label.text = "Status: Deadlock Flank Past - Moving Forward"
        update_usd_transform(current_pos, current_yaw)
    elif state == "FORCE_RETREAT_BACK":
        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] -= fwd_x * FORWARD_SPEED * dt
        current_pos[1] -= fwd_y * FORWARD_SPEED * dt
        backup_progress += FORWARD_SPEED * dt
        if (
            backup_progress >= force_retreat_target
            or room_along(current_pos, current_yaw) < 0.02
        ):
            print("[Force Retreat Complete]: No lane shift, just resuming forward.")
            emergency_triggered = False
            backup_progress = 0.0
            avoidance_sub_phase = "IDLE"
            if get_progress_dir_x() > 0:
                target_yaw = 0.0
            else:
                target_yaw = 180.0
            current_yaw = target_yaw
            state = "MOVING"
            step_start_pos = list(current_pos)
            update_usd_transform(current_pos, current_yaw)
            return
        update_usd_transform(current_pos, current_yaw)
    elif state in (
        "TURNING_1",
        "LANE_SHIFTING",
        "TURNING_2",
        "BIG_SKIP_TURN",
        "BIG_SKIP_DRIVE",
        "BIG_SKIP_PAUSE",
        "BIG_SKIP_ADVANCE_TURN",
        "BIG_SKIP_ADVANCE_DRIVE",
        "BIG_SKIP_ADVANCE_PAUSE",
        "BIG_SKIP_FINAL_TURN",
    ):
        # Guard a legacy state restored by an external caller; never shift lanes.
        finish_run("Lane shifting disabled")
        return
    elif state == "DEADLOCK_RESET_TURN":
        yaw_diff = get_shortest_angle_diff(lane_base_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot
        if abs(yaw_diff) <= TURN_SPEED * dt:
            current_yaw = lane_base_yaw
            state = "DEADLOCK_RESET_DRIVE"
            deadlock_retreat_count = 0
        update_usd_transform(current_pos, current_yaw)
    elif state == "DEADLOCK_RESET_DRIVE":
        step_dist = current_forward_speed() * dt
        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] += fwd_x * step_dist
        current_pos[1] += fwd_y * step_dist
        backup_progress += step_dist
        if backup_progress >= 0.5:
            print("[Deadlock Reset Complete]: Triggering standard lane shift.")
            backup_progress = 0.0
            stop_blocked_lane()
            return
        update_usd_transform(current_pos, current_yaw)


update_usd_transform(current_pos, current_yaw)
print("[VLM] Classification uses only " + VLM_MODEL)
log_decision("trial_started")
_phantomx_sub = (
    omni.kit.app.get_app()
    .get_update_event_stream()
    .create_subscription_to_pop(zig_zag_update)
)
