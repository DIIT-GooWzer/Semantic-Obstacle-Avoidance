"""
PhantomX Target-Directed Controller with Class-Based VLM Obstacle Avoidance
---------------------------------------------------------------------------------------
Features:
1. Multi-Class VLM Classification: Classifies obstacles into 4 categories:
    BIOLOGICAL, STATIC_BAR, MINOR_SURFACE, or CLEAR.
2. Category-Specific Strategies:
    - BIOLOGICAL: Yield & Wait (Stop and monitor until clear).
    - STATIC_BAR: Standard Bypass (Diverge, flank, return to lane).
    - MINOR_SURFACE: Scan & Proceed (Slow down, traverse directly over cracks/potholes).
    - CLEAR: False detection recovery (Instantly resume lane).
3. Boundary-Aware Edge Skip: When an obstacle is detected near the edge of the road,
    it cleanly terminates the lane and executes an early shift into the next sweep lane.
4. Auto-Resume Hardening: Only monitors clear depth during biological yield wait to
    prevent premature state abortions during bypass divergence turns.
5. NavMesh Boundary Clamping: Constrains all micro-steps within valid road limits.
"""

import os
import time
import math
import json
import base64
import threading
import re
import urllib.request
import urllib.error
import numpy as np
import omni.usd
import omni.timeline
import omni.kit.app
import omni.ui as ui
from pxr import Usd, UsdGeom, Gf, Sdf

# Cleanup prior loop instances on script reload
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
    except Exception:
        pass

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

try:
    import cv2
    from cv_bridge import CvBridge
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
SPAWN_POSITION     = (0.11952272590602997, 1.8697408479533721, 0.1)
SPAWN_ROTATION     = (0.0, 0.0, -90.0)
TARGET_DESTINATION = (9, -1.8266811603307502)
TARGET_RADIUS      = 0.3

STEP_DISTANCE      = 0.4                             # Sweep pause step size (m)
PAUSE_DURATION     = 0.5                             # Sweep pause duration (s)
FORWARD_SPEED      = 0.5                            # Base translation speed (m/s)
TURN_SPEED         = 180.0                           # Base rotation speed (deg/s)
LANE_SHIFT         = 0.5                             # Lane shift width (m)
OBSTACLE_THRESHOLD = 0.65                            # Avoidance trigger distance (m)
MIN_CHASSIS_CLEAR  = 0.40                            # Minimum lateral distance to clear chassis (m)
PRINT_INTERVAL     = 2.0                             # Telemetry print interval (s)
YIELD_WAIT_DURATION = 3.0                            # Time in seconds to wait before checking if animal moved
MAX_YIELD_WAIT = 10.0            # Maximum seconds to wait for a biological to clear

# Avoidance micro-step settings
AVOIDANCE_STEP_SIZE   = 0.30
DIVERGE_ANGLE_INC     = 31.0
DIVERGE_ESCALATION_LIMIT = 3  
MIN_VLM_COOLDOWN      = 2.0
MIN_VLM_COOLDOWN = 15.0   # was 2.0

LANE_STUCK_THRESHOLD = 2       # after this many failed shifts at ~same spot, big-skip
LANE_STUCK_POS_TOL = 0.30      # meters — "same spot" tolerance
BIG_SKIP_X_DISTANCE = 1.5 

# Emergency Backup Settings
EMERGENCY_BACKUP_DIST = 0.3        # Distance to back up (m)
EMERGENCY_BACKUP_SPEED = 0.5        # Speed while backing up (m/s)
EMERGENCY_COOLDOWN_DIST = 1.3       # Must travel this far forward before allowing another emergency stop on same obstacle
EMERGENCY_COOLDOWN_TIME = 5.0       # Or wait this long (seconds) before allowing another emergency stop

# Head-On Center Evasion Parameters
CRITICAL_CENTER_THRESH = 1.0  # Meter threshold to trigger target-directed evade
SIDE_SAFETY_THRESH     = 0.5  # Minimum lateral space required to make the turn
EVADE_TURN_ANGLE       = 30.0 # Degrees to rotate toward the target per evasion step

EMERGENCY_STOP_DIST = 0.35        # If obstacle closer than this, stop immediately
EMERGENCY_BACKUP_DIST = 0.3        # This is no longer used, keep it for reference
EMERGENCY_BACKUP_CLEAR_DIST = 0.6  # NEW: Back up until obstacle is at least this far away
EMERGENCY_BACKUP_MAX_DIST = 2.5    # NEW: Safety cap - if it backs up this much without clearing, give up
# ============================================================
# CATEGORY-SPECIFIC FLANK DISTANCES
# ============================================================
# Small objects such as traffic cones should not cause the
# robot to drive several metres past them.
#
# Larger objects/animals get a longer clearance distance.

CLEAR_PASS_SAFETY_PAD = 0.30
VERIFY_SKIP_DIST = 2.0 


FLANK_EXTRA_DISTANCE = {
    "STATIC_BAR": 0.35,       # Cone, bollard, small barrier
    "BIOLOGICAL": 0.80,       # Cow / animal
    "MINOR_SURFACE": 0.0
}

FLANK_MIN_DISTANCE = {
    "STATIC_BAR": 0.70,
    "BIOLOGICAL": 1.20
}

FLANK_MAX_DISTANCE = {
    "STATIC_BAR": 1.50,
    "BIOLOGICAL": 2.50
}

# Fallback values for unknown categories
DEFAULT_FLANK_EXTRA = 0.50
DEFAULT_FLANK_MIN = 0.90
DEFAULT_FLANK_MAX = 2.00

# ============================================================
# FALSE-ALARM PUSH SETTINGS
# ============================================================
# Minimum forward movement after a false detection.
FALSE_ALARM_PUSH_MIN = 0.30

# Maximum forward movement after a false detection.
FALSE_ALARM_PUSH_MAX = 0.80

# Extra distance beyond the obstacle trigger threshold.
FALSE_ALARM_PUSH_MARGIN = 0.40                         # Extra margin added to flank bypass (m)

# # VLM API Configuration
# VLM_API_URL        = "https://openrouter.ai/api/v1/chat/completions"
# VLM_API_KEY        = os.getenv("OPENROUTER_API_KEY", "sk-or-v1-xxxx")
# VLM_MODEL          = "dots-studio/dots-3-note-preview:free"

# VLM_API_URL = "http://localhost:8000/v1/chat/completions"
# VLM_API_KEY = os.getenv("FREEBUFF_API_KEY", "231970c3-bd57-41a8-8368-1e2d81176618")
# VLM_MODEL = "mimo/mimo-v2.5"


# VLM_API_URL = "http://localhost:8000/v1/chat/completions"
# VLM_API_KEY = os.getenv("FREEBUFF_API_KEY", "231970c3-bd57-41a8-8368-1e2d81176618")
# VLM_MODEL = "deepseek/deepseek-v4-flash"


VLM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
VLM_API_KEY = os.getenv("LM_STUDIO_API_KEY", "sk-lm-xxxx")
VLM_MODEL   = "lmstudio-community/Qwen3.5-9B-GGUF"  

VLM_TIMEOUT     = 180.0             # Max seconds to wait for model response
VLM_MAX_RETRIES = 0                 # 0 retries recommended for local LLM
VLM_MAX_TOKENS  = 8             # Max tokens to generate




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
navmesh_prim = stage.GetPrimAtPath("/World/NavMeshVolume")

min_x, max_x = -10.0, 12.0
min_y, max_y = -2.0, 2.0
if navmesh_prim.IsValid():
    bound = bbox_cache.ComputeWorldBound(navmesh_prim).ComputeAlignedBox()
    min_x, max_x = float(bound.GetMin()[0]), float(bound.GetMax()[0])
    min_y, max_y = float(bound.GetMin()[1]), float(bound.GetMax()[1])

# Safety boundary margin to keep hexapod completely inside NavMesh
# Safety boundary margins
ROBOT_SAFETY_MARGIN = 0.35
SAFE_MIN_Y = min_y + ROBOT_SAFETY_MARGIN
SAFE_MAX_Y = max_y - ROBOT_SAFETY_MARGIN
SAFE_MIN_X = min_x + ROBOT_SAFETY_MARGIN
SAFE_MAX_X = max_x - ROBOT_SAFETY_MARGIN


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

def is_pushing_into_edge(pos, side, margin=0.10):
    """True if moving further toward `side` (+1 = toward SAFE_MAX_Y, -1 = toward SAFE_MIN_Y)
    would push the robot past the navmesh boundary."""
    if side > 0 and pos[1] >= SAFE_MAX_Y - margin:
        return True
    if side < 0 and pos[1] <= SAFE_MIN_Y + margin:
        return True
    return False

# Global list to track breadcrumb positions
breadcrumb_counter = 0
BREADCRUMB_SIZE = 0.15      # width/depth of each plate (m)
BREADCRUMB_THICKNESS = 0.01 # keep it thin so it sits flush on the ground

def add_breadcrumb(pos, color=(0.0, 1.0, 0.0)):
    global breadcrumb_counter
    stage = omni.usd.get_context().get_stage()

    prim_path = f"/World/Breadcrumbs/plate_{breadcrumb_counter}"
    breadcrumb_counter += 1

    plate = UsdGeom.Cube.Define(stage, prim_path)
    plate.CreateSizeAttr(2.5)  # unit cube; scaled to final size below

    xform = UsdGeom.Xformable(plate)
    xform.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], BREADCRUMB_THICKNESS / 2))
    xform.AddScaleOp().Set(Gf.Vec3d(BREADCRUMB_SIZE, BREADCRUMB_SIZE, BREADCRUMB_THICKNESS))

    plate.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    prim = plate.GetPrim()
    # Guide purpose: shown in the viewport for you, excluded from
    # camera/sensor render products by default -> invisible to the robot
    prim.CreateAttribute("purpose", Sdf.ValueTypeNames.Token).Set(UsdGeom.Tokens.guide)
    # No physics interaction
    prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)

def compute_dist_2d(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    


def get_target_angle_diff(current_pose, current_heading_deg, target_pose):
    """
    Computes the relative angle between robot heading and target destination.
    Positive result -> Target is to the LEFT.
    Negative result -> Target is to the RIGHT.
    """
    dx = target_pose[0] - current_pose[0]
    dy = target_pose[1] - current_pose[1]
    
    # Calculate global goal angle in degrees (0..360)
    target_bearing_deg = math.degrees(math.atan2(dy, dx)) % 360.0
    
    # Compute relative angle difference clipped to [-180, 180]
    diff = (target_bearing_deg - current_heading_deg + 180.0) % 360.0 - 180.0
    return diff

# ============================================================
# 5. MEMORY-AUGMENTED VISION LLM CLIENT WITH CLASSIFICATION
# ============================================================
global vlm_request_in_flight, last_vlm_query_time, llm_avoidance_history
vlm_request_in_flight = False
last_vlm_query_time = 0.0
llm_avoidance_history = []  # Short-term session memory for current obstacle

def encode_frame_to_base64(rgb_array):
    if _has_cv2 and rgb_array is not None:
        h, w = rgb_array.shape[:2]
        if w > 640:
            new_w = 640
            new_h = int(h * (640 / w))
            rgb_array = cv2.resize(rgb_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode('.jpg', rgb_array, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return base64.b64encode(buffer).decode('utf-8')
    return ""

def async_query_vlm_agent(rgb_array, obstacle_dist, current_sub_phase, detour_offset, callback_fn):
    global vlm_request_in_flight, last_vlm_query_time, llm_avoidance_history, vlm_requested_phase

    vlm_requested_phase = current_sub_phase 

    now = time.time()

    if state in ["STOPPED", "USER_PAUSED"]:
        return

    # Already mid-request -> just skip, don't double-fire deterministic either,
    # since a real VLM response is still on its way and will resolve this.
    if vlm_request_in_flight:
        print(f"[VLM Busy]: Request already in flight, skipping duplicate call for phase={current_sub_phase}.")
        return

    # Still inside the cooldown window -> don't touch the VLM at all,
    # resolve immediately with the deterministic rule instead.
    if (now - last_vlm_query_time) < MIN_VLM_COOLDOWN:
        remaining = MIN_VLM_COOLDOWN - (now - last_vlm_query_time)
        print(f"[VLM Cooldown]: {remaining:.1f}s remaining, using deterministic fallback for phase={current_sub_phase}.")
        execute_deterministic_fallback(
            current_sub_phase, obstacle_dist, detour_offset, callback_fn,
            reason=f"Deterministic fallback (VLM cooldown, {remaining:.1f}s remaining)."
        )
        return

    # ---- OPTIMIZATION: Skip VLM if depth is clearly clear ----
    if current_sub_phase in ["VERIFY", "FLANK_INSPECT", "FLANK_PRE_CHECK"] and obstacle_dist > VERIFY_SKIP_DIST:
        print(f"[{current_sub_phase}]: Depth clear ({obstacle_dist:.2f}m), skipping VLM.")
        callback_fn({
            "command": "return_to_lane",
            "obstacle_visible": False,
            "category": "CLEAR",
            "reasoning": f"Depth sensors show clear path ({obstacle_dist:.2f}m), VLM skipped."
        })
        return

    # ---- Proceed with normal VLM query ----
    vlm_request_in_flight = True
    last_vlm_query_time = now
    print(f"[VLM Dispatch]: in_flight=True | phase={current_sub_phase} | t={now:.2f}")
    def worker():
        global vlm_request_in_flight, llm_avoidance_history
        try:
            max_retries = VLM_MAX_RETRIES
            attempt = 0
            while attempt <= max_retries:
                try:
                    if state not in ["STOPPED", "USER_PAUSED"]:
                        print("[VLM Max Retries/Timeout Exceeded]: Executing deterministic fallback rule.")
                        execute_deterministic_fallback(
                            current_sub_phase, obstacle_dist, detour_offset, callback_fn,
                            reason="Deterministic fallback safety rule post timeout/error."
                        )

                    b64_img = encode_frame_to_base64(rgb_array)
                    if not b64_img:
                        print("[Perception Error]: RGB image frame is empty or invalid.")
                        return

                    history_summary = ""
                    if len(llm_avoidance_history) > 0:
                        history_summary = "Past avoidance decisions for this obstacle:\n" + "\n".join([f"- {h}" for h in llm_avoidance_history[-3:]]) + "\n"

                    if current_sub_phase == "CLASSIFY_OBSTACLE":
                        prompt = (
                            f"You are the visual perception brain for an autonomous sweeper robot analyzing an obstacle detected at {obstacle_dist:.2f}m.\n"
                            f"If the path is actually clear or it is a false detection, categorize as CLEAR.\n"
                            f"Double check: If the image you are looking at, doesn't resemble a sky, a road, any form that doesn't look like a terrain, then the thing you are directly looking at is an obstacle.\n"
                            f"Examine the image carefully for 3D rendered asset features:\n\n"
                            f"CRITICAL VISUAL RULES:\n"
                            f"- BIOLOGICAL: Any animal, quadruped, cow, cattle, bull, or livestock model (look for 4 legs, torso, snout, ears, tail, black/white or brown hide pattern). Do NOT classify animals or quadrupeds as traffic cones or static barriers even if captured up close!\n"
                            f"- STATIC_BAR: Traffic cones, bright orange/red plastic markers, construction barriers, bollards, or solid sharp-edged blocks, etc.\n"
                            f"- MINOR_SURFACE: Potholes, road cracks, asphalt patches, ground depressions, or flat surface defects.\n"
                            f"- CLEAR: No obstacle blocking forward progress or clear open corridor.\n\n"
                            f"Additionally, divide the image horizontally into 3 equal vertical slices (Left 0â€‘33%, Center 33â€‘66%, Right 66â€‘100%).\n"
                            f"Provide sector analysis:\n"
                            f"- 'center_blocked': true if ANY obstacle (any category) intrudes into the Center slice.\n"
                            f"- 'left_sector_coverage_pct': integer 0â€‘100 representing obstacle coverage in Left slice.\n"
                            f"- 'right_sector_coverage_pct': integer 0â€‘100 representing obstacle coverage in Right slice.\n\n"
                            f"Return ONLY valid JSON with the following structure:\n"
                            f"{{\n"
                            f'  "category": "BIOLOGICAL" | "STATIC_BAR" | "MINOR_SURFACE" | "LARGE_OBJECT" | "CLEAR",\n'
                            f'  "obstacle_class": "Cow/Animal" | "Traffic Cone" | "Pothole/Crack" | "None",\n'
                            f'  "command": "yield_and_wait" | "standard_bypass" | "scan_and_proceed" | "resume_path",\n'
                            f'  "reasoning": "Detailed visual evidence (e.g., detected 4 legs and bovine body shape)",\n'
                            f'  "center_blocked": bool,\n'
                            f'  "left_sector_coverage_pct": int,\n'
                            f'  "right_sector_coverage_pct": int\n'
                            f"}}"
                        )

                    elif current_sub_phase == "DIVERGE_SCAN":
                        left_d = _ros_node.depth_left_dist
                        ctr_d = _ros_node.depth_center_dist
                        right_d = _ros_node.depth_right_dist
                        
                        prompt = (
                            f"You are navigating a sweeper robot bypassing a PRIMARY target ({active_obstacle_category}).\n"
                            f"Current depths: Left={left_d:.2f}m, Center={ctr_d:.2f}m, Right={right_d:.2f}m.\n"
                            f"Lateral detour offset: {detour_offset:.2f}m.\n\n"
                            f"MULTI-OBSTACLE CORRIDOR RULES:\n"
                            f"1. PRIMARY TARGET: Focus strictly on clearing the INITIAL obstacle (e.g., traffic cone).\n"
                            f"2. SECONDARY OBSTACLES: If a car, wall, or secondary object appears on the other side, "
                            f"DO NOT turn further into it if there is a clear passage/gap available straight ahead.\n"
                            f"3. THREADING THE GAP: If the primary obstacle has drifted to the side and the center corridor "
                            f"has an open gap between objects, command 'flank_forward' to thread through.\n"
                            f"4. ONLY command 'diverge_more' if the primary target itself is still blocking the central path.\n\n"
                            f"Return ONLY valid JSON:\n"
                            f"{{\n"
                            f'  "primary_obstacle_cleared": bool,\n'
                            f'  "corridor_gap_open": bool,\n'
                            f'  "command": "diverge_more" | "flank_forward",\n'
                            f'  "reasoning": "Explain primary target clearance and corridor gap status"\n'
                            f"}}"
                        )

                    elif current_sub_phase == "WIDE_OBSTACLE_DIVERGE":
                        prompt = (
                            f"CRITICAL: Pre-flank verification FAILED. A wide obstacle (car/vehicle) is blocking the forward corridor.\n"
                            f"{history_summary}"
                            f"Current distance: {obstacle_dist:.2f}m. Lateral detour offset: {detour_offset:.2f}m.\n"
                            f"Task: Inspect the wide field. Are we far enough laterally to clear the entire front/rear bumper and chassis?\n"
                            f"Commands:\n"
                            f"- 'diverge_more': If the vehicle body still overlaps our lane corridor.\n"
                            f"- 'flank_forward': If we have completely shifted beyond the outer edge of the car.\n"
                            f"Return ONLY valid JSON: {{\"obstacle_visible\": bool, \"command\": \"diverge_more\"|\"flank_forward\", \"reasoning\": string}}"
                        )

                    elif current_sub_phase == "FLANK_INSPECT":
                        prompt = (
                            f"Verifying flank clearance of obstacle.\n"
                            f"{history_summary}"
                            f"Current distance reading: {obstacle_dist:.2f}m. Lateral displacement: {detour_offset:.2f}m.\n"
                            f"Task: Is the obstacle still alongside/ahead, or has it passed behind?\n"
                            f"Double check: If the image you are looking at, doesn't resemble a sky, a road, any form that doesn't look like a terrain, then the thing you are directly looking at is an obstacle.\n"
                            f"Commands:\n"
                            f"- 'flank_forward': Obstacle still alongside.\n"
                            f"- 'return_to_lane': Obstacle completely passed; return to lane.\n"
                            f"Return ONLY valid JSON: {{\"obstacle_visible\": bool, \"command\": \"flank_forward\"|\"return_to_lane\", \"reasoning\": string}}"
                        )

                    elif current_sub_phase == "FLANK_PRE_CHECK":
                        left_d = _ros_node.depth_left_dist
                        ctr_d = _ros_node.depth_center_dist
                        right_d = _ros_node.depth_right_dist
                        
                        prompt = (
                            f"You are verifying corridor clearance for an autonomous sweeper.\n"
                            f"Sensor depths: Left={left_d:.2f}m, Center={ctr_d:.2f}m, Right={right_d:.2f}m. Lateral detour: {detour_offset:.2f}m.\n\n"
                            f"Analyze the image divided horizontally into 3 equal columns:\n"
                            f"- LEFT Sector (0% to 33% width)\n"
                            f"- CENTER Sector (33% to 66% width)\n"
                            f"- RIGHT Sector (66% to 100% width)\n\n"
                            f"CRITICAL VISUAL RULES:\n"
                            f"1. Any vehicle parts (rusted panels, car bumpers, headlights, tires) belong to the EXTERNAL OBSTACLE, NOT this sweeper robot.\n"
                            f"2. Estimate the obstacle coverage percentage (0-100%) in the Left and Right sectors.\n"
                            f"3. Mark 'center_blocked': true if ANY obstacle intrudes into the Center corridor.\n\n"
                            f"Return ONLY valid JSON:\n"
                            f"{{\n"
                            f'  "center_blocked": bool,\n'
                            f'  "left_sector_coverage_pct": int,\n'
                            f'  "right_sector_coverage_pct": int,\n'
                            f'  "obstacle_visible": bool,\n'
                            f'  "command": "diverge_more" | "flank_forward",\n'
                            f'  "reasoning": "Explain sector coverage (e.g., right sector 60% covered by car body)"\n'
                            f"}}"
                        )

                    else:  # VERIFY
                        prompt = (
                            f"Verifying return path to original sweep lane.\n"
                            f"{history_summary}"
                            f"Lateral offset to cancel: {detour_offset:.2f}m.\n"
                            f"Task: Is the return line clear of obstacles?\n"
                            f"Commands:\n"
                            f"- 'return_to_lane': Return corridor completely clear.\n"
                            f"- 'flank_forward': Return corridor obstructed.\n"
                            f"Return ONLY valid JSON: {{\"obstacle_visible\": bool, \"command\": \"return_to_lane\"|\"flank_forward\", \"reasoning\": string}}"
                        )

                    payload = {
                        "model": VLM_MODEL,
                        "temperature": 0.0,
                        "max_tokens": VLM_MAX_TOKENS,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                                ]
                            }
                        ]
                    }
                    if "localhost:1234" not in VLM_API_URL and "127.0.0.1:1234" not in VLM_API_URL:
                        payload["response_format"] = {"type": "json_object"}

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
                    print(f"[VLM Query]: Querying {VLM_MODEL} for '{current_sub_phase}' (waiting up to {VLM_TIMEOUT:.0f}s)...")
                    with urllib.request.urlopen(req, timeout=VLM_TIMEOUT) as resp:
                        if state in ["STOPPED", "USER_PAUSED"]:
                            return

                        result = json.loads(resp.read().decode('utf-8'))
                        raw_reply = result['choices'][0]['message']['content'].strip()

                        # Robust JSON extraction (strip code blocks, find outer braces)
                        if "```" in raw_reply:
                            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_reply, re.DOTALL)
                            if match:
                                raw_reply = match.group(1).strip()
                        first_brace = raw_reply.find('{')
                        last_brace = raw_reply.rfind('}')
                        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                            raw_reply = raw_reply[first_brace:last_brace + 1]

                        parsed = json.loads(raw_reply)

                        log_entry = f"Phase {current_sub_phase}: Decided '{parsed.get('command')}' (Cat={parsed.get('category', 'N/A')}). Reason: {parsed.get('reasoning')}"
                        llm_avoidance_history.append(log_entry)

                        callback_fn(parsed)
                        return  # Success: Exit thread

                except urllib.error.HTTPError as e:
                    attempt += 1
                    err_body = ""
                    try:
                        err_body = e.read().decode('utf-8', errors='ignore')
                    except Exception:
                        pass
                    print(f"[VLM HTTP Error {e.code} (Attempt {attempt}/{max_retries + 1})]: {e.reason} | Details: {err_body}")
                    time.sleep(1.0)
                except urllib.error.URLError as e:
                    attempt += 1
                    err_msg = str(e.reason).lower()
                    if "timed out" in err_msg or "timeout" in err_msg:
                        print(f"[VLM Timeout (Attempt {attempt}/{max_retries + 1})]: Model response took longer than {VLM_TIMEOUT:.0f}s.")
                    else:
                        print(f"[VLM Server Connection Error (Attempt {attempt}/{max_retries + 1})]: {e.reason}")
                    time.sleep(1.0)
                except Exception as e:
                    attempt += 1
                    print(f"[VLM Unexpected Error (Attempt {attempt}/{max_retries + 1})]: {e}")
                    time.sleep(1.0)

            # Fallback executed ONLY after all retries and full timeouts expire
            if state not in ["STOPPED", "USER_PAUSED"]:
                print("[VLM Max Retries/Timeout Exceeded]: Executing deterministic fallback rule.")
                if current_sub_phase == "CLASSIFY_OBSTACLE":
                    cmd, cat = "standard_bypass", "STATIC_BAR"
                elif current_sub_phase in ["DIVERGE_SCAN", "FLANK_PRE_CHECK", "WIDE_OBSTACLE_DIVERGE"]:
                    cmd = "diverge_more" if (abs(detour_offset) < MIN_CHASSIS_CLEAR or obstacle_dist < 1.5) else "flank_forward"
                    cat = "STATIC_BAR"
                elif current_sub_phase == "FLANK_INSPECT":
                    cmd = "return_to_lane" if obstacle_dist > 1.2 else "flank_forward"
                    cat = "STATIC_BAR"
                elif current_sub_phase == "VERIFY":
                    cmd = "return_to_lane"
                    cat = "STATIC_BAR"
                else:
                    cmd, cat = "return_to_lane", "STATIC_BAR"

                callback_fn({
                    "category": cat,
                    "obstacle_visible": True,
                    "command": cmd,
                    "reasoning": "Deterministic fallback safety rule post timeout."
                })
        finally:
            # Guaranteed to run no matter which return/exception path was taken above,
            # so a completed or abandoned query can never permanently wedge the gate.
            print(f"[VLM Dispatch]: in_flight=False | phase={current_sub_phase} | t={time.time():.2f}")
            vlm_request_in_flight = False

    threading.Thread(target=worker, daemon=True).start()


# ============================================================
# 6. STATE VARIABLES & DISPLACEMENT LEDGER
# ============================================================
global state, saved_pre_pause_state, current_pos, current_yaw, sweep_direction_y
global step_start_pos, target_yaw, pause_start_time, last_time, shifted_distance, last_print_time
global lane_base_yaw, lateral_detour_ledger, micro_step_progress, micro_step_target, detour_side
global avoidance_sub_phase, initial_detected_distance, required_flank_distance, flank_progress, diverge_turn_count
global active_obstacle_category, yield_wait_start_time
global push_progress, push_target
global pre_check_settle_start
global yield_wait_total_start_time
global emergency_triggered, emergency_cooldown_start, emergency_trigger_pos
global deadlock_retreat_count, force_retreat_target
global wide_diverge_count
global edge_stuck_time, vlm_requested_phase
global flank_step_start_pos, flank_step_progress, flank_pause_start_time
global lane_stuck_count, lane_stuck_pos, big_skip_target_y
global big_skip_step_progress, big_skip_advance_progress, big_skip_pause_start




big_skip_step_progress = 0.0      # tracks distance within current STEP_DISTANCE chunk
big_skip_advance_progress = 0.0   # tracks total distance advanced into the new lane
big_skip_pause_start = 0.0

lane_stuck_count = 0
lane_stuck_pos = None

big_skip_target_y = 0.0   # new: where BIG_SKIP_DRIVE is heading


edge_stuck_time = 0.0
vlm_requested_phase = ""   # track the phase for which the VLM was dispatched

deadlock_retreat_count = 0
force_retreat_target = 1.5
wide_diverge_count = 0  


emergency_triggered = False
emergency_cooldown_start = 0.0
emergency_trigger_pos = [0.0, 0.0]
backup_progress = 0.0


deadlock_retreat_count = 0
force_retreat_target = 1.5


yield_wait_total_start_time = 0.0

pre_check_settle_start = 0.0


push_progress = 0.0
push_target = 0.0

current_pos = list(SPAWN_POSITION)
current_yaw = SPAWN_ROTATION[2]
state = "MOVING"
saved_pre_pause_state = "MOVING"

flank_step_start_pos = list(current_pos)   # start of current flank step
flank_step_progress = 0.0                  # accumulated distance in current step
flank_pause_start_time = 0.0


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

active_obstacle_category = "UNKNOWN"
yield_wait_start_time = 0.0

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

def initiate_emergency_backup():
    """Back up away from obstacle, then re-evaluate."""
    global state, emergency_triggered, emergency_cooldown_start, emergency_trigger_pos
    global backup_progress, avoidance_sub_phase, lateral_detour_ledger
    
    print(f"[Emergency Backup]: Obstacle too close! Backing up {EMERGENCY_BACKUP_DIST}m...")
    
    # Record where we triggered the emergency
    emergency_trigger_pos = list(current_pos)
    emergency_triggered = True
    emergency_cooldown_start = time.time()
    backup_progress = 0.0
    
    # Clear any ongoing avoidance state
    avoidance_sub_phase = "IDLE"
    lateral_detour_ledger = 0.0
    
    # Enter backup state
    state = "EMERGENCY_BACKUP"
    
    if status_label:
        status_label.text = f"Status: EMERGENCY BACKUP ({EMERGENCY_BACKUP_DIST:.1f}m)"

def execute_deterministic_fallback(current_sub_phase, obstacle_dist, detour_offset, callback_fn, reason="Deterministic fallback rule."):
    if current_sub_phase == "CLASSIFY_OBSTACLE":
        cmd, cat = "standard_bypass", "STATIC_BAR"
    elif current_sub_phase in ["DIVERGE_SCAN", "FLANK_PRE_CHECK", "WIDE_OBSTACLE_DIVERGE"]:
        cmd = "diverge_more" if (abs(detour_offset) < MIN_CHASSIS_CLEAR or obstacle_dist < 1.5) else "flank_forward"
        cat = "STATIC_BAR"
    elif current_sub_phase == "FLANK_INSPECT":
        cmd = "return_to_lane" if obstacle_dist > 1.2 else "flank_forward"
        cat = "STATIC_BAR"
    elif current_sub_phase == "VERIFY":
        cmd = "return_to_lane"
        cat = "STATIC_BAR"
    else:
        cmd, cat = "return_to_lane", "STATIC_BAR"

    callback_fn({
        "category": cat,
        "obstacle_visible": True,
        "command": cmd,
        "reasoning": reason
    })

def check_emergency_proximity(dist_ahead):
    """True if an emergency backup should fire right now, respecting cooldown."""
    if state in ["STOPPED", "USER_PAUSED", "EMERGENCY_BACKUP", "VLM_INSPECTING"]:
        return False
    if dist_ahead >= EMERGENCY_STOP_DIST:
        return False
    distance_from_trigger = compute_dist_2d(current_pos, emergency_trigger_pos)
    time_since_trigger = time.time() - emergency_cooldown_start
    if emergency_triggered and (distance_from_trigger < EMERGENCY_COOLDOWN_DIST or time_since_trigger < EMERGENCY_COOLDOWN_TIME):
        return False
    return True

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
        ui.Label("PhantomX Class-Based Sweeper", height=20, style={"font_size": 14, "color": 0xFF00FFFF})
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
                global vlm_request_in_flight, llm_avoidance_history, active_obstacle_category
                global emergency_triggered, emergency_cooldown_start, emergency_trigger_pos
                global wide_diverge_count          # NEW

                wide_diverge_count = 0             # NEW
                state = "MOVING"
                vlm_request_in_flight = False
                llm_avoidance_history.clear()
                emergency_triggered = False
                emergency_cooldown_start = 0.0
                emergency_trigger_pos = list(SPAWN_POSITION)
                current_pos = list(SPAWN_POSITION)
                current_yaw = SPAWN_ROTATION[2]
                lane_base_yaw = current_yaw
                sweep_direction_y = 1.0 if SPAWN_ROTATION[2] == 90.0 else -1.0
                lateral_detour_ledger = 0.0
                avoidance_sub_phase = "IDLE"
                active_obstacle_category = "UNKNOWN"
                step_start_pos = list(current_pos)
                shifted_distance = 0.0
                last_time = time.time()
                stage = omni.usd.get_context().get_stage()
                trail_prim = stage.GetPrimAtPath("/World/Breadcrumbs")
                if trail_prim.IsValid():
                    stage.RemovePrim("/World/Breadcrumbs")
                
                # If using the Points approach, also clear the lists:
                global breadcrumb_positions, breadcrumb_colors
                breadcrumb_positions = []
                breadcrumb_colors = []
                update_usd_transform(current_pos, current_yaw)
                if status_label:
                    status_label.text = "Status: RESET & MOVING"
                if vlm_label:
                    vlm_label.text = "VLM: Idle"

            ui.Button("Stop", clicked_fn=on_emergency_stop, style={"background_color": 0xFF2222DD})
            ui.Button("Pause/Play", clicked_fn=on_pause_resume, style={"background_color": 0xFFDD8822})
            ui.Button("Reset", clicked_fn=on_reset, style={"background_color": 0xFF22AA22})

# ============================================================
# 8. VLM DECISION HANDLER & STRATEGY EXECUTION
# ============================================================
def trigger_early_lane_shift():
    """Skips the rest of this lane and begins shifting to the next lane."""
    global state, target_yaw, avoidance_sub_phase, step_start_pos, shifted_distance, sweep_direction_y
    print("[Edge Obstacle]: Obstacle at road boundary. Skipping directly to next lane.")
    
    target_yaw = get_lane_shift_heading()
    avoidance_sub_phase = "IDLE"
    
    # If already facing the shift direction, force a jump to the next lane
    if abs(get_shortest_angle_diff(target_yaw, current_yaw)) < 1.0:
        # Force complete the shift and go to the next sweep lane immediately
        state = "TURNING_2"
        sweep_direction_y = -sweep_direction_y
        target_yaw = 90.0 if sweep_direction_y > 0 else 270.0
        shifted_distance = 0.0
        step_start_pos = list(current_pos)
    else:
        # Otherwise, turn to the shift heading
        state = "TURNING_1"

def trigger_big_lane_skip():
    global state, target_yaw, sweep_direction_y, lane_stuck_count, lane_stuck_pos
    global avoidance_sub_phase, big_skip_target_y
    global big_skip_step_progress, big_skip_advance_progress   # ADD THIS LINE

    print("[BIG SKIP]: Repeated lane-shift deadlock at edge. Driving across to opposite side of road.")

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
    lane_stuck_count = 0
    lane_stuck_pos = None
    big_skip_step_progress = 0.0        # ADD
    big_skip_advance_progress = 0.0     # ADD
    state = "BIG_SKIP_TURN"

    if status_label:
        status_label.text = "Status: Big Skip - Driving to opposite side"

def enter_wide_obstacle_diverge(angle=65.0, step=0.75, reason=""):
    """
    Single entry point for escalating into WIDE_OBSTACLE_DIVERGE.
    If a wide diverge has already been attempted once for this obstacle,
    give up on diverging further and skip the lane instead of repeating it.
    """
    global avoidance_sub_phase, target_yaw, micro_step_target, micro_step_progress
    global state, wide_diverge_count

    if wide_diverge_count >= 1:
        print(f"[Wide Diverge Limit]: Already attempted wide diverge once for this obstacle "
              f"({reason}). Giving up — skipping lane.")
        trigger_early_lane_shift()
        return

    wide_diverge_count += 1
    print(f"[Wide Diverge]: Escalating (attempt {wide_diverge_count}). {reason}")
    avoidance_sub_phase = "WIDE_OBSTACLE_DIVERGE"
    target_yaw = (lane_base_yaw + (detour_side * angle)) % 360.0
    micro_step_target = step
    micro_step_progress = 0.0
    state = "AVOID_ROTATE_THEN_STEP"

def initiate_bypass_sequence():
    global state, avoidance_sub_phase, target_yaw, detour_side, diverge_turn_count
    global wide_diverge_count, lane_base_yaw
    diverge_turn_count = 1
    wide_diverge_count = 0   
    lane_base_yaw = current_yaw

    # Check if road edge is too close on current lane heading to bypass
    dist_to_top = SAFE_MAX_Y - current_pos[1]
    dist_to_bot = current_pos[1] - SAFE_MIN_Y
    if (sweep_direction_y > 0 and dist_to_top < 0.60) or (sweep_direction_y < 0 and dist_to_bot < 0.60):
        trigger_early_lane_shift()
        return

    left_d = _ros_node.depth_left_dist
    ctr_d = _ros_node.depth_center_dist
    right_d = _ros_node.depth_right_dist

    # --- TARGET-FIRST DIVERGE LOGIC ---
    # Always prefer the side that turns the robot toward the destination.
    # Only override with the sensor-clear side if the target-facing side
    # is actually blocked (obstacle-safety takes precedence over routing preference).
    target_angle_diff = get_target_angle_diff(current_pos, current_yaw, TARGET_DESTINATION)
    # diff > 0 -> target requires increasing yaw (CCW) -> that's LEFT in this convention
    preferred_side = 1.0 if target_angle_diff > 0 else -1.0
    preferred_depth = right_d if preferred_side < 0 else left_d
    opposite_depth  = left_d  if preferred_side < 0 else right_d

    MIN_VIABLE_CLEAR = MIN_CHASSIS_CLEAR + 0.2  # 0.60m

    if preferred_depth < MIN_VIABLE_CLEAR and opposite_depth > preferred_depth + 0.3:
        detour_side = -preferred_side
        print(f"[Target-First Diverge]: Target side blocked ({preferred_depth:.2f}m < {MIN_VIABLE_CLEAR:.2f}m). "
            f"Rerouting to open side: {'LEFT' if detour_side > 0 else 'RIGHT'}")
    else:
        detour_side = preferred_side
        print(f"[Target-First Diverge]: Turning toward target -> "
            f"{'LEFT' if detour_side > 0 else 'RIGHT'} (angle_diff={target_angle_diff:.1f}°, "
            f"depth={preferred_depth:.2f}m)")

    # 2D NavMesh Boundary Check (Check both X and Y for predicted detour)
    test_yaw = (lane_base_yaw + (detour_side * DIVERGE_ANGLE_INC)) % 360.0
    fwd_x, fwd_y, _ = get_forward_vector(test_yaw)
    pred_x = current_pos[0] + (fwd_x * 0.6)
    pred_y = current_pos[1] + (fwd_y * 0.6)

    if pred_x < SAFE_MIN_X or pred_x > SAFE_MAX_X or pred_y < SAFE_MIN_Y or pred_y > SAFE_MAX_Y:
        detour_side = -detour_side
        print(f"[NavMesh Safety]: Boundary blocked on chosen side. Re-routing detour to: {'LEFT' if detour_side > 0 else 'RIGHT'}")

    avoidance_sub_phase = "DIVERGE_SCAN"
    target_yaw = (lane_base_yaw + (detour_side * DIVERGE_ANGLE_INC)) % 360.0
    state = "AVOID_INITIAL_TURN"
    if status_label:
        status_label.text = "Status: Executing Bypass Diverge..."


def on_vlm_agent_decision(decision_dict):
    global state, avoidance_sub_phase, target_yaw, current_yaw, micro_step_target, micro_step_progress
    global detour_side, flank_progress, required_flank_distance, initial_detected_distance, diverge_turn_count
    global active_obstacle_category, yield_wait_start_time
    global push_progress, push_target
    global vlm_requested_phase
    global flank_step_progress, flank_step_start_pos   # add this

    # Ignore the callback if the phase changed since the request was sent
    if vlm_requested_phase != avoidance_sub_phase:
        print(f"[VLM Callback Ignored]: phase mismatch (requested {vlm_requested_phase}, current {avoidance_sub_phase})")
        return

    if state in ["STOPPED", "USER_PAUSED"]:
        return

    if avoidance_sub_phase in ["IDLE", "CLASSIFY_OBSTACLE"] and _ros_node.closest_obstacle_dist == float('inf'):
        print(f"[VLM Callback Ignored]: Sensor shows clear (inf), but callback says {category}. Treating as false alarm.")
        return

    cmd = decision_dict.get("command", "standard_bypass")
    category = decision_dict.get("category", active_obstacle_category)
    reasoning = decision_dict.get("reasoning", "")
    obs_visible = decision_dict.get("obstacle_visible", True)
    
    print(f"\n[VLM Callback]: Phase='{avoidance_sub_phase}' | Category='{category}' | Command='{cmd}' | Reason: {reasoning}")
    if vlm_label:
        vlm_label.text = f"VLM: [{category}] -> {cmd}"

    # ONLY abort avoidance if we are in CLASSIFY phase and it was a false alarm
    # ONLY handle false alarm / clear corridor in CLASSIFY phase
    if avoidance_sub_phase == "CLASSIFY_OBSTACLE" and (cmd == "resume_path" or category == "CLEAR" or not obs_visible):
        center_blocked = decision_dict.get("center_blocked", False)
        left_pct = decision_dict.get("left_sector_coverage_pct", 0)
        right_pct = decision_dict.get("right_sector_coverage_pct", 0)

        # Ground-truth check: only trust the VLM's sector claim if the depth
        # sensor actually agrees something is close in that sector.
        depth_center_blocked = _ros_node.depth_center_dist < OBSTACLE_THRESHOLD
        depth_left_blocked   = _ros_node.depth_left_dist   < OBSTACLE_THRESHOLD
        depth_right_blocked  = _ros_node.depth_right_dist  < OBSTACLE_THRESHOLD

        sector_override = (
            (center_blocked and depth_center_blocked) or
            (left_pct >= 40 and depth_left_blocked) or
            (right_pct >= 40 and depth_right_blocked)
        )

        if sector_override:
            print(f"[VLM Override]: VLM said CLEAR but sector analysis AND depth both confirm blockage "
                f"(Ctr={center_blocked}/{_ros_node.depth_center_dist:.2f}m, "
                f"L={left_pct}%/{_ros_node.depth_left_dist:.2f}m, "
                f"R={right_pct}%/{_ros_node.depth_right_dist:.2f}m). Treating as real obstacle.")
            active_obstacle_category = "STATIC_BAR"
            initiate_bypass_sequence()
            return
        elif center_blocked or left_pct >= 40 or right_pct >= 40:
            print(f"[VLM Override Skipped]: VLM claimed blockage but depth sensors disagree "
                f"(Ctr={_ros_node.depth_center_dist:.2f}m, L={_ros_node.depth_left_dist:.2f}m, "
                f"R={_ros_node.depth_right_dist:.2f}m). Trusting depth, proceeding as false alarm.")
        


    # --- 0. INITIAL CLASSIFICATION RESOLUTION ---
    if avoidance_sub_phase == "CLASSIFY_OBSTACLE":
        active_obstacle_category = category

        if category == "BIOLOGICAL" or cmd == "yield_and_wait":
            state = "YIELD_WAITING"
            yield_wait_start_time = time.time()
            yield_wait_total_start_time = time.time()
            
            if status_label:
                status_label.text = "Status: BIOLOGICAL -> Yielding & Waiting"
            print("[Strategy - Yield & Wait]: Animal detected. Stopping and waiting for path to clear...")

        elif category in ["STATIC_BAR", "LARGE_OBJECT"] or cmd == "standard_bypass":
            print(f"[Strategy - Standard Bypass]: Obstacle ({category}) detected. Executing standard bypass...")
            initiate_bypass_sequence()

        elif category == "MINOR_SURFACE" or cmd == "scan_and_proceed":
            state = "SCAN_AND_PROCEED"
            if status_label:
                status_label.text = "Status: MINOR_SURFACE -> Walking Directly Over Feature"
            print("[Strategy - Scan & Proceed]: Surface defect/pothole detected. Continuing directly in lane...")

        else:
            print(f"[Strategy - Default Bypass]: Category '{category}' -> Executing standard bypass...")
            initiate_bypass_sequence()

    # --- 1. DIVERGE PHASE RESOLUTION ---
    elif avoidance_sub_phase == "DIVERGE_SCAN":
        gap_open = decision_dict.get("corridor_gap_open", False)
        primary_cleared = decision_dict.get("primary_obstacle_cleared", False)

        critical_side_depth = _ros_node.depth_left_dist if detour_side > 0 else _ros_node.depth_right_dist
        
        if critical_side_depth < 0.45:
            print(f"[Multi-Obstacle Guard]: Proximity warning to secondary obstacle ({critical_side_depth:.2f}m). Forcing Flank Forward.")
            target_yaw = lane_base_yaw
            state = "AVOID_ALIGN_TO_FLANK"
            return

        if (cmd == "diverge_more" and not primary_cleared) and not gap_open:
            diverge_turn_count += 1

            # NEW: repeated diverge_more decisions mean the small-obstacle
            # incremental turn isn't working — escalate to a wide maneuver.
            if diverge_turn_count > DIVERGE_ESCALATION_LIMIT:
                enter_wide_obstacle_diverge(
                    angle=65.0, step=0.75,
                    reason=f"{diverge_turn_count - 1} consecutive 'diverge_more' in DIVERGE_SCAN."
                )
                return

            max_angle = min(45.0, DIVERGE_ANGLE_INC + (diverge_turn_count * 5.0))
            target_yaw = (lane_base_yaw + (detour_side * max_angle)) % 360.0
            micro_step_target = AVOIDANCE_STEP_SIZE
            micro_step_progress = 0.0
            state = "AVOID_ROTATE_THEN_STEP"
        else:
            print("[Multi-Obstacle Resolution]: Open corridor detected between obstacles. Re-aligning to drive forward.")
            target_yaw = lane_base_yaw
            state = "AVOID_ALIGN_TO_FLANK"
    # --- 1.5 PRE-FLANK FORWARD VERIFICATION WITH SECTOR COVERAGE ---
    elif avoidance_sub_phase == "FLANK_PRE_CHECK":
        current_dist = _ros_node.closest_obstacle_dist
        center_blocked = decision_dict.get("center_blocked", False)
        left_pct = decision_dict.get("left_sector_coverage_pct", 0)
        right_pct = decision_dict.get("right_sector_coverage_pct", 0)

        # Sector threshold evaluation:
        # If detour_side == 1.0 (detoured Left), the car is on our RIGHT.
        # If detour_side == -1.0 (detoured Right), the car is on our LEFT.
        side_blocked = False
        if detour_side > 0:
            # Check Right sector intrusion
            if right_pct >= 40 or _ros_node.depth_right_dist < 0.60:
                side_blocked = True
                print(f"[Sector Check]: Right sector {right_pct}% covered (Depth: {_ros_node.depth_right_dist:.2f}m) >= 40% threshold.")
        else:
            # Check Left sector intrusion
            if left_pct >= 40 or _ros_node.depth_left_dist < 0.60:
                side_blocked = True
                print(f"[Sector Check]: Left sector {left_pct}% covered (Depth: {_ros_node.depth_left_dist:.2f}m) >= 40% threshold.")

        # Diverge if center corridor is blocked, side sector exceeds threshold, or depth is critically close
        if center_blocked or side_blocked or cmd == "diverge_more" or current_dist < 0.50:
            print(f"[Flank Check Failed]: Obstacle intrusion (Ctr={center_blocked}, Left={left_pct}%, Right={right_pct}%). Diverging further...")

            predicted_y = current_pos[1] + (detour_side * 0.75)
            if predicted_y > SAFE_MAX_Y or predicted_y < SAFE_MIN_Y:
                print("[NavMesh Boundary]: Road too narrow to bypass wide vehicle. Skipping lane.")
                trigger_early_lane_shift()
                return

            # CHANGED: was a direct state/phase assignment, now goes through the gatekeeper
            enter_wide_obstacle_diverge(
                angle=65.0, step=0.75,
                reason="FLANK_PRE_CHECK failed (sector coverage / depth)."
            )
            return
        else:
            print(f"[Flank Check Passed]: Corridor clear (Left={left_pct}%, Right={right_pct}% < 40%). Driving flank...")
            avoidance_sub_phase = "FLANK_DRIVE"
            flank_progress = 0.0
            # ========================================================
            # CATEGORY-SPECIFIC FLANK DISTANCE
            # ========================================================

            extra_distance = FLANK_EXTRA_DISTANCE.get(
                active_obstacle_category,
                DEFAULT_FLANK_EXTRA
            )

            minimum_distance = FLANK_MIN_DISTANCE.get(
                active_obstacle_category,
                DEFAULT_FLANK_MIN
            )

            maximum_distance = FLANK_MAX_DISTANCE.get(
                active_obstacle_category,
                DEFAULT_FLANK_MAX
            )

            # Base distance depends on the original detection distance,
            # but category-specific limits prevent small obstacles such
            # as cones from causing a huge unnecessary detour.

            required_flank_distance = max(
                minimum_distance,
                min(
                    maximum_distance,
                    initial_detected_distance
                    + CLEAR_PASS_SAFETY_PAD
                    + extra_distance
                )
            )

            print(
                f"[Flank Distance]: "
                f"Category={active_obstacle_category} | "
                f"Initial Detection={initial_detected_distance:.2f}m | "
                f"Required Flank={required_flank_distance:.2f}m"
            )

            state = "FLANK_STEP"

    # --- 1.6 WIDE OBSTACLE RECOVERY RESOLUTION ---
    elif avoidance_sub_phase == "WIDE_OBSTACLE_DIVERGE":
        if cmd == "diverge_more" and obs_visible:
            # Further divergence required
            target_yaw = (lane_base_yaw + (detour_side * 75.0)) % 360.0
            micro_step_target = 0.50
            micro_step_progress = 0.0
            state = "AVOID_ROTATE_THEN_STEP"
        else:
            # Re-align to lane heading to re-verify corridor
            avoidance_sub_phase = "FLANK_PRE_CHECK"
            target_yaw = lane_base_yaw
            state = "AVOID_ALIGN_TO_FLANK"

    # --- 2. FLANK INWARD INSPECTION RESOLUTION ---
    elif avoidance_sub_phase == "FLANK_INSPECT":
        if cmd == "flank_forward" and obs_visible:
            avoidance_sub_phase = "FLANK_DRIVE"
            target_yaw = lane_base_yaw
            flank_progress = 0.0
            required_flank_distance = 0.50
            state = "AVOID_ALIGN_TO_FLANK"
        else:
            # Obstacle passed -> look back toward original lane
            avoidance_sub_phase = "VERIFY"
            return_steer_side = -1.0 if lateral_detour_ledger > 0.0 else 1.0
            target_yaw = (lane_base_yaw + (return_steer_side * 45.0)) % 360.0
            state = "LOOK_BACK_ROTATE"

    # --- 3. RETURN VERIFICATION RESOLUTION ---
    elif avoidance_sub_phase == "VERIFY":
        if cmd == "return_to_lane" or not obs_visible:
            avoidance_sub_phase = "RETURNING"
            state = "RETURN_DRIVE_TO_LANE"
        else:
            avoidance_sub_phase = "FLANK_DRIVE"
            target_yaw = lane_base_yaw
            flank_progress = 0.0
            required_flank_distance = 0.40
            state = "AVOID_ALIGN_TO_FLANK"

def is_target_x_reached():
    progress_dir = get_progress_dir_x()
    if progress_dir > 0 and current_pos[0] >= TARGET_DESTINATION[0]:
        return True
    if progress_dir < 0 and current_pos[0] <= TARGET_DESTINATION[0]:
        return True
    return False

def zig_zag_update(event):
    global state, saved_pre_pause_state, current_pos, current_yaw, sweep_direction_y
    global step_start_pos, target_yaw, pause_start_time, last_time, shifted_distance, last_print_time
    global lane_base_yaw, lateral_detour_ledger, micro_step_progress, micro_step_target, detour_side
    global avoidance_sub_phase, initial_detected_distance, required_flank_distance, flank_progress, diverge_turn_count
    global llm_avoidance_history, yield_wait_start_time, active_obstacle_category
    global pre_check_settle_start
    global push_progress, push_target
    global yield_wait_total_start_time
    global state, emergency_triggered, emergency_cooldown_start, emergency_trigger_pos
    global backup_progress, avoidance_sub_phase, lateral_detour_ledger
    global deadlock_retreat_count, force_retreat_target
    global wide_diverge_count
    global edge_stuck_time, vlm_requested_phase
    global flank_step_start_pos, flank_step_progress, flank_pause_start_time
    global lane_stuck_count, lane_stuck_pos, big_skip_target_y,lane_base_yaw
    global big_skip_step_progress, big_skip_advance_progress, big_skip_pause_start


    rclpy.spin_once(_ros_node, timeout_sec=0.0)

    if state in ["STOPPED", "USER_PAUSED"]:
        return

    now = time.time()
    dt = min(now - last_time, 0.1)
    last_time = now

    min_dist_ahead = _ros_node.closest_obstacle_dist
    rgb_frame = _ros_node.latest_rgb

    if state not in ["MOVING", "STOPPED", "USER_PAUSED", "EMERGENCY_BACKUP", "VLM_INSPECTING"]:
        if check_emergency_proximity(min_dist_ahead):
            print(f"[EMERGENCY STOP - Global Guard]: {min_dist_ahead:.2f}m while in '{state}'. Backing up.")
            initiate_emergency_backup()
            return

    if sensor_label:
        sensor_label.text = f"Obstacle Dist: {min_dist_ahead:.2f} m" if min_dist_ahead != float('inf') else "Obstacle Dist: Clear (>10m)"

    if ledger_label:
        ledger_label.text = f"Detour Ledger: {lateral_detour_ledger:.2f} m"

    if (now - last_print_time) >= PRINT_INTERVAL:
        last_print_time = now
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        print("-" * 65)
        print(f"[{timestamp_str}] [Telemetry] State: {state} | Cat: {active_obstacle_category} | Phase: {avoidance_sub_phase}")
        print(f"            Pose: ({current_pos[0]:.2f}, {current_pos[1]:.2f}) | Heading: {current_yaw:.1f}\u00b0 | Obstacle: {min_dist_ahead:.2f}m")
        print(f"            Sectors: Left: {_ros_node.depth_left_dist:.2f}m, Ctr: {_ros_node.depth_center_dist:.2f}m, Right: {_ros_node.depth_right_dist:.2f}m")
        print("-" * 65)

    # ============================================================
    # AUTOMATIC RESUME FUNCTION: Triggered ONLY when biological clears
    # ============================================================
    if state == "YIELD_WAITING":
        if min_dist_ahead >= (OBSTACLE_THRESHOLD + 0.40):
            print(f"[Auto Resume]: Path completely clear ({min_dist_ahead:.2f}m). Resuming sweep operations.")
            state = "MOVING"
            avoidance_sub_phase = "IDLE"
            active_obstacle_category = "UNKNOWN"
            lateral_detour_ledger = 0.0
            target_yaw = lane_base_yaw
            current_yaw = lane_base_yaw
            step_start_pos = list(current_pos)
            llm_avoidance_history.clear()
            if status_label:
                status_label.text = "Status: Path Clear -> Resuming Operations"
            update_usd_transform(current_pos, current_yaw)
            return

    # ============================================================
    # 1. NORMAL SWEEP FORWARD & OBSTACLE DETECTION TRIGGER
    # ============================================================
    if state == "MOVING":
        lane_base_yaw = current_yaw

        if compute_dist_2d(current_pos, TARGET_DESTINATION) <= TARGET_RADIUS:
            stop()
            if status_label:
                status_label.text = "Status: DESTINATION REACHED"
            return

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

        if state == "MOVING" and (now - edge_stuck_time) > 3.0:
            if sweep_direction_y > 0 and abs(current_pos[1] - (max_y - margin_y)) < 0.05:
                # stuck at top edge (green sweep)
                target_yaw = get_lane_shift_heading()
                state = "TURNING_1"
                edge_stuck_time = now
                add_breadcrumb(current_pos, color=(0.0, 0.0, 1.0))   # blue
                update_usd_transform(current_pos, current_yaw)
                return
            elif sweep_direction_y < 0 and abs(current_pos[1] - (min_y + margin_y)) < 0.05:
                # stuck at bottom edge (red sweep)
                target_yaw = get_lane_shift_heading()
                state = "TURNING_1"
                edge_stuck_time = now
                add_breadcrumb(current_pos, color=(0.0, 0.0, 1.0))   # blue
                update_usd_transform(current_pos, current_yaw)
                return
        if edge_hit:
            target_yaw = get_lane_shift_heading()
            state = "TURNING_1"
            if status_label:
                status_label.text = "Status: Edge Hit -> Shifting Lane"
            color = (0.0, 0.0, 1.0)   # blue for edge points (or any distinct colour)
            add_breadcrumb(current_pos, color=color)
            update_usd_transform(current_pos, current_yaw)
            return

        # Obstacle detected -> Trigger VLM Classification
        # Target-Directed Center Evade & VLM Trigger
        ctr_d = _ros_node.depth_center_dist
        left_d = _ros_node.depth_left_dist
        right_d = _ros_node.depth_right_dist

        if min_dist_ahead < OBSTACLE_THRESHOLD:
            # ---- EMERGENCY STOP ----
            if min_dist_ahead < EMERGENCY_STOP_DIST:
                # Check cooldown: don't re-trigger on same obstacle too soon
                distance_from_trigger = compute_dist_2d(current_pos, emergency_trigger_pos)
                time_since_trigger = time.time() - emergency_cooldown_start
                
                if emergency_triggered and (distance_from_trigger < EMERGENCY_COOLDOWN_DIST or time_since_trigger < EMERGENCY_COOLDOWN_TIME):
                    # Still in cooldown - we're already backing up or just backed up
                    print(f"[Emergency Cooldown]: Ignoring re-trigger. Dist from trigger: {distance_from_trigger:.2f}m, Time: {time_since_trigger:.1f}s")
                    # Continue with normal obstacle handling (VLM classification) instead of emergency stop
                    # This prevents the back-and-forth loop
                    initial_detected_distance = min_dist_ahead
                    lateral_detour_ledger = 0.0
                    llm_avoidance_history.clear()
                    active_obstacle_category = "STATIC_BAR"
                    print("[Obstacle Detected]: Deterministic STATIC_BAR bypass initiated.")
                    initiate_bypass_sequence()
                    return
                
                # Trigger emergency backup
                print(f"[EMERGENCY STOP]: Obstacle at {min_dist_ahead:.2f}m – too close! Initiating backup.")
                initiate_emergency_backup()
                return

            # ---- Deterministic Classification (No VLM) ----
            # We default to STATIC_BAR and start bypass immediately for speed.
            initial_detected_distance = min_dist_ahead
            lateral_detour_ledger = 0.0
            llm_avoidance_history.clear()
            active_obstacle_category = "STATIC_BAR"
            print("[Obstacle Detected]: Deterministic STATIC_BAR bypass initiated.")
            initiate_bypass_sequence()
            return

        move_forward(speed=FORWARD_SPEED, dt=dt)

        hard_limit_hit = (current_pos[0] > max_x) or (current_pos[0] < min_x)

        if compute_dist_2d(current_pos, TARGET_DESTINATION) <= TARGET_RADIUS or is_target_x_reached() or hard_limit_hit:
            print(f"[Destination Reached]: X={current_pos[0]:.2f} >= {TARGET_DESTINATION[0]:.2f}")
            stop()
            if status_label:
                status_label.text = "Status: DESTINATION REACHED"
            return

        dist_in_step = compute_dist_2d(current_pos, step_start_pos)

        if dist_in_step >= STEP_DISTANCE:
            state = "PAUSED"
            pause_start_time = now
            lane_stuck_count = 0
            lane_stuck_pos = None

            if status_label:
                status_label.text = f"Status: Step Paused ({PAUSE_DURATION}s)"
            # --- ADD BREADCRUMB HERE ---
            color = (0.0, 1.0, 0.0) if sweep_direction_y > 0 else (1.0, 0.0, 0.0)  # green for +Y, red for -Y
            add_breadcrumb(current_pos, color=color)
        update_usd_transform(current_pos, current_yaw)

    # ============================================================
    # 2. STRATEGY EXECUTIONS (Yield & Wait / Scan & Proceed)
    # ============================================================
    elif state == "YIELD_WAITING":
    # Check if we have waited too long
        if (now - yield_wait_total_start_time) >= MAX_YIELD_WAIT:
            print("[Yield Timeout]: Animal still present after 10s. Initiating bypass as if STATIC_BAR.")
            # Override category and start a standard bypass
            active_obstacle_category = "STATIC_BAR"
            lateral_detour_ledger = 0.0
            llm_avoidance_history.clear()
            # Reset timers (they will be overwritten later if needed)
            yield_wait_start_time = 0.0
            yield_wait_total_start_time = 0.0
            initiate_bypass_sequence()   # sets state to AVOID_INITIAL_TURN, etc.
            # No further movement in this frame; update will continue next tick
            return

        # Normal yield check: see if animal cleared
        if (now - yield_wait_start_time) >= YIELD_WAIT_DURATION:
            if min_dist_ahead >= OBSTACLE_THRESHOLD:
                print("[Yield Clear]: Animal moved away. Resuming sweep.")
                state = "MOVING"
                avoidance_sub_phase = "IDLE"
                active_obstacle_category = "UNKNOWN"
                yield_wait_start_time = 0.0
                yield_wait_total_start_time = 0.0
            else:
                # Still blocked â€“ reset short timer for next check, but keep total timer running
                yield_wait_start_time = now
                print("[Yield Waiting]: Animal still present. Continuing wait...")

    # ============================================================
    # 2.5 FALSE ALARM RECOVERY PUSH
    # ============================================================
    elif state == "FALSE_ALARM_PUSH":
        if min_dist_ahead < EMERGENCY_STOP_DIST:
            # Check cooldown before triggering another emergency
            distance_from_trigger = compute_dist_2d(current_pos, emergency_trigger_pos)
            time_since_trigger = time.time() - emergency_cooldown_start
            
            if emergency_triggered and (distance_from_trigger < EMERGENCY_COOLDOWN_DIST or time_since_trigger < EMERGENCY_COOLDOWN_TIME):
                print("[Emergency Cooldown]: Continuing push despite proximity. Cooldown active.")
                # Skip emergency, continue push
                pass
            else:
                print("[EMERGENCY STOP] False-alarm push aborted – obstacle too close.")
                initiate_emergency_backup()
                return

        step_dist = FORWARD_SPEED * 0.6 * dt
        move_forward_step(step_dist, dt)
        push_progress += step_dist

        if push_progress >= push_target:
            state = "MOVING"
            step_start_pos = list(current_pos)
            if status_label:
                status_label.text = "Status: Sweep Resumed"
            print("[False Alarm Recovery]: Push complete. Standard sweeping resumed.")
            # Reset cooldown after successful push
            emergency_triggered = False

        update_usd_transform(current_pos, current_yaw)

    # ============================================================
    # 3. STEP PAUSE & BYPASS SEQUENCES
    # ============================================================
    elif state == "PAUSED":
        if (now - pause_start_time) >= PAUSE_DURATION:
            state = "MOVING"
            step_start_pos = list(current_pos)

    
    elif state == "FLANK_STEP":
        # Move forward by a small increment each frame
        step_dist = FORWARD_SPEED * 0.6 * dt   # same speed as normal steps
        move_forward_step(step_dist, dt)
        flank_step_progress += step_dist
        flank_progress += step_dist

        # Place a breadcrumb when the step completes (every STEP_DISTANCE)
        if flank_step_progress >= STEP_DISTANCE:
            # Ensure we don't overshoot too much
            overshoot = flank_step_progress - STEP_DISTANCE
            flank_step_progress = 0.0
            # Adjust position to exactly STEP_DISTANCE (optional, but can keep)
            # For simplicity, just add a marker
            add_breadcrumb(current_pos, color=(0.0, 1.0, 1.0))   # cyan
            # Enter pause
            state = "FLANK_PAUSED"
            flank_pause_start_time = now
            if status_label:
                status_label.text = "Status: Flank Paused"

        # Edge check during the step
        if (detour_side > 0 and current_pos[1] >= SAFE_MAX_Y - 0.05) or \
        (detour_side < 0 and current_pos[1] <= SAFE_MIN_Y + 0.05):
            print("[Edge Blocked during Flank Step]: Skipping lane.")
            trigger_early_lane_shift()
            return

        update_usd_transform(current_pos, current_yaw)

    elif state == "FLANK_PAUSED":
        if (now - flank_pause_start_time) >= PAUSE_DURATION:   # or shorter e.g. 0.3s
            # Check if we have flanked enough
            if flank_progress >= required_flank_distance:
                # Done flanking – turn to inspect
                avoidance_sub_phase = "FLANK_INSPECT"
                inspect_angle_side = -1.0 if detour_side > 0 else 1.0
                target_yaw = (lane_base_yaw + (inspect_angle_side * 35.0)) % 360.0
                state = "AVOID_ROTATE_TO_INSPECT_FLANK"
                if status_label:
                    status_label.text = "Status: Flank Complete – Inspecting"
            else:
                # Take another step
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

        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] += fwd_x * FORWARD_SPEED * 0.5 * dt
        current_pos[1] += fwd_y * FORWARD_SPEED * 0.5 * dt

        # NEW: bail before we grind into the road edge
        if is_pushing_into_edge(current_pos, detour_side):
            print("[Edge Blocked during Diverge Turn]: Skipping lane instead of diverging into the boundary.")
            trigger_early_lane_shift()
            return

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
            current_pos[0] += fwd_x * 0.20
            current_pos[1] += fwd_y * 0.20

            add_breadcrumb(current_pos, color=(1.0, 1.0, 0.0))

            # NEW: re-check after the push too
            if is_pushing_into_edge(current_pos, detour_side):
                print("[Edge Blocked after Diverge Push]: Skipping lane.")
                trigger_early_lane_shift()
                return

            critical_side_depth = _ros_node.depth_left_dist if detour_side > 0 else _ros_node.depth_right_dist
            if critical_side_depth < 0.60:
                diverge_turn_count += 1
                max_angle = min(45.0, DIVERGE_ANGLE_INC + (diverge_turn_count * 5.0))
                target_yaw = (lane_base_yaw + (detour_side * max_angle)) % 360.0
                micro_step_target = AVOIDANCE_STEP_SIZE
                micro_step_progress = 0.0
                state = "AVOID_ROTATE_THEN_STEP"
            else:
                target_yaw = lane_base_yaw
                state = "AVOID_ALIGN_TO_FLANK"

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

        if check_emergency_proximity(min_dist_ahead):
            print(f"[EMERGENCY STOP - Step Guard]: {min_dist_ahead:.2f}m in '{state}'. Backing up.")
            initiate_emergency_backup()
            return

        # NEW: bail before we grind into the road edge
        if is_pushing_into_edge(current_pos, detour_side):
            print("[Edge Blocked during Diverge Step]: Skipping lane instead of diverging into the boundary.")
            trigger_early_lane_shift()
            return

        if micro_step_progress >= micro_step_target:
            add_breadcrumb(current_pos, color=(1.0, 1.0, 0.0))
            critical_side_depth = _ros_node.depth_left_dist if detour_side > 0 else _ros_node.depth_right_dist
            if critical_side_depth < 0.60:
                diverge_turn_count += 1
                max_angle = min(45.0, DIVERGE_ANGLE_INC + (diverge_turn_count * 5.0))
                target_yaw = (lane_base_yaw + (detour_side * max_angle)) % 360.0
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

        # ONLY proceed once heading is completely aligned to target_yaw
        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            state = "AVOID_FLANK_PRE_CHECK_SETTLE"
            pre_check_settle_start = now
            if status_label:
                status_label.text = "Status: Aligned to Flank. Settling camera..."

        update_usd_transform(current_pos, current_yaw)

    elif state == "AVOID_FLANK_PRE_CHECK_SETTLE":
        # Wait 0.4s to guarantee the ROS camera topic publishes the new 90-degree frame
        if check_emergency_proximity(min_dist_ahead):
            print(f"[EMERGENCY STOP - Step Guard]: {min_dist_ahead:.2f}m in '{state}'. Backing up.")
            initiate_emergency_backup()
            return 
        if (now - pre_check_settle_start) >= 0.4:
            # Deterministic Flank Pre-Check
            center_clear = _ros_node.depth_center_dist > 0.60
            side_clear = _ros_node.depth_right_dist > 0.60 if detour_side > 0 else _ros_node.depth_left_dist > 0.60
            
            if center_clear and side_clear:
                print("[Flank Pre-Check Passed]: Corridor clear. Driving flank...")
                avoidance_sub_phase = "FLANK_DRIVE"
                flank_progress = 0.0
                flank_step_progress = 0.0
                flank_step_start_pos = list(current_pos)
                state = "FLANK_STEP"
            else:
                # Blocked, escalate to wide diverge
                enter_wide_obstacle_diverge(angle=65.0, step=0.75, reason="Deterministic pre-check failed.")

        update_usd_transform(current_pos, current_yaw)

  

    elif state == "AVOID_ROTATE_TO_INSPECT_FLANK":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            if min_dist_ahead < 0.80:
                # Obstacle still alongside, continue flanking
                avoidance_sub_phase = "FLANK_DRIVE"
                target_yaw = lane_base_yaw
                flank_progress = 0.0
                required_flank_distance = 0.50
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
            print("[Target X Reached]: Stopping bypass.")
            stop()
            return

        step_dist = FORWARD_SPEED * 0.8 * dt
        move_forward_step(step_dist, dt)

        if check_emergency_proximity(min_dist_ahead):
            print(f"[EMERGENCY STOP - Step Guard]: {min_dist_ahead:.2f}m in '{state}'. Backing up.")
            initiate_emergency_backup()
            return

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
            deadlock_retreat_count = 0
            state = "MOVING"
            avoidance_sub_phase = "IDLE"
            llm_avoidance_history.clear()
            step_start_pos = list(current_pos)

        update_usd_transform(current_pos, current_yaw)

    # ============================================================
    # EMERGENCY BACKUP STATE
    # ============================================================
    elif state == "EMERGENCY_BACKUP":
        # Back up (move in reverse)
        step_dist = EMERGENCY_BACKUP_SPEED * dt
        backup_progress += step_dist
        
        # Move backward (negative forward direction)
        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] -= fwd_x * step_dist
        current_pos[1] -= fwd_y * step_dist
        
        if backup_progress >= EMERGENCY_BACKUP_DIST:
            print("[Emergency Backup]: Backup complete. Re-evaluating obstacle.")
            
            # Check if obstacle is still too close after backing up
            min_dist_ahead = _ros_node.closest_obstacle_dist
            
            if min_dist_ahead < OBSTACLE_THRESHOLD:
                deadlock_retreat_count += 1
                print(f"[Deadlock Counter]: Attempt {deadlock_retreat_count}/2. Obstacle still at {min_dist_ahead:.2f}m.")
                
                if deadlock_retreat_count >= 2:
                    print("[DEADLOCK RESOLVED]: Backing off to lane and skipping current lane.")
                    deadlock_retreat_count = 0
                    emergency_triggered = False
                    backup_progress = 0.0
                    state = "FORCE_RETREAT_BACK"
                else:
                    emergency_triggered = False
                    state = "VLM_INSPECTING"
                    avoidance_sub_phase = "CLASSIFY_OBSTACLE"
                    if status_label:
                        status_label.text = "Status: Reassessing Obstacle with VLM..."
                    # REPLACED: Run VLM ONLY HERE (after first backup)
                    print("[Emergency Backup]: Calling VLM to reassess obstacle after backup...")
                    async_query_vlm_agent(rgb_frame, min_dist_ahead, "CLASSIFY_OBSTACLE", lateral_detour_ledger, on_vlm_agent_decision)
            else:
                print(f"[Emergency Backup]: Path cleared after backup ({min_dist_ahead:.2f}m >= {OBSTACLE_THRESHOLD:.2f}m). Resuming.")
                deadlock_retreat_count = 0
                emergency_triggered = False
                backup_progress = 0.0
                state = "MOVING"
                if status_label:
                    status_label.text = "Status: MOVING"
        
        update_usd_transform(current_pos, current_yaw)

    elif state == "VLM_INSPECTING":
        # Waiting stationary for VLM decision
        update_usd_transform(current_pos, current_yaw)


    elif state == "FORCE_RETREAT_BACK":
    # Back up straight (reverse) until we are well clear of the car's wedge
        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] -= fwd_x * FORWARD_SPEED * dt
        current_pos[1] -= fwd_y * FORWARD_SPEED * dt
        backup_progress += FORWARD_SPEED * dt
        
        # Once we have backed up enough, force skip the lane
        if backup_progress >= force_retreat_target:
            print("[Force Retreat Complete]: Skipping to next lane.")
            target_yaw = get_lane_shift_heading()
            avoidance_sub_phase = "IDLE"
            state = "TURNING_1"
            emergency_triggered = False
            step_start_pos = list(current_pos)
        
        update_usd_transform(current_pos, current_yaw)

    # ============================================================
    # 4. SWEEP TURN ROUTINES
    # ============================================================
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
            print("[Target X Reached]: Stopping lane shift.")
            stop()
            return

        if _ros_node.closest_obstacle_dist < 0.60:
            if lane_stuck_pos is not None and compute_dist_2d(current_pos, lane_stuck_pos) < LANE_STUCK_POS_TOL:
                lane_stuck_count += 1
            else:
                lane_stuck_count = 1
            lane_stuck_pos = list(current_pos)

            print(f"[LANE_SHIFT BLOCKED]: Obstacle detected during shift. Stuck count={lane_stuck_count}/{LANE_STUCK_THRESHOLD}.")

            if lane_stuck_count >= LANE_STUCK_THRESHOLD:
                trigger_big_lane_skip()
                return

            state = "TURNING_2"
            sweep_direction_y = -sweep_direction_y
            target_yaw = 90.0 if sweep_direction_y > 0 else 270.0
            shifted_distance = 0.0
            step_start_pos = list(current_pos)
            return

        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        dist = FORWARD_SPEED * dt
        current_pos[0] += fwd_x * dist
        current_pos[1] += fwd_y * dist
        shifted_distance += dist

        if shifted_distance >= LANE_SHIFT:
            state = "TURNING_2"
            sweep_direction_y = -sweep_direction_y
            target_yaw = 90.0 if sweep_direction_y > 0 else 270.0
            lane_stuck_count = 0
            lane_stuck_pos = None

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
            color = (0.0, 0.0, 1.0)   # blue for edge points (or any distinct colour)
            add_breadcrumb(current_pos, color=color)
        update_usd_transform(current_pos, current_yaw)

    elif state == "BIG_SKIP_TURN":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            state = "BIG_SKIP_DRIVE"

        update_usd_transform(current_pos, current_yaw)

    elif state == "BIG_SKIP_DRIVE":
        step_dist = FORWARD_SPEED * 0.6 * dt
        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] += fwd_x * step_dist
        current_pos[1] += fwd_y * step_dist
        big_skip_step_progress += step_dist

        if check_emergency_proximity(min_dist_ahead):
            print(f"[EMERGENCY STOP - Step Guard]: {min_dist_ahead:.2f}m in '{state}'. Backing up.")
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
            add_breadcrumb(current_pos, color=(0.6, 0.0, 1.0))  # purple: big-skip edge reached
            # Turn toward the target's forward direction to advance into a FRESH lane
            target_yaw = get_lane_shift_heading()
            state = "BIG_SKIP_ADVANCE_TURN"
            if status_label:
                status_label.text = "Status: Big Skip Edge Reached - Turning to Advance Lane"
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
        step_dist = FORWARD_SPEED * 0.6 * dt
        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] += fwd_x * step_dist
        current_pos[1] += fwd_y * step_dist
        big_skip_step_progress += step_dist
        big_skip_advance_progress += step_dist

        if check_emergency_proximity(min_dist_ahead):
            print(f"[EMERGENCY STOP - Step Guard]: {min_dist_ahead:.2f}m in '{state}'. Backing up.")
            initiate_emergency_backup()
            return

        if big_skip_advance_progress >= LANE_SHIFT:
            add_breadcrumb(current_pos, color=(0.6, 0.0, 1.0))
            # Advance complete -> turn to resume sweeping in the fresh lane
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

# Register update loop
global _phantomx_sub
_phantomx_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(zig_zag_update) 