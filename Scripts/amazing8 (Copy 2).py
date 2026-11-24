

"""
PhantomX Target-Directed Controller with Class-Based VLM Obstacle Avoidance
+ Continuous Stress ("Cortisol") System
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
6. STRESS SYSTEM (NEW): A continuous 0-10 "stress" value is derived from obstacle
    proximity (depth) and VLM-perceived threat, smoothed via exponential approach
    (no instant jumps), and locked from decreasing while actively engaged with an
    obstacle (cortisol doesn't drop until the encounter is actually over). The
    resulting value is bucketed into LOW / MEDIUM / HIGH bands with hysteresis,
    which gate (Low), partially override (Medium), or fully override (High) the
    category-driven avoidance strategy above -- including robot speed and how
    often the LLM is re-consulted mid-maneuver. A "potential" metric (speed x
    band urgency) is plotted alongside stress on a live strip chart in the UI.
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
import onnxruntime as ort
import cv2
from pxr import Usd, UsdGeom, Gf, Sdf, Semantics



stage = omni.usd.get_context().get_stage()

# Map each obstacle prim path to its class name
LABELS = {
    "/World/Cow_01":       "cow",
    "/World/Cow_02":       "cow",       
    "/World/Cone_01":      "cone",
    "/World/Cone_02":      "cone",
    "/World/Person_01":    "person",
    "/World/Car_01":       "car",
    # ... add every obstacle instance
}

for path, label in LABELS.items():
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        print(f"SKIP (not found): {path}")
        continue
    sem = Semantics.SemanticsAPI.Apply(prim, "class")
    sem.CreateSemanticTypeAttr().Set("class")
    sem.CreateSemanticDataAttr().Set(label)
    print(f"Labeled {path} -> {label}")

# Cleanup prior loop instances on script reload
if globals().get("_phantomx_sub"):
    try:
        globals()["_phantomx_sub"].unsubscribe()
    except Exception:
        pass
    globals()["_phantomx_sub"] = None

if globals().get("_yolo_generation") is not None:
    globals()["yolo_thread_running"] = False
    globals()["_yolo_generation"] += 1

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

ROBOT_LENGTH = 0.55
DIVERGE_CLEARANCE_MARGINE = 0.20
MIN_DIVERGE_STEP = ROBOT_LENGTH + DIVERGE_CLEARANCE_MARGINE

STEP_DISTANCE      = 0.4                             # Sweep pause step size (m)
PAUSE_DURATION     = 0.5                             # Sweep pause duration (s)
FORWARD_SPEED      = 0.5                            # Base translation speed (m/s)
TURN_SPEED         = 180.0                           # Base rotation speed (deg/s)
LANE_SHIFT         = 0.5                             # Lane shift width (m)
OBSTACLE_THRESHOLD = 0.8                            # Avoidance trigger distance (m)
MIN_CHASSIS_CLEAR  = 0.40                            # Minimum lateral distance to clear chassis (m)
PRINT_INTERVAL     = 2.0                             # Telemetry print interval (s)
YIELD_WAIT_DURATION = 3.0                            # Time in seconds to wait before checking if animal moved
MAX_YIELD_WAIT = 10.0            # Maximum seconds to wait for a biological to clear

# Avoidance micro-step settings
AVOIDANCE_STEP_SIZE   = MIN_DIVERGE_STEP 
MIN_DIVERGE_STEP = ROBOT_LENGTH + DIVERGE_CLEARANCE_MARGINE
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

MEMORY_MATCH_RADIUS = 0.8          # meters — how close a new detection must be to a cached entry to reuse it
BIOLOGICAL_MEMORY_EXPIRY = 30.0    # seconds — biological cache entries older than this are re-verified

EMERGENCY_STOP_DIST = 0.25        # If obstacle closer than this, stop immediately
EMERGENCY_BACKUP_DIST = 0.3        # This is no longer used, keep it for reference
EMERGENCY_BACKUP_CLEAR_DIST = MIN_DIVERGE_STEP   # NEW: Back up until obstacle is at least this far away
EMERGENCY_BACKUP_MAX_DIST = 2.5    # NEW: Safety cap - if it backs up this much without clearing, give up

EMERGENCY_BACKUP_STUCK_TIME = 5.0
EMERGENCY_ESCAPE_DIST = 0.6
EMERGENCY_MAX_ESCAPE_TURNS  = 1     # escape attempts per emergency
EMERGENCY_ESCAPE_MIN_ROOM = 0.30   # m of free travel required in the chosen backing direction
EMERGENCY_WALL_PIN_TIME   = 1.5    # s clamped by a boundary while backing => treat as stuck
                                   # (set equal to EMERGENCY_BACKUP_STUCK_TIME to keep pure 5s)


DIVERGE_GRACE_PERIOD = 0.6     # seconds — protects a deliberate diverge from being instantly undone
CRITICAL_TOUCH_DIST = 0.18     # even during grace, still back up if truly about to collide
# near other globals

REPEATED_BACKUP_RADIUS = 0.5
REPEATED_BACKUP_LIMIT = 2

MEMORY_MATCH_RADIUS = 0.8          # meters — how close a new detection must be to a cached entry to reuse it
MOBILE_MEMORY_EXPIRY = 30.0        # seconds — cache entries for MOVING obstacles are re-verified after this long
# (STATIONARY obstacles are cached indefinitely — no expiry)

# Risk-score modulation (0-10 scale from VLM)
RISK_FLANK_MULTIPLIER_RANGE = (0.75, 1.35)   # at risk=0 -> 0.75x flank distance, at risk=10 -> 1.35x
RISK_YIELD_MULTIPLIER_RANGE = (0.70, 1.40)   # same idea for yield wait duration

YOLO_CONF_THRESHOLD = 0.45
YOLO_NMS_IOU = 0.5
YOLO_INPUT_SIZE = (640, 640)
TRACK_MAX_AGE = 30          # frames before a track is dropped

ATTACK_WEIGHT = 1.5          # scales the attack term into 0-10 stress
ATTACK_CLOSING_THRESHOLD = 0.8   # m/s — below this, not an "attack"
ATTACK_TTI_THRESHOLD = 1.5       # seconds to impact — below this, attack suspected
ATTACK_AREA_GROWTH_THRESHOLD = 0.35  # 35% bbox growth in one frame

VLM_PRIORITY_MODE = True   # halt on every new obstacle and wait for the VLM reply
VLM_TIMEOUT = 90           # was 360: the robot would sit up to 6 min before falling back

MANEUVER_SPEED_FACTOR = {
    "LOW":    0.60,
    "MEDIUM": 0.60,
    "HIGH":   1.10,   # actually escape
}

# ============================================================
# STRESS / CORTISOL SIMULATION SYSTEM
# ============================================================
# Design summary (see accompanying discussion):
#   target_stress = max(PROXIMITY_WEIGHT * proximity_term, THREAT_WEIGHT * threat_term)
#   stress += (effective_target - stress) * STRESS_SMOOTHING_RATE * dt      [exponential approach]
#   Decreases in `stress` are LOCKED while an obstacle encounter is actively in
#   progress (avoidance_sub_phase != IDLE) -- cortisol only tapers once the
#   encounter fully resolves. This also prevents band flapping mid-maneuver.
#   Band membership uses hysteresis (different up/down thresholds) so dwelling
#   near a boundary doesn't cause chatter.
#   potential = (FORWARD_SPEED * band_speed_multiplier) * band_urgency_multiplier
#   -- decouples "how fast" from "how urgent" so a fast Low cruise and a fast
#   High-band active retreat land in different places on the (time, potential)
#   strip chart.
# All numeric constants below are reasonable starting points, NOT validated
# values -- calibrate STRESS_SMOOTHING_RATE and the band thresholds from
# logged `stress_level` traces across representative test runs (cone, passive
# animal, approaching animal, parked car, etc.) rather than trusting them as-is.
# ============================================================
STRESS_MAX = 10.0
STRESS_MIN = 0.0

# Weight applied to each normalized (0-10) input term before taking the max.
# Both terms are pre-normalized to 0-10, so start both weights at 1.0 and only
# adjust if testing shows one input dominating more than it should.
PROXIMITY_WEIGHT = 1.0
THREAT_WEIGHT = 1.0

# Proximity term maps OBSTACLE_THRESHOLD (0 stress) -> 0m (max stress) onto 0-10.
# Reuses the existing, already-tuned OBSTACLE_THRESHOLD rather than a new constant.
PROXIMITY_ZERO_STRESS_DIST = OBSTACLE_THRESHOLD

# Extra threat bump added to VLM risk_score when the obstacle is judged to be
# actively approaching / agitated (currently_stationary == False) rather than
# merely present.
THREAT_AGGRESSION_BUMP = 2.0

# Exponential-approach smoothing rate (1/seconds). Higher = snaps to target
# faster. ~2.0 gives roughly a 1-1.5s visible rise to a sudden high target.
# Tune from logged runs to hit whatever "feels" like a realistic hormone ramp.
STRESS_SMOOTHING_RATE = 2.0

# Band thresholds WITH hysteresis: the up-crossing point differs from the
# down-crossing point so the band doesn't flap when stress hovers near a
# boundary. (On top of this, decreases are also fully locked during an active
# encounter -- see STRESS SYSTEM CORE section below.)
STRESS_LOW_MEDIUM_UP     = 4.0   # crossing up out of LOW into MEDIUM
STRESS_LOW_MEDIUM_DOWN   = 3.0   # crossing down out of MEDIUM into LOW
STRESS_MEDIUM_HIGH_UP    = 7.0   # crossing up out of MEDIUM into HIGH
STRESS_MEDIUM_HIGH_DOWN  = 5.5   # crossing down out of HIGH into MEDIUM

# Per-band speed multiplier RANGE (continuous scaling within the band, based
# on where stress sits inside that band's span -- same pattern as the
# existing scale_by_risk() usage elsewhere in this script).
# Note HIGH's range is NOT "slower" -- an active retreat should move fast.
BAND_SPEED_MULTIPLIER_RANGE = {
    "LOW":    (1.00, 1.00),   # full speed throughout Low
    "MEDIUM": (0.75, 0.50),   # progressively cautious as stress climbs within Medium
    "HIGH" : (1.00, 1.25),
}

# Per-band urgency multiplier for `potential` -- this is what actually
# distinguishes "fast because routine" (Low) from "fast because fleeing" (High)
# on the strip chart, since raw speed alone can't tell those apart.
BAND_URGENCY_MULTIPLIER = {
    "LOW":    0.15,
    "MEDIUM": 0.55,
    "HIGH":   1.20,
}

# While in HIGH band, allow re-querying the VLM more often than the normal
# MIN_VLM_COOLDOWN (which is tuned for routine, non-urgent classification).
HIGH_STRESS_VLM_COOLDOWN = 4.0

STRESS_HISTORY_MAXLEN = 300   # strip-chart sample buffer length

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

VLM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
VLM_API_KEY = os.getenv("LM_STUDIO_API_KEY", "sk-lm-xxxx")
VLM_MODEL  = "qwen3.5-9b" 
#
#VLM_API_URL = "http://127.0.0.1:8083/v1/chat/completions"
#VLM_API_KEY = "your-secret-api-key"
#VLM_MODEL   = "mimo-v2.6-flash-free"  

VLM_TIMEOUT     = 360       # Max seconds to wait for model response
VLM_MAX_RETRIES = 0                 # 0 retries recommended for local LLM
VLM_MAX_TOKENS  = 3000            # Max tokens to generate


AUTO_COLLECT_ENABLED    = True
AUTO_COLLECT_DIR        = "/home/user/Desktop/llmforhexapod/dataset/auto_collect"
AUTO_COLLECT_MIN_INTERVAL = 1.0   # seconds -- rate limit so one obstacle can't flood disk


RGB_TOPICS   = ["/rgb", "/camera/rgb", "/phantomx/camera/rgb"]
DEPTH_TOPICS = ["/depth", "/camera/depth", "/phantomx/camera/depth"]

AUTO_LOOP_ENABLED = True
AUTO_LOOP_DELAY   = 2.0   # seconds to sit at "finished" before restarting

_loop_restart_pending = False
_loop_restart_time = 0.0


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

def room_along(pos, yaw_deg, travel_dir=-1.0):
    """Free distance before hitting the SAFE_* box, moving along travel_dir*forward(yaw).
    X and Y are checked independently. travel_dir=-1 means backing up."""
    fx, fy, _ = get_forward_vector(yaw_deg)
    vx, vy = fx * travel_dir, fy * travel_dir
    room = float('inf')
    if vx > 1e-6:    room = min(room, (SAFE_MAX_X - pos[0]) / vx)
    elif vx < -1e-6: room = min(room, (SAFE_MIN_X - pos[0]) / vx)
    if vy > 1e-6:    room = min(room, (SAFE_MAX_Y - pos[1]) / vy)
    elif vy < -1e-6: room = min(room, (SAFE_MIN_Y - pos[1]) / vy)
    return max(0.0, room)

def pick_escape_yaw():
    """Returns (yaw, room) for the heading whose REAR direction has the most free room,
    or (None, 0). Only headings that back AWAY from the obstacle are tried."""
    best_yaw, best_room = None, 0.0
    for offset in (90.0, -90.0, 45.0, -45.0):
        cand = (current_yaw + offset) % 360.0
        room = min(room_along(current_pos, cand), EMERGENCY_ESCAPE_DIST * 1.5)
        if room > best_room + 1e-6:
            best_room, best_yaw = room, cand
    if best_room < EMERGENCY_ESCAPE_MIN_ROOM:
        return None, 0.0
    return best_yaw, best_room
# Section 4 additions — spatial memory helpers
def estimate_obstacle_world_pos(pos, yaw_deg, dist):
    """Estimate the world (x, y) of an obstacle currently in front of the robot."""
    fwd_x, fwd_y, _ = get_forward_vector(yaw_deg)
    return (pos[0] + fwd_x * dist, pos[1] + fwd_y * dist)

def find_cached_obstacle(world_pos):
    """Return the matching known_obstacles entry within MEMORY_MATCH_RADIUS, or None.
    Non-stationary entries older than MOBILE_MEMORY_EXPIRY are treated as stale (ignored) —
    stationary entries are cached indefinitely."""
    now = time.time()
    for entry in known_obstacles:
        if entry.get("category") in ("UNKNOWN", "", None):
            continue
        d = compute_dist_2d(world_pos, entry["pos"])
        if d <= MEMORY_MATCH_RADIUS:
            if not entry.get("stationary", True) and (now - entry["last_seen"]) > MOBILE_MEMORY_EXPIRY:
                continue  # stale mobile sighting — don't trust it, fall through to re-classify
            return entry
    return None

def maneuver_speed_factor():
    base = MANEUVER_SPEED_FACTOR.get(stress_band, 0.6)
    if stress_band == "HIGH" and _ros_node.closest_obstacle_dist < 1.0:
        return min(base, 0.75)   # don't rush while still close to the obstacle itself
    return base


def find_confident_yolo_match(obstacle_dist):
    """Look for a YOLO track that has been classified as the SAME class for
    YOLO_CONFIDENCE_HITS_REQUIRED+ consecutive frames, at high confidence,
    with a depth reading close to the depth-sensor's detected obstacle
    distance. Returns a dict ready to feed into on_vlm_agent_decision, or None
    if nothing confident enough was found (caller should fall back to VLM)."""
    if not YOLO_FASTPATH_ENABLED or yolo_session is None:
        return None

    best = None
    best_diff = YOLO_MATCH_DIST_TOLERANCE
    with yolo_lock:
        for t in yolo_tracks:
            cls_name = t.get("class_name")
            if cls_name not in YOLO_CLASS_TO_CATEGORY:
                continue
            if t.get("class_hits", 0) < YOLO_CONFIDENCE_HITS_REQUIRED:
                continue
            if t.get("confidence", 0.0) < YOLO_MIN_CONFIDENCE:
                continue
            t_dist = t.get("distance", float('inf'))
            if t_dist == float('inf'):
                continue
            diff = abs(t_dist - obstacle_dist)
            if diff < best_diff:
                best_diff = diff
                best = t

    if best is None:
        return None

    category = YOLO_CLASS_TO_CATEGORY[best["class_name"]]
    return {
        "category": category,
        "class_name": best["class_name"],
        "confidence": best.get("confidence", 0.0),
        "risk_score": YOLO_DEFAULT_RISK.get(category, 5),
        "currently_stationary": category != "BIOLOGICAL",  # conservative; VLM refines this
        "track_id": best.get("track_id"),
    }

def scale_by_risk(base_value, risk_score, multiplier_range):
    """Linearly map risk_score (0-10) to a multiplier within multiplier_range, applied to base_value."""
    lo, hi = multiplier_range
    risk_norm = max(0, min(10, risk_score)) / 10.0
    multiplier = lo + (hi - lo) * risk_norm
    return base_value * multiplier

def record_obstacle_memory(world_pos, category, risk_score, stationary):
    """Cache or refresh a classified obstacle's world position, category, risk, and mobility."""
    now = time.time()
    for entry in known_obstacles:
        if compute_dist_2d(world_pos, entry["pos"]) <= MEMORY_MATCH_RADIUS:
            entry["category"] = category
            entry["risk_score"] = risk_score
            entry["stationary"] = stationary
            entry["last_seen"] = now
            return
    known_obstacles.append({
        "pos": world_pos, "category": category, "risk_score": risk_score,
        "stationary": stationary, "last_seen": now
    })
    print(f"[Memory] Cached new obstacle: {category} (risk={risk_score}, stationary={stationary}) "
        f"at ({world_pos[0]:.2f}, {world_pos[1]:.2f})")
# Global list to track breadcrumb positions
breadcrumb_counter = 0
BREADCRUMB_SIZE = 0.15      # width/depth of each plate (m)
BREADCRUMB_THICKNESS = 0.01 # keep it thin so it sits flush on the ground

def add_breadcrumb(pos, color=(0.0, 1.0, 0.0)):
    global breadcrumb_counter
    stage = omni.usd.get_context().get_stage()

    prim_path = f"/World/Breadcrumbs/plate_{breadcrumb_counter}"
    breadcrumb_counter += 1

    # If a prim already lives at this path (script reload, stale stage, etc.),
    # nuke it first so Define() gives us a genuinely fresh prim.
    existing = stage.GetPrimAtPath(prim_path)
    if existing.IsValid():
        stage.RemovePrim(prim_path)

    plate = UsdGeom.Cube.Define(stage, prim_path)
    plate.CreateSizeAttr(2.5)

    xform = UsdGeom.Xformable(plate)

    # Get-or-create the ops instead of blindly Add-ing them.
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
# YOLO MODEL
# ============================================================
def init_yolo(model_path):
    global yolo_session
    providers = ['CPUExecutionProvider']
    yolo_session = ort.InferenceSession(model_path, providers=providers)
    print(f"[YOLO] Model loaded: {model_path}")
    print(f"[YOLO] Providers: {yolo_session.get_providers()}")

_yolo_generation = globals().get("_yolo_generation", 0)

def start_yolo_worker():
    global _yolo_generation, yolo_thread_running
    _yolo_generation += 1              # invalidates any older worker
    yolo_thread_running = True
    threading.Thread(target=yolo_worker, args=(_yolo_generation,), daemon=True).start()

def shutdown_yolo_worker():
    global yolo_thread_running, _yolo_generation
    yolo_thread_running = False
    _yolo_generation += 1

def preprocess_yolo(rgb_bgr):
    """Letterbox resize + normalize to (1,3,640,640) float32."""
    h, w = rgb_bgr.shape[:2]
    scale = min(YOLO_INPUT_SIZE[0] / w, YOLO_INPUT_SIZE[1] / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(rgb_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((YOLO_INPUT_SIZE[1], YOLO_INPUT_SIZE[0], 3), 114, dtype=np.uint8)
    dx = (YOLO_INPUT_SIZE[0] - nw) // 2
    dy = (YOLO_INPUT_SIZE[1] - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0   # BGR->RGB, normalize
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]  # HWC -> 1CHW
    return blob, scale, dx, dy


# Class names — must match your custom.yaml
YOLO_CLASS_NAMES = {
    0: "cow", 1: "person", 2: "cone", 3: "car", 4: "pothole"
}
# ============================================================
# YOLO FAST-PATH CLASSIFICATION
# ============================================================
YOLO_CLASS_TO_CATEGORY = {
    "cow":     "BIOLOGICAL",
    "person":  "BIOLOGICAL",
    "cone":    "STATIC_BAR",
    "car":     "STATIC_BAR",
    "pothole": "MINOR_SURFACE",
}

# Conservative defaults used ONLY until the background VLM refine call lands
# (or times out) -- YOLO can't judge "how agitated does this animal look",
# so BIOLOGICAL starts at a cautious mid-risk rather than trusting YOLO fully.
YOLO_DEFAULT_RISK = {
    "BIOLOGICAL":    6,
    "STATIC_BAR":    3,
    "MINOR_SURFACE": 1,
}

YOLO_CONFIDENCE_HITS_REQUIRED = 5     # consecutive frames of the SAME class on the SAME track
YOLO_MIN_CONFIDENCE           = 0.60  # per-detection confidence floor to trust the fast-path
YOLO_MATCH_DIST_TOLERANCE     = 0.6   # meters -- how close YOLO's depth-slice distance must be
                                       # to the depth-sensor obstacle distance to call it a match
YOLO_FASTPATH_ENABLED         = True

def yolo_worker():
    """Runs continuously in a daemon thread."""
    global yolo_tracks, yolo_thread_running, next_track_id, track_memory
    yolo_thread_running = True
    print("[YOLO Worker] Started.")

    last_frame_time = time.time()   
    last_dbg = 0.0

    while yolo_thread_running:
        if state in ("STOPPED", "USER_PAUSED"):
            time.sleep(0.1)
            last_frame_time = time.time()   # avoid a huge dt_frame on resume
            continue


        rgb = _ros_node.latest_rgb
        if rgb is None or yolo_session is None:
            time.sleep(0.02)
            continue

        try:
            now_frame = time.time()
            dt_frame = now_frame - last_frame_time      
            last_frame_time = now_frame                  
            if dt_frame <= 0:
                dt_frame = 1e-3
            blob, scale, dx, dy = preprocess_yolo(rgb)
            input_name = yolo_session.get_inputs()[0].name
            outputs = yolo_session.run(None, {input_name: blob})
            dets = outputs[0]  # (1, 300, 6) for end2end, or (1, 84, 8400) for raw

            # ---- Decode (end2end assumed) ----
            raw = dets[0]  # (300, 6)
            frame_dets = []
            for row in raw:
                conf = float(row[4])
                if conf < YOLO_CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = [float(v) for v in row[:4]]
                cls_id = int(row[5])
                # Undo letterbox
                x1 = (x1 - dx) / scale
                y1 = (y1 - dy) / scale
                x2 = (x2 - dx) / scale
                y2 = (y2 - dy) / scale
                frame_dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": YOLO_CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                })

            # ---- Centroid tracker (IoU-free, fast) ----
            new_memory = {}
            for det in frame_dets:
                bx1, by1, bx2, by2 = det["bbox"]
                cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                best_id, best_dist = None, 120.0   # max match distance in pixels
                for tid, mem in track_memory.items():
                    mcx, mcy = mem["centroid"]
                    d = math.hypot(cx - mcx, cy - mcy)
                    if d < best_dist:
                        best_dist, best_id = d, tid
                if best_id is not None:

                    prev_class = track_memory[best_id].get("class_name")
                    # Streak resets whenever the predicted class flips -- a track that keeps
                    # flip-flopping between "cone" and "car" never builds enough confidence
                    # to trigger the fast-path, which is the point.
                    class_hits = (track_memory[best_id].get("class_hits", 0) + 1
                                if prev_class == det["class_name"] else 1)
                    track_memory[best_id]["centroid"] = (cx, cy)
                    track_memory[best_id]["bbox"] = det["bbox"]
                    track_memory[best_id]["age"] = 0
                    track_memory[best_id]["class_name"] = det["class_name"]
                    track_memory[best_id]["class_hits"] = class_hits
                    track_memory[best_id]["confidence"] = det["confidence"]
                    det["track_id"] = best_id
                    det["class_hits"] = class_hits
                    new_memory[best_id] = track_memory[best_id]
                else:
                    det["track_id"] = next_track_id
                    det["class_hits"] = 1
                    new_memory[next_track_id] = {
                        "centroid": (cx, cy), "bbox": det["bbox"],
                        "age": 0, "class_name": det["class_name"],
                        "class_hits": 1, "confidence": det["confidence"]
                    }
                    next_track_id += 1

            # Age out old tracks
            for tid, mem in track_memory.items():
                if tid not in new_memory:
                    mem["age"] += 1
                    if mem["age"] <= TRACK_MAX_AGE:
                        new_memory[tid] = mem
            track_memory = new_memory

            # ---- Compute per-track kinematics ----
            with yolo_lock:
                prev_tracks = {t["track_id"]: t for t in yolo_tracks}
                h, w = rgb.shape[:2]
                for det in frame_dets:
                    prev = prev_tracks.get(det["track_id"])
                    bx1, by1, bx2, by2 = det["bbox"]
                    cx = (bx1 + bx2) / 2
                    if cx < w * 0.33:
                        det["distance"] = _ros_node.depth_left_dist
                    elif cx > w * 0.66:
                        det["distance"] = _ros_node.depth_right_dist
                    else:
                        det["distance"] = _ros_node.depth_center_dist

                    det["area_growth"] = 0.0
                    det["closing_velocity"] = 0.0
                    if prev is not None:
                        pa = (prev["bbox"][2] - prev["bbox"][0]) * (prev["bbox"][3] - prev["bbox"][1])
                        ca = (bx2 - bx1) * (by2 - by1)
                        if pa > 1:
                            det["area_growth"] = (ca - pa) / pa
                        d0, d1 = prev.get("distance", float('inf')), det["distance"]
                        if d0 != float('inf') and d1 != float('inf'):
                            det["closing_velocity"] = max(0.0, (d0 - d1) / dt_frame)
                yolo_tracks = frame_dets

            if time.time() - last_dbg > 1.0:      # throttled debug: once per second
                last_dbg = time.time()
                if frame_dets:
                    top = max(frame_dets, key=lambda d: d["confidence"])
                    print(f"[YOLO Debug] {len(frame_dets)} dets | best: {top['class_name']} "
                        f"conf={top['confidence']:.2f} hits={top.get('class_hits')} dist={top['distance']:.2f}")
                else:
                    print("[YOLO Debug] 0 detections")
        except Exception as e:
            print(f"[YOLO Worker Error]: {e}")

        time.sleep(0.033)  # ~30 FPS cap# Start after model is exported
YOLO_ONNX_PATH = "/home/user/Desktop/llmforhexapod/dataset/runs/detect/train-2/weights/best.onnx"



# ============================================================
# TRAUMA SECTION
# ============================================================

def record_trauma(world_pos, category, max_closing_velocity):
    """Called when an attack encounter ends."""
    if category in ("UNKNOWN", "", None):
        print(f"[TRAUMA]: Skipping — category still UNKNOWN at ({world_pos[0]:.2f}, {world_pos[1]:.2f}), not recording.")
        return
    trauma_memory.append({
        "pos": world_pos,
        "category": category,
        "closing_velocity": max_closing_velocity,
        "timestamp": time.time(),
    })
    print(f"[TRAUMA]: Recorded attack by '{category}' at "
        f"({world_pos[0]:.2f}, {world_pos[1]:.2f}), v={max_closing_velocity:.2f}m/s")

    # Also write into known_obstacles with traumatic flag
    known_obstacles.append({
        "pos": world_pos,
        "category": category,
        "risk_score": 10,
        "stationary": False,
        "last_seen": time.time(),
        "traumatic": True,
        "trauma_velocity": max_closing_velocity,
    })

def find_trauma_match(world_pos):
    """Check if a new obstacle is near a known trauma site."""
    for entry in trauma_memory:
        if entry.get("category") in ("UNKNOWN", "", None):
            continue                      # never fire trauma on an UNKNOWN entry
        d = compute_dist_2d(world_pos, entry["pos"])
        if d <= MEMORY_MATCH_RADIUS * 1.5:   # slightly wider radius for trauma
            return entry
    return None


# ============================================================
# 4.5 STRESS SYSTEM CORE
# ============================================================
def compute_proximity_term(dist):
    """Map obstacle distance to a 0-10 proximity stress term.
    0 at PROXIMITY_ZERO_STRESS_DIST (or farther), 10 as distance -> 0.
    This is the fast, per-frame, depth-driven input -- deliberately NOT
    VLM/image-size based, since apparent size is unreliable at close range
    (e.g. a large animal up close can fill the frame with an out-of-context
    close-up that a classifier can't size correctly)."""
    if dist == float('inf') or dist >= PROXIMITY_ZERO_STRESS_DIST:
        return 0.0
    frac = 1.0 - (dist / PROXIMITY_ZERO_STRESS_DIST)
    return max(0.0, min(STRESS_MAX, frac * STRESS_MAX))

def compute_threat_term(risk_score, is_aggressive=False):
    """VLM risk_score is already 0-10 by prompt design; bump it if the VLM
    judged the obstacle to be actively approaching/agitated rather than
    merely present (currently_stationary == False for a BIOLOGICAL)."""
    val = float(risk_score)
    if is_aggressive:
        val += THREAT_AGGRESSION_BUMP
    return max(0.0, min(STRESS_MAX, val))

def resolve_stress_band(value, current_band):
    """Hysteresis band resolution: the up-crossing threshold differs from the
    down-crossing threshold so the band doesn't flap when stress hovers near
    a boundary. (Decreases are additionally locked during an active encounter
    -- see update_stress_system -- so in practice a band rarely even gets the
    chance to flap mid-maneuver.)"""
    if current_band == "LOW":
        if value >= STRESS_LOW_MEDIUM_UP:
            return "MEDIUM"
        return "LOW"
    elif current_band == "MEDIUM":
        if value >= STRESS_MEDIUM_HIGH_UP:
            return "HIGH"
        if value < STRESS_LOW_MEDIUM_DOWN:
            return "LOW"
        return "MEDIUM"
    else:  # HIGH
        if value < STRESS_MEDIUM_HIGH_DOWN:
            return "MEDIUM"
        return "HIGH"

def get_band_speed_multiplier(band, value):
    """Continuous scaling of speed WITHIN the current band's stress span
    (same pattern as the existing scale_by_risk() flank-distance scaling)."""
    lo_mult, hi_mult = BAND_SPEED_MULTIPLIER_RANGE.get(band, (1.0, 1.0))
    if band == "LOW":
        span_lo, span_hi = 0.0, STRESS_LOW_MEDIUM_UP
    elif band == "MEDIUM":
        span_lo, span_hi = STRESS_LOW_MEDIUM_DOWN, STRESS_MEDIUM_HIGH_UP
    else:
        span_lo, span_hi = STRESS_MEDIUM_HIGH_DOWN, STRESS_MAX

    span = max(0.001, span_hi - span_lo)
    frac = max(0.0, min(1.0, (value - span_lo) / span))
    return lo_mult + (hi_mult - lo_mult) * frac

def current_forward_speed():
    """FORWARD_SPEED scaled by the active stress band's speed multiplier.
    Use this instead of the raw FORWARD_SPEED constant anywhere the robot is
    doing routine/avoidance travel. Deliberately NOT used for the dedicated
    EMERGENCY_BACKUP_SPEED path, which is a separate hard safety layer that
    should stay independent of the stress model."""
    mult = get_band_speed_multiplier(stress_band, stress_level)
    return FORWARD_SPEED * mult

def update_stress_system(dt):
    """Call once per frame (top of zig_zag_update). Recomputes target_stress
    from current sensor/VLM state, advances `stress_level` toward it via
    exponential smoothing, applies the decrease-lock while an obstacle
    encounter is active, resolves the band with hysteresis, computes
    `potential`, and records a strip-chart history sample."""
    global stress_level, stress_target, stress_band, stress_locked
    global last_speed_multiplier, last_potential
    global stress_history_time, stress_history_stress, stress_history_potential


    if YOLO_TEST_MODE:
        stress_level = 0.0
        stress_band = "LOW"
        last_speed_multiplier = 1.0
        last_potential = 0.0
        return

    dist = _ros_node.closest_obstacle_dist
    proximity_term = compute_proximity_term(dist)

    # Only count the VLM threat term while genuinely engaged with an obstacle
    # -- otherwise a stale active_obstacle_risk (default 5) would leak stress
    # in on a perfectly clear path.
    engaged = (avoidance_sub_phase != "IDLE") or (state in ["YIELD_WAITING", "VLM_INSPECTING"])
    if engaged:
        threat_term = compute_threat_term(active_obstacle_risk, is_aggressive=active_obstacle_aggressive)
    else:
        threat_term = 0.0

    with yolo_lock:
        attack_term = compute_attack_term(yolo_tracks, dt)

    depth_result = depth_monitor.update(
        dist,
        _ros_node.depth_left_dist,
        _ros_node.depth_center_dist,
        _ros_node.depth_right_dist,
        dt,
    )
    depth_attack_term = depth_result["score"]

    # Persistence filter: a single noisy frame (ego-motion, dt division, bad
    # depth sample) must NOT be able to spike stress. Require the depth attack
    # term to stay above threshold for 2 consecutive frames before it counts.
    global _depth_attack_streak
    if depth_attack_term > 5.0:
        _depth_attack_streak += 1
    else:
        _depth_attack_streak = 0

    if _depth_attack_streak < 2:
        depth_attack_term = 0.0

    # Stash for telemetry
    globals()["_last_depth_result"] = depth_result


    new_target = max(
        PROXIMITY_WEIGHT * proximity_term,
        THREAT_WEIGHT * threat_term,
        ATTACK_WEIGHT * attack_term,
        ATTACK_WEIGHT * depth_attack_term,     # <-- depth-only attack
    )

    stress_target = new_target

    # Attack overrides the decrease lock — stress cannot decay while being charged.
    # NOTE: do NOT include VLM_INSPECTING or CLASSIFY_OBSTACLE here. Those are
    # pre-decision states; locking on them means a single noisy depth frame at
    # the moment of detection freezes stress HIGH for the entire encounter
    # (even after the depth signal self-corrects one frame later).
    # Only lock once we're actually mid-maneuver, or a real attack is ongoing.
    actively_avoiding = (
        avoidance_sub_phase not in ("IDLE", "CLASSIFY_OBSTACLE")
        or state == "YIELD_WAITING"
        or attack_suspected
    )
    
    stress_locked = actively_avoiding
    if stress_locked and stress_target < stress_level:
        effective_target = stress_level
    else:
        effective_target = stress_target

    stress_level += (effective_target - stress_level) * STRESS_SMOOTHING_RATE * dt
    stress_level = max(STRESS_MIN, min(STRESS_MAX, stress_level))

    stress_band = resolve_stress_band(stress_level, stress_band)

    speed_mult = get_band_speed_multiplier(stress_band, stress_level)
    urgency = BAND_URGENCY_MULTIPLIER.get(stress_band, 0.2)
    potential = (FORWARD_SPEED * speed_mult) * urgency

    last_speed_multiplier = speed_mult
    last_potential = potential

    now_t = time.time()
    stress_history_time.append(now_t)
    stress_history_stress.append(stress_level)
    stress_history_potential.append(potential)
    if len(stress_history_time) > STRESS_HISTORY_MAXLEN:
        stress_history_time.pop(0)
        stress_history_stress.pop(0)
        stress_history_potential.pop(0)

def update_strip_chart():
    """Push the latest history buffers into the UI plot widgets, if present."""
    try:
        if stress_plot is not None and len(stress_history_stress) > 1:
            stress_plot.set_data(*stress_history_stress)
    except Exception as e:
        print(f"[Strip Chart]: stress_plot update failed ({e}). See ui.Plot API note in comments.")
    try:
        if potential_plot is not None and len(stress_history_potential) > 1:
            potential_plot.set_data(*stress_history_potential)
    except Exception as e:
        print(f"[Strip Chart]: potential_plot update failed ({e}). See ui.Plot API note in comments.")

# ============================================================
# 5. MEMORY-AUGMENTED VISION LLM CLIENT WITH CLASSIFICATION
# ============================================================
global vlm_request_in_flight, last_vlm_query_time, llm_avoidance_history
vlm_request_in_flight = False
last_vlm_query_time = 0.0
llm_avoidance_history = []  # Short-term session memory for current obstacle

# STRESS SYSTEM: STRESS_RECHECK gets its OWN in-flight flag + cooldown timer,
# fully separate from the CLASSIFY_OBSTACLE ones above. This local VLM does
# heavy internal chain-of-thought reasoning and can take tens of seconds (or
# longer) to respond -- if recheck shared the classify lock, a slow recheck
# would silently swallow a brand-new obstacle's CLASSIFY_OBSTACLE request
# (the "already in flight" branch returns without ever calling callback_fn),
# permanently wedging the FSM in VLM_INSPECTING. Keeping these separate means
# a slow/stuck recheck can never starve the critical classification path.
global stress_recheck_in_flight, last_stress_recheck_query_time
stress_recheck_in_flight = False
last_stress_recheck_query_time = 0.0

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

_last_collect_time = 0.0

BURST_FRAMES   = 5
BURST_INTERVAL = 0.25    # seconds between frames
BURST_MIN_GAP  = 3.0     # min seconds between bursts
_last_burst_time = 0.0

def save_burst_snapshots(first_frame, folder_name, tag="x"):
    global _last_burst_time
    if not AUTO_COLLECT_ENABLED:
        return
    now = time.time()
    if now - _last_burst_time < BURST_MIN_GAP:
        return
    _last_burst_time = now

    def _run():
        try:
            folder = os.path.join(AUTO_COLLECT_DIR, folder_name)
            os.makedirs(folder, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
            safe = re.sub(r'[^a-zA-Z0-9_-]', '', tag)[:20]
            for i in range(BURST_FRAMES):
                frame = first_frame if (i == 0 and first_frame is not None) else _ros_node.latest_rgb
                if frame is not None:
                    d = _ros_node.closest_obstacle_dist
                    cv2.imwrite(os.path.join(folder, f"{stamp}_f{i}_{d:.2f}m_{safe}.jpg"), frame)
                time.sleep(BURST_INTERVAL)
            print(f"[Data Collect]: burst of {BURST_FRAMES} saved -> {folder}")
        except Exception as e:
            print(f"[Data Collect Error]: {e}")

    threading.Thread(target=_run, daemon=True).start()

def async_query_vlm_agent(rgb_array, obstacle_dist, current_sub_phase, detour_offset, callback_fn):
    global vlm_request_in_flight, last_vlm_query_time, llm_avoidance_history, vlm_requested_phase
    global stress_recheck_in_flight, last_stress_recheck_query_time
    global yolo_refine_in_flight, last_yolo_refine_query_time

    vlm_requested_phase = current_sub_phase 

    now = time.time()

    if state in ["STOPPED", "USER_PAUSED"]:
        return

    # STRESS SYSTEM: STRESS_RECHECK uses its own in-flight/cooldown pair so it
    # can never block or be blocked by the critical CLASSIFY_OBSTACLE path.
    is_recheck = (current_sub_phase == "STRESS_RECHECK")

        # Already mid-request -> just skip, don't double-fire deterministic either,
        # since a real VLM response is still on its way and will resolve this.
    is_refine = (current_sub_phase == "YOLO_REFINE")

    if is_recheck:
        if stress_recheck_in_flight:
            print(f"[VLM Busy]: STRESS_RECHECK already in flight, skipping duplicate recheck call.")
            return
    elif is_refine:
        if yolo_refine_in_flight:
            print(f"[VLM Busy]: YOLO_REFINE already in flight, skipping duplicate refine call.")
            return
    else:
        if vlm_request_in_flight:
            print(f"[VLM Busy]: Request already in flight, skipping duplicate call for phase={current_sub_phase}.")
            return

    # Still inside the cooldown window -> don't touch the VLM at all,
    # resolve immediately with the deterministic rule instead.
    # STRESS SYSTEM: while in HIGH band, use a shorter cooldown so the robot
    # can re-consult the LLM more frequently during active-alert situations.
    effective_cooldown = HIGH_STRESS_VLM_COOLDOWN if stress_band == "HIGH" else MIN_VLM_COOLDOWN
    if is_recheck:
        last_query_time = last_stress_recheck_query_time
    elif is_refine:
        last_query_time = last_yolo_refine_query_time
    else:
        last_query_time = last_vlm_query_time
    skip_cooldown = VLM_PRIORITY_MODE and current_sub_phase == "CLASSIFY_OBSTACLE"
    if (now - last_query_time) < effective_cooldown and not skip_cooldown:
        remaining = effective_cooldown - (now - last_query_time)
        if is_refine:
            print(f"[YOLO Refine]: cooldown active ({remaining:.1f}s), skipping this refinement.")
            return
        print(f"[VLM Cooldown]: {remaining:.1f}s remaining ({stress_band} band), using deterministic fallback for phase={current_sub_phase}.")
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
    if is_recheck:
        stress_recheck_in_flight = True
        last_stress_recheck_query_time = now
    elif is_refine:
        yolo_refine_in_flight = True
        last_yolo_refine_query_time = now
    else:
        vlm_request_in_flight = True
        last_vlm_query_time = now
        print(f"[VLM Dispatch]: in_flight=True | phase={current_sub_phase} | recheck={is_recheck} | t={now:.2f}")
    def worker():
        global vlm_request_in_flight, llm_avoidance_history, stress_recheck_in_flight, yolo_refine_in_flight
        try:
            max_retries = VLM_MAX_RETRIES
            attempt = 0
            while attempt <= max_retries:
                try:
                    b64_img = encode_frame_to_base64(rgb_array)
                    if not b64_img:
                        print("[Perception Error]: RGB image frame is empty or invalid.")
                        return

                    history_summary = ""
                    if len(llm_avoidance_history) > 0:
                        history_summary = "Past avoidance decisions for this obstacle:\n" + "\n".join([f"- {h}" for h in llm_avoidance_history[-3:]]) + "\n"

                    if current_sub_phase in ("CLASSIFY_OBSTACLE", "YOLO_REFINE"):
                        prompt = (
                            f"You are the visual perception brain for an autonomous sweeper robot analyzing an obstacle detected at {obstacle_dist:.2f}m.\n"
                            f"If the path is actually clear or it is a false detection, categorize as CLEAR.\n"
                            f"Double check: If the image you are looking at, doesn't resemble a sky, a road, any form that doesn't look like a terrain, then the thing you are directly looking at is an obstacle.\n"
                            f"Examine the image carefully for 3D rendered asset features:\n\n"
                            f"CRITICAL VISUAL RULES:\n"
                            f"- BIOLOGICAL: Any animal, quadruped, cow, cattle, bull, or livestock model. Do NOT classify animals as traffic cones or static barriers even if captured up close!\n"
                            f"- STATIC_BAR: Traffic cones, bright orange/red plastic markers, construction barriers, bollards, vehicles, or solid sharp-edged blocks, etc.\n"
                            f"- MINOR_SURFACE: Potholes, road cracks, asphalt patches, ground depressions, or flat surface defects.\n"
                            f"- CLEAR: No obstacle blocking forward progress or clear open corridor.\n\n"
                            f"RISK SCORE (0-10 integer) — rate how much EXTRA CAUTION this specific instance warrants:\n"
                            f"0-2: No unpredictability at all (painted line, shallow crack, distant/irrelevant object)\n"
                            f"3-5: Requires deliberate avoidance but behavior is fully predictable (typical cone, barrier, parked vehicle)\n"
                            f"6-8: Some unpredictability present (animal not currently agitated, but capable of sudden movement)\n"
                            f"9-10: High unpredictability or imminent risk (animal alert/moving toward the robot, very close proximity, erratic object)\n\n"
                            f"STATIONARY CHECK: Set 'currently_stationary' to true ONLY if this specific instance shows NO signs of independent motion "
                            f"capability right now (a parked car with no indication of moving, a cone, a pothole). Set it to false for ANY animal, "
                            f"or any vehicle showing signs of being actively driven (engine cues, wheels turned, mid-motion posture).\n\n"
                            f"Additionally, divide the image horizontally into 3 equal vertical slices (Left 0-33%, Center 33-66%, Right 66-100%).\n"
                            f"Provide sector analysis:\n"
                            f"- 'center_blocked': true if ANY obstacle (any category) intrudes into the Center slice.\n"
                            f"- 'left_sector_coverage_pct': integer 0-100 representing obstacle coverage in Left slice.\n"
                            f"- 'right_sector_coverage_pct': integer 0-100 representing obstacle coverage in Right slice.\n\n"
                            f"Return ONLY valid JSON with the following structure:\n"
                            f"{{\n"
                            f'  "category": "BIOLOGICAL" | "STATIC_BAR" | "MINOR_SURFACE" | "CLEAR",\n'
                            f'  "obstacle_class": "Cow/Animal" | "Traffic Cone" | "Vehicle" | "Pothole/Crack" | "None",\n'
                            f'  "risk_score": <int 0-10>,\n'
                            f'  "currently_stationary": bool,\n'
                            f'  "command": "yield_and_wait" | "standard_bypass" | "scan_and_proceed" | "resume_path",\n'
                            f'  "reasoning": "Detailed visual evidence for BOTH the category and the risk score",\n'
                            f'  "center_blocked": bool,\n'
                            f'  "left_sector_coverage_pct": int,\n'
                            f'  "right_sector_coverage_pct": int\n'
                            f"}}"
                        )
                    elif current_sub_phase == "STRESS_RECHECK":
                        # STRESS SYSTEM: lightweight mid-maneuver recheck used by
                        # MEDIUM band (once, before committing to return-to-lane)
                        # and HIGH band (at select avoidance sub-phase
                        # transitions) to let the LLM decide whether the robot
                        # should keep executing its current avoidance plan or
                        # abandon it and actively back off instead.
                        prompt = (
                            f"You are re-checking an obstacle the robot is already in the process of avoiding, "
                            f"currently {obstacle_dist:.2f}m away.\n"
                            f"{history_summary}"
                            f"Look at the current image and judge whether the obstacle is approaching the robot, "
                            f"staying still, or moving away.\n"
                            f"Return ONLY valid JSON with the following structure:\n"
                            f"{{\n"
                            f'  "obstacle_visible": bool,\n'
                            f'  "trend": "approaching" | "stationary" | "receding",\n'
                            f'  "command": "abandon_and_retreat" | "continue" | "return_to_lane",\n'
                            f'  "reasoning": "Brief visual evidence for the trend and command"\n'
                            f"}}"
                        )
                    else:
                        prompt = (
                            f"Re-evaluate the obstacle at {obstacle_dist:.2f}m during avoidance phase '{current_sub_phase}'. "
                            f"Return ONLY valid JSON: "
                            f'{{"obstacle_visible": bool, "command": "diverge_more" | "flank_forward" | "return_to_lane", "reasoning": "..."}}'
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
                if current_sub_phase == "YOLO_REFINE":
                    print("[YOLO Refine]: VLM unreachable after retries, keeping YOLO's decision.")
                else:
                    print("[VLM Max Retries/Timeout Exceeded]: Executing deterministic fallback rule.")
                    if current_sub_phase == "CLASSIFY_OBSTACLE":
                        cmd, cat = "standard_bypass", "STATIC_BAR"
                    elif current_sub_phase == "STRESS_RECHECK":
                        # STRESS SYSTEM: conservative-by-default fallback -- if we
                        # can't reach the LLM during a stress-driven recheck,
                        # default to the SAFER option (abandon/back off), not
                        # "assume it's fine and keep going".
                        cmd, cat = "abandon_and_retreat", "N/A"
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
            # STRESS SYSTEM: release the LOCK THAT WAS ACTUALLY ACQUIRED for this
            # dispatch (classify vs recheck) -- releasing the wrong one would
            # either leave the real lock stuck forever or spuriously clear a
            # lock some other in-flight request still owns.
            print(f"[VLM Dispatch]: in_flight=False | phase={current_sub_phase} | recheck={is_recheck} | t={time.time():.2f}")
            if is_recheck:
                stress_recheck_in_flight = False
            elif is_refine:
                yolo_refine_in_flight = False
            else:
                vlm_request_in_flight = False

    threading.Thread(target=worker, daemon=True).start()


# ============================================================
# TEMPORARY: YOLO ISOLATION TEST MODE
# ============================================================


YOLO_TEST_MODE = False   # set False to restore normal behavior
# ============================================================
# 6. STATE VARIABLES & DISPLACEMENT LEDGER
# ============================================================
global state, saved_pre_pause_state, current_pos, current_yaw, sweep_direction_y
global step_start_pos, target_yaw, pause_start_time, last_time, shifted_distance, last_print_time
global lane_base_yaw, lateral_detour_ledger, micro_step_progress, micro_step_target, detour_side
global avoidance_sub_phase, initial_detected_distance, required_flank_distance, flank_progress, diverge_turn_count
global active_obstacle_category, yield_wait_start_time, active_obstacle_risk
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
global known_obstacles, pending_classify_world_pos

# STRESS SYSTEM globals
global stress_level, stress_target, stress_band, stress_locked
global current_encounter_id, active_obstacle_aggressive
global last_speed_multiplier, last_potential
global stress_history_time, stress_history_stress, stress_history_potential
global pending_band_abandon
global attack_suspected, attack_start_time, attacked_tracks
global trauma_memory
global max_closing_velocity_observed

# Yolo globals
global yolo_refine_in_flight, last_yolo_refine_query_time
global yolo_session, yolo_tracks, yolo_lock, yolo_thread_running

global pending_classify_rgb_frame
global repeated_backup_count, repeated_backup_pos
global diverge_grace_until

# Emergency response
global emergency_backup_start_time, emergency_escape_turns
global emergency_original_yaw, escape_progress, emergency_pin_time, escape_target_dist

vlm_inspect_start = 0.0
emergency_pin_time = 0.0
escape_target_dist = EMERGENCY_ESCAPE_DIST
emergency_backup_start_time = 0.0
emergency_escape_turns = 0
emergency_original_yaw = 0.0
escape_progress = 0.0

pending_classify_rgb_frame = None

diverge_grace_until = 0.0

repeated_backup_pos = None
repeated_backup_count = 0

yolo_refine_in_flight = False
last_yolo_refine_query_time = 0.0
lane_stuck_count = 0
lane_stuck_pos = None
consecutive_blocked_shifts = 0   # position-independent: increments on EVERY blocked shift
_depth_attack_streak = 0
max_closing_velocity_observed = 0.0

trauma_memory = []   # list of {pos, category, closing_velocity, timestamp, track_ids}

attack_suspected = False
attack_start_time = 0.0
attacked_tracks = []         # track IDs that triggered attack


yolo_session = None
yolo_tracks = []            # list of dicts: {id, class, bbox, confidence, distance, closing_velocity}
yolo_lock = threading.Lock()
yolo_thread_running = False

track_memory = {}           # id -> {"centroid": (cx, cy), "bbox": ..., "age": 0, "class": ...}
next_track_id = 0


active_obstacle_category = "UNKNOWN"
active_obstacle_risk = 5          # NEW — default mid-risk until VLM says otherwise
active_obstacle_aggressive = False  # STRESS SYSTEM: True if VLM judged obstacle non-stationary/agitated
yield_wait_start_time = 0.0

known_obstacles = []                 # [{"pos": (x,y), "category": str, "last_seen": float}, ...]
pending_classify_world_pos = None  

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

# STRESS SYSTEM initial state
stress_level = 0.0
stress_target = 0.0
stress_band = "LOW"
stress_locked = False
current_encounter_id = 0
last_speed_multiplier = 1.0
last_potential = 0.0
stress_history_time = []
stress_history_stress = []
stress_history_potential = []
pending_band_abandon = False

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
    """Back up away from obstacle, then re-evaluate.
    STRESS SYSTEM: also reused as the concrete action for a HIGH-band
    'abandon_and_retreat' LLM decision -- it already does exactly that."""
    global state, emergency_triggered, emergency_cooldown_start, emergency_trigger_pos
    global backup_progress, avoidance_sub_phase, lateral_detour_ledger
    global repeated_backup_pos, repeated_backup_count
    global emergency_backup_start_time, emergency_escape_turns, emergency_original_yaw
    emergency_backup_start_time = time.time()
    emergency_escape_turns = 0
    emergency_original_yaw = current_yaw
    emergency_pin_time = 0.0

    if repeated_backup_pos is not None and compute_dist_2d(current_pos, repeated_backup_pos) < REPEATED_BACKUP_RADIUS:
        repeated_backup_count += 1
    else:
        repeated_backup_count = 1
        repeated_backup_pos = list(current_pos)
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
    if current_sub_phase == "YOLO_REFINE":
        return  # no VLM available -- just keep YOLO's fast-path decision as-is
    if current_sub_phase == "CLASSIFY_OBSTACLE":
        cmd, cat = "standard_bypass", "STATIC_BAR"
    elif current_sub_phase == "STRESS_RECHECK":
        # STRESS SYSTEM: conservative default -- can't reach LLM, assume the
        # safer option during a stress-driven recheck.
        cmd, cat = "abandon_and_retreat", "N/A"
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

# ============================================================
# DEPTH-ONLY ATTACK DETECTOR (works without YOLO)
# ============================================================
BUMPER_DISTANCE = 0.25                    # meters — counts as virtual contact
DEPTH_ATTACK_CLOSING_THRESHOLD = 0.6      # m/s
DEPTH_ATTACK_TTI_THRESHOLD = 1.5          # seconds

class DepthThreatMonitor:
    """Continuous depth-only threat signature detector. No YOLO required."""

    def __init__(self, history_len=30):
        self.history_len = history_len
        self.reset()

    def reset(self):
        self.prev_dist = float('inf')
        self.closing_velocity = 0.0
        self.acceleration = 0.0
        self.sector_history = []
        self.bumper_dwell = 0.0
        self.bumper_hits = 0
        self.last_signature = "NONE"
        self.last_score = 0.0

    def update(self, dist, dl, dc, dr, dt, ego_speed=0.0):
        if dt <= 1e-4:
            dt = 1e-4
        if dist == float('inf') or dist > 3.0:
            self.prev_dist = dist
            self.closing_velocity = 0.0
            self.acceleration = 0.0
            return {
                "score": 0.0, "signature": "NONE",
                "closing_velocity": 0.0, "acceleration": 0.0,
                "bumper_dwell": 0.0, "bumper_hits": self.bumper_hits,
                "pop_in": False, "tti": 999.0,
            }
        # --- Closing velocity (smoothed) ---
        if self.prev_dist != float('inf') and dist != float('inf'):
            raw_v = (self.prev_dist - dist) / dt
            raw_v -= ego_speed 
            new_v = 0.7 * self.closing_velocity + 0.3 * raw_v
            self.acceleration = (new_v - self.closing_velocity) / dt
            self.closing_velocity = new_v
        elif dist == float('inf'):
            self.closing_velocity = 0.0
            self.acceleration = 0.0

        # --- Sector deltas ---
        sector_now = (dl, dc, dr)
        if self.sector_history:
            prev = self.sector_history[-1]
            dl_delta = (prev[0] - dl) / dt if dl != float('inf') and prev[0] != float('inf') else 0.0
            dc_delta = (prev[1] - dc) / dt if dc != float('inf') and prev[1] != float('inf') else 0.0
            dr_delta = (prev[2] - dr) / dt if dr != float('inf') and prev[2] != float('inf') else 0.0
        else:
            dl_delta = dc_delta = dr_delta = 0.0

        self.sector_history.append(sector_now)
        if len(self.sector_history) > self.history_len:
            self.sector_history.pop(0)

        # --- Virtual bumper ---
        if dist < BUMPER_DISTANCE:
            if self.bumper_dwell == 0.0:
                self.bumper_hits += 1
            self.bumper_dwell += dt
        else:
            self.bumper_dwell = 0.0

        # --- Pop-in (single-frame discontinuity) ---
        pop_in = (
            self.prev_dist != float('inf')
            and self.prev_dist > 2.0
            and dist < 0.5
        )

        # --- Signature classification ---
        sig = "NONE"
        if self.closing_velocity > 0.4:
            c_closing = dc_delta > 0.4
            l_closing = dl_delta > 0.4
            r_closing = dr_delta > 0.4
            if c_closing and l_closing and r_closing:
                sig = "SURROUND"
            elif c_closing:
                sig = "HEAD_ON"
            elif l_closing or r_closing:
                sig = "FLANK"
            elif abs(self.acceleration) < 0.3:
                sig = "STALKING"
        if pop_in:
            sig = "POP_IN"

        # --- Threat score (0-10) ---
        tti = dist / max(self.closing_velocity, 1e-3) if self.closing_velocity > 0 else 999.0
        tti_score = max(0.0, 10.0 * (1.0 - min(tti, 2.0) / 2.0))
        accel_score = min(10.0, max(0.0, self.acceleration * 3.0))
        bumper_score = min(10.0, self.bumper_dwell * 4.0)
        pop_score = 9.0 if pop_in else 0.0
        score = max(tti_score, accel_score, bumper_score, pop_score)

        self.last_signature = sig
        self.last_score = score
        self.prev_dist = dist

        return {
            "score": score,
            "signature": sig,
            "closing_velocity": self.closing_velocity,
            "acceleration": self.acceleration,
            "bumper_dwell": self.bumper_dwell,
            "bumper_hits": self.bumper_hits,
            "pop_in": pop_in,
            "tti": tti,
        }
# Instantiate the depth monitor (must come AFTER the class definition)
depth_monitor = DepthThreatMonitor()
_prev_bumper_dwell = 0.0
def compute_attack_term(tracks, frame_dt):
    """0-10. High when a tracked entity is closing on the robot fast.
    Uses both closing velocity (from depth) and bbox area growth (from YOLO)."""
    global attack_suspected, attack_start_time, attacked_tracks
    global max_closing_velocity_observed


    worst = 0.0
    new_attacked = []

    for t in tracks:
        dist = t.get("distance", float('inf'))
        if dist > 2.0:
            continue
        cv = t.get("closing_velocity", 0.0)
        ag = t.get("area_growth", 0.0)
        # Must be closing OR growing fast
        if cv < ATTACK_CLOSING_THRESHOLD and ag < ATTACK_AREA_GROWTH_THRESHOLD:
            continue

        # Time-to-impact from closing velocity
        tti = dist / max(cv, 1e-3) if cv > 0 else 999.0
        if tti > ATTACK_TTI_THRESHOLD and ag < ATTACK_AREA_GROWTH_THRESHOLD * 2:
            continue

        # Score: higher as TTI shrinks
        tti_score = max(0.0, 10.0 * (1.0 - min(tti, ATTACK_TTI_THRESHOLD) / ATTACK_TTI_THRESHOLD))
        # Score: higher as area grows faster
        ag_score = min(10.0, ag * 30.0) 
        score = max(tti_score, ag_score)
        worst = max(worst, score)
        new_attacked.append(t["track_id"])

    if worst > 4.0:
        if not attack_suspected:
            attack_suspected = True
            attack_start_time = time.time()
            print(f"[ATTACK SUSPECTED]: {len(new_attacked)} track(s) closing. "
                f"IDs={new_attacked}")
    else:
        if attack_suspected:
            attack_suspected = False
            attack_start_time = 0.0
            print("[Attack ended]: No closing tracks above threshold.")

    attacked_tracks = new_attacked

    if attack_suspected:
        highest = max((t.get("closing_velocity", 0.0) for t in tracks), default=0.0)
        max_closing_velocity_observed = max(max_closing_velocity_observed, highest)
    else:
        # Reset when attack ends so next encounter starts fresh
        # (but only after zig_zag_update has consumed it for trauma recording)
        pass
    
    return worst

def check_emergency_proximity(dist_ahead):
    if state in ["STOPPED", "USER_PAUSED", "EMERGENCY_BACKUP", "VLM_INSPECTING",
                "EMERGENCY_ESCAPE_TURN", "EMERGENCY_ESCAPE_BACK", "EMERGENCY_ESCAPE_REALIGN"]:
        return False
    if time.time() < diverge_grace_until:
        return dist_ahead < CRITICAL_TOUCH_DIST   # grace: only a genuine near-collision gets through
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
# STRESS SYSTEM: mid-maneuver LLM recheck (abandon vs continue)
# ============================================================
def trigger_stress_recheck(phase_tag):
    """Fire an asynchronous LLM re-check of the current obstacle WITHOUT
    disturbing the active FSM sub-phase. Used by:
    - MEDIUM band: once, right before committing to return-to-lane.
    - HIGH band: at select avoidance sub-phase transitions (VERIFY entry,
        and the FLANK_PRE_CHECK deterministic settle point). Extend the call
        sites below if you want denser HIGH-band scanning at additional
        transitions (e.g. DIVERGE_SCAN, WIDE_OBSTACLE_DIVERGE).
    Staleness is guarded via current_encounter_id rather than the
    avoidance_sub_phase match used by on_vlm_agent_decision, since this can
    legitimately span multiple sub-phases before a response arrives."""
    if state in ["STOPPED", "USER_PAUSED"]:
        return
    rgb_frame = _ros_node.latest_rgb
    captured_id = current_encounter_id

    def _cb(decision_dict):
        on_stress_recheck_decision(decision_dict, captured_id)

    print(f"[Stress Recheck]: Dispatching ({phase_tag}, band={stress_band}, encounter={captured_id})...")
    async_query_vlm_agent(rgb_frame, _ros_node.closest_obstacle_dist, "STRESS_RECHECK", lateral_detour_ledger, _cb)

def on_stress_recheck_decision(decision_dict, captured_encounter_id):
    global pending_band_abandon
    if captured_encounter_id != current_encounter_id:
        print(f"[Stress Recheck Ignored]: stale response for encounter {captured_encounter_id}, current is {current_encounter_id}.")
        return
    if state in ["STOPPED", "USER_PAUSED"]:
        return

    cmd = decision_dict.get("command", "continue")
    reasoning = decision_dict.get("reasoning", "")
    print(f"[Stress Recheck Result]: command='{cmd}' | {reasoning}")
    if vlm_label:
        vlm_label.text = f"VLM(recheck): {cmd}"

    if cmd == "abandon_and_retreat":
        pending_band_abandon = True

def perform_full_reset():
    """Full state reset back to spawn. Used by the Reset button AND by the
    auto-loop restart. Also clears spatial/trauma memory so each loop pass
    re-classifies obstacles fresh (more varied training snapshots) instead of
    silently reusing cached categories."""
    global state, current_pos, current_yaw, step_start_pos, last_time, shifted_distance
    global sweep_direction_y, lane_base_yaw, lateral_detour_ledger, avoidance_sub_phase
    global vlm_request_in_flight, llm_avoidance_history, active_obstacle_category
    global emergency_triggered, emergency_cooldown_start, emergency_trigger_pos
    global wide_diverge_count
    global stress_level, stress_target, stress_band, stress_locked
    global current_encounter_id, active_obstacle_aggressive
    global stress_history_time, stress_history_stress, stress_history_potential
    global pending_band_abandon
    global stress_recheck_in_flight, last_stress_recheck_query_time
    global consecutive_blocked_shifts
    global yolo_refine_in_flight, last_yolo_refine_query_time
    global pending_classify_rgb_frame, pending_classify_world_pos
    global known_obstacles, trauma_memory
    global track_memory, next_track_id, yolo_tracks
    global breadcrumb_positions, breadcrumb_colors

    yolo_refine_in_flight = False
    last_yolo_refine_query_time = 0.0
    wide_diverge_count = 0
    state = "MOVING"
    vlm_request_in_flight = False
    stress_recheck_in_flight = False
    last_stress_recheck_query_time = 0.0
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
    active_obstacle_aggressive = False
    step_start_pos = list(current_pos)
    shifted_distance = 0.0
    last_time = time.time()
    consecutive_blocked_shifts = 0
    pending_classify_rgb_frame = None
    pending_classify_world_pos = None

    # Fresh perception memory each loop -- forces real re-classification
    # instead of reusing cached categories from the previous pass.
    known_obstacles = []
    trauma_memory = []
    track_memory = {}
    next_track_id = 0
    yolo_tracks = []

    stress_level = 0.0
    stress_target = 0.0
    stress_band = "LOW"
    stress_locked = False
    current_encounter_id = 0
    pending_band_abandon = False
    stress_history_time = []
    stress_history_stress = []
    stress_history_potential = []

    stage = omni.usd.get_context().get_stage()
    trail_prim = stage.GetPrimAtPath("/World/Breadcrumbs")
    if trail_prim.IsValid():
        stage.RemovePrim("/World/Breadcrumbs")
    breadcrumb_positions = []
    breadcrumb_colors = []

    update_usd_transform(current_pos, current_yaw)
    if status_label:
        status_label.text = "Status: RESET & MOVING"
    if vlm_label:
        vlm_label.text = "VLM: Idle"
    if stress_label:
        stress_label.text = "Stress: 0.0/10 [LOW]  Potential: 0.00"


def finish_run(reason="Destination Reached"):
    """Called at every natural end-of-run point (destination reached, target
    X reached). If AUTO_LOOP_ENABLED, parks briefly then restarts from spawn
    for unattended data collection -- unlike a manual Stop, which halts for
    good."""
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
#
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

_phantomx_window = ui.Window("PhantomX Turn-Analyze-Verify Controller", width=440, height=430, flags=ui.WINDOW_FLAGS_NO_SAVED_SETTINGS)
_phantomx_window.position_x = 80
_phantomx_window.position_y = 120
_phantomx_window.visible = True
_phantomx_window.focus()

status_label = None
sensor_label = None
ledger_label = None
vlm_label = None
stress_label = None
stress_plot = None
potential_plot = None

with _phantomx_window.frame:
    with ui.VStack(spacing=6, style={"margin": 10}):
        ui.Label("PhantomX Class-Based Sweeper", height=20, style={"font_size": 14, "color": 0xFF00FFFF})
        ui.Label(f"Target: ({TARGET_DESTINATION[0]:.1f}, {TARGET_DESTINATION[1]:.1f})", height=18, style={"font_size": 12, "color": 0xFFAAAAFF})
        sensor_label = ui.Label("Obstacle Dist: -- m", height=18, style={"font_size": 12, "color": 0xFFFFFF00})
        vlm_label = ui.Label("VLM: Idle", height=18, style={"font_size": 11, "color": 0xFFFFAA88})
        ledger_label = ui.Label("Detour Ledger: 0.00 m", height=18, style={"font_size": 11, "color": 0xFF88FF88})
        stress_label = ui.Label("Stress: 0.0/10 [LOW]  Potential: 0.00", height=18, style={"font_size": 12, "color": 0xFFFF66CC})

        # STRESS SYSTEM: live strip chart. NOTE: ui.Plot's exact constructor
        # signature/behavior can vary slightly across Isaac Sim / Kit
        # versions -- if this errors on your build, check the local
        # omni.ui.Plot docs for the correct args and adjust set_data() calls
        # in update_strip_chart() above accordingly. A text-only readout
        # (stress_label above) will keep working regardless.
        ui.Label("Stress (0-10) over time:", height=14, style={"font_size": 10, "color": 0xFFAAAAAA})
        try:
            stress_plot = ui.Plot(
                ui.Type.LINE, 0.0, STRESS_MAX,
                *([0.0] * STRESS_HISTORY_MAXLEN),
                height=55,
                style={"color": 0xFFFF66CC, "background_color": 0xFF1A1A1A}
            )
        except Exception as e:
            print(f"[UI]: stress_plot creation failed ({e}); continuing without strip chart.")
            stress_plot = None

        ui.Label("Potential (speed x urgency) over time:", height=14, style={"font_size": 10, "color": 0xFFAAAAAA})
        try:
            potential_max_guess = max(1.0, FORWARD_SPEED * 1.2 * max(BAND_URGENCY_MULTIPLIER.values()))
            potential_plot = ui.Plot(
                ui.Type.LINE, 0.0, potential_max_guess,
                *([0.0] * STRESS_HISTORY_MAXLEN),
                height=45,
                style={"color": 0xFF66CCFF, "background_color": 0xFF1A1A1A}
            )
        except Exception as e:
            print(f"[UI]: potential_plot creation failed ({e}); continuing without strip chart.")
            potential_plot = None

        status_label = ui.Label("Status: MOVING", height=20, style={"font_size": 13})
        
        with ui.HStack(spacing=8, height=36):
            def on_emergency_stop():
                global _loop_restart_pending
                _loop_restart_pending = False
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
                perform_full_reset()

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
    if step is None:
        step = MIN_DIVERGE_STEP * 1.3
    global avoidance_sub_phase, target_yaw, micro_step_target, micro_step_progress
    global state, wide_diverge_count
    global diverge_grace_until
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
    diverge_grace_until = time.time() + DIVERGE_GRACE_PERIOD
    state = "AVOID_ROTATE_THEN_STEP"

def initiate_bypass_sequence(force_wide=False):
    """STRESS SYSTEM: `force_wide=True` (used for HIGH band) skips the normal
    incremental diverge-and-verify sequence and goes straight into a wide
    evasive maneuver -- 'bigger edge steps trying to completely avoid the
    threat', per the High-alert spec."""
    global state, avoidance_sub_phase, target_yaw, detour_side, diverge_turn_count
    global wide_diverge_count, lane_base_yaw
    global diverge_grace_until
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

    if force_wide:
        # STRESS SYSTEM (HIGH band): skip the incremental DIVERGE_SCAN ladder
        # entirely and commit straight to a wide evasive maneuver.
        avoidance_sub_phase = "WIDE_OBSTACLE_DIVERGE"
        target_yaw = (lane_base_yaw + (detour_side * 65.0)) % 360.0
        micro_step_target = MIN_DIVERGE_STEP * 1.33 
        micro_step_progress = 0.0
        wide_diverge_count = 1
        diverge_grace_until = time.time() + DIVERGE_GRACE_PERIOD
        state = "AVOID_ROTATE_THEN_STEP"
        if status_label:
            status_label.text = "Status: HIGH STRESS - Wide Evasive Maneuver"
        print("[High Stress Override]: Skipping incremental diverge, committing to wide evasive maneuver.")
        return

    avoidance_sub_phase = "DIVERGE_SCAN"
    target_yaw = (lane_base_yaw + (detour_side * DIVERGE_ANGLE_INC)) % 360.0
    state = "AVOID_INITIAL_TURN"
    if status_label:
        status_label.text = "Status: Executing Bypass Diverge..."

def on_vlm_refine_decision(decision_dict, captured_encounter_id, captured_world_pos):
    """Fired when the background VLM refine call (dispatched after a YOLO
    fast-path decision) eventually returns. Does NOT redirect the FSM under
    normal circumstances -- the robot is already mid-maneuver on YOLO's call.
    The one exception is a safety escalation: if the VLM sees BIOLOGICAL where
    YOLO called something else, that's promoted immediately via the stress
    system rather than silently ignored."""
    global active_obstacle_risk, active_obstacle_aggressive, active_obstacle_category
    global stress_target

    if captured_encounter_id != current_encounter_id:
        print(f"[YOLO Refine Ignored]: stale response for encounter {captured_encounter_id}, "
              f"current is {current_encounter_id}.")
        return
    if state in ["STOPPED", "USER_PAUSED"]:
        return

    vlm_category = decision_dict.get("category", active_obstacle_category)
    risk = decision_dict.get("risk_score", active_obstacle_risk)
    stationary = decision_dict.get("currently_stationary", True)

    if vlm_category == "BIOLOGICAL" and active_obstacle_category != "BIOLOGICAL":
        print(f"[YOLO Refine]: VLM disagrees with YOLO fast-path -- reclassifying as "
              f"BIOLOGICAL, bumping risk so the stress system reacts.")
        active_obstacle_category = "BIOLOGICAL"
        active_obstacle_risk = max(risk, 7)
        active_obstacle_aggressive = not stationary
    else:
        active_obstacle_risk = risk
        active_obstacle_aggressive = (active_obstacle_category == "BIOLOGICAL" and not stationary)
        print(f"[YOLO Refine]: VLM confirmed -> risk={risk}, stationary={stationary}")

    if captured_world_pos is not None:
        record_obstacle_memory(captured_world_pos, active_obstacle_category,
                                active_obstacle_risk, stationary)

def on_vlm_agent_decision(decision_dict):
    global state, avoidance_sub_phase, target_yaw, current_yaw, micro_step_target, micro_step_progress
    global detour_side, flank_progress, required_flank_distance, initial_detected_distance, diverge_turn_count
    global active_obstacle_category, yield_wait_start_time, active_obstacle_risk
    global push_progress, push_target
    global vlm_requested_phase
    global flank_step_progress, flank_step_start_pos   
    global pending_classify_world_pos, pending_classify_rgb_frame       
    global category   
    global active_obstacle_aggressive
    global stress_band, stress_level, stress_target, stress_locked
    global yield_wait_total_start_time
    global repeated_backup_count, repeated_backup_pos

    # Ignore the callback if the phase changed since the request was sent
    if vlm_requested_phase != avoidance_sub_phase:
        print(f"[VLM Callback Ignored]: phase mismatch (requested {vlm_requested_phase}, current {avoidance_sub_phase})")
        return

    if state in ["STOPPED","USER_PAUSED"]:
        return

    cmd = decision_dict.get("command", "standard_bypass")
    category = decision_dict.get("category", active_obstacle_category)
    reasoning = decision_dict.get("reasoning", "")
    obs_visible = decision_dict.get("obstacle_visible", True)

    if avoidance_sub_phase in ["IDLE", "CLASSIFY_OBSTACLE"] and _ros_node.closest_obstacle_dist == float('inf'):
        print(f"[VLM Callback Ignored]: Sensor shows clear (inf), but callback says {category}. Treating as false alarm.")
        state = "MOVING"
        avoidance_sub_phase = "IDLE"
        active_obstacle_category = "UNKNOWN"
        pending_classify_world_pos = None
        pending_classify_rgb_frame = None
        step_start_pos = list(current_pos)
        if status_label:
            status_label.text = "Status: False Alarm -> Resuming Sweep"
        return
    
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
            initiate_bypass_sequence(force_wide=(stress_band == "HIGH"))
            return
        elif center_blocked or left_pct >= 40 or right_pct >= 40:
            print(f"[VLM Override Skipped]: VLM claimed blockage but depth sensors disagree "
                f"(Ctr={_ros_node.depth_center_dist:.2f}m, L={_ros_node.depth_left_dist:.2f}m, "
                f"R={_ros_node.depth_right_dist:.2f}m). Trusting depth, proceeding as false alarm.")
        


    # --- 0. INITIAL CLASSIFICATION RESOLUTION ---
    if avoidance_sub_phase == "CLASSIFY_OBSTACLE":
        active_obstacle_category = category
        active_obstacle_risk = decision_dict.get("risk_score", 5)
        currently_stationary = decision_dict.get("currently_stationary", True)
        active_obstacle_aggressive = (category == "BIOLOGICAL" and not currently_stationary)

        snapshot_frame = pending_classify_rgb_frame if pending_classify_rgb_frame is not None else _ros_node.latest_rgb
        folder = "UNVERIFIED" if "fallback" in reasoning.lower() else category
        save_burst_snapshots(snapshot_frame, folder, cmd)
        pending_classify_rgb_frame = None

        if pending_classify_world_pos is not None and category in ("BIOLOGICAL", "STATIC_BAR", "MINOR_SURFACE"):
            record_obstacle_memory(pending_classify_world_pos, category, active_obstacle_risk, currently_stationary)

        # --- TRAUMA CHECK FIRST ---
        if pending_classify_world_pos is not None:
            obs_pos = pending_classify_world_pos
        else:
            obs_pos = estimate_obstacle_world_pos(
                current_pos, current_yaw, _ros_node.closest_obstacle_dist
            )
        trauma = find_trauma_match(obs_pos)
        if trauma is not None:
            print(f"[TRAUMA RECALL]: Known attack site. Instant HIGH-band response.")
            stress_level = STRESS_MAX
            stress_target = STRESS_MAX
            stress_band = "HIGH"
            stress_locked = True
            pending_classify_world_pos = None
            initiate_bypass_sequence(force_wide=True)
            return

        pending_classify_world_pos = None

        # ============================================================
        # STRESS SYSTEM: HIGH band OVERRIDES category-driven strategy.
        # A robot in high alert takes active evasive/retreat steps rather
        # than passively yielding, even for a biological -- especially if
        # the VLM has flagged it as approaching/agitated.
        # ============================================================
        if stress_band == "HIGH" and category not in ["CLEAR", "MINOR_SURFACE"]:
            print(f"[HIGH STRESS OVERRIDE]: Category='{category}', aggressive={active_obstacle_aggressive}. "
                f"Skipping normal strategy, committing to active evasive retreat.")
            if status_label:
                status_label.text = "Status: HIGH ALERT -> Active Evasion"
            initiate_bypass_sequence(force_wide=True)
            return

        if category == "BIOLOGICAL" or cmd == "yield_and_wait":
            state = "YIELD_WAITING"
            yield_wait_start_time = time.time()
            yield_wait_total_start_time = time.time()
            
            if status_label:
                status_label.text = "Status: BIOLOGICAL -> Yielding & Waiting"
            print("[Strategy - Yield & Wait]: Animal detected. Stopping and waiting for path to clear...")

        elif category in ["STATIC_BAR", "LARGE_OBJECT"] or cmd == "standard_bypass":
            print(f"[Strategy - Standard Bypass]: Obstacle ({category}) detected. Executing standard bypass...")
            initiate_bypass_sequence(force_wide=False)

        elif category == "MINOR_SURFACE" or cmd == "scan_and_proceed":
            state = "SCAN_AND_PROCEED"
            push_progress = 0.0
            push_target = 1.0
            if status_label:
                status_label.text = "Status: MINOR_SURFACE -> Walking Directly Over Feature"
            print("[Strategy - Scan & Proceed]: Surface defect/pothole detected. Continuing directly in lane...")

        else:
            print(f"[Strategy - Default Bypass]: Category '{category}' -> Executing standard bypass...")
            initiate_bypass_sequence(force_wide=False)


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
                    scale_by_risk(
                        initial_detected_distance + CLEAR_PASS_SAFETY_PAD + extra_distance,
                        active_obstacle_risk,
                        RISK_FLANK_MULTIPLIER_RANGE
                    )
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
            micro_step_target = MIN_DIVERGE_STEP 
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
            # STRESS SYSTEM: Medium (end-check) / High (per-transition) rescan
            if stress_band in ["MEDIUM", "HIGH"]:
                trigger_stress_recheck("VERIFY_ENTRY_FLANK_INSPECT")

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
    global known_obstacles, pending_classify_world_pos
    global stress_level, stress_target, stress_band, stress_locked
    global current_encounter_id, active_obstacle_aggressive
    global last_speed_multiplier, last_potential
    global pending_band_abandon
    global attack_suspected, attack_start_time, attacked_tracks
    global max_closing_velocity_observed
    global _prev_bumper_dwell
    global consecutive_blocked_shifts
    global _loop_restart_pending, _loop_restart_time   # <-- ADD
    global pending_classify_rgb_frame
    global repeated_backup_count, repeated_backup_pos
    global emergency_backup_start_time, emergency_escape_turns
    global emergency_original_yaw, escape_progress
    global vlm_inspect_start

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
    rgb_frame = _ros_node.latest_rgb

    # ============================================================
    # STRESS SYSTEM: update every frame, before any state handling, so
    # current_forward_speed() / stress_band reflect the latest value in
    # whichever branch runs below this point.
    # ============================================================
    update_stress_system(dt)

    if not attack_suspected and attacked_tracks:
        if _ros_node.closest_obstacle_dist != float('inf'):
            obs_pos = estimate_obstacle_world_pos(
                current_pos, current_yaw, _ros_node.closest_obstacle_dist
            )
            record_trauma(obs_pos, active_obstacle_category, max_closing_velocity_observed)
        attacked_tracks = []
        max_closing_velocity_observed = 0.0
        # Depth-only trauma: fire when bumper contact ends
        _dr = globals().get("_last_depth_result")
        if _dr:
            curr_bumper = _dr["bumper_dwell"]
            global _prev_bumper_dwell
            if _prev_bumper_dwell > 0.5 and curr_bumper == 0.0:
                if _ros_node.closest_obstacle_dist != float('inf'):
                    obs_pos = estimate_obstacle_world_pos(
                        current_pos, current_yaw, _ros_node.closest_obstacle_dist
                    )
                    record_trauma(obs_pos, active_obstacle_category, _dr["closing_velocity"])
                depth_monitor.bumper_hits = 0
            _prev_bumper_dwell = curr_bumper


    if stress_label:
        stress_label.text = f"Stress: {stress_level:.1f}/10 [{stress_band}]  Potential: {last_potential:.2f}"
    update_strip_chart()

    # A HIGH-band mid-maneuver LLM recheck can recommend abandoning the
    # current avoidance plan entirely -- handle that before anything else
    # this frame, reusing the existing emergency-backup mechanism.
    if pending_band_abandon and state not in ["STOPPED", "USER_PAUSED", "EMERGENCY_BACKUP"]:
        pending_band_abandon = False
        print("[Band Escalation]: LLM recommended abandon & retreat. Backing off.")
        initiate_emergency_backup()
        return


    if state not in ["MOVING", "STOPPED", "USER_PAUSED", "EMERGENCY_BACKUP", "VLM_INSPECTING", "EMERGENCY_ESCAPE_TURN", "EMERGENCY_ESCAPE_BACK", "EMERGENCY_ESCAPE_REALIGN"]:
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
        print(f"            Stress: {stress_level:.2f}/10 [{stress_band}] | Potential: {last_potential:.3f} | SpeedMult: {last_speed_multiplier:.2f}")
        dr = globals().get("_last_depth_result")
        if dr:
            print(f"            DepthThreat: {dr['signature']:8s} "
                f"score={dr['score']:.1f} "
                f"v={dr['closing_velocity']:+.2f} m/s "
                f"a={dr['acceleration']:+.2f} m/s² "
                f"bumper={dr['bumper_dwell']:.2f}s "
                f"(hits={dr['bumper_hits']})")
            print("-" * 65)

    # ============================================================
    # AUTOMATIC RESUME FUNCTION: Triggered ONLY when biological clears
    # ============================================================
    if state == "YIELD_WAITING":
        effective_max_wait = scale_by_risk(
            MAX_YIELD_WAIT, active_obstacle_risk, RISK_YIELD_MULTIPLIER_RANGE
        )
        effective_check_interval = scale_by_risk(
            YIELD_WAIT_DURATION, active_obstacle_risk, RISK_YIELD_MULTIPLIER_RANGE
        )

        # 1. Hard timeout: waited too long overall
        if (now - yield_wait_total_start_time) >= effective_max_wait:
            print(f"[Yield Timeout]: Waited {effective_max_wait:.1f}s (risk-scaled), still present. "
                f"Initiating bypass as if STATIC_BAR.")
            active_obstacle_category = "STATIC_BAR"
            lateral_detour_ledger = 0.0
            llm_avoidance_history.clear()
            yield_wait_start_time = 0.0
            yield_wait_total_start_time = 0.0
            initiate_bypass_sequence(force_wide=(stress_band == "HIGH"))
            return

        # 2. Only re-check the path every effective_check_interval seconds
        if (now - yield_wait_start_time) >= effective_check_interval:
            yield_wait_start_time = now  # reset the check timer

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
            else:
                print(f"[Yield Waiting]: Obstacle still present (risk={active_obstacle_risk}). Continuing wait...")
                
    # ============================================================
    # 1. NORMAL SWEEP FORWARD & OBSTACLE DETECTION TRIGGER
    # ============================================================
    if state == "MOVING":
        lane_base_yaw = current_yaw

        if compute_dist_2d(current_pos, TARGET_DESTINATION) <= TARGET_RADIUS:
            finish_run("Destination Reached")
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
                    current_encounter_id += 1
                    print("[Obstacle Detected]: Deterministic STATIC_BAR bypass initiated.")
                    initiate_bypass_sequence(force_wide=(stress_band == "HIGH"))
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
            # STRESS SYSTEM: a new obstacle encounter begins here -- bump the
            # encounter id so stale mid-maneuver rechecks from a PREVIOUS
            # obstacle get discarded rather than acted on.
            current_encounter_id += 1

            obstacle_world_pos = estimate_obstacle_world_pos(current_pos, current_yaw, min_dist_ahead)

            # --- 1. YOLO fast-path FIRST (fastest, no network, no stale cache) ---
            yolo_match = None if VLM_PRIORITY_MODE else find_confident_yolo_match(min_dist_ahead)
            if yolo_match is not None:
                print(f"[YOLO Fast-Path]: Confident '{yolo_match['class_name']}' -> "
                    f"{yolo_match['category']} (conf={yolo_match['confidence']:.2f}, "
                    f"track={yolo_match['track_id']}). Acting immediately, VLM refines in background.")

                record_obstacle_memory(obstacle_world_pos, yolo_match["category"],
                                        yolo_match["risk_score"], yolo_match["currently_stationary"])

                pending_classify_world_pos = obstacle_world_pos
                pending_classify_rgb_frame = rgb_frame
                avoidance_sub_phase = "CLASSIFY_OBSTACLE"
                vlm_requested_phase = "CLASSIFY_OBSTACLE"
                on_vlm_agent_decision({
                    "category": yolo_match["category"],
                    "risk_score": yolo_match["risk_score"],
                    "currently_stationary": yolo_match["currently_stationary"],
                    "obstacle_visible": True,
                    "command": {
                        "BIOLOGICAL": "yield_and_wait",
                        "MINOR_SURFACE": "scan_and_proceed",
                    }.get(yolo_match["category"], "standard_bypass"),
                    "reasoning": f"YOLO fast-path: '{yolo_match['class_name']}' held for "
                                f"{YOLO_CONFIDENCE_HITS_REQUIRED}+ consecutive frames at "
                                f"{yolo_match['confidence']:.2f} confidence."
                })

                captured_id = current_encounter_id
                captured_pos = obstacle_world_pos
                def _refine_cb(d, _id=captured_id, _pos=captured_pos):
                    on_vlm_refine_decision(d, _id, _pos)
                async_query_vlm_agent(rgb_frame, min_dist_ahead, "YOLO_REFINE", lateral_detour_ledger, _refine_cb)
                return

            # --- 2. Memory cache second (now guaranteed to have a real category) ---
            cached = None if VLM_PRIORITY_MODE else find_cached_obstacle(obstacle_world_pos)
            if cached is not None:
                print(f"[Memory Hit]: Obstacle at ({obstacle_world_pos[0]:.2f}, {obstacle_world_pos[1]:.2f}) "
                    f"matches known '{cached['category']}' (risk={cached.get('risk_score', 5)}) — skipping VLM.")
                pending_classify_world_pos = obstacle_world_pos
                pending_classify_rgb_frame = rgb_frame
                avoidance_sub_phase = "CLASSIFY_OBSTACLE"
                vlm_requested_phase = "CLASSIFY_OBSTACLE"
                on_vlm_agent_decision({
                    "category": cached["category"],
                    "risk_score": cached.get("risk_score", 5),
                    "currently_stationary": cached.get("stationary", True),
                    "obstacle_visible": True,
                    "command": {
                        "BIOLOGICAL": "yield_and_wait",
                        "MINOR_SURFACE": "scan_and_proceed",
                    }.get(cached["category"], "standard_bypass"),
                    "reasoning": "Reused from spatial obstacle memory (no new VLM query)."
                })
                return

            # --- 3. VLM fallback last ---
            print(f"[Memory Miss]: New obstacle at ({obstacle_world_pos[0]:.2f}, {obstacle_world_pos[1]:.2f}). "
                f"YOLO not confident -- querying VLM directly...")
            pending_classify_world_pos = obstacle_world_pos
            pending_classify_rgb_frame = rgb_frame
            avoidance_sub_phase = "CLASSIFY_OBSTACLE"
            vlm_requested_phase = "CLASSIFY_OBSTACLE"
            state = "VLM_INSPECTING"
            vlm_inspect_start = now     
            if status_label:
                status_label.text = "Status: Classifying New Obstacle (VLM)..."
            async_query_vlm_agent(rgb_frame, min_dist_ahead, "CLASSIFY_OBSTACLE", lateral_detour_ledger, on_vlm_agent_decision)
            return
         
        move_forward(speed=current_forward_speed(), dt=dt)

        hard_limit_hit = (current_pos[0] > max_x) or (current_pos[0] < min_x)


        if compute_dist_2d(current_pos, TARGET_DESTINATION) <= TARGET_RADIUS or is_target_x_reached() or hard_limit_hit:
            print(f"[Destination Reached]: X={current_pos[0]:.2f} >= {TARGET_DESTINATION[0]:.2f}")
            finish_run("Destination Reached" if not hard_limit_hit else "Hard Limit Reached")
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
    # 2 FALSE ALARM RECOVERY PUSH
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

        step_dist = current_forward_speed() * 0.6 * dt
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

    elif state == "SCAN_AND_PROCEED":
        step_dist = current_forward_speed() * 0.5 * dt
        move_forward_step(step_dist, dt)
        push_progress += step_dist
        if push_progress >= push_target:
            state = "MOVING"
            avoidance_sub_phase = "IDLE"
            step_start_pos = list(current_pos)
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
        step_dist = current_forward_speed() * maneuver_speed_factor() * dt   # same speed as normal steps, stress-scaled
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

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            micro_step_target = AVOIDANCE_STEP_SIZE
            micro_step_progress = 0.0
            state = "AVOID_DIVERGE_STEP"      # drives with speed*dt, updates ledger

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
        step_dist = current_forward_speed() * maneuver_speed_factor() * dt
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
            REQUIRED_LATERAL = MIN_CHASSIS_CLEAR + 0.30   # tune per obstacle size
            lateral_ok = abs(lateral_detour_ledger) >= REQUIRED_LATERAL

            if critical_side_depth < 0.60 or not lateral_ok:
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
            # STRESS SYSTEM: HIGH band re-consults the LLM at this
            # avoidance-stage transition rather than relying purely on the
            # deterministic depth check below.
            if stress_band == "HIGH":
                avoidance_sub_phase = "FLANK_PRE_CHECK"
                trigger_stress_recheck("FLANK_PRE_CHECK_SETTLE")

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
                # STRESS SYSTEM: Medium (end-check) / High (per-transition) rescan
                if stress_band in ["MEDIUM", "HIGH"]:
                    trigger_stress_recheck("VERIFY_ENTRY_INSPECT_FLANK")

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
            finish_run("Target X Reached")
            return
        step_dist = current_forward_speed() * maneuver_speed_factor() * dt
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
            avoidance_sub_phase = "IDLE"   # STRESS SYSTEM: this unlocks stress decay (encounter resolved)
            llm_avoidance_history.clear()
            step_start_pos = list(current_pos)

        update_usd_transform(current_pos, current_yaw)

    # ============================================================
    # EMERGENCY BACKUP STATE
    # ============================================================
    elif state == "EMERGENCY_BACKUP":
        # Back up continuously until the obstacle is clear (>= 0.6m)
        step_dist = EMERGENCY_BACKUP_SPEED * dt
        backup_progress += step_dist
        
        # Move backward (negative forward direction)
        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] -= fwd_x * step_dist
        current_pos[1] -= fwd_y * step_dist
        
        # Update the current sensor reading
        min_dist_ahead = _ros_node.closest_obstacle_dist
        
        if (now - emergency_backup_start_time) >= EMERGENCY_BACKUP_STUCK_TIME \
                and emergency_escape_turns < EMERGENCY_MAX_ESCAPE_TURNS:
            esc_yaw = pick_escape_yaw()
            if esc_yaw is not None:
                print(f"[Emergency Backup]: Backing for {now - emergency_backup_start_time:.1f}s "
                      f"without clearing (likely against edge). Turning away to {esc_yaw:.0f}°.")
                emergency_escape_turns += 1
                emergency_original_yaw = current_yaw
                target_yaw = esc_yaw
                escape_progress = 0.0
                state = "EMERGENCY_ESCAPE_TURN"
                if status_label:
                    status_label.text = "Status: Emergency Backup stuck -> Turning Away"
                update_usd_transform(current_pos, current_yaw)
                return        # Condition to STOP backing up: obstacle is clear enough OR we've backed up too far

        if min_dist_ahead >= EMERGENCY_BACKUP_CLEAR_DIST:
            print(f"[Emergency Backup]: Obstacle cleared after backing up {backup_progress:.2f}m. Resuming.")
            deadlock_retreat_count = 0
            emergency_triggered = False
            backup_progress = 0.0
            if repeated_backup_count >= REPEATED_BACKUP_LIMIT:
                print(f"[Emergency Backup]: {repeated_backup_count} repeated backups near same spot. Skipping lane instead of retrying.")
                repeated_backup_count = 0
                repeated_backup_pos = None
                trigger_early_lane_shift()
                return
            state = "MOVING"

        elif backup_progress >= EMERGENCY_BACKUP_MAX_DIST:
            # We backed up a LOT and it's still blocked - give up on this lane
            print("[Emergency Backup]: Max backup distance reached, obstacle still present. Giving up on this lane.")
            deadlock_retreat_count += 1
            emergency_triggered = False
            backup_progress = 0.0
            
            # NEW COMBINED LOGIC: On the THIRD max backup, 
            # 1. Face forward (break 90-degree trap)
            # 2. Drive forward a bit (clear the chokehold)
            # 3. Trigger the standard lane shift (handle horizontal car)
            if deadlock_retreat_count >= 3:
                print("[DEADLOCK RESET]: 3 backups reached. Resetting heading to forward, driving clear, then shifting lane.")
                deadlock_retreat_count = 0
                avoidance_sub_phase = "IDLE"

                # Set the new lane_base_yaw to face forward towards the target
                if get_progress_dir_x() > 0:
                    lane_base_yaw = 0.0
                else:
                    lane_base_yaw = 180.0
                
                # Set state to turn to the new lane direction
                target_yaw = lane_base_yaw
                state = "DEADLOCK_RESET_TURN"
                
                if status_label:
                    status_label.text = "Status: Deadlock Reset - Turning Forward"
                return
            else:
                # Standard second backup behavior (go to force retreat)
                print("[DEADLOCK RESOLVED]: Backing off to lane and skipping current lane.")
                deadlock_retreat_count = 0
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
        # Blind back-up along the new heading, off the wall. No clear-check:
        # the camera is now facing sideways, so its reading is meaningless here.
        step_dist = EMERGENCY_BACKUP_SPEED * dt
        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] -= fwd_x * step_dist
        current_pos[1] -= fwd_y * step_dist
        escape_progress += step_dist
        if escape_progress >= escape_target_dist or room_along(current_pos, current_yaw) < 0.02:
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
            deadlock_retreat_count = 0
            avoidance_sub_phase = "IDLE"
            state = "MOVING"
            step_start_pos = list(current_pos)
        update_usd_transform(current_pos, current_yaw)

    elif state == "VLM_INSPECTING":
        waited = now - vlm_inspect_start
        if status_label:
            status_label.text = f"Status: Halted - waiting for VLM ({waited:.0f}s)"
        if waited > VLM_TIMEOUT + 10.0:
            print("[VLM Watchdog]: no reply, falling back to standard bypass.")
            on_vlm_agent_decision({
                "category": "STATIC_BAR", "obstacle_visible": True,
                "command": "standard_bypass",
                "reasoning": "Deterministic fallback: VLM watchdog timeout."
            })
        update_usd_transform(current_pos, current_yaw)

    elif state == "DEADLOCK_FLANK_PAST":
        yaw_diff = get_shortest_angle_diff(lane_base_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
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
        
        # Stop retreating, face forward, and resume normal sweeping
        if backup_progress >= force_retreat_target or room_along(current_pos, current_yaw) < 0.02:
            print("[Force Retreat Complete]: No lane shift, just resuming forward.")
            emergency_triggered = False
            backup_progress = 0.0
            avoidance_sub_phase = "IDLE"
            # Face the target direction
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
            finish_run("Target X Reached")
            return

        if _ros_node.closest_obstacle_dist < 0.60:
            # Position-independent counter: a robot that inches forward between
            # blocks is still stuck. Do NOT reset on small position deltas —
            # that's what prevented big-lane-skip from ever firing.
            consecutive_blocked_shifts += 1
            lane_stuck_pos = list(current_pos)

            print(f"[LANE_SHIFT BLOCKED]: Obstacle detected during shift. "
                  f"Consecutive blocks={consecutive_blocked_shifts}/{LANE_STUCK_THRESHOLD}.")

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
        dist = current_forward_speed() * dt
        current_pos[0] += fwd_x * dist
        current_pos[1] += fwd_y * dist
        shifted_distance += dist

        if shifted_distance >= LANE_SHIFT:
            state = "TURNING_2"
            sweep_direction_y = -sweep_direction_y
            target_yaw = 90.0 if sweep_direction_y > 0 else 270.0
            lane_stuck_count = 0
            lane_stuck_pos = None
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
        step_dist = current_forward_speed() * 0.6 * dt
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
        step_dist = current_forward_speed() * 0.6 * dt
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

    elif state == "DEADLOCK_RESET_TURN":
        yaw_diff = get_shortest_angle_diff(lane_base_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = lane_base_yaw
            state = "DEADLOCK_RESET_DRIVE"
            deadlock_retreat_count = 0 # reset for next time
        
        update_usd_transform(current_pos, current_yaw)

    elif state == "DEADLOCK_RESET_DRIVE":
        # Drive forward just enough to get out of the chokehold (0.5m)
        step_dist = current_forward_speed() * dt
        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] += fwd_x * step_dist
        current_pos[1] += fwd_y * step_dist
        backup_progress += step_dist  # reuse this variable to track the drive distance

        if backup_progress >= 0.5:
            print("[Deadlock Reset Complete]: Triggering standard lane shift.")
            backup_progress = 0.0
            # NOW we trigger the normal horizontal car solution
            trigger_early_lane_shift()
            return

        update_usd_transform(current_pos, current_yaw)

if os.path.exists(YOLO_ONNX_PATH):
    init_yolo(YOLO_ONNX_PATH)
    threading.Thread(target=yolo_worker, daemon=True).start()
    print("[YOLO] Model loaded and worker started.")
else:
    print(f"[YOLO] Disabled — no model at {YOLO_ONNX_PATH}. "
        f"Attack detection will report zero until a model exists.")

# Register update loop
global _phantomx_sub
_phantomx_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(zig_zag_update)

