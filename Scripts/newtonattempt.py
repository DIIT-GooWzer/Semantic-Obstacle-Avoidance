#!/usr/bin/env python3
"""
PhantomX Target-Directed Controller with Class-Based VLM Obstacle Avoidance
(Isaac Sim NVIDIA Newton Physics & Joint Actuation Edition)
==============================================================================
Features:
1. Joint-Actuated Locomotion (NO Kinematic Teleportation):
   - Uses Isaac Sim Newton physics with angular DriveAPI on all 18 revolute joints
     (coxa, femur, tibia across 6 legs: LF, LM, LR, RF, RM, RR).
   - Driven by 6-oscillator Hopf CPG + PPO Actor checkpoint (with automatic pure
     CPG fallback if checkpoint is absent).
   - Dynamic velocity command integration (cmd_vx, cmd_vy, cmd_wz) at 50 Hz.
2. Real-Physics Odometry Feedback:
   - Reads authoritative world pose and heading directly from USD transform cache.
3. Multi-Class VLM Obstacle Classification:
   - BIOLOGICAL: Yield & wait (monitors clear depth until path opens).
   - STATIC_BAR: Standard bypass (smart divergence, flanking, lane return).
   - MINOR_SURFACE: Scan & proceed directly over cracks/depressions.
   - CLEAR: False detection recovery with forward push.
4. Smart Avoidance & Edge-Aware Navigation:
   - Distinguishes small obstacles (cone) vs wide obstacles (vehicle).
   - Boundary-aware lane skip when obstacles hit road edges.
   - Emergency backup cooldown and deadlock recovery.
5. Designed for execution in Isaac Sim GUI Script Editor.
"""

from __future__ import annotations

import os
import sys
import time
import math
import json
import base64
import threading
import urllib.request
import numpy as np

# Ensure attempt folder and llmforhexapod are available on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "/home/user/Downloads/isaacsim_preview-20260903T081251Z-1-001/isaacsim_preview/attempt"
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if "/home/user/Desktop/llmforhexapod" not in sys.path:
    sys.path.insert(0, "/home/user/Desktop/llmforhexapod")

import torch
import torch.nn as nn
import omni.usd
import omni.timeline
import omni.kit.app
import omni.ui as ui
from pxr import Usd, UsdGeom, UsdPhysics, Gf

from cpg_controller import ACTION_LAYOUT, CPGJointController, map_ik_to_newton_asset
from inverse_kinematics_urdf import neutral_joint_angles

try:
    import isaacsim.physics.newton as newton_ext
    _ns = newton_ext.acquire_stage()
    if _ns is not None:
        _ns.cfg.collapse_fixed_joints = True
        _ns.cfg.joint_damping = 0.05
        if hasattr(_ns.cfg.solver_cfg, "nconmax"):
            _ns.cfg.solver_cfg.nconmax = 600
        if hasattr(_ns.cfg.solver_cfg, "njmax"):
            _ns.cfg.solver_cfg.njmax = 4096
        print("[Newton]: Configured contact buffer (nconmax=600, njmax=4096).")
except Exception as _e:
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


if "_ros_node" in globals() and _ros_node is not None:
    try:
        _ros_node.destroy_node()
        print("[Cleanup]: Previous ROS perception node destroyed.")
    except Exception as exc:
        print(f"[ROS Cleanup Warning]: {exc}")
    _ros_node = None

global newton_actuators

if "newton_actuators" in globals() and newton_actuators is not None:
    try:
        newton_actuators.close()
        print("[Cleanup]: Previous Newton actuator manager closed.")
    except Exception as exc:
        print(f"[Actuator Cleanup Warning]: {exc}")
    newton_actuators = None



def ros_img_to_numpy(msg, desired_encoding="passthrough"):
    if _has_cv2:
        try:
            bridge = CvBridge()
            return bridge.imgmsg_to_cv2(msg, desired_encoding=desired_encoding)
        except Exception:
            pass

    if msg.encoding in ["32FC1", "32f"]:
        return np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
    elif msg.encoding in ["rgb8", "bgr8", "8UC3"]:
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        if desired_encoding == "bgr8" and msg.encoding == "rgb8":
            arr = arr[:, :, ::-1]
        return arr
    elif msg.encoding in ["mono8", "8UC1"]:
        return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
    else:
        return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))


stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()
if not timeline.is_playing():
    timeline.play()

# ============================================================
# 1. CONFIGURATION & NAVIGATION CONSTANTS
# ============================================================
ROBOT_PATH         = "/World/PhantomX"
SPAWN_POSITION     = (0.1195227, 1.4619421832676414, 0.185)
SPAWN_ROTATION     = (0.0, 0.0, -90.0)
TARGET_DESTINATION = (6.1992602, -1.8266811)
TARGET_RADIUS      = 0.35

STEP_DISTANCE       = 0.40                            # Sweep pause step size (m)
PAUSE_DURATION      = 0.50                            # Sweep pause duration (s)
FORWARD_SPEED       = 0.22                            # Physical translation speed (m/s)
BACKUP_SPEED        = 0.15                            # Reverse translation speed (m/s)
MAX_TURN_RATE_RAD   = 1.20                            # Max yaw-rate command (rad/s)
TURN_SPEED_DEG      = 60.0                            # Nominal turning rate (deg/s)
LANE_SHIFT          = 0.50                            # Lane shift width (m)
OBSTACLE_THRESHOLD  = 0.0001                           # Avoidance trigger distance (m)
MIN_CHASSIS_CLEAR   = 0.40                            # Minimum lateral distance to clear chassis (m)
PRINT_INTERVAL      = 2.0                             # Telemetry print interval (s)
YIELD_WAIT_DURATION = 3.0                             # Seconds to wait before checking if animal moved
MAX_YIELD_WAIT      = 10.0                            # Max seconds to wait before bypassing biological

# Avoidance micro-step settings
AVOIDANCE_STEP_SIZE   = 0.30
DIVERGE_ANGLE_INC     = 35.0
MIN_VLM_COOLDOWN      = 2.0

# Emergency Backup Settings
EMERGENCY_STOP_DIST     = 0.32
EMERGENCY_BACKUP_DIST   = 0.45
EMERGENCY_BACKUP_SPEED  = 0.15
EMERGENCY_COOLDOWN_DIST = 1.20
EMERGENCY_COOLDOWN_TIME = 4.0

CLEAR_PASS_SAFETY_PAD = 0.30
VERIFY_SKIP_DIST      = 2.0

FLANK_EXTRA_DISTANCE = {
    "STATIC_BAR": 0.35,
    "BIOLOGICAL": 0.80,
    "MINOR_SURFACE": 0.0,
}

FLANK_MIN_DISTANCE = {
    "STATIC_BAR": 0.70,
    "BIOLOGICAL": 1.20,
}

FLANK_MAX_DISTANCE = {
    "STATIC_BAR": 1.50,
    "BIOLOGICAL": 2.50,
}

DEFAULT_FLANK_EXTRA = 0.50
DEFAULT_FLANK_MIN   = 0.90
DEFAULT_FLANK_MAX   = 2.00

FALSE_ALARM_PUSH_MIN    = 0.30
FALSE_ALARM_PUSH_MAX    = 0.80
FALSE_ALARM_PUSH_MARGIN = 0.40

# VLM API Configuration
VLM_API_URL = "http://localhost:8000/v1/chat/completions"
VLM_API_KEY = os.getenv("FREEBUFF_API_KEY", "231970c3-bd57-41a8-8368-1e2d81176618")
VLM_MODEL   = "deepseek/deepseek-v4-flash"

RGB_TOPICS   = ["/rgb", "/camera/rgb", "/phantomx/camera/rgb"]
DEPTH_TOPICS = ["/depth", "/camera/depth", "/phantomx/camera/depth"]

# Canonical joint order matching cpg_controller.step()
LEG_ORDER = ("LF", "LM", "LR", "RF", "RM", "RR")
CANONICAL_JOINT_NAMES = tuple(
    f"{joint}_joint_{leg}" for joint in ("coxa", "femur", "tibia") for leg in LEG_ORDER
)
LOWER_LIMITS = np.radians([-40] * 6 + [-100] * 6 + [-145] * 6).astype(np.float32)
UPPER_LIMITS = np.radians([40] * 6 + [85] * 6 + [-5] * 6).astype(np.float32)

# ============================================================
# 2. LOCOMOTION ENGINE (PPO POLICY + HOPF CPG)
# ============================================================
class ActorNetwork(nn.Module):
    RAW_MEAN_LIMIT = 1.25

    def __init__(self, obs_dim=1120, action_dim=37):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
        )
        self.actor_mean = nn.Linear(256, action_dim)

    def forward(self, observation):
        raw = self.actor_mean(self.actor(observation))
        return self.RAW_MEAN_LIMIT * torch.tanh(raw / self.RAW_MEAN_LIMIT)


class PhantomXNewtonPolicy:
    """Combines 6-oscillator Hopf CPG with optional trained PPO policy checkpoint."""

    def __init__(self, checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dt = 0.02
        self.cpg = CPGJointController(worlds=1, seed=20260903)
        self.actor = None
        self.obs_history = np.zeros((8, 140), dtype=np.float32)

        candidates = [
            checkpoint_path,
            "/home/user/Downloads/ppo_final.pt",
            os.path.join(SCRIPT_DIR, "ppo_final.pt"),
        ]

        loaded = False
        for path_str in candidates:
            if path_str and os.path.isfile(path_str):
                try:
                    saved = torch.load(path_str, map_location=self.device, weights_only=False)
                    actor_obs_dim = int(saved.get("actor_obs_dim", 1120))
                    action_dim = int(saved.get("action_dim", 37))
                    self.actor = ActorNetwork(obs_dim=actor_obs_dim, action_dim=action_dim).to(self.device)

                    state_dict = saved.get("model", saved)
                    actor_state = {}
                    for k, v in state_dict.items():
                        if k.startswith("actor.") or k.startswith("actor_mean."):
                            actor_state[k] = v
                    if actor_state:
                        self.actor.load_state_dict(actor_state, strict=False)
                    self.actor.eval()
                    loaded = True
                    print(f"[Policy Loaded]: PPO checkpoint loaded from {path_str}")
                    break
                except Exception as exc:
                    print(f"[Policy Load Warning]: Failed to load {path_str}: {exc}")

        if not loaded:
            print("[Locomotion Engine]: Operating in validated pure analytical CPG mode (no checkpoint).")

    def compute_joint_targets(self, cmd_vx: float, cmd_vy: float, cmd_wz: float) -> np.ndarray:
        if self.actor is not None:
            try:
                obs_tensor = torch.from_numpy(self.obs_history.flatten()).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    actions = self.actor(obs_tensor).cpu().squeeze(0).numpy()
                self.cpg.set_policy_action(np.expand_dims(actions, 0))
            except Exception:
                pass

        commands = np.array([[cmd_vx, cmd_vy, cmd_wz]], dtype=np.float32)
        raw_targets = self.cpg.step(self.dt, commands)[0]
        return np.clip(raw_targets, LOWER_LIMITS, UPPER_LIMITS)


global policy_engine
policy_engine = PhantomXNewtonPolicy()

# ============================================================
# 3. ROBOT SPAWNING & NEWTON ARTICULATION DRIVE SETUP
# ============================================================
global joint_drives
joint_drives = {}

PHYSICS_PROXY_NAMES = frozenset({"collision", "foot_collider", "inertial_proxy"})


def hide_physics_proxies(stage: Usd.Stage, root_path: str):
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in stage.Traverse():
            path_str = str(prim.GetPath())
            if path_str.startswith(f"{root_path}/") and prim.GetName() in PHYSICS_PROXY_NAMES:
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.MakeInvisible()


def find_robot_physics_root(stage, outer_path):
    candidates = []
 
    for prim in stage.Traverse():
        path = str(prim.GetPath())
 
        if path == outer_path or path.startswith(outer_path + "/"):
            joints_path = f"{path}/Joints"
            if stage.GetPrimAtPath(joints_path).IsValid():
                candidates.append(path)
 
    if not candidates:
        return None
 
    # Prefer the deepest path containing the actual joint hierarchy.
    return max(candidates, key=len)



def spawn_phantomx(stage, position=SPAWN_POSITION, rotation_euler=SPAWN_ROTATION):
    # Hide or deactivate any legacy non-articulated robot prim
    legacy_prim = stage.GetPrimAtPath("/World/phantomX_base")
    if legacy_prim.IsValid():
        imageable = UsdGeom.Imageable(legacy_prim)
        if imageable:
            imageable.MakeInvisible()

    outer_path = ROBOT_PATH
    outer_prim = stage.GetPrimAtPath(outer_path)

    robot_usd_candidates = [
        "/home/user/Desktop/llmforhexapod/netwontraineddata/phantomx_newton.usda",
        "/home/user/Desktop/llmforhexapod/netwontraineddata/phantomx_newton_viewer.usda",
        os.path.join(SCRIPT_DIR, "phantomx_newton.usda"),
    ]

    robot_asset_path = None
    for cand in robot_usd_candidates:
        if os.path.isfile(cand):
            robot_asset_path = cand
            break

    if not robot_asset_path:
        raise FileNotFoundError("Could not locate phantomx_newton.usda articulation asset.")

    if not outer_prim.IsValid():
        outer_prim = stage.DefinePrim(outer_path, "Xform")
        outer_prim.GetReferences().AddReference(assetPath=robot_asset_path, primPath="/World/PhantomX")

    if not outer_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(outer_prim)

    try:
        if not outer_prim.HasAPI("NewtonArticulationAPI"):
            outer_prim.AddAppliedSchema("NewtonArticulationAPI")
    except Exception:
        pass

    xform = UsdGeom.Xformable(outer_prim)
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

    cur_pos = translate_op.Get()
    if cur_pos is None or (abs(cur_pos[0]) < 0.01 and abs(cur_pos[1]) < 0.01):
        translate_op.Set(Gf.Vec3d(*position))
        rotate_op.Set(Gf.Vec3d(*rotation_euler))

    # Configure Angular DriveAPI for each joint
    global joint_drives
    joint_drives = {}

    active_robot_path = find_robot_physics_root(stage, outer_path)
    
    if active_robot_path is None:
        raise RuntimeError(
            f"Could not find a Joints hierarchy below {outer_path}"
        )
    
    print(f"[PhantomX Spawner]: Using physics root {active_robot_path}")


    for jname in CANONICAL_JOINT_NAMES:
        jprim = stage.GetPrimAtPath(f"{active_robot_path}/Joints/{jname}")
        if jprim.IsValid():
            drive = UsdPhysics.DriveAPI.Get(jprim, "angular")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(jprim, "angular")
                drive.CreateStiffnessAttr(300.0)
                drive.CreateDampingAttr(8.0)
                drive.CreateMaxForceAttr(20.0)


            target_attr = drive.GetTargetPositionAttr()
            if not target_attr.IsValid():
                drive.CreateTargetPositionAttr(0.0)
            joint_drives[jname] = drive

    hide_physics_proxies(stage, active_robot_path)
    print(
    f"[PhantomX Spawner]: active_robot_path={active_robot_path} | "
    f"Configured {len(joint_drives)}/18 Newton joint drives."
)


    return stage.GetPrimAtPath(active_robot_path), translate_op, rotate_op



global robot_articulation, dof_indices
robot_articulation = None
dof_indices = None


def get_robot_articulation():
    global robot_articulation, dof_indices
 
    if robot_articulation is not None and dof_indices is not None:
        return robot_articulation
 
    try:
        if newton_actuators is not None:
            robot_articulation = newton_actuators.articulation
        else:
            from isaacsim.core.experimental.prims import Articulation
            robot_articulation = Articulation("/World/PhantomX")
 
        # Newton uses all 18 DOFs in the discovered order.
        dof_indices = np.arange(18, dtype=np.int32)
 
        print(
            f"[Articulation]: Using all Newton DOFs "
            f"count={len(dof_indices)}"
        )
 
    except Exception as exc:
        print(f"[Articulation ERROR]: {exc}")
        robot_articulation = None
        dof_indices = None
 
    return robot_articulation

def set_robot_joint_targets(target_angles_rad: np.ndarray):
    ra = get_robot_articulation()
 
    targets = np.asarray(
        target_angles_rad, dtype=np.float32
    ).reshape(1, 18)
 
    if ra is not None and dof_indices is not None:
        try:
            ra.set_dof_position_targets(
                targets,
                dof_indices=dof_indices,
            )
            return
        except Exception as exc:
            print(f"[Articulation Target ERROR]: {exc}")
 
    print("[Actuation Warning]: Falling back to USD DriveAPI.")
 
    for idx, jname in enumerate(CANONICAL_JOINT_NAMES):
        drive = joint_drives.get(jname)
        if drive is None:
            continue
 
        # USD angular drive targets are degrees.
        drive.GetTargetPositionAttr().Set(
            float(np.degrees(target_angles_rad[idx]))
        )


def get_physics_pose(prim) -> tuple[list[float], float] | tuple[None, None]:
    """Reads the authoritative world translation and yaw from the physics transform."""
    try:
        body_prim = prim
        if prim.IsValid() and prim.GetChild("Links").IsValid() and prim.GetChild("Links").GetChild("body_link").IsValid():
            body_prim = prim.GetChild("Links").GetChild("body_link")

        xform = UsdGeom.Xformable(body_prim)
        world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        trans = world_transform.ExtractTranslation()
        rot = world_transform.ExtractRotation()
        dir_vec = rot.TransformDir(Gf.Vec3d(1, 0, 0))
        yaw = math.degrees(math.atan2(dir_vec[1], dir_vec[0]))
        return [float(trans[0]), float(trans[1]), float(trans[2])], float(yaw)
    except Exception:
        return None, None


robot_prim, translate_op, rotate_op = spawn_phantomx(stage)
 
# Activate the NewtonActuator definitions authored in the USD asset.

newton_actuators = None

 
try:
    from isaacsim.core.experimental.actuators import ArticulationActuators
 
    newton_actuators = ArticulationActuators(
        "/World/PhantomX",
        auto_step_pre_physics=True,
    )
 
    print(
        f"[Newton Actuators]: Discovered "
        f"{len(newton_actuators.actuators)} actuators | "
        f"DOFs={newton_actuators.actuated_dof_indices}"
    )
except Exception as exc:
    newton_actuators = None
    print(f"[Newton Actuators ERROR]: {exc}")
 
# Neutral stance target initialization
neutral_rad = map_ik_to_newton_asset(neutral_joint_angles()).reshape(18)
set_robot_joint_targets(neutral_rad)

# NavMesh Bounds Detection
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
navmesh_candidates = [
    "/World/Terrain/section_00_asphalt/NavMeshVolume",
    "/World/NavMeshVolume",
    "/NavMeshVolume",
]
navmesh_prim = None
for path in navmesh_candidates:
    p = stage.GetPrimAtPath(path)
    if p.IsValid():
        navmesh_prim = p
        break

min_x, max_x = -10.0, 12.0
min_y, max_y = -2.0, 2.0
if navmesh_prim and navmesh_prim.IsValid():
    bound = bbox_cache.ComputeWorldBound(navmesh_prim).ComputeAlignedBox()
    min_x, max_x = float(bound.GetMin()[0]), float(bound.GetMax()[0])
    min_y, max_y = float(bound.GetMin()[1]), float(bound.GetMax()[1])

ROBOT_SAFETY_MARGIN = 0.35
SAFE_MIN_Y = min_y + ROBOT_SAFETY_MARGIN
SAFE_MAX_Y = max_y - ROBOT_SAFETY_MARGIN
SAFE_MIN_X = min_x + ROBOT_SAFETY_MARGIN
SAFE_MAX_X = max_x - ROBOT_SAFETY_MARGIN

# ============================================================
# 4. ROS 2 PERCEPTION NODE WITH SPATIAL DEPTH SLICING
# ============================================================
if not rclpy.ok():
    rclpy.init()


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("phantomx_isaac_perception")

        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.latest_rgb = None
        self.latest_depth = None
        self.closest_obstacle_dist = float("inf")

        self.depth_left_dist = float("inf")
        self.depth_center_dist = float("inf")
        self.depth_right_dist = float("inf")

        self.subs = []
        for topic in RGB_TOPICS:
            self.subs.append(self.create_subscription(Image, topic, self._rgb_callback, qos_best_effort))
            self.subs.append(self.create_subscription(Image, topic, self._rgb_callback, qos_reliable))

        for topic in DEPTH_TOPICS:
            self.subs.append(self.create_subscription(Image, topic, self._depth_callback, qos_best_effort))
            self.subs.append(self.create_subscription(Image, topic, self._depth_callback, qos_reliable))

    def _rgb_callback(self, msg):
        try:
            self.latest_rgb = ros_img_to_numpy(msg, desired_encoding="bgr8")
        except Exception:
            pass

    def _depth_callback(self, msg):
        try:
            depth_frame = ros_img_to_numpy(msg, desired_encoding="32FC1")
            self.latest_depth = depth_frame
            h, w = depth_frame.shape

            row_top, row_bottom = int(h * 0.35), int(h * 0.65)
            left_col = depth_frame[row_top:row_bottom, : int(w * 0.33)]
            center_col = depth_frame[row_top:row_bottom, int(w * 0.33) : int(w * 0.66)]
            right_col = depth_frame[row_top:row_bottom, int(w * 0.66) :]

            def extract_min(roi):
                valid = roi[(np.isfinite(roi)) & (roi > 0.05)]
                return float(np.min(valid)) if len(valid) > 0 else float("inf")

            self.depth_left_dist = extract_min(left_col)
            self.depth_center_dist = extract_min(center_col)
            self.depth_right_dist = extract_min(right_col)
            self.closest_obstacle_dist = min(
                self.depth_left_dist, self.depth_center_dist, self.depth_right_dist
            )
        except Exception:
            pass


global ros_node
_ros_node = PerceptionNode()

# ============================================================
# 5. DYNAMIC TARGET & DIRECTION HELPERS
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
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_target_angle_diff(current_pose, current_heading_deg, target_pose):
    dx = target_pose[0] - current_pose[0]
    dy = target_pose[1] - current_pose[1]
    target_bearing_deg = math.degrees(math.atan2(dy, dx)) % 360.0
    diff = (target_bearing_deg - current_heading_deg + 180.0) % 360.0 - 180.0
    return diff

# ============================================================
# 6. VISION LLM CLIENT WITH CLASSIFICATION
# ============================================================
global vlm_request_in_flight, last_vlm_query_time, llm_avoidance_history
vlm_request_in_flight = False
last_vlm_query_time = 0.0
llm_avoidance_history = []


def encode_frame_to_base64(rgb_array):
    if _has_cv2 and rgb_array is not None:
        h, w = rgb_array.shape[:2]
        if w > 640:
            new_w = 640
            new_h = int(h * (640 / w))
            rgb_array = cv2.resize(rgb_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode(".jpg", rgb_array, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return base64.b64encode(buffer).decode("utf-8")
    return ""


def async_query_vlm_agent(rgb_array, obstacle_dist, current_sub_phase, detour_offset, callback_fn):
    global vlm_request_in_flight, last_vlm_query_time, llm_avoidance_history

    now = time.time()
    if vlm_request_in_flight or (now - last_vlm_query_time) < MIN_VLM_COOLDOWN:
        return

    if state in ["STOPPED", "USER_PAUSED"]:
        return

    # Skip VLM if distance indicates the corridor is wide open
    if current_sub_phase in ["VERIFY", "FLANK_INSPECT", "FLANK_PRE_CHECK"] and obstacle_dist > VERIFY_SKIP_DIST:
        print(f"[{current_sub_phase}]: Depth clear ({obstacle_dist:.2f}m), skipping VLM.")
        callback_fn({
            "command": "return_to_lane",
            "obstacle_visible": False,
            "category": "CLEAR",
            "reasoning": f"Depth sensors confirm clear path ({obstacle_dist:.2f}m).",
        })
        return

    vlm_request_in_flight = True
    last_vlm_query_time = now
    print(f"[VLM Dispatch]: in_flight=True | phase={current_sub_phase} | t={now:.2f}")

    def worker():
        global vlm_request_in_flight, llm_avoidance_history
        try:
            max_retries = 1
            attempt = 0
            while attempt <= max_retries:
                try:
                    if state in ["STOPPED", "USER_PAUSED"]:
                        return

                    b64_img = encode_frame_to_base64(rgb_array)
                    if not b64_img:
                        print("[Perception Error]: RGB image frame is empty or invalid.")
                        return

                    history_summary = ""
                    if len(llm_avoidance_history) > 0:
                        history_summary = "Past avoidance decisions:\n" + "\n".join([f"- {h}" for h in llm_avoidance_history[-3:]]) + "\n"

                    if current_sub_phase == "CLASSIFY_OBSTACLE":
                        prompt = (
                            f"You are the visual perception brain for an autonomous sweeper robot analyzing an obstacle detected at {obstacle_dist:.2f}m.\n"
                            f"If the path is clear or it is a false detection, categorize as CLEAR.\n"
                            f"CRITICAL VISUAL RULES:\n"
                            f"- BIOLOGICAL: Any animal, quadruped, cow, cattle, bull, or livestock model.\n"
                            f"- STATIC_BAR: Traffic cones, bright orange/red plastic markers, construction barriers, bollards, or solid blocks.\n"
                            f"- MINOR_SURFACE: Potholes, road cracks, asphalt patches, ground depressions, or flat surface defects.\n"
                            f"- CLEAR: No obstacle blocking forward progress or clear open corridor.\n\n"
                            f"Divide the image into 3 vertical slices (Left 0-33%, Center 33-66%, Right 66-100%).\n"
                            f"Return ONLY valid JSON:\n"
                            f"{{\n"
                            f'  "category": "BIOLOGICAL" | "STATIC_BAR" | "MINOR_SURFACE" | "CLEAR",\n'
                            f'  "obstacle_class": "Cow/Animal" | "Traffic Cone" | "Pothole/Crack" | "None",\n'
                            f'  "command": "yield_and_wait" | "standard_bypass" | "scan_and_proceed" | "resume_path",\n'
                            f'  "reasoning": "Detailed visual evidence",\n'
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
                            f"Navigating sweeper robot bypassing target ({active_obstacle_category}).\n"
                            f"Depths: Left={left_d:.2f}m, Center={ctr_d:.2f}m, Right={right_d:.2f}m.\n"
                            f"Detour offset: {detour_offset:.2f}m.\n"
                            f"Return ONLY valid JSON:\n"
                            f"{{\n"
                            f'  "primary_obstacle_cleared": bool,\n'
                            f'  "corridor_gap_open": bool,\n'
                            f'  "command": "diverge_more" | "flank_forward",\n'
                            f'  "reasoning": string\n'
                            f"}}"
                        )
                    elif current_sub_phase == "FLANK_INSPECT":
                        prompt = (
                            f"Verifying flank clearance of obstacle.\n"
                            f"Distance: {obstacle_dist:.2f}m. Detour offset: {detour_offset:.2f}m.\n"
                            f"Commands:\n"
                            f"- 'flank_forward': Obstacle still alongside.\n"
                            f"- 'return_to_lane': Obstacle passed; return to lane.\n"
                            f"Return ONLY valid JSON: {{\"obstacle_visible\": bool, \"command\": \"flank_forward\"|\"return_to_lane\", \"reasoning\": string}}"
                        )
                    elif current_sub_phase == "FLANK_PRE_CHECK":
                        prompt = (
                            f"Corridor clearance check for sweeper.\n"
                            f"Sensor depths: Left={_ros_node.depth_left_dist:.2f}m, Center={_ros_node.depth_center_dist:.2f}m, Right={_ros_node.depth_right_dist:.2f}m.\n"
                            f"Return ONLY valid JSON:\n"
                            f"{{\n"
                            f'  "center_blocked": bool,\n'
                            f'  "left_sector_coverage_pct": int,\n'
                            f'  "right_sector_coverage_pct": int,\n'
                            f'  "obstacle_visible": bool,\n'
                            f'  "command": "diverge_more" | "flank_forward",\n'
                            f'  "reasoning": string\n'
                            f"}}"
                        )
                    else:  # VERIFY
                        prompt = (
                            f"Verifying return path to original sweep lane.\n"
                            f"Detour offset: {detour_offset:.2f}m.\n"
                            f"Return ONLY valid JSON: {{\"obstacle_visible\": bool, \"command\": \"return_to_lane\"|\"flank_forward\", \"reasoning\": string}}"
                        )

                    payload = {
                        "model": VLM_MODEL,
                        "temperature": 0.0,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                                ],
                            }
                        ],
                        "response_format": {"type": "json_object"},
                    }

                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {VLM_API_KEY}",
                        "User-Agent": "IsaacSim-PhantomX-Agent/1.0",
                    }

                    req = urllib.request.Request(
                        VLM_API_URL, data=json.dumps(payload).encode("utf-8"), headers=headers
                    )

                    with urllib.request.urlopen(req, timeout=15.0) as resp:
                        if state in ["STOPPED", "USER_PAUSED"]:
                            return
                        result = json.loads(resp.read().decode("utf-8"))
                        raw_reply = result["choices"][0]["message"]["content"]
                        parsed = json.loads(raw_reply)

                        log_entry = f"Phase {current_sub_phase}: Decided '{parsed.get('command')}' ({parsed.get('category', 'N/A')})."
                        llm_avoidance_history.append(log_entry)
                        callback_fn(parsed)
                        return

                except urllib.error.URLError as exc:
                    attempt += 1
                    print(f"[VLM Connection Error (Attempt {attempt}/{max_retries + 1})]: {exc.reason}")
                    time.sleep(0.5)
                except Exception as exc:
                    attempt += 1
                    print(f"[VLM Error (Attempt {attempt}/{max_retries + 1})]: {exc}")
                    time.sleep(0.5)

            # Deterministic safety fallback
            if state not in ["STOPPED", "USER_PAUSED"]:
                print("[VLM Fallback]: Executing deterministic safety rule.")
                if current_sub_phase == "CLASSIFY_OBSTACLE":
                    cmd, cat = "standard_bypass", "STATIC_BAR"
                elif current_sub_phase in ["DIVERGE_SCAN", "FLANK_PRE_CHECK"]:
                    cmd = "diverge_more" if abs(detour_offset) < MIN_CHASSIS_CLEAR else "flank_forward"
                    cat = "STATIC_BAR"
                elif current_sub_phase == "FLANK_INSPECT":
                    cmd = "return_to_lane" if obstacle_dist > 1.2 else "flank_forward"
                    cat = "STATIC_BAR"
                else:
                    cmd, cat = "return_to_lane", "STATIC_BAR"

                callback_fn({
                    "category": cat,
                    "obstacle_visible": True,
                    "command": cmd,
                    "reasoning": "Deterministic fallback safety rule.",
                })
        finally:
            print(f"[VLM Dispatch]: in_flight=False | phase={current_sub_phase} | t={time.time():.2f}")
            vlm_request_in_flight = False

    threading.Thread(target=worker, daemon=True).start()

# ============================================================
# 7. STATE VARIABLES & ODOMETRY LEDGER
# ============================================================
global state, saved_pre_pause_state, current_pos, current_yaw, sweep_direction_y
global step_start_pos, target_yaw, pause_start_time, last_time, shifted_distance, last_print_time
global lane_base_yaw, lateral_detour_ledger, micro_step_progress, micro_step_target, detour_side
global avoidance_sub_phase, initial_detected_distance, required_flank_distance, flank_progress, diverge_turn_count
global active_obstacle_category, yield_wait_start_time, yield_wait_total_start_time
global push_progress, push_target, pre_check_settle_start
global emergency_triggered, emergency_cooldown_start, emergency_trigger_pos
global deadlock_retreat_count, force_retreat_target, backup_progress
global active_robot_path

active_robot_path = ROBOT_PATH


current_pos = list(SPAWN_POSITION)
current_yaw = float(SPAWN_ROTATION[2])
state = "MOVING"
saved_pre_pause_state = "MOVING"

sweep_direction_y = 1.0 if SPAWN_ROTATION[2] == 90.0 else -1.0
step_start_pos = list(current_pos)
target_yaw = current_yaw
pause_start_time = 0.0
last_time = time.time()
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

emergency_triggered = False
emergency_cooldown_start = 0.0
emergency_trigger_pos = [0.0, 0.0]
backup_progress = 0.0
deadlock_retreat_count = 0
force_retreat_target = 1.2


def update_usd_transform(pos, yaw_deg):
    """Kinematic teleportation is DISABLED. Physics drives movement."""
    pass


def record_motion_to_ledger(dx, dy):
    global lateral_detour_ledger, lane_base_yaw
    rad = math.radians(lane_base_yaw)
    cos_b, sin_b = math.cos(rad), math.sin(rad)
    d_lateral = (-dx * sin_b) + (dy * cos_b)
    lateral_detour_ledger += d_lateral


def initiate_emergency_backup():
    global state, emergency_triggered, emergency_cooldown_start, emergency_trigger_pos
    global backup_progress, avoidance_sub_phase, lateral_detour_ledger

    print(f"[Emergency Backup]: Obstacle too close! Backing up {EMERGENCY_BACKUP_DIST:.2f}m...")
    emergency_trigger_pos = list(current_pos)
    emergency_triggered = True
    emergency_cooldown_start = time.time()
    backup_progress = 0.0
    avoidance_sub_phase = "IDLE"
    lateral_detour_ledger = 0.0
    state = "EMERGENCY_BACKUP"
    if status_label:
        status_label.text = f"Status: EMERGENCY BACKUP ({EMERGENCY_BACKUP_DIST:.1f}m)"


def check_emergency_proximity(dist_ahead):
    if state in ["STOPPED", "USER_PAUSED", "EMERGENCY_BACKUP"]:
        return False
    if dist_ahead >= EMERGENCY_STOP_DIST:
        return False
    dist_from_trigger = compute_dist_2d(current_pos, emergency_trigger_pos)
    time_since_trigger = time.time() - emergency_cooldown_start
    if emergency_triggered and (
        dist_from_trigger < EMERGENCY_COOLDOWN_DIST or time_since_trigger < EMERGENCY_COOLDOWN_TIME
    ):
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
    print("[PhantomX]: All robot operations halted.")

# ============================================================
# 8. UI PANEL
# ============================================================
global _phantomx_window
if "_phantomx_window" in globals() and _phantomx_window:
    try:
        _phantomx_window.visible = False
        _phantomx_window.destroy()
    except Exception:
        pass
    _phantomx_window = None

_phantomx_window = ui.Window(
    "PhantomX Newton VLM Controller",
    width=440,
    height=290,
    flags=ui.WINDOW_FLAGS_NO_SAVED_SETTINGS,
)
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
        ui.Label("PhantomX Newton Joint-Actuated Sweeper", height=20, style={"font_size": 14, "color": 0xFF00FFFF})
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

                state = "MOVING"
                vlm_request_in_flight = False
                llm_avoidance_history.clear()
                emergency_triggered = False
                emergency_cooldown_start = 0.0
                emergency_trigger_pos = list(SPAWN_POSITION)
                current_pos = list(SPAWN_POSITION)
                current_yaw = float(SPAWN_ROTATION[2])
                lane_base_yaw = current_yaw
                sweep_direction_y = 1.0 if SPAWN_ROTATION[2] == 90.0 else -1.0
                lateral_detour_ledger = 0.0
                avoidance_sub_phase = "IDLE"
                active_obstacle_category = "UNKNOWN"
                step_start_pos = list(current_pos)
                shifted_distance = 0.0
                last_time = time.time()

                # Reset joint positions to neutral stance
                neutral = map_ik_to_newton_asset(neutral_joint_angles()).reshape(18)
                set_robot_joint_targets(neutral)

                # Reset base pose
                if translate_op and rotate_op:
                    translate_op.Set(Gf.Vec3d(*SPAWN_POSITION))
                    rotate_op.Set(Gf.Vec3d(*SPAWN_ROTATION))

                if status_label:
                    status_label.text = "Status: RESET & MOVING"
                if vlm_label:
                    vlm_label.text = "VLM: Idle"

            ui.Button("Stop", clicked_fn=on_emergency_stop, style={"background_color": 0xFF2222DD})
            ui.Button("Pause/Play", clicked_fn=on_pause_resume, style={"background_color": 0xFFDD8822})
            ui.Button("Reset", clicked_fn=on_reset, style={"background_color": 0xFF22AA22})

# ============================================================
# 9. VLM DECISION HANDLER & STRATEGY EXECUTION
# ============================================================
def trigger_early_lane_shift():
    global state, target_yaw, avoidance_sub_phase, step_start_pos, shifted_distance, sweep_direction_y
    print("[Edge Obstacle]: Obstacle at road boundary. Skipping directly to next lane.")
    target_yaw = get_lane_shift_heading()
    avoidance_sub_phase = "IDLE"

    if abs(get_shortest_angle_diff(target_yaw, current_yaw)) < 5.0:
        state = "TURNING_2"
        sweep_direction_y = -sweep_direction_y
        target_yaw = 90.0 if sweep_direction_y > 0 else 270.0
        shifted_distance = 0.0
        step_start_pos = list(current_pos)
    else:
        state = "TURNING_1"


def initiate_bypass_sequence():
    global state, avoidance_sub_phase, target_yaw, detour_side, diverge_turn_count
    diverge_turn_count = 1

    dist_to_top = SAFE_MAX_Y - current_pos[1]
    dist_to_bot = current_pos[1] - SAFE_MIN_Y
    if (sweep_direction_y > 0 and dist_to_top < 0.60) or (sweep_direction_y < 0 and dist_to_bot < 0.60):
        trigger_early_lane_shift()
        return

    left_d = _ros_node.depth_left_dist
    ctr_d = _ros_node.depth_center_dist
    right_d = _ros_node.depth_right_dist

    is_wide_obstacle = (ctr_d < 1.5) or (left_d < 1.5) or (right_d < 1.5)
    if (ctr_d < OBSTACLE_THRESHOLD) and (left_d > 1.5) and (right_d > 1.5):
        is_wide_obstacle = False
    if deadlock_retreat_count > 0:
        is_wide_obstacle = True

    if is_wide_obstacle:
        target_diff = get_target_angle_diff(current_pos, current_yaw, TARGET_DESTINATION)
        detour_side = -1.0 if target_diff > 0 else 1.0
        print(f"[Smart Diverge]: Wide obstacle -> Diverging {'LEFT' if detour_side > 0 else 'RIGHT'} towards goal.")
    else:
        detour_side = 1.0 if left_d >= right_d else -1.0
        print(f"[Smart Diverge]: Small obstacle -> Diverging {'LEFT' if detour_side > 0 else 'RIGHT'} to open side.")

    test_yaw = (lane_base_yaw + (detour_side * DIVERGE_ANGLE_INC)) % 360.0
    fwd_x, fwd_y, _ = get_forward_vector(test_yaw)
    pred_x = current_pos[0] + (fwd_x * 0.6)
    pred_y = current_pos[1] + (fwd_y * 0.6)

    if pred_x < SAFE_MIN_X or pred_x > SAFE_MAX_X or pred_y < SAFE_MIN_Y or pred_y > SAFE_MAX_Y:
        detour_side = -detour_side
        print(f"[NavMesh Safety]: Re-routing detour to {'LEFT' if detour_side > 0 else 'RIGHT'}.")

    avoidance_sub_phase = "DIVERGE_SCAN"
    target_yaw = (lane_base_yaw + (detour_side * DIVERGE_ANGLE_INC)) % 360.0
    state = "AVOID_INITIAL_TURN"
    if status_label:
        status_label.text = "Status: Executing Bypass Diverge..."


def on_vlm_agent_decision(decision_dict):
    global state, avoidance_sub_phase, target_yaw, current_yaw, micro_step_target, micro_step_progress
    global detour_side, flank_progress, required_flank_distance, initial_detected_distance, diverge_turn_count
    global active_obstacle_category, yield_wait_start_time, push_progress, push_target

    if state in ["STOPPED", "USER_PAUSED"]:
        return

    cmd = decision_dict.get("command", "standard_bypass")
    category = decision_dict.get("category", active_obstacle_category)
    reasoning = decision_dict.get("reasoning", "")
    obs_visible = decision_dict.get("obstacle_visible", True)

    print(f"\n[VLM Callback]: Phase='{avoidance_sub_phase}' | Cat='{category}' | Cmd='{cmd}' | Reason: {reasoning}")
    if vlm_label:
        vlm_label.text = f"VLM: [{category}] -> {cmd}"

    if avoidance_sub_phase == "CLASSIFY_OBSTACLE" and (
        cmd == "resume_path" or category == "CLEAR" or not obs_visible
    ):
        center_blocked = decision_dict.get("center_blocked", False)
        left_pct = decision_dict.get("left_sector_coverage_pct", 0)
        right_pct = decision_dict.get("right_sector_coverage_pct", 0)

        if center_blocked or left_pct >= 40 or right_pct >= 40:
            print(f"[VLM Override]: Sector analysis shows blockage. Treating as static obstacle.")
            active_obstacle_category = "STATIC_BAR"
            initiate_bypass_sequence()
            return

        current_dist = _ros_node.closest_obstacle_dist
        print(f"[VLM Decision]: Path clear ({current_dist:.2f}m).")
        avoidance_sub_phase = "IDLE"
        active_obstacle_category = "UNKNOWN"

        if current_dist < OBSTACLE_THRESHOLD:
            push_target = max(
                FALSE_ALARM_PUSH_MIN,
                min(FALSE_ALARM_PUSH_MAX, (OBSTACLE_THRESHOLD - current_dist) + FALSE_ALARM_PUSH_MARGIN),
            )
            push_progress = 0.0
            state = "FALSE_ALARM_PUSH"
        else:
            state = "MOVING"
        return

    state = "MOVING"

    # 0. Initial Classification Resolution
    if avoidance_sub_phase == "CLASSIFY_OBSTACLE":
        active_obstacle_category = category
        if category == "BIOLOGICAL" or cmd == "yield_and_wait":
            state = "YIELD_WAITING"
            yield_wait_start_time = time.time()
            yield_wait_total_start_time = time.time()
            if status_label:
                status_label.text = "Status: BIOLOGICAL -> Yielding"
            print("[Strategy - Yield]: Animal detected. Waiting for path to clear...")
        elif category == "STATIC_BAR" or cmd == "standard_bypass":
            initiate_bypass_sequence()
        elif category == "MINOR_SURFACE" or cmd == "scan_and_proceed":
            state = "SCAN_AND_PROCEED"
            if status_label:
                status_label.text = "Status: MINOR_SURFACE -> Walking Over Feature"

    # 1. Diverge Scan Resolution
    elif avoidance_sub_phase == "DIVERGE_SCAN":
        gap_open = decision_dict.get("corridor_gap_open", False)
        primary_cleared = decision_dict.get("primary_obstacle_cleared", False)
        critical_side_depth = _ros_node.depth_left_dist if detour_side > 0 else _ros_node.depth_right_dist

        if critical_side_depth < 0.45:
            print(f"[Multi-Obstacle Guard]: Proximity to obstacle ({critical_side_depth:.2f}m). Forcing Flank.")
            target_yaw = lane_base_yaw
            state = "AVOID_ALIGN_TO_FLANK"
            return

        if (cmd == "diverge_more" and not primary_cleared) and not gap_open:
            diverge_turn_count += 1
            max_angle = min(45.0, DIVERGE_ANGLE_INC + (diverge_turn_count * 5.0))
            target_yaw = (lane_base_yaw + (detour_side * max_angle)) % 360.0
            micro_step_target = AVOIDANCE_STEP_SIZE
            micro_step_progress = 0.0
            state = "AVOID_ROTATE_THEN_STEP"
        else:
            target_yaw = lane_base_yaw
            state = "AVOID_ALIGN_TO_FLANK"

    # 2. Flank Pre-Check Resolution
    elif avoidance_sub_phase == "FLANK_PRE_CHECK":
        current_dist = _ros_node.closest_obstacle_dist
        center_blocked = decision_dict.get("center_blocked", False)
        left_pct = decision_dict.get("left_sector_coverage_pct", 0)
        right_pct = decision_dict.get("right_sector_coverage_pct", 0)

        side_blocked = False
        if detour_side > 0:
            if right_pct >= 40 or _ros_node.depth_right_dist < 0.60:
                side_blocked = True
        else:
            if left_pct >= 40 or _ros_node.depth_left_dist < 0.60:
                side_blocked = True

        if center_blocked or side_blocked or cmd == "diverge_more" or current_dist < 0.50:
            target_yaw = (lane_base_yaw + (detour_side * 55.0)) % 360.0
            micro_step_target = 0.50
            micro_step_progress = 0.0
            state = "AVOID_ROTATE_THEN_STEP"
        else:
            avoidance_sub_phase = "FLANK_DRIVE"
            flank_progress = 0.0
            extra_d = FLANK_EXTRA_DISTANCE.get(active_obstacle_category, DEFAULT_FLANK_EXTRA)
            min_d = FLANK_MIN_DISTANCE.get(active_obstacle_category, DEFAULT_FLANK_MIN)
            max_d = FLANK_MAX_DISTANCE.get(active_obstacle_category, DEFAULT_FLANK_MAX)
            required_flank_distance = max(
                min_d, min(max_d, initial_detected_distance + CLEAR_PASS_SAFETY_PAD + extra_d)
            )
            state = "AVOID_FLANK_DRIVE"

    # 3. Flank Inward Inspect Resolution
    elif avoidance_sub_phase == "FLANK_INSPECT":
        if cmd == "flank_forward" and obs_visible:
            avoidance_sub_phase = "FLANK_DRIVE"
            target_yaw = lane_base_yaw
            flank_progress = 0.0
            required_flank_distance = 0.40
            state = "AVOID_ALIGN_TO_FLANK"
        else:
            avoidance_sub_phase = "VERIFY"
            return_steer_side = -1.0 if lateral_detour_ledger > 0.0 else 1.0
            target_yaw = (lane_base_yaw + (return_steer_side * 40.0)) % 360.0
            state = "LOOK_BACK_ROTATE"

    # 4. Return Verification Resolution
    elif avoidance_sub_phase == "VERIFY":
        if cmd == "return_to_lane" or not obs_visible:
            avoidance_sub_phase = "RETURNING"
            state = "RETURN_DRIVE_TO_LANE"
        else:
            avoidance_sub_phase = "FLANK_DRIVE"
            target_yaw = lane_base_yaw
            flank_progress = 0.0
            required_flank_distance = 0.35
            state = "AVOID_ALIGN_TO_FLANK"

# ============================================================
# 10. MAIN PHYSICS & CONTROLLER UPDATE LOOP
# ============================================================
def zig_zag_update(event):
    global state, saved_pre_pause_state, current_pos, current_yaw, sweep_direction_y
    global step_start_pos, target_yaw, pause_start_time, last_time, shifted_distance, last_print_time
    global lane_base_yaw, lateral_detour_ledger, micro_step_progress, micro_step_target, detour_side
    global avoidance_sub_phase, initial_detected_distance, required_flank_distance, flank_progress, diverge_turn_count
    global llm_avoidance_history, yield_wait_start_time, active_obstacle_category
    global pre_check_settle_start, push_progress, push_target, yield_wait_total_start_time
    global emergency_triggered, emergency_cooldown_start, emergency_trigger_pos
    global backup_progress, deadlock_retreat_count, force_retreat_target

    # 1. Read true physics pose from robot prim
    phys_pos, phys_yaw = get_physics_pose(robot_prim)
    if phys_pos is not None:
        dx = phys_pos[0] - current_pos[0]
        dy = phys_pos[1] - current_pos[1]
        dist_traveled = math.sqrt(dx * dx + dy * dy)

        # Update odometry ledgers based on real motion
        record_motion_to_ledger(dx, dy)
        if state == "AVOID_DIVERGE_STEP":
            micro_step_progress += dist_traveled
        elif state == "AVOID_FLANK_DRIVE":
            flank_progress += dist_traveled
        elif state == "EMERGENCY_BACKUP":
            backup_progress += dist_traveled
        elif state == "FALSE_ALARM_PUSH":
            push_progress += dist_traveled
        elif state == "LANE_SHIFTING":
            shifted_distance += dist_traveled

        current_pos = phys_pos
        current_yaw = phys_yaw

    rclpy.spin_once(_ros_node, timeout_sec=0.0)

    # Commanded velocities for this tick
    cmd_vx = 0.0
    cmd_vy = 0.0
    cmd_wz = 0.0

    if state in ["STOPPED", "USER_PAUSED"]:
        # Zero velocity, neutral stance holding
        neutral = map_ik_to_newton_asset(neutral_joint_angles()).reshape(18)
        set_robot_joint_targets(neutral)
        return

    now = time.time()
    dt = min(now - last_time, 0.1)
    last_time = now

    min_dist_ahead = _ros_node.closest_obstacle_dist
    rgb_frame = _ros_node.latest_rgb

    # Global emergency proximity guard
    if state not in ["MOVING", "STOPPED", "USER_PAUSED", "EMERGENCY_BACKUP"]:
        if check_emergency_proximity(min_dist_ahead):
            print(f"[EMERGENCY STOP - Global Guard]: {min_dist_ahead:.2f}m in '{state}'. Backing up.")
            initiate_emergency_backup()
            return

    if sensor_label:
        sensor_label.text = f"Obstacle Dist: {min_dist_ahead:.2f} m" if min_dist_ahead != float("inf") else "Obstacle Dist: Clear (>10m)"
    if ledger_label:
        ledger_label.text = f"Detour Ledger: {lateral_detour_ledger:.2f} m"

    if (now - last_print_time) >= PRINT_INTERVAL:
        last_print_time = now
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        print("-" * 65)
        print(f"[{timestamp_str}] [Telemetry] State: {state} | Cat: {active_obstacle_category} | Phase: {avoidance_sub_phase}")
        print(f"            Pose: ({current_pos[0]:.2f}, {current_pos[1]:.2f}) | Heading: {current_yaw:.1f}\u00b0 | Dist: {min_dist_ahead:.2f}m")
        print(f"            Ledger: {lateral_detour_ledger:.2f}m | Left: {_ros_node.depth_left_dist:.2f}m | Ctr: {_ros_node.depth_center_dist:.2f}m | Right: {_ros_node.depth_right_dist:.2f}m")
        print("-" * 65)

    # ------------------------------------------------------------
    # A. AUTOMATIC RESUME: Biological Clears
    # ------------------------------------------------------------
    if state == "YIELD_WAITING":
        if min_dist_ahead >= (OBSTACLE_THRESHOLD + 0.40):
            print(f"[Auto Resume]: Path clear ({min_dist_ahead:.2f}m). Resuming operations.")
            state = "MOVING"
            avoidance_sub_phase = "IDLE"
            active_obstacle_category = "UNKNOWN"
            lateral_detour_ledger = 0.0
            target_yaw = lane_base_yaw
            step_start_pos = list(current_pos)
            llm_avoidance_history.clear()
            if status_label:
                status_label.text = "Status: Path Clear -> Resumed"
            return

        if (now - yield_wait_total_start_time) >= MAX_YIELD_WAIT:
            print("[Yield Timeout]: Animal still present after 10s. Initiating standard bypass.")
            active_obstacle_category = "STATIC_BAR"
            lateral_detour_ledger = 0.0
            initiate_bypass_sequence()
            return

        if (now - yield_wait_start_time) >= YIELD_WAIT_DURATION:
            if min_dist_ahead >= OBSTACLE_THRESHOLD:
                state = "MOVING"
                avoidance_sub_phase = "IDLE"
            else:
                yield_wait_start_time = now

    # ------------------------------------------------------------
    # B. NORMAL SWEEP FORWARD
    # ------------------------------------------------------------
    elif state == "MOVING":
        lane_base_yaw = target_yaw

        if compute_dist_2d(current_pos, TARGET_DESTINATION) <= TARGET_RADIUS:
            stop()
            if status_label:
                status_label.text = "Status: DESTINATION REACHED"
            return

        # NavMesh road boundary edge check
        edge_hit = False
        margin_y = ROBOT_SAFETY_MARGIN
        if current_pos[1] >= (max_y - margin_y) and sweep_direction_y > 0:
            edge_hit = True
        elif current_pos[1] <= (min_y + margin_y) and sweep_direction_y < 0:
            edge_hit = True

        if edge_hit:
            target_yaw = get_lane_shift_heading()
            state = "TURNING_1"
            if status_label:
                status_label.text = "Status: Edge Hit -> Shifting Lane"
            return

        # Obstacle proximity check
        if min_dist_ahead < OBSTACLE_THRESHOLD:
            if min_dist_ahead < EMERGENCY_STOP_DIST:
                dist_from_trigger = compute_dist_2d(current_pos, emergency_trigger_pos)
                time_since_trigger = time.time() - emergency_cooldown_start
                if emergency_triggered and (
                    dist_from_trigger < EMERGENCY_COOLDOWN_DIST or time_since_trigger < EMERGENCY_COOLDOWN_TIME
                ):
                    pass
                else:
                    print(f"[EMERGENCY STOP]: Obstacle at {min_dist_ahead:.2f}m. Initiating backup.")
                    initiate_emergency_backup()
                    return

            initial_detected_distance = min_dist_ahead
            lateral_detour_ledger = 0.0
            llm_avoidance_history.clear()
            avoidance_sub_phase = "CLASSIFY_OBSTACLE"
            state = "VLM_INSPECTING"
            if status_label:
                status_label.text = "Status: Classifying Obstacle..."
            async_query_vlm_agent(
                rgb_frame, min_dist_ahead, avoidance_sub_phase, lateral_detour_ledger, on_vlm_agent_decision
            )
            return

        # Drive forward along current sweep line
        cmd_vx = FORWARD_SPEED
        yaw_diff = get_shortest_angle_diff(lane_base_yaw, current_yaw)
        cmd_wz = float(np.clip(math.radians(yaw_diff) * 1.5, -MAX_TURN_RATE_RAD, MAX_TURN_RATE_RAD))

        dist_in_step = compute_dist_2d(current_pos, step_start_pos)
        if dist_in_step >= STEP_DISTANCE:
            state = "PAUSED"
            pause_start_time = now
            if status_label:
                status_label.text = f"Status: Step Paused ({PAUSE_DURATION}s)"

    # ------------------------------------------------------------
    # C. STEP PAUSE & FALSE ALARM PUSH
    # ------------------------------------------------------------
    elif state == "PAUSED":
        if (now - pause_start_time) >= PAUSE_DURATION:
            state = "MOVING"
            step_start_pos = list(current_pos)

    elif state == "FALSE_ALARM_PUSH":
        if min_dist_ahead < EMERGENCY_STOP_DIST:
            initiate_emergency_backup()
            return
        cmd_vx = FORWARD_SPEED * 0.7
        yaw_diff = get_shortest_angle_diff(lane_base_yaw, current_yaw)
        cmd_wz = float(np.clip(math.radians(yaw_diff) * 1.5, -MAX_TURN_RATE_RAD, MAX_TURN_RATE_RAD))
        if push_progress >= push_target:
            state = "MOVING"
            step_start_pos = list(current_pos)
            emergency_triggered = False

    # ------------------------------------------------------------
    # D. TURNING & ALIGNMENT STATES
    # ------------------------------------------------------------
    elif state in [
        "AVOID_INITIAL_TURN",
        "AVOID_ROTATE_THEN_STEP",
        "AVOID_ALIGN_TO_FLANK",
        "AVOID_ROTATE_TO_INSPECT_FLANK",
        "LOOK_BACK_ROTATE",
        "RETURN_FINAL_ALIGN",
        "TURNING_1",
        "TURNING_2",
    ]:
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        cmd_wz = float(np.clip(math.radians(yaw_diff) * 2.0, -MAX_TURN_RATE_RAD, MAX_TURN_RATE_RAD))

        if abs(yaw_diff) <= 4.0:
            if state == "AVOID_INITIAL_TURN":
                state = "VLM_INSPECTING"
                async_query_vlm_agent(
                    rgb_frame, min_dist_ahead, avoidance_sub_phase, lateral_detour_ledger, on_vlm_agent_decision
                )
            elif state == "AVOID_ROTATE_THEN_STEP":
                state = "AVOID_DIVERGE_STEP"
                micro_step_progress = 0.0
            elif state == "AVOID_ALIGN_TO_FLANK":
                state = "AVOID_FLANK_PRE_CHECK_SETTLE"
                pre_check_settle_start = now
            elif state in ["AVOID_ROTATE_TO_INSPECT_FLANK", "LOOK_BACK_ROTATE"]:
                state = "VLM_INSPECTING"
                async_query_vlm_agent(
                    rgb_frame, min_dist_ahead, avoidance_sub_phase, lateral_detour_ledger, on_vlm_agent_decision
                )
            elif state == "RETURN_FINAL_ALIGN":
                deadlock_retreat_count = 0
                state = "MOVING"
                avoidance_sub_phase = "IDLE"
                step_start_pos = list(current_pos)
            elif state == "TURNING_1":
                state = "LANE_SHIFTING"
                shifted_distance = 0.0
            elif state == "TURNING_2":
                lane_base_yaw = current_yaw
                state = "MOVING"
                step_start_pos = list(current_pos)

    # ------------------------------------------------------------
    # E. FLANK & DETOUR DRIVING STATES
    # ------------------------------------------------------------
    elif state == "AVOID_DIVERGE_STEP":
        cmd_vx = FORWARD_SPEED * 0.7
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        cmd_wz = float(np.clip(math.radians(yaw_diff) * 1.5, -MAX_TURN_RATE_RAD, MAX_TURN_RATE_RAD))

        if micro_step_progress >= micro_step_target:
            state = "VLM_INSPECTING"
            async_query_vlm_agent(
                rgb_frame, min_dist_ahead, avoidance_sub_phase, lateral_detour_ledger, on_vlm_agent_decision
            )

    elif state == "AVOID_FLANK_PRE_CHECK_SETTLE":
        if (now - pre_check_settle_start) >= 0.40:
            state = "VLM_INSPECTING"
            avoidance_sub_phase = "FLANK_PRE_CHECK"
            async_query_vlm_agent(
                rgb_frame, min_dist_ahead, avoidance_sub_phase, lateral_detour_ledger, on_vlm_agent_decision
            )

    elif state == "AVOID_FLANK_DRIVE":
        cmd_vx = FORWARD_SPEED * 0.8
        yaw_diff = get_shortest_angle_diff(lane_base_yaw, current_yaw)
        cmd_wz = float(np.clip(math.radians(yaw_diff) * 1.5, -MAX_TURN_RATE_RAD, MAX_TURN_RATE_RAD))

        if (detour_side > 0 and current_pos[1] >= SAFE_MAX_Y - 0.05) or (
            detour_side < 0 and current_pos[1] <= SAFE_MIN_Y + 0.05
        ):
            trigger_early_lane_shift()
            return

        if flank_progress >= required_flank_distance:
            avoidance_sub_phase = "FLANK_INSPECT"
            inspect_angle_side = -1.0 if detour_side > 0 else 1.0
            target_yaw = (lane_base_yaw + (inspect_angle_side * 35.0)) % 360.0
            state = "AVOID_ROTATE_TO_INSPECT_FLANK"

    elif state == "RETURN_DRIVE_TO_LANE":
        cmd_vx = FORWARD_SPEED * 0.8
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        cmd_wz = float(np.clip(math.radians(yaw_diff) * 1.5, -MAX_TURN_RATE_RAD, MAX_TURN_RATE_RAD))

        if abs(lateral_detour_ledger) <= 0.08 or (lateral_detour_ledger * detour_side <= 0.0):
            lateral_detour_ledger = 0.0
            target_yaw = lane_base_yaw
            state = "RETURN_FINAL_ALIGN"

    # ------------------------------------------------------------
    # F. LANE SHIFTING
    # ------------------------------------------------------------
    elif state == "LANE_SHIFTING":
        if _ros_node.closest_obstacle_dist < 0.50:
            state = "TURNING_2"
            sweep_direction_y = -sweep_direction_y
            target_yaw = 90.0 if sweep_direction_y > 0 else 270.0
            shifted_distance = 0.0
            step_start_pos = list(current_pos)
            return

        cmd_vx = FORWARD_SPEED * 0.8
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        cmd_wz = float(np.clip(math.radians(yaw_diff) * 1.5, -MAX_TURN_RATE_RAD, MAX_TURN_RATE_RAD))

        if shifted_distance >= LANE_SHIFT:
            state = "TURNING_2"
            sweep_direction_y = -sweep_direction_y
            target_yaw = 90.0 if sweep_direction_y > 0 else 270.0

    # ------------------------------------------------------------
    # G. EMERGENCY BACKUP & FORCE RETREAT
    # ------------------------------------------------------------
    elif state == "EMERGENCY_BACKUP":
        cmd_vx = -EMERGENCY_BACKUP_SPEED
        if backup_progress >= EMERGENCY_BACKUP_DIST:
            print("[Emergency Backup]: Backup complete. Re-evaluating obstacle.")
            min_dist_ahead = _ros_node.closest_obstacle_dist
            if min_dist_ahead < OBSTACLE_THRESHOLD:
                deadlock_retreat_count += 1
                if deadlock_retreat_count >= 2:
                    print("[DEADLOCK RESOLVED]: Forcing retreat to skip lane.")
                    deadlock_retreat_count = 0
                    emergency_triggered = False
                    backup_progress = 0.0
                    state = "FORCE_RETREAT_BACK"
                else:
                    emergency_triggered = False
                    initiate_bypass_sequence()
            else:
                state = "MOVING"

    elif state == "FORCE_RETREAT_BACK":
        cmd_vx = -EMERGENCY_BACKUP_SPEED * 1.2
        if backup_progress >= force_retreat_target:
            target_yaw = get_lane_shift_heading()
            avoidance_sub_phase = "IDLE"
            state = "TURNING_1"
            emergency_triggered = False
            step_start_pos = list(current_pos)

    elif state == "SCAN_AND_PROCEED":
        cmd_vx = FORWARD_SPEED * 0.6
        yaw_diff = get_shortest_angle_diff(lane_base_yaw, current_yaw)
        cmd_wz = float(np.clip(math.radians(yaw_diff) * 1.5, -MAX_TURN_RATE_RAD, MAX_TURN_RATE_RAD))
        if min_dist_ahead > (OBSTACLE_THRESHOLD + 0.20):
            state = "MOVING"

    elif state == "VLM_INSPECTING":
        cmd_vx = 0.0
        cmd_vy = 0.0
        cmd_wz = 0.0

    # ------------------------------------------------------------
    # H. COMPUTE & ACTUATE REVOLUTE JOINT TARGETS
    # ------------------------------------------------------------
    target_angles = policy_engine.compute_joint_targets(cmd_vx, cmd_vy, cmd_wz)
    set_robot_joint_targets(target_angles)


# Register update loop subscription
global _phantomx_sub
if "_phantomx_sub" in globals() and _phantomx_sub is not None:
    try:
        _phantomx_sub.unsubscribe()
        print("[Cleanup]: Previous update callback unsubscribed.")
    except Exception as exc:
        print(f"[Cleanup Warning]: Could not unsubscribe old callback: {exc}")
    _phantomx_sub = None


_phantomx_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(zig_zag_update)
print("✅ [PhantomX Controller]: Initialized and running with Newton joint actuation.")
