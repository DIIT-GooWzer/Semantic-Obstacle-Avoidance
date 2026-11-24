"""
PhantomX Target-Directed Zig-Zag Controller
------------------------------------------
Sweeps an area back-and-forth, with the initial lane shift
intelligently directed toward a user-defined TARGET_DESTINATION.
"""

import time
import math
import omni.usd
import omni.kit.app
import omni.ui as ui
from pxr import Usd, UsdGeom, Gf

stage = omni.usd.get_context().get_stage()

# ============================================================
# CONFIGURATION
# ============================================================
SPAWN_POSITION     = (0.3295789538104168, -1.681203353653679, 0.10692057425096305)  # (X, Y, Z)
SPAWN_ROTATION     = (0.0, 0.0, 90.0)                                                 # (Roll, Pitch, Yaw)
TARGET_DESTINATION = (10.0, 1.5)                                                      # (X, Y) Destination goal
TARGET_RADIUS = 0.3 

STEP_DISTANCE  = 0.5                      # Distance per step (m)
PAUSE_DURATION = 2.0                      # Pause duration (s)
FORWARD_SPEED  = 1.0                      # Speed (m/s)
TURN_SPEED     = 180.0                    # Turn speed (deg/s)
LANE_SHIFT     = 0.5                      # Lane shift width (m)

def spawn_phantomx(stage, prim_path="/World/phantomX_base", position=(0.2, -1.5, 0.25), rotation_euler=(0.0, 0.0, 0.0)):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(
            "/home/user/Desktop/llmforhexapod/usd_to_stl_conversion/phantomX_base.usd"
        )

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp()
    rotate_op = xform.AddRotateXYZOp()
    translate_op.Set(Gf.Vec3d(*position))
    rotate_op.Set(Gf.Vec3d(*rotation_euler))
    return prim, translate_op, rotate_op

robot_prim, translate_op, rotate_op = spawn_phantomx(stage, position=SPAWN_POSITION, rotation_euler=SPAWN_ROTATION)

bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
navmesh_prim = stage.GetPrimAtPath("/World/Terrain/section_00_asphalt/NavMeshVolume")

min_x, max_x = 0.0, 12.0
min_y, max_y = -2.0, 2.0
if navmesh_prim.IsValid():
    bound = bbox_cache.ComputeWorldBound(navmesh_prim).ComputeAlignedBox()
    min_x, max_x = float(bound.GetMin()[0]), float(bound.GetMax()[0])
    min_y, max_y = float(bound.GetMin()[1]), float(bound.GetMax()[1])

# ============================================================
# ORIENTATION & VECTOR HELPERS
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
    """
    Computes whether the target is to the LEFT (+1.0) or RIGHT (-1.0)
    of the robot's current heading using 2D cross product.
    """
    fwd_x, fwd_y, _ = get_forward_vector(current_yaw_deg)
    to_target_x = target_p[0] - current_p[0]
    to_target_y = target_p[1] - current_p[1]

    cross_z = (fwd_x * to_target_y) - (fwd_y * to_target_x)
    return 1.0 if cross_z >= 0.0 else -1.0

# ============================================================
# STATE VARIABLES
# ============================================================
global state, saved_pre_pause_state, current_pos, current_yaw, turn_sign, initial_turn_calculated
global step_start_pos, target_yaw, pause_start_time, last_time, shifted_distance

current_pos = list(SPAWN_POSITION)
current_yaw = SPAWN_ROTATION[2]
state = "MOVING"
saved_pre_pause_state = "MOVING"

turn_sign = 1.0
initial_turn_calculated = False  # Calculated upon hitting the first boundary

step_start_pos = list(current_pos)
target_yaw = current_yaw
pause_start_time = 0.0
last_time = time.time()
shifted_distance = 0.0

def update_usd_transform(pos, yaw_deg):
    translate_op.Set(Gf.Vec3d(*pos))
    rotate_op.Set(Gf.Vec3d(0.0, 0.0, yaw_deg))

update_usd_transform(current_pos, current_yaw)

# ============================================================
# UI PANEL
# ============================================================
global _phantomx_window
if "_phantomx_window" in globals() and _phantomx_window:
    try:
        _phantomx_window.visible = False
        _phantomx_window.destroy()
    except Exception:
        pass
    _phantomx_window = None

_phantomx_window = ui.Window("PhantomX Target Sweep Controller", width=380, height=200, flags=ui.WINDOW_FLAGS_NO_SAVED_SETTINGS)
_phantomx_window.position_x = 80
_phantomx_window.position_y = 120
_phantomx_window.visible = True
_phantomx_window.focus()

try:
    ui.Workspace.show_window("PhantomX Target Sweep Controller", True)
except Exception:
    pass

status_label = None

with _phantomx_window.frame:
    with ui.VStack(spacing=8, style={"margin": 12}):
        ui.Label("🤖 PhantomX Target-Directed Controller", height=24, style={"font_size": 15, "color": 0xFF00FFFF})
        ui.Label(f"🎯 Target: ({TARGET_DESTINATION[0]:.1f}, {TARGET_DESTINATION[1]:.1f})", height=18, style={"font_size": 12, "color": 0xFFAAAAFF})
        status_label = ui.Label("Status: ▶️ MOVING (Forward)", height=22, style={"font_size": 14})
        
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
                global state, current_pos, current_yaw, step_start_pos, last_time, shifted_distance, initial_turn_calculated
                state = "MOVING"
                current_pos = list(SPAWN_POSITION)
                current_yaw = SPAWN_ROTATION[2]
                initial_turn_calculated = False
                step_start_pos = list(current_pos)
                shifted_distance = 0.0
                last_time = time.time()
                update_usd_transform(current_pos, current_yaw)
                if status_label:
                    status_label.text = "Status: 🔄 RESET & MOVING"

            ui.Button("🛑 Stop", clicked_fn=on_emergency_stop, style={"background_color": 0xFF2222DD})
            ui.Button("⏯️ Pause/Play", clicked_fn=on_pause_resume, style={"background_color": 0xFFDD8822})
            ui.Button("🔄 Reset", clicked_fn=on_reset, style={"background_color": 0xFF22AA22})

# ============================================================
# MAIN UPDATE LOOP
# ============================================================
def zig_zag_update(event):
    global state, saved_pre_pause_state, current_pos, current_yaw, turn_sign, initial_turn_calculated
    global step_start_pos, target_yaw, pause_start_time, last_time, shifted_distance

    if state == "STOPPED" or state == "USER_PAUSED":
        return

    now = time.time()
    dt = now - last_time
    last_time = now

    # 1. SWEEP FORWARD
    if state == "MOVING":
        dist_to_target = compute_dist_2d(current_pos, TARGET_DESTINATION)
        if dist_to_target <= TARGET_RADIUS:
            state = "STOPPED"
            if status_label:
                status_label.text = "Status: 🎯 DESTINATION REACHED"
            print(f"🎯 Target ({TARGET_DESTINATION[0]}, {TARGET_DESTINATION[1]}) reached! Robot stopped.")
            update_usd_transform(current_pos, current_yaw)
            return
        
        fwd_x, fwd_y, _ = get_forward_vector(current_yaw)
        current_pos[0] += fwd_x * FORWARD_SPEED * dt
        current_pos[1] += fwd_y * FORWARD_SPEED * dt

        # Boundary checks against NavMesh bounds
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

        dist_in_step = compute_dist_2d(current_pos, step_start_pos)
        
        if edge_hit:
            # Check target direction on the first boundary encounter
            if not initial_turn_calculated:
                turn_sign = calculate_turn_toward_target(current_pos, current_yaw, TARGET_DESTINATION)
                initial_turn_calculated = True

            target_yaw = (current_yaw + turn_sign * 90.0) % 360.0
            state = "TURNING_1"
            turn_name = "Left" if turn_sign > 0 else "Right"
            if status_label:
                status_label.text = f"Status: 🔄 Edge hit -> Turn 1 ({turn_name} 90°)"
        elif dist_in_step >= STEP_DISTANCE:
            state = "PAUSED"
            pause_start_time = now
            if status_label:
                status_label.text = "Status: 📌 Step Paused"

        update_usd_transform(current_pos, current_yaw)

    # 2. STEP INTERVAL PAUSE
    elif state == "PAUSED":
        if (now - pause_start_time) >= PAUSE_DURATION:
            state = "MOVING"
            step_start_pos = list(current_pos)
            if status_label:
                status_label.text = "Status: ▶️ MOVING (Forward)"

    # 3. FIRST 90° TURN
    elif state == "TURNING_1":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            state = "LANE_SHIFTING"
            shifted_distance = 0.0
            if status_label:
                status_label.text = "Status: ➡️ Shifting Lane toward Target"

        update_usd_transform(current_pos, current_yaw)

    # 4. LANE SHIFT
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
                status_label.text = f"Status: 🔄 Turn 2 ({turn_name} 90°)"

        update_usd_transform(current_pos, current_yaw)

    # 5. SECOND 90° TURN (RE-ALIGNING)
    elif state == "TURNING_2":
        yaw_diff = get_shortest_angle_diff(target_yaw, current_yaw)
        step_rot = math.copysign(min(abs(yaw_diff), TURN_SPEED * dt), yaw_diff)
        current_yaw += step_rot

        if abs(yaw_diff) <= (TURN_SPEED * dt):
            current_yaw = target_yaw
            # Flip turn direction for subsequent alternating passes
            turn_sign = -turn_sign
            state = "MOVING"
            step_start_pos = list(current_pos)
            if status_label:
                status_label.text = "Status: ▶️ MOVING (New Lane)"

        update_usd_transform(current_pos, current_yaw)

global _phantomx_sub
_phantomx_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(zig_zag_update)