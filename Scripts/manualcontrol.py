"""
PhantomX Relative Heading Controller
-------------------------------------
Prototype movement controller using USD translation/rotation.

Available movement functions:
    move_forward()
    move_backward()
    turn_left()
    turn_right()
    stop()

The robot is currently moved by directly changing its USD transform.
This is a prototype and does NOT yet use joints/IK/physics locomotion.
"""

import time
import math
import omni.usd
import omni.kit.app
import omni.ui as ui
from pxr import Usd, UsdGeom, Gf


# ============================================================
# ISAAC SIM / STAGE
# ============================================================

stage = omni.usd.get_context().get_stage()


# ============================================================
# CONFIGURATION
# ============================================================

ROBOT_PATH = "/World/phantomX_base"

SPAWN_POSITION = (  
    0.3295789538104168,
    -1.681203353653679,
    0.10692057425096305
)

# X, Y, Z rotation
SPAWN_ROTATION = (
    0.0,
    0.0,
    90.0
)

# Movement
MOVE_SPEED = 1.0          # metres / second
TURN_SPEED = 180.0        # degrees / second

# Zig-zag
STEP_DISTANCE = 0.5
PAUSE_DURATION = 2.0
LANE_SHIFT = 0.5


# ============================================================
# SPAWN / GET ROBOT
# ============================================================

def spawn_phantomx(
    stage,
    prim_path=ROBOT_PATH,
    position=SPAWN_POSITION,
    rotation_euler=SPAWN_ROTATION
):

    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():

        prim = stage.DefinePrim(
            prim_path,
            "Xform"
        )

        prim.GetReferences().AddReference(
            "/home/user/Desktop/llmforhexapod/"
            "usd_to_stl_conversion/phantomX_base.usd"
        )

    xform = UsdGeom.Xformable(prim)

    xform.ClearXformOpOrder()

    translate_op = xform.AddTranslateOp()
    rotate_op = xform.AddRotateXYZOp()

    translate_op.Set(
        Gf.Vec3d(*position)
    )

    rotate_op.Set(
        Gf.Vec3d(*rotation_euler)
    )

    return prim, translate_op, rotate_op


robot_prim, translate_op, rotate_op = spawn_phantomx(
    stage
)


# ============================================================
# ROBOT STATE
# ============================================================

current_pos = list(SPAWN_POSITION)

current_yaw = SPAWN_ROTATION[2]

state = "STOPPED"

target_yaw = current_yaw

step_start_pos = list(current_pos)

pause_start_time = 0.0

last_time = time.time()

shifted_distance = 0.0


# ============================================================
# TRANSFORM UPDATE
# ============================================================

def update_usd_transform(pos, yaw_deg):

    translate_op.Set(
        Gf.Vec3d(
            pos[0],
            pos[1],
            pos[2]
        )
    )

    rotate_op.Set(
        Gf.Vec3d(
            0.0,
            0.0,
            yaw_deg
        )
    )


# ============================================================
# ORIENTATION FUNCTIONS
# ============================================================

def get_forward_vector(yaw_deg):

    rad = math.radians(yaw_deg)

    return (
        math.cos(rad),
        math.sin(rad),
        0.0
    )


def get_shortest_angle_diff(target, current):

    diff = (target - current) % 360.0

    if diff > 180.0:
        diff -= 360.0

    return diff


def compute_dist_2d(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
# MOVEMENT FUNCTIONS
# ============================================================

def move_forward():

    global state

    state = "MOVING_FORWARD"

    if status_label:
        status_label.text = "Status: ▶️ MOVING FORWARD"


def move_backward():

    global state

    state = "MOVING_BACKWARD"

    if status_label:
        status_label.text = "Status: ◀️ MOVING BACKWARD"


def turn_left():

    global state
    global target_yaw

    target_yaw = (current_yaw + 90.0) % 360.0

    state = "TURNING_LEFT"

    if status_label:
        status_label.text = "Status: ↩️ TURNING LEFT"


def turn_right():

    global state
    global target_yaw

    target_yaw = (current_yaw - 90.0) % 360.0

    state = "TURNING_RIGHT"

    if status_label:
        status_label.text = "Status: ↪️ TURNING RIGHT"


def stop():

    global state

    state = "STOPPED"

    if status_label:
        status_label.text = "Status: 🛑 STOPPED"


# ============================================================
# NAVIGATION AREA
# ============================================================

bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_]
)

navmesh_prim = stage.GetPrimAtPath(
    "/World/Terrain/section_00_asphalt/NavMeshVolume"
)

min_x, max_x = 0.0, 12.0
min_y, max_y = -2.0, 2.0

if navmesh_prim.IsValid():

    bound = (
        bbox_cache
        .ComputeWorldBound(navmesh_prim)
        .ComputeAlignedBox()
    )

    min_x = float(bound.GetMin()[0])
    max_x = float(bound.GetMax()[0])

    min_y = float(bound.GetMin()[1])
    max_y = float(bound.GetMax()[1])


# ============================================================
# UI
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
    "PhantomX Motion Control",
    width=420,
    height=300,
    flags=ui.WINDOW_FLAGS_NO_SAVED_SETTINGS
)

_phantomx_window.position_x = 80
_phantomx_window.position_y = 120

_phantomx_window.visible = True
_phantomx_window.focus()


status_label = None


# ============================================================
# UI BUTTONS
# ============================================================

with _phantomx_window.frame:

    with ui.VStack(
        spacing=8,
        style={"margin": 12}
    ):

        ui.Label(
            "🤖 PhantomX Movement Controller",
            height=28,
            style={
                "font_size": 16,
                "color": 0xFF00FFFF
            }
        )

        status_label = ui.Label(
            "Status: 🛑 STOPPED",
            height=24,
            style={
                "font_size": 14
            }
        )

        # ----------------------------------------------------
        # Forward / Backward
        # ----------------------------------------------------

        with ui.HStack(
            spacing=10,
            height=40
        ):

            ui.Button(
                "⬆️ Forward",
                clicked_fn=move_forward
            )

            ui.Button(
                "⬇️ Backward",
                clicked_fn=move_backward
            )

        # ----------------------------------------------------
        # Turning
        # ----------------------------------------------------

        with ui.HStack(
            spacing=10,
            height=40
        ):

            ui.Button(
                "⬅️ Left",
                clicked_fn=turn_left
            )

            ui.Button(
                "➡️ Right",
                clicked_fn=turn_right
            )

        # ----------------------------------------------------
        # Stop
        # ----------------------------------------------------

        with ui.HStack(
            spacing=10,
            height=40
        ):

            ui.Button(
                "🛑 STOP",
                clicked_fn=stop,
                style={
                    "background_color": 0xFF2222DD
                }
            )


# ============================================================
# RESET FUNCTION
# ============================================================

def reset_robot():

    global state
    global current_pos
    global current_yaw
    global step_start_pos
    global last_time

    state = "STOPPED"

    current_pos = list(
        SPAWN_POSITION
    )

    current_yaw = SPAWN_ROTATION[2]

    step_start_pos = list(
        current_pos
    )

    last_time = time.time()

    update_usd_transform(
        current_pos,
        current_yaw
    )

    if status_label:

        status_label.text = (
            "Status: 🔄 RESET"
        )


# ============================================================
# UPDATE LOOP
# ============================================================

def movement_update(event):

    global state
    global current_pos
    global current_yaw
    global last_time
    global target_yaw

    now = time.time()

    dt = now - last_time

    last_time = now

    # Prevent huge movement if simulation pauses
    dt = min(dt, 0.1)

    # --------------------------------------------------------
    # STOPPED
    # --------------------------------------------------------

    if state == "STOPPED":

        return


    # --------------------------------------------------------
    # MOVE FORWARD
    # --------------------------------------------------------

    if state == "MOVING_FORWARD":

        forward_x, forward_y, _ = (
            get_forward_vector(current_yaw)
        )

        current_pos[0] += (
            forward_x *
            MOVE_SPEED *
            dt
        )

        current_pos[1] += (
            forward_y *
            MOVE_SPEED *
            dt
        )

        update_usd_transform(
            current_pos,
            current_yaw
        )


    # --------------------------------------------------------
    # MOVE BACKWARD
    # --------------------------------------------------------

    elif state == "MOVING_BACKWARD":

        forward_x, forward_y, _ = (
            get_forward_vector(current_yaw)
        )

        current_pos[0] -= (
            forward_x *
            MOVE_SPEED *
            dt
        )

        current_pos[1] -= (
            forward_y *
            MOVE_SPEED *
            dt
        )

        update_usd_transform(
            current_pos,
            current_yaw
        )


    # --------------------------------------------------------
    # TURN LEFT
    # --------------------------------------------------------

    elif state == "TURNING_LEFT":

        yaw_diff = get_shortest_angle_diff(
            target_yaw,
            current_yaw
        )

        rotation = (
            TURN_SPEED *
            dt
        )

        if abs(yaw_diff) <= rotation:

            current_yaw = target_yaw

            state = "STOPPED"

        else:

            current_yaw += (
                math.copysign(
                    rotation,
                    yaw_diff
                )
            )

        update_usd_transform(
            current_pos,
            current_yaw
        )


    # --------------------------------------------------------
    # TURN RIGHT
    # --------------------------------------------------------

    elif state == "TURNING_RIGHT":

        yaw_diff = get_shortest_angle_diff(
            target_yaw,
            current_yaw
        )

        rotation = (
            TURN_SPEED *
            dt
        )

        if abs(yaw_diff) <= rotation:

            current_yaw = target_yaw

            state = "STOPPED"

        else:

            current_yaw += (
                math.copysign(
                    rotation,
                    yaw_diff
                )
            )

        update_usd_transform(
            current_pos,
            current_yaw
        )


# ============================================================
# START UPDATE LOOP
# ============================================================

global _phantomx_sub

_phantomx_sub = (
    omni.kit.app
    .get_app()
    .get_update_event_stream()
    .create_subscription_to_pop(
        movement_update
    )
)


# ============================================================
# INITIALIZE
# ============================================================

update_usd_transform(
    current_pos,
    current_yaw
)

print("======================================")
print("PhantomX Movement Controller Started")
print("======================================")
print("Available functions:")
print("  move_forward()")
print("  move_backward()")
print("  turn_left()")
print("  turn_right()")
print("  stop()")
print("======================================")