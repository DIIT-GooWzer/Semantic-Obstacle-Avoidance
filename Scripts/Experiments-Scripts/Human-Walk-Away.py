"""Isaac Script Editor: load ONE Standard pipeline, then this script.

Call reset_walkaway_trial({...}) in the same namespace to start both actors.
Kinematic translation only; no walking gait. Human starts on the robot's lane
centreline, waits a fixed simulation-time interval, then departs diagonally ahead
and left. Departure never waits for model classification. Other scene obstacles
must be positioned separately; model height and orientation are preserved.
"""

import math
import omni.usd
import omni.timeline
import omni.kit.app
import omni.ui as ui
from pxr import Usd, UsdGeom, Gf


class HumanWalkAwayController:
    PRIM_PATH = "/jesus_gta"
    ROBOT_PATH = "/World/phantomX_base"
    START_AHEAD_M = 1.3  # Pivot offset from robot spawn; preserves human asset height.
    SPEED_MPS = 0.35
    START_WAIT_SECONDS = 1.5
    LEAVE_FORWARD_M = 2.0
    LEAVE_SIDE_M = 2.0  # Positive = left of robot's initial heading.

    def __init__(self):
        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()
        self.subscription = None
        self.window = None
        self.running = False
        self.log_event = None
        self.robot_namespace = None
        for value in (self.START_AHEAD_M, self.SPEED_MPS, self.START_WAIT_SECONDS,
                      self.LEAVE_FORWARD_M, self.LEAVE_SIDE_M):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Walk-away scenario parameters must be positive and finite")
        prim = self.stage.GetPrimAtPath(self.PRIM_PATH)
        robot = self.stage.GetPrimAtPath(self.ROBOT_PATH)
        if not prim.IsValid() or not robot.IsValid():
            raise RuntimeError("Scene must contain /jesus_gta and /World/phantomX_base")
        self.prim, self.robot = prim, robot
        self.xform = UsdGeom.Xformable(prim)
        ops = self.xform.GetOrderedXformOps()
        name = "xformOp:translate:humanApproachOffset"
        self.offset = next((op for op in ops if op.GetOpName() == name), None)
        if self.offset is None:
            self.offset = self.xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble,
                                                    "humanApproachOffset")
        self.xform.SetXformOpOrder(
            [self.offset] + [op for op in ops if op.GetOpName() != name],
            self.xform.GetResetXformStack(),
        )
        # Reuse the approach scenario's offset so switching behaviors does not
        # accumulate multiple translation offsets on /jesus_gta.
        self.offset.Set(Gf.Vec3d(0, 0, 0))
        self.base_translation = self.xform.GetLocalTransformation().ExtractTranslation()
        self.original_z = float(self.base_translation[2])
        self.phase = "READY"
        self.position = tuple(self.base_translation)
        self.last_sim_time = float(self.timeline.get_current_time())
        self.window = ui.Window("Human starts in lane / walks away", width=440, height=130)
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label(f"/jesus_gta | {self.SPEED_MPS:g} m/s | wait {self.START_WAIT_SECONDS:g} s")
                self.label = ui.Label("Ready: call reset_walkaway_trial({...})")
                ui.Button("Stop human", clicked_fn=self.pause)
        self.subscription = (omni.kit.app.get_app().get_update_event_stream()
                             .create_subscription_to_pop(self.update))

    def set_position(self, position):
        self.position = tuple(position)
        self.offset.Set(Gf.Vec3d(*(self.position[i] - self.base_translation[i]
                                 for i in range(3))))

    def gap(self):
        # Fresh cache each observation: USD transforms are edited every frame.
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        human = cache.ComputeWorldBound(self.prim).ComputeAlignedBox()
        robot = cache.ComputeWorldBound(self.robot).ComputeAlignedBox()
        lo_h, hi_h = human.GetMin(), human.GetMax()
        lo_r, hi_r = robot.GetMin(), robot.GetMax()
        values = [float(v[i]) for v in (lo_h, hi_h, lo_r, hi_r) for i in (0, 1)]
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError("Missing or nonfinite actor bounding boxes")
        return math.hypot(*(max(0.0, float(lo_h[i] - hi_r[i]),
                                    float(lo_r[i] - hi_h[i])) for i in (0, 1)))

    def event(self, phase):
        if self.log_event is not None:
            self.log_event("human_phase", human_phase=phase,
                           human_prim_path=self.PRIM_PATH,
                           human_position_m=list(self.position),
                           actor_aabb_gap_m=self.gap())

    def reset(self, spawn, yaw):
        self.running = False
        self.log_event = None
        rad = math.radians(yaw)
        self.forward = (math.cos(rad), math.sin(rad))
        left = (-self.forward[1], self.forward[0])
        displacement = tuple(self.forward[i] * self.LEAVE_FORWARD_M + left[i] * self.LEAVE_SIDE_M
                             for i in (0, 1))
        self.leave_length = math.hypot(*displacement)
        self.leave_direction = tuple(v / self.leave_length for v in displacement)
        self.start_position = (spawn[0] + self.forward[0] * self.START_AHEAD_M,
                               spawn[1] + self.forward[1] * self.START_AHEAD_M,
                               self.original_z)
        self.set_position(self.start_position)
        self.wait_elapsed = self.leave_progress = 0.0
        self.phase = "READY"
        self.last_sim_time = float(self.timeline.get_current_time())
        self.label.text = "Ready in lane"

    def start(self):
        self.phase = "WAIT"
        self.last_sim_time = float(self.timeline.get_current_time())
        self.event(self.phase)
        self.running = True
        if not self.timeline.is_playing():
            self.timeline.play()

    def pause(self):
        self.running = False
        self.label.text = "Human stopped; use reset_walkaway_trial({...}) to restart"

    def advance(self, delta, gap):
        if self.phase == "WAIT":
            waiting = min(delta, max(0.0, self.START_WAIT_SECONDS - self.wait_elapsed))
            self.wait_elapsed += waiting
            delta -= waiting
            if self.wait_elapsed < self.START_WAIT_SECONDS:
                return
            self.phase = "WALK_AWAY"
            self.event(self.phase)
        if self.phase == "WALK_AWAY":
            distance = min(self.SPEED_MPS * delta, self.leave_length - self.leave_progress)
            self.leave_progress += distance
            self.set_position((
                self.start_position[0] + self.leave_direction[0] * self.leave_progress,
                self.start_position[1] + self.leave_direction[1] * self.leave_progress,
                self.start_position[2],
            ))
            if self.leave_progress >= self.leave_length - 1e-9:
                self.phase = "DONE"
                self.running = False
                self.event(self.phase)

    def update(self, event):
        if omni.usd.get_context().get_stage() != self.stage:
            self.close()
            return
        now = float(self.timeline.get_current_time())
        delta = now - self.last_sim_time
        self.last_sim_time = now
        if delta < 0:
            self.pause()
            self.label.text = "Timeline rewound: reset the trial"
            return
        if not self.running or not self.timeline.is_playing() or delta == 0:
            return
        if self.robot_namespace.get("state") == "STOPPED":
            self.pause()
            return
        try:
            self.advance(delta, self.gap())
            self.label.text = f"{self.phase} | bounding-box gap {self.gap():.2f} m"
        except Exception:
            self.pause()
            self.robot_namespace["stop"](reason="human_scenario_failed")
            raise

    def close(self):
        self.running = False
        if self.subscription is not None:
            self.subscription.unsubscribe()
            self.subscription = None
        if self.window is not None:
            self.window.visible = False


# A previous controller must not keep moving the same asset during this trial.
for _other_human_key in ("_human_crossing_controller", "_human_approach_controller"):
    _other_human_controller = globals().get(_other_human_key)
    if _other_human_controller is not None:
        _other_human_controller.pause()
_old_walkaway_controller = globals().get("_human_walkaway_controller")
if _old_walkaway_controller is not None:
    _old_walkaway_controller.close()
_human_walkaway_controller = HumanWalkAwayController()


def reset_walkaway_trial(trial_context):
    """Reset robot + /jesus_gta and start a labeled simulation-time trial."""
    reset_robot, log_event = globals().get("perform_full_reset"), globals().get("log_decision")
    if not callable(reset_robot) or not callable(log_event) or not callable(globals().get("navigation_time")):
        raise RuntimeError("Run the simulation-time Standard-VLM.py or Standard-YOLO.py first")
    human = _human_walkaway_controller
    if human.subscription is None or omni.usd.get_context().get_stage() != human.stage:
        raise RuntimeError("Rerun Human-Walk-Away.py for the active stage")
    human.reset(globals()["SPAWN_POSITION"], globals()["SPAWN_ROTATION"][2])
    reset_robot(trial_context)
    if human.gap() <= globals().get("EMERGENCY_STOP_DIST", 0.25):
        globals()["stop"](reason="human_start_too_close")
        raise RuntimeError("Human starts too close; increase START_AHEAD_M")
    human.log_event = log_event
    human.robot_namespace = reset_robot.__globals__
    log_event("scenario_configured", human_prim_path=human.PRIM_PATH,
              human_start_m=list(human.start_position), human_speed_mps=human.SPEED_MPS,
              human_start_wait_s=human.START_WAIT_SECONDS,
              human_leave_forward_m=human.LEAVE_FORWARD_M,
              human_leave_side_m=human.LEAVE_SIDE_M,
              human_leave_distance_m=human.leave_length,
              human_leave_direction=list(human.leave_direction),
              human_clock="simulation_time", coordination="reset_together_start_human_immediately")
    human.start()
