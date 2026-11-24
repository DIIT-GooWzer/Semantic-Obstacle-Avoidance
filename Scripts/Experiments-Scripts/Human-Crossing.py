"""Run in Isaac Sim Script Editor alongside ONE Standard robot pipeline.

The existing human slides kinematically from A to B; this does not generate a
walking gait. Motion uses timeline seconds and continues during robot inference
and yielding. Running this file starts the crossing immediately. It does not
synchronize robot startup or guarantee an encounter within three robot steps.

After loading the robot and this script in the same Script Editor namespace, use
reset_crossing_trial({...}) instead of perform_full_reset({...}) to reset both.
"""

import math
import omni.usd
import omni.timeline
import omni.kit.app
import omni.ui as ui
from pxr import UsdGeom, Gf


class HumanCrossingController:
    PRIM_PATH = "/Rugged_Winter_Traveler"
    START = (1.8807980725731495, 0.33112147104726, 0.4820345021416584)
    END = (-1.5034274817465252, -0.5971399678209076, 0.4820345021416566)
    SPEED_MPS = 0.35

    def __init__(self):
        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()
        self.subscription = None
        self.window = None
        self.running = False
        if not math.isfinite(self.SPEED_MPS) or self.SPEED_MPS <= 0:
            raise ValueError("Human crossing speed must be positive and finite")
        prim = self.stage.GetPrimAtPath(self.PRIM_PATH)
        if not prim.IsValid():
            raise RuntimeError("Human prim missing: " + self.PRIM_PATH)
        self.xform = UsdGeom.Xformable(prim)
        if not self.xform:
            raise RuntimeError("Human prim must be transformable")
        # A leading offset preserves the asset's scale, orientation and pivot.
        # This prim is at the stage root, so parent coordinates are world coordinates.
        ops = self.xform.GetOrderedXformOps()
        name = "xformOp:translate:humanCrossingOffset"
        self.offset = next((op for op in ops if op.GetOpName() == name), None)
        if self.offset is None:
            self.offset = self.xform.AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble, "humanCrossingOffset"
            )
        reset_stack = self.xform.GetResetXformStack()
        self.xform.SetXformOpOrder(
            [self.offset] + [op for op in ops if op.GetOpName() != name], reset_stack
        )
        self.offset.Set(Gf.Vec3d(0, 0, 0))
        self.base_translation = self.xform.GetLocalTransformation().ExtractTranslation()
        self.length = math.dist(self.START, self.END)
        self.elapsed = 0.0
        self.last_sim_time = self.timeline.get_current_time()
        self.window = ui.Window("Human crossing", width=370, height=140)
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label(f"Pedestrian A -> B | {self.SPEED_MPS:g} m/s | simulation time")
                self.label = ui.Label("Ready")
                with ui.HStack():
                    ui.Button("Start / Resume", clicked_fn=self.start)
                    ui.Button("Pause", clicked_fn=self.pause)
                    ui.Button("Reset to A", clicked_fn=self.reset)
        self.reset()
        self.subscription = (
            omni.kit.app.get_app().get_update_event_stream()
            .create_subscription_to_pop(self.update)
        )
        self.start()

    def set_position(self):
        fraction = min(1.0, self.elapsed * self.SPEED_MPS / self.length) if self.length else 1.0
        position = tuple(a + fraction * (b - a) for a, b in zip(self.START, self.END))
        self.offset.Set(Gf.Vec3d(*(position[i] - self.base_translation[i] for i in range(3))))
        self.label.text = f"{'Moving' if self.running else 'Paused'}: {fraction * 100:.1f}%"
        if fraction >= 1.0:
            self.running = False
            self.label.text = "Finished: human at B"

    def start(self):
        if self.elapsed * self.SPEED_MPS >= self.length:
            self.reset()
        self.last_sim_time = self.timeline.get_current_time()
        self.running = True
        if not self.timeline.is_playing():
            self.timeline.play()

    def pause(self):
        self.running = False
        self.set_position()

    def reset(self):
        self.running = False
        self.elapsed = 0.0
        self.last_sim_time = self.timeline.get_current_time()
        self.set_position()

    def update(self, event):
        if omni.usd.get_context().get_stage() != self.stage:
            self.close()
            return
        now = self.timeline.get_current_time()
        delta = now - self.last_sim_time
        self.last_sim_time = now
        if delta < 0:
            # Rewinding invalidates this execution; explicitly restart from A.
            self.reset()
            self.label.text = "Timeline rewound: reset to A; press Start"
            return
        if self.running and self.timeline.is_playing():
            self.elapsed += delta
            self.set_position()

    def close(self):
        self.running = False
        if self.subscription is not None:
            self.subscription.unsubscribe()
            self.subscription = None
        if self.window is not None:
            self.window.visible = False


# Dedicated global avoids interfering with the robot's rerun cleanup and workers.
_old_human_crossing = globals().get("_human_crossing_controller")
if _old_human_crossing is not None:
    _old_human_crossing.close()
_human_crossing_controller = HumanCrossingController()


def reset_crossing_trial(trial_context):
    """Reset both actors, label the robot trial, then release the pedestrian.

    Robot and human scripts must have been executed in the same globals namespace.
    This coordinates reset/start only; robot sensor readiness can delay its motion.
    """
    reset_robot = globals().get("perform_full_reset")
    log_event = globals().get("log_decision")
    if callable(reset_robot) and not callable(log_event):
        raise RuntimeError(
            "The loaded robot controller lacks experiment logging. The legacy "
            "amazing8.py is incompatible with labeled crossing trials. Run "
            "Scripts/Experiments-Scripts/Standard-VLM.py or Standard-YOLO.py, "
            "then rerun Human-Crossing.py and reset_crossing_trial({...})."
        )
    if not callable(reset_robot) or not callable(log_event):
        raise RuntimeError(
            "Run Standard-VLM.py or Standard-YOLO.py first, then Human-Crossing.py and this call "
            "in the same Script Editor namespace."
        )
    human = _human_crossing_controller
    if human.subscription is None or omni.usd.get_context().get_stage() != human.stage:
        raise RuntimeError("Human controller is inactive; rerun Human-Crossing.py")
    human.reset()
    reset_robot(trial_context)
    # Record actor settings as trial evidence; do not pass them to either model.
    log_event(
        "scenario_configured",
        human_prim_path=human.PRIM_PATH,
        human_start_m=list(human.START),
        human_end_m=list(human.END),
        human_speed_mps=human.SPEED_MPS,
        human_clock="simulation_time",
        human_start_delay_s=0.0,    
        coordination="reset_together_start_human_immediately",
    )
    human.start()
