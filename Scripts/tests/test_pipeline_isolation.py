"""Headless regression tests; Isaac/ROS/ONNX are mocked, not simulated."""

import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]


class FakeNode:
    def __init__(self, *args):
        pass

    def create_subscription(self, *args):
        return object()

    def destroy_node(self):
        pass


def load_controller(filename, navmesh_paths=None, onnx_names=None):
    """Execute every top-level statement to catch initialization-order errors."""
    modules = {}
    for name in [
        "omni",
        "omni.usd",
        "omni.timeline",
        "omni.kit",
        "omni.kit.app",
        "omni.ui",
        "pxr",
        "rclpy",
        "rclpy.node",
        "rclpy.qos",
        "sensor_msgs",
        "sensor_msgs.msg",
        "onnxruntime",
    ]:
        modules[name] = types.ModuleType(name)
    for name, module in modules.items():
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    stage = MagicMock()
    if navmesh_paths is None:
        navmesh_paths = {"/World/NavMeshVolume"}

    def get_prim(path):
        prim = MagicMock()
        prim.IsValid.return_value = path in navmesh_paths
        return prim

    stage.GetPrimAtPath.side_effect = get_prim
    stage.DefinePrim.return_value.IsValid.return_value = True
    modules["omni.usd"].get_context = lambda: types.SimpleNamespace(
        get_stage=lambda: stage
    )
    modules["omni.timeline"].get_timeline_interface = MagicMock()
    modules[
        "omni.timeline"
    ].get_timeline_interface.return_value.get_current_time.return_value = 12.5
    modules["omni.kit.app"].get_app = MagicMock()
    for name in ["Window", "VStack", "HStack", "Label", "Button"]:
        setattr(modules["omni.ui"], name, MagicMock())
    for name in ["Usd", "UsdGeom", "Gf", "Sdf", "Semantics"]:
        setattr(modules["pxr"], name, MagicMock())
    modules["pxr"].UsdGeom.Xformable.return_value.GetOrderedXformOps.return_value = []
    bound = modules[
        "pxr"
    ].UsdGeom.BBoxCache.return_value.ComputeWorldBound.return_value.ComputeAlignedBox.return_value
    bound.GetMin.return_value = (-10.0, -2.0, 0.0)
    bound.GetMax.return_value = (12.0, 2.0, 1.0)
    modules["pxr"].Gf.Vec3d.side_effect = lambda *args: args
    modules["rclpy"].ok = lambda: True
    modules["rclpy"].spin_once = lambda *args, **kwargs: None
    modules["rclpy.node"].Node = FakeNode
    for name in ["QoSProfile", "ReliabilityPolicy", "HistoryPolicy"]:
        setattr(modules["rclpy.qos"], name, MagicMock())
    modules["sensor_msgs.msg"].Image = object
    session = MagicMock()
    session.get_outputs.return_value = [types.SimpleNamespace(shape=[1, 300, 6])]
    session.get_inputs.return_value = [types.SimpleNamespace(name="images")]
    session.run.return_value = [np.zeros((1, 0, 6), dtype=np.float32)]
    modules["onnxruntime"].__version__ = "mock-test-runtime"
    session.get_providers.return_value = ["CPUExecutionProvider"]
    session.get_modelmeta.return_value.custom_metadata_map = {
        "names": repr(onnx_names) if onnx_names is not None else "{}"
    }
    modules["onnxruntime"].InferenceSession = MagicMock(return_value=session)
    env = {"__name__": "__test_controller__"}
    with tempfile.TemporaryDirectory() as folder, patch(
        "os.path.expanduser", return_value=folder
    ), patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
        exec(compile((SCRIPTS / filename).read_text(), filename, "exec"), env)
    env["RESULT_LOG_PATH"] = "/dev/null"
    env["_ros_node"].latest_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    # Provide a valid already-settled observation for generic controller tests.
    # Dedicated sensor tests exercise the real callback/reset gate separately.
    if hasattr(env["_ros_node"], "sensor_reset_sim_ns"):
        env["_ros_node"].sensor_reset_sim_ns = 0
        env["_ros_node"].sensor_accept_after_wall = 0.0
        env["_ros_node"]._rgb_stamp = env["_ros_node"]._depth_stamp = (13, 0)
        env["_ros_node"].latest_depth = np.ones((100, 100), dtype=np.float32)
    env["_ros_node"].rgb_received_at = env["_ros_node"].depth_received_at = (
        time.monotonic()
    )
    env["_ros_node"].closest_obstacle_dist = 0.7
    env["_ros_node"].depth_left_dist = env["_ros_node"].depth_right_dist = 2.0
    env["_ros_node"].depth_center_dist = 0.7
    return env


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.controllers = []
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        for env in self.controllers:
            if "_worker_stop" in env:
                env["_worker_stop"].set()
                env["_inference_thread"].join(timeout=1)
        self.output.__exit__(None, None, None)

    def load(self, filename):
        env = load_controller(filename)
        self.controllers.append(env)
        return env

    def test_model_dependencies_are_exclusive(self):
        for filename, forbidden in [
            ("amazing8.py", "onnxruntime"),
            ("amazing8-novlm.py", "urllib"),
        ]:
            tree = ast.parse((SCRIPTS / filename).read_text())
            imports = [
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            ]
            self.assertFalse(any(name.startswith(forbidden) for name in imports))
            names = {
                node.name for node in tree.body if isinstance(node, ast.FunctionDef)
            }
            if filename == "amazing8.py":
                self.assertNotIn("find_confident_yolo_match", names)
            else:
                self.assertNotIn("parse_vlm_reply", names)

    def test_initialization_and_all_navigation_states(self):
        for filename in ["amazing8.py", "amazing8-novlm.py"]:
            env = self.load(filename)
            self.assertAlmostEqual(env["AVOIDANCE_STEP_SIZE"], 0.75)
            tree = ast.parse((SCRIPTS / filename).read_text())
            states = {
                node.comparators[0].value
                for node in ast.walk(tree)
                if isinstance(node, ast.Compare)
                and isinstance(node.left, ast.Name)
                and node.left.id == "state"
                and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant)
                and isinstance(node.comparators[0].value, str)
            }
            for state in states:
                env["reset_navigation_state"]()
                env["state"] = state
                env["start_selected_model"] = lambda: None
                env["zig_zag_update"](None)

    def test_mapping_clear_and_failure_stop(self):
        for filename in ["amazing8.py", "amazing8-novlm.py"]:
            env = self.load(filename)
            for category, expected in [
                ("CLEAR", "MOVING"),
                ("BIOLOGICAL", "YIELD_WAITING"),
                ("MINOR_SURFACE", "SCAN_AND_PROCEED"),
                ("STATIC_BAR", "AVOID_INITIAL_TURN"),
            ]:
                env["reset_navigation_state"]()
                env["current_pos"] = [0.0, 0.0, 0.1]
                env["current_yaw"] = 0.0
                env["state"], env["avoidance_sub_phase"] = (
                    "CLASSIFYING",
                    "CLASSIFY_OBSTACLE",
                )
                env["_active_request"] = 1
                env["classify_started_at"] = time.monotonic()
                env["apply_category"](category)
                self.assertEqual(env["state"], expected, (filename, category))
            env["_active_request"] = 2
            env["state"] = "CLASSIFYING"
            env["fail_classification"]("no prediction")
            self.assertEqual(env["state"], "STOPPED")
            self.assertIsNone(env["_active_request"])

    def test_vlm_parsing_and_stale_result(self):
        env = self.load("amazing8.py")
        self.assertEqual(
            env["parse_vlm_reply"]('```json\n{"category":"CLEAR"}\n```')["category"],
            "CLEAR",
        )
        for raw in ["{}", "[]", '{"category":"COW"}', "not json"]:
            with self.assertRaises((ValueError, TypeError)):
                env["parse_vlm_reply"](raw)
        env["state"], env["_active_request"] = "CLASSIFYING", 2
        env["classify_started_at"] = time.monotonic()
        env["_result_queue"].put(
            (env["_model_token"], 1, {"category": "STATIC_BAR"}, "", 0.1, None)
        )
        env["drain_model_results"]()
        self.assertEqual(env["state"], "CLASSIFYING")
        env["_result_queue"].put(
            (env["_model_token"], 2, {"category": "CLEAR"}, "{}", 0.1, None)
        )
        env["drain_model_results"]()
        self.assertEqual(env["state"], "MOVING")

    def test_vlm_http_request_and_error_are_not_detector_fallbacks(self):
        env = self.load("amazing8.py")
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": '{"category":"BIOLOGICAL"}'}}]}
        ).encode()
        with patch("urllib.request.urlopen", return_value=response) as transport:
            env["begin_classification"](0.7)
            deadline = time.monotonic() + 2
            while env["_result_queue"].empty() and time.monotonic() < deadline:
                time.sleep(0.01)
            env["drain_model_results"]()
            self.assertEqual(env["state"], "YIELD_WAITING")
            payload = json.loads(transport.call_args.args[0].data)
            self.assertEqual(payload["temperature"], 0.0)
            self.assertEqual(payload["model"], env["VLM_MODEL"])
        with patch("urllib.request.urlopen", side_effect=OSError("server unavailable")):
            env["begin_classification"](0.7)
            deadline = time.monotonic() + 2
            while env["_result_queue"].empty() and time.monotonic() < deadline:
                time.sleep(0.01)
            env["drain_model_results"]()
            self.assertEqual(env["state"], "STOPPED")

    def test_kinematic_navigation_and_action_policy_match_between_scripts(self):
        trees = [
            ast.parse((SCRIPTS / name).read_text())
            for name in ("amazing8.py", "amazing8-novlm.py")
        ]
        for name in (
            "zig_zag_update",
            "initiate_bypass_sequence",
            "apply_category",
            "move_forward",
            "reset_navigation_state",
            "capture_observation",
            "log_decision",
            "begin_classification",
            "perform_full_reset",
            "resolve_navmesh",
        ):
            functions = [
                next(
                    n
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == name
                )
                for tree in trees
            ]
            self.assertEqual(ast.dump(functions[0]), ast.dump(functions[1]), name)

    def test_yolo_confidence_and_reset(self):
        env = self.load("amazing8-novlm.py")
        now = time.monotonic()
        for sequence in range(1, 6):
            env["update_tracks"](
                sequence,
                now,
                (0.7, 0.7, 0.7),
                [
                    dict(
                        bbox=[30.0, 30.0, 60.0, 60.0],
                        confidence=0.8,
                        class_name="person",
                    )
                ],
                0.01,
            )
        self.assertEqual(env["find_confident_yolo_match"](0.7)["class_hits"], 5)
        env["state"], env["_active_request"] = "CLASSIFYING", 1
        env["classify_started_at"] = now
        env["poll_classification"]()
        self.assertEqual(env["state"], "YIELD_WAITING")
        old_generation = env["_sensor_generation"]
        env["perform_full_reset"]()
        self.assertGreater(env["_sensor_generation"], old_generation)
        self.assertIsNone(env["_ros_node"].latest_rgb)
        self.assertEqual(env["_yolo_tracks"], [])
        env["state"], env["_active_request"] = "CLASSIFYING", 2
        env["classify_started_at"] = now - 10
        env["poll_classification"]()
        self.assertEqual(env["state"], "STOPPED")

    def test_duplicate_rgb_and_one_to_one_tracks(self):
        env = self.load("amazing8-novlm.py")
        node = env["_ros_node"]
        msg = types.SimpleNamespace(
            header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=1, nanosec=1)),
            encoding="bgr8",
            height=2,
            width=2,
            data=bytes(12),
        )
        node._rgb_callback(msg)
        seq = node.rgb_sequence
        node._rgb_callback(msg)
        self.assertEqual(node.rgb_sequence, seq)
        node.latest_rgb = np.zeros((100, 100, 3), np.uint8)
        for sequence in range(1, 3):
            env["update_tracks"](
                sequence,
                time.monotonic(),
                (0.7, 0.7, 0.7),
                [
                    dict(
                        bbox=[30.0, 30.0, 60.0, 60.0], confidence=0.8, class_name="cone"
                    ),
                    dict(
                        bbox=[32.0, 32.0, 62.0, 62.0], confidence=0.8, class_name="cone"
                    ),
                ],
                0.01,
            )
        self.assertEqual(len({track["track_id"] for track in env["_yolo_tracks"]}), 2)
        with self.assertRaises(ValueError):
            env["decode_detections"](np.zeros((1, 9, 8400)), 1, 0, 0)

    def test_phase1_schema_and_frozen_start_snapshot(self):
        common_keys = None
        for filename in ("amazing8.py", "amazing8-novlm.py"):
            env = self.load(filename)
            env["start_selected_model"] = lambda: None
            with tempfile.TemporaryDirectory() as folder:
                env["RESULT_LOG_PATH"] = str(Path(folder) / "events.jsonl")
                env["begin_classification"](0.7)
                start_pos = list(env["current_pos"])
                env["current_pos"][0] += 0.1
                env["_ros_node"].closest_obstacle_dist = float("inf")
                env["apply_category"](
                    "CLEAR", inference_s=0.02, example_model_field="value"
                )
                records = [
                    json.loads(line, parse_constant=lambda value: self.fail(value))
                    for line in Path(env["RESULT_LOG_PATH"]).read_text().splitlines()
                ]
                self.assertEqual(
                    [r["status"] for r in records], ["detected", "predicted"]
                )
                self.assertEqual(
                    records[0]["classification_start"],
                    records[1]["classification_start"],
                )
                self.assertEqual(
                    records[1]["classification_start"]["robot_pos"], start_pos
                )
                self.assertNotEqual(records[1]["robot_pos"], start_pos)
                self.assertIsNone(records[1]["obstacle_distance_m"])
                self.assertEqual(records[1]["sim_time"], 12.5)
                self.assertEqual(records[1]["inference_s"], 0.02)
                self.assertEqual(
                    records[1]["model_metadata"], {"example_model_field": "value"}
                )
                self.assertEqual(records[0]["request_id"], records[1]["request_id"])
                self.assertEqual(records[0]["trial_id"], records[1]["trial_id"])
                self.assertEqual(records[0]["session_id"], records[1]["session_id"])
                for record in records:
                    if common_keys is None:
                        common_keys = set(record)
                    self.assertEqual(set(record), common_keys)

    def test_yolo_track_reacquires_across_misses_without_counting_them(self):
        env = self.load("amazing8-novlm.py")
        captured = time.monotonic() - 0.5
        original_id = None
        for sequence in range(1, 10):
            detected = sequence % 2 == 1
            detections = (
                [
                    dict(
                        bbox=[30.0, 30.0, 60.0, 60.0],
                        confidence=0.8,
                        class_name="person",
                    )
                ]
                if detected
                else []
            )
            env["update_tracks"](
                sequence, captured + sequence * 0.01, (0.7, 0.7, 0.7), detections, 0.01
            )
            track = next(iter(env["_track_memory"].values()))
            if original_id is None:
                original_id = track["track_id"]
            self.assertEqual(track["track_id"], original_id)
            self.assertEqual(track["class_hits"], (sequence + 1) // 2)
            self.assertEqual(track["age"], 0 if detected else 1)
            if not detected:
                self.assertEqual(track["sequence"], sequence - 1)
                self.assertAlmostEqual(
                    track["captured"], captured + (sequence - 1) * 0.01
                )
                self.assertIsNone(env["find_confident_yolo_match"](0.7))
        self.assertEqual(env["find_confident_yolo_match"](0.7)["class_hits"], 5)
        env["update_tracks"](10, captured + 0.1, (0.7, 0.7, 0.7), [], 0.01)
        self.assertEqual(next(iter(env["_track_memory"].values()))["class_hits"], 5)
        self.assertIsNone(env["find_confident_yolo_match"](0.7))

    def test_yolo_track_expires_by_frame_count_and_elapsed_time(self):
        env = self.load("amazing8-novlm.py")

        def detection():
            return [
                dict(bbox=[30.0, 30.0, 60.0, 60.0], confidence=0.8, class_name="person")
            ]

        captured = time.monotonic() - 0.5
        env["update_tracks"](1, captured, (0.7, 0.7, 0.7), detection(), 0.01)
        old_id = env["_yolo_tracks"][0]["track_id"]
        for age in range(1, env["TRACK_MAX_AGE"] + 1):
            env["update_tracks"](
                1 + age, captured + age * 0.001, (0.7, 0.7, 0.7), [], 0.01
            )
        self.assertEqual(env["_track_memory"][old_id]["age"], env["TRACK_MAX_AGE"])
        env["update_tracks"](12, captured + 0.02, (0.7, 0.7, 0.7), [], 0.01)
        self.assertEqual(env["_track_memory"], {})
        env["update_tracks"](13, captured + 0.03, (0.7, 0.7, 0.7), detection(), 0.01)
        self.assertNotEqual(env["_yolo_tracks"][0]["track_id"], old_id)
        env["reset_selected_model"]()
        env["update_tracks"](1, captured, (0.7, 0.7, 0.7), detection(), 0.01)
        env["update_tracks"](
            2, captured + env["SENSOR_MAX_AGE"] + 0.01, (0.7, 0.7, 0.7), [], 0.01
        )
        self.assertEqual(env["_track_memory"], {})

    def test_yolo_duplicate_results_and_low_confidence_cannot_inflate_hits(self):
        env = self.load("amazing8-novlm.py")
        captured = time.monotonic() - 0.5

        def update(sequence, confidence=0.8, category="person"):
            env["update_tracks"](
                sequence,
                captured + sequence * 0.01,
                (0.7, 0.7, 0.7),
                [
                    dict(
                        bbox=[30.0, 30.0, 60.0, 60.0],
                        confidence=confidence,
                        class_name=category,
                    )
                ],
                0.01,
            )

        update(1)
        update(1)
        update(0)
        self.assertEqual(env["_yolo_tracks"][0]["class_hits"], 1)
        update(2, 0.5)
        self.assertEqual(env["_yolo_tracks"][0]["class_hits"], 0)
        update(3)
        self.assertEqual(env["_yolo_tracks"][0]["class_hits"], 1)
        update(4, category="cone")
        self.assertEqual(env["_yolo_tracks"][0]["class_hits"], 1)
        env["reset_selected_model"]()
        update(1)
        self.assertEqual(env["_yolo_tracks"][0]["class_hits"], 1)

    def test_phase1_reset_ids_and_manual_abort(self):
        sessions = set()
        for filename in ("amazing8.py", "amazing8-novlm.py"):
            env = self.load(filename)
            sessions.add(env["_session_id"])
            env["start_selected_model"] = lambda: None
            with tempfile.TemporaryDirectory() as folder:
                env["RESULT_LOG_PATH"] = str(Path(folder) / "events.jsonl")
                old_trial = env["_trial_id"]
                env["begin_classification"](0.7)
                old_encounter = env["current_encounter_id"]
                env["perform_full_reset"](
                    {
                        "scenario_id": "stationary_human",
                        "scenario_instance_id": "pair-01",
                        "scenario_config_id": "config-v1",
                        "expected_initial_command": "yield_and_wait",
                    }
                )
                self.assertNotEqual(env["_trial_id"], old_trial)
                self.assertEqual(env["current_encounter_id"], old_encounter)
                env["begin_classification"](0.7)
                self.assertEqual(env["current_encounter_id"], old_encounter + 1)
                env["stop"]()
                records = [
                    json.loads(line)
                    for line in Path(env["RESULT_LOG_PATH"]).read_text().splitlines()
                ]
                starts = [r for r in records if r["status"] == "trial_started"]
                self.assertEqual(starts[0]["scenario_instance_id"], "pair-01")
                self.assertIsNone(starts[0]["encounter_id"])
                finished = [r for r in records if r["status"] == "run_finished"]
                self.assertEqual(len(finished), 2)
                self.assertTrue(all(r["outcome"] == "aborted" for r in finished))
                self.assertEqual(len({r["session_id"] for r in records}), 1)
                self.assertEqual(
                    [r["event_sequence"] for r in records],
                    sorted({r["event_sequence"] for r in records}),
                )
        self.assertEqual(len(sessions), 2)

    def test_phase1_yield_timeout_is_policy_override(self):
        for filename in ("amazing8.py", "amazing8-novlm.py"):
            env = self.load(filename)
            env["start_selected_model"] = lambda: None
            env["current_pos"] = [0.0, 0.0, 0.1]
            env["current_yaw"] = 0.0
            with tempfile.TemporaryDirectory() as folder:
                env["RESULT_LOG_PATH"] = str(Path(folder) / "events.jsonl")
                env["begin_classification"](0.7)
                env["apply_category"]("BIOLOGICAL")
                env["yield_wait_total_start_time"] = time.monotonic() - 11
                env["zig_zag_update"](None)
                records = [
                    json.loads(line)
                    for line in Path(env["RESULT_LOG_PATH"]).read_text().splitlines()
                ]
                override = next(r for r in records if r["status"] == "policy_override")
                self.assertEqual(override["category"], "BIOLOGICAL")
                self.assertEqual(override["command"], "standard_bypass")
                self.assertEqual(override["reason"], "yield_timeout")
                self.assertEqual(env["active_obstacle_category"], "BIOLOGICAL")
                self.assertEqual(env["state"], "AVOID_INITIAL_TURN")

    def test_phase1_shared_navmesh_and_no_auto_loop(self):
        for filename in ("amazing8.py", "amazing8-novlm.py"):
            env = self.load(filename)
            self.assertFalse(env["AUTO_LOOP_ENABLED"])
            env["finish_run"]()
            self.assertFalse(env["_loop_restart_pending"])
            self.assertEqual(env["NAVMESH_PATH"], "/World/NavMeshVolume")
            missing = MagicMock()
            missing.GetPrimAtPath.return_value.IsValid.return_value = False
            with self.assertRaisesRegex(RuntimeError, "No NavMeshVolume"):
                env["resolve_navmesh"](missing)
            both = MagicMock()
            both.GetPrimAtPath.return_value.IsValid.return_value = True
            self.assertEqual(
                env["resolve_navmesh"](both)[0], env["NAVMESH_CANDIDATE_PATHS"][0]
            )

    def test_phase1_log_write_failure_stops_execution(self):
        env = self.load("amazing8.py")
        with patch("builtins.open", side_effect=OSError("disk unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Decision logging failed"):
                env["log_decision"]("detected")
        self.assertEqual(env["state"], "STOPPED")

    def test_lane_end_stops_in_both_directions_without_turning(self):
        for filename in ("amazing8.py", "amazing8-novlm.py"):
            env = self.load(filename)
            for direction in (-1.0, 1.0):
                env["reset_navigation_state"]()
                env["_trial_finished"] = False
                env["sweep_direction_y"] = direction
                env["current_yaw"] = -90.0 if direction < 0 else 90.0
                edge = env["SAFE_MIN_Y"] if direction < 0 else env["SAFE_MAX_Y"]
                env["current_pos"] = [0.0, edge - direction * 0.005, 0.1]
                env["last_time"] = time.monotonic() - 0.05
                env["_ros_node"].closest_obstacle_dist = float("inf")
                env["_ros_node"].rgb_received_at = env[
                    "_ros_node"
                ].depth_received_at = time.monotonic()
                yaw = env["current_yaw"]
                env["zig_zag_update"](None)
                self.assertEqual(env["state"], "STOPPED")
                self.assertEqual(env["current_yaw"], yaw)
                self.assertAlmostEqual(env["current_pos"][0], 0.0)
                self.assertAlmostEqual(env["current_pos"][1], edge)
                self.assertIn("Lane End Reached", env["status_label"].text)
                self.assertFalse(env["_loop_restart_pending"])
                stopped_pos = list(env["current_pos"])
                env["zig_zag_update"](None)
                self.assertEqual(env["current_pos"], stopped_pos)

    def test_blocked_lane_and_legacy_shift_states_stop(self):
        for filename in ("amazing8.py", "amazing8-novlm.py"):
            env = self.load(filename)
            env["stop_blocked_lane"]()
            self.assertEqual(env["state"], "STOPPED")
            for state in ("TURNING_1", "TURNING_2", "LANE_SHIFTING", "BIG_SKIP_DRIVE"):
                env["reset_navigation_state"]()
                env["_trial_finished"] = False
                env["state"] = state
                env["_ros_node"].closest_obstacle_dist = float("inf")
                before = list(env["current_pos"])
                yaw = env["current_yaw"]
                env["zig_zag_update"](None)
                self.assertEqual(env["state"], "STOPPED")
                self.assertEqual(env["current_pos"], before)
                self.assertEqual(env["current_yaw"], yaw)


if __name__ == "__main__":
    unittest.main()
