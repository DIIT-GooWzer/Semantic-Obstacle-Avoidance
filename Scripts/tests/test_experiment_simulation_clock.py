"""Headless checks of experimental controllers; no live Isaac simulation."""

import contextlib
import io
import time
import unittest
from unittest.mock import patch

from test_pipeline_isolation import load_controller


class SimulationClockTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.controllers = []

    def tearDown(self):
        for env in self.controllers:
            if "_worker_stop" in env:
                env["_worker_stop"].set()
                env["_inference_thread"].join(timeout=1)
        self.output.__exit__(None, None, None)

    def load(self, pipeline):
        env = load_controller(f"Experiments-Scripts/Standard-{pipeline}.py")
        self.controllers.append(env)
        env["drain_model_results"] = lambda: None
        env["current_pos"] = [0.0, 0.0, 0.1]
        env["step_start_pos"] = list(env["current_pos"])
        env["current_yaw"] = 0.0
        env["last_time"] = 12.5
        env["_ros_node"].closest_obstacle_dist = float("inf")
        return env

    def tick(self, env, sim, wall, playing=True):
        env["timeline"].get_current_time.return_value = sim
        env["timeline"].is_playing.return_value = playing
        node = env["_ros_node"]
        node.rgb_received_at = node.depth_received_at = wall
        with patch.object(time, "monotonic", return_value=wall):
            env["zig_zag_update"](None)

    def test_same_simulation_motion_despite_different_wall_elapsed_time(self):
        for pipeline in ("VLM", "YOLO"):
            positions = []
            for wall_gap in (0.01, 5.0):
                env = self.load(pipeline)
                self.tick(env, 12.7, 100.0 + wall_gap)
                self.tick(env, 12.9, 100.0 + wall_gap * 2)
                positions.append(env["current_pos"][0])
            self.assertAlmostEqual(positions[0], 0.2)
            self.assertAlmostEqual(positions[0], positions[1])

    def test_timeline_pause_and_resume_do_not_accumulate_wall_motion(self):
        for pipeline in ("VLM", "YOLO"):
            env = self.load(pipeline)
            self.tick(env, 12.5, 100.0, playing=False)
            self.tick(env, 12.5, 200.0, playing=False)
            self.assertEqual(env["current_pos"][0], 0.0)
            self.tick(env, 12.6, 201.0)
            self.assertAlmostEqual(env["current_pos"][0], 0.05)

    def test_yield_policy_uses_simulation_seconds(self):
        for pipeline in ("VLM", "YOLO"):
            env = self.load(pipeline)
            env["state"] = "YIELD_WAITING"
            env["yield_wait_total_start_time"] = 12.5
            env["yield_wait_start_time"] = 12.5
            env["_ros_node"].closest_obstacle_dist = 0.7
            calls = []
            env["initiate_bypass_sequence"] = lambda: calls.append("bypass")
            self.tick(env, 12.6, 1000.0)
            self.assertEqual(calls, [])
            self.tick(env, 22.6, 1000.01)
            self.assertEqual(calls, ["bypass"])

    def test_inference_deadlines_remain_wall_time_while_simulation_paused(self):
        for pipeline in ("VLM", "YOLO"):
            env = self.load(pipeline)
            env["state"] = "CLASSIFYING"
            env["_active_request"] = 1
            env["classify_started_at"] = 100.0
            self.tick(env, 12.5, 500.0, playing=False)
            self.assertEqual(env["state"], "STOPPED")

    def test_rewind_stops_and_reset_rebases_navigation_clock(self):
        for pipeline in ("VLM", "YOLO"):
            env = self.load(pipeline)
            self.tick(env, 1.0, 100.0)
            self.assertEqual(env["state"], "STOPPED")
            env["perform_full_reset"]()
            self.assertEqual(env["last_time"], 1.0)
            self.assertEqual(env["state"], "MOVING")


if __name__ == "__main__":
    unittest.main()
