"""Headless checks of the real walk-away motion methods."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'Experiments-Scripts/Human-Walk-Away.py'


class HumanWalkAwayTests(unittest.TestCase):
    def make_controller(self):
        tree = ast.parse(SOURCE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        ns = {'math': math}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), 'exec'), ns)
        c = ns['HumanWalkAwayController'].__new__(ns['HumanWalkAwayController'])
        c.original_z = 0.48
        c.timeline = SimpleNamespace(get_current_time=lambda: 12.0)
        c.label = SimpleNamespace(text='')
        c.set_position = lambda p: setattr(c, 'position', p)
        c.reset((0.12, 1.87, 0.1), -90.0)
        c.phase, c.running = 'WAIT', True
        c.events = []
        c.event = lambda phase: c.events.append(phase)
        return c

    def test_starts_in_lane_and_waits(self):
        c = self.make_controller()
        self.assertAlmostEqual(c.position[0], 0.12)
        self.assertAlmostEqual(c.position[1], 0.57)
        initial = c.position
        c.advance(1.0, 1.0)
        self.assertEqual(c.phase, 'WAIT')
        self.assertEqual(c.position, initial)

    def test_motion_consumes_only_time_after_wait(self):
        c = self.make_controller()
        initial = c.position
        c.advance(c.START_WAIT_SECONDS + 1.0, 1.0)
        self.assertEqual(c.phase, 'WALK_AWAY')
        self.assertAlmostEqual(math.dist(initial, c.position), c.SPEED_MPS)
        self.assertGreater(c.position[0], initial[0])
        self.assertLess(c.position[1], initial[1])
        self.assertEqual(c.position[2], initial[2])

    def test_endpoint_and_frame_rate_independence(self):
        a, b = self.make_controller(), self.make_controller()
        a.advance(100, 1.0)
        for _ in range(1000):
            b.advance(0.1, 1.0)
        self.assertEqual(a.phase, 'DONE')
        self.assertFalse(a.running)
        self.assertEqual(a.events, ['WALK_AWAY', 'DONE'])
        for x, y in zip(a.position, b.position):
            self.assertAlmostEqual(x, y)
        self.assertAlmostEqual(a.position[0], 2.12)
        self.assertAlmostEqual(a.position[1], -1.43)


if __name__ == '__main__':
    unittest.main()
