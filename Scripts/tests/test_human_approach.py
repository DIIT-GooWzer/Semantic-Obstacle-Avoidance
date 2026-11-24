"""Exercise the real scenario state machine without Isaac dependencies."""
import ast
import math
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'Experiments-Scripts/Human-Approach-Stop-Leave.py'


class HumanApproachTests(unittest.TestCase):
    def make_controller(self):
        tree = ast.parse(SOURCE.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        namespace = {'math': math}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), 'exec'), namespace)
        c = namespace['HumanApproachController'].__new__(namespace['HumanApproachController'])
        c.position = (0.0, 2.0, 0.48)
        c.forward = (0.0, 1.0)
        c.leave_direction = (1.0, 0.0)
        c.phase = 'APPROACH'
        c.running = True
        c.leave_progress = c.hold_elapsed = 0.0
        c.set_position = lambda pos: setattr(c, 'position', pos)
        c.gap = lambda: c.position[1]
        c.events = []
        c.event = lambda phase: c.events.append((phase, c.position))
        return c

    def test_approach_clamps_at_separation(self):
        c = self.make_controller()
        c.advance(100.0, c.gap())
        self.assertAlmostEqual(c.gap(), c.STOP_SEPARATION_M)
        self.assertEqual(c.phase, 'HOLD')
        self.assertEqual(c.events[0][0], 'HOLD')
        self.assertEqual(c.position[2], 0.48)

    def test_hold_stays_fixed_then_leaves_and_stops(self):
        c = self.make_controller()
        c.advance(100.0, c.gap())
        held = c.position
        c.advance(2.0, c.gap())
        self.assertEqual(c.phase, 'HOLD')
        self.assertEqual(c.position, held)
        c.advance(1.0, c.gap())
        self.assertEqual(c.phase, 'LEAVE')
        self.assertEqual(c.position, held)
        c.advance(100.0, c.gap())
        self.assertAlmostEqual(c.position[0], c.LEAVE_DISTANCE_M)
        self.assertEqual(c.phase, 'DONE')
        self.assertFalse(c.running)
        self.assertEqual([phase for phase, _ in c.events], ['HOLD', 'LEAVE', 'DONE'])

    def test_approach_speed_and_already_close(self):
        c = self.make_controller()
        c.advance(0.2, c.gap())
        self.assertAlmostEqual(c.position[1], 1.95)
        c.position = (0.0, 0.5, 0.48)
        c.advance(0.1, c.gap())
        self.assertEqual(c.position, (0.0, 0.5, 0.48))
        self.assertEqual(c.phase, 'HOLD')


if __name__ == '__main__':
    unittest.main()
