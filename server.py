#!/usr/bin/env python3
"""
Hexapod Robot Mission Control & Scanning Web Server
----------------------------------------------------
Provides:
1. Real-time WebSocket + REST API for Hexapod robot control.
2. Area-directed mission dispatch ("Move to Area A and Scan").
3. Directional scanning & manual teleoperation.
4. GroundMapper 2D defect accumulation & connected-component defect extraction.
5. Digital Twin / Simulation engine matching PhantomX characteristics.
6. Live Isaac Sim WebSocket Bridge integration.
7. Terrain management & custom area configuration.
"""

import os
import sys
import time
import math
import json
import base64
import random
import asyncio
import threading
from typing import Dict, List, Any, Optional

import numpy as np
import tornado.ioloop
import tornado.web
import tornado.websocket

try:
    from scipy import ndimage
    _has_scipy = True
except ImportError:
    _has_scipy = False

try:
    from PIL import Image as PILImage, ImageDraw as PILImageDraw
    _has_pil = True
except ImportError:
    _has_pil = False

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================
DEFAULT_PORT = 3001
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Default Terrain / Bounds for the Operational Area (Hexapod NavMesh Track)
DEFAULT_TERRAIN = {
    "name": "Hexapod NavMesh Inspection Track",
    "min_x": -10.0,
    "max_x": 12.0,
    "min_y": -2.5,
    "max_y": 2.5,
    "safety_margin": 0.35,
    "spawn_pos": [0.12, 1.87],
    "spawn_yaw": -90.0,
}

# Default Battery Charging Station (Autonomous Recharge Depot 🔋)
DEFAULT_CHARGING_STATION = {
    "x": 0.12,
    "y": 1.87,
    "radius": 0.8,
    "name": "Spawn Docking Base 🔋",
    "emoji": "🔋"
}

# Pre-defined Scanning Areas within the Operational Zone
DEFAULT_AREAS = [
    {
        "id": "A",
        "name": "Area A (Entry Staging)",
        "x": 1.5,
        "y": 0.0,
        "radius": 1.2,
        "color": "#38bdf8",
        "description": "Initial staging and sensor calibration zone"
    },
    {
        "id": "B",
        "name": "Area B (Mid-Track Obstacle Sector)",
        "x": 4.5,
        "y": 0.0,
        "radius": 1.5,
        "color": "#ec4899",
        "description": "Mid-track hazardous section with obstacles"
    },
    {
        "id": "C",
        "name": "Area C (Target Destination)",
        "x": 9.0,
        "y": -1.0,
        "radius": 1.2,
        "color": "#22c55e",
        "description": "Primary mission destination goal"
    },
    {
        "id": "D",
        "name": "Area D (Surface Inspection Zone)",
        "x": 7.0,
        "y": 0.5,
        "radius": 1.2,
        "color": "#a855f7",
        "description": "Pothole, crack, and defect inspection zone"
    }
]

# Ground Mapper parameters
MAP_RESOLUTION_M = 0.05
MAP_MIN_HITS = 3
MAP_MIN_BLOB_CELLS = 4

# Robot dynamics
FORWARD_SPEED = 1.0       # m/s (boosted from 0.5 for responsive walking and scanning)
TURN_SPEED = 180.0        # deg/s
STEP_DISTANCE = 0.4       # m
PAUSE_DURATION = 0.5      # s
LANE_SHIFT = 0.5          # m
OBSTACLE_THRESHOLD = 0.65 # m
EMERGENCY_STOP_DIST = 0.35# m
TARGET_RADIUS = 0.35      # m

# Camera parameters
CAM_HEIGHT_M = 0.45
CAM_FORWARD_OFFSET_M = 0.10
CAM_PITCH_DEG = 12.0
CAM_HFOV_DEG = 90.0
CAM_VFOV_DEG = 60.0
DEFECT_DEPTH_TOL_M = 0.05


# ============================================================
# GROUND MAPPER
# ============================================================
class GroundMapper:
    """World-aligned 2D accumulator grid for surface defects."""
    def __init__(self, min_x, max_x, min_y, max_y, resolution=MAP_RESOLUTION_M):
        self.res = resolution
        self.update_bounds(min_x, max_x, min_y, max_y)

    def update_bounds(self, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.nx = max(10, int((max_x - min_x) / self.res) + 1)
        self.ny = max(10, int((max_y - min_y) / self.res) + 1)
        self.accumulator = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.frame_count = 0
        self.total_points = 0

    def integrate(self, world_x, world_y, weights=None):
        if len(world_x) == 0:
            return
        gx = ((np.array(world_x) - self.min_x) / self.res).astype(np.int32)
        gy = ((np.array(world_y) - self.min_y) / self.res).astype(np.int32)
        valid = (gx >= 0) & (gx < self.nx) & (gy >= 0) & (gy < self.ny)
        if np.any(valid):
            w = 1.0 if weights is None else np.array(weights)[valid]
            np.add.at(self.accumulator, (gy[valid], gx[valid]), w)
            self.total_points += int(np.count_nonzero(valid))
        self.frame_count += 1

    def extract_defects(self, min_hits=MAP_MIN_HITS, min_cells=MAP_MIN_BLOB_CELLS):
        binary = (self.accumulator >= min_hits).astype(np.uint8)
        if binary.sum() == 0:
            return []

        defects = []
        if _has_scipy:
            labeled_array, num_features = ndimage.label(binary)
            for i in range(1, num_features + 1):
                mask = (labeled_array == i)
                cells = int(np.count_nonzero(mask))
                if cells < min_cells:
                    continue
                y_indices, x_indices = np.where(mask)
                cx_pix = float(np.mean(x_indices))
                cy_pix = float(np.mean(y_indices))
                hit_sum = float(self.accumulator[mask].sum())
                wx = self.min_x + cx_pix * self.res
                wy = self.min_y + cy_pix * self.res
                area_m2 = round(cells * (self.res ** 2), 4)
                severity = min(1.0, round(float(np.max(self.accumulator[mask]) / 15.0), 2))
                defects.append({
                    "id": f"DET-{len(defects) + 1:03d}",
                    "world_x": round(float(wx), 3),
                    "world_y": round(float(wy), 3),
                    "area_m2": area_m2,
                    "cells": cells,
                    "total_hits": int(hit_sum),
                    "severity": severity,
                })
        else:
            # Fallback BFS connected components
            visited = np.zeros_like(binary, dtype=bool)
            defects = []
            for r in range(self.ny):
                for c in range(self.nx):
                    if binary[r, c] and not visited[r, c]:
                        component = []
                        q = [(r, c)]
                        visited[r, c] = True
                        while q:
                            cr, cc = q.pop(0)
                            component.append((cr, cc))
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < self.ny and 0 <= nc < self.nx:
                                    if binary[nr, nc] and not visited[nr, nc]:
                                        visited[nr, nc] = True
                                        q.append((nr, nc))
                        if len(component) >= min_cells:
                            cells = len(component)
                            cx_pix = sum(c for _, c in component) / cells
                            cy_pix = sum(r for r, _ in component) / cells
                            hit_sum = sum(self.accumulator[r, c] for r, c in component)
                            wx = self.min_x + cx_pix * self.res
                            wy = self.min_y + cy_pix * self.res
                            defects.append({
                                "id": f"DET-{len(defects) + 1:03d}",
                                "world_x": round(float(wx), 3),
                                "world_y": round(float(wy), 3),
                                "area_m2": round(cells * (self.res ** 2), 4),
                                "cells": cells,
                                "total_hits": int(hit_sum),
                                "severity": min(1.0, round(float(max(self.accumulator[r, c] for r, c in component) / 15.0), 2)),
                            })

        defects.sort(key=lambda d: -d["cells"])
        return defects

    def get_sparse_heatmap(self, threshold=1.0):
        """Returns non-zero cells formatted for lightweight client rendering."""
        ys, xs = np.where(self.accumulator >= threshold)
        points = []
        max_val = float(np.max(self.accumulator)) if len(xs) > 0 else 1.0
        for y, x in zip(ys, xs):
            val = float(self.accumulator[y, x])
            points.append([
                round(float(self.min_x + x * self.res), 3),
                round(float(self.min_y + y * self.res), 3),
                round(val / max_val, 2)
            ])
        return points

    def reset(self):
        self.accumulator.fill(0.0)
        self.frame_count = 0
        self.total_points = 0


# ============================================================
# ROBOT SIMULATION & MISSION ENGINE (DIGITAL TWIN)
# ============================================================
class RobotEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.terrain = dict(DEFAULT_TERRAIN)
        self.areas = list(DEFAULT_AREAS)
        self.is_isaac_sim_connected = False
        self.last_isaac_packet_time = 0.0

        # Robot kinematic state
        self.pos = list(DEFAULT_TERRAIN["spawn_pos"])
        self.z = 0.1
        self.yaw = DEFAULT_TERRAIN["spawn_yaw"]
        self.state = "IDLE"
        self.sub_phase = "IDLE"
        self.status_text = "System Ready - Select Area or Teleoperate"

        # Parameters
        self.forward_speed = FORWARD_SPEED
        self.turn_speed = TURN_SPEED
        self.step_distance = STEP_DISTANCE
        self.pause_duration = PAUSE_DURATION
        self.lane_shift = LANE_SHIFT
        self.obstacle_threshold = OBSTACLE_THRESHOLD
        self.emergency_stop_dist = EMERGENCY_STOP_DIST

        # Navigation & Scanning Mission
        self.target_destination: Optional[List[float]] = None
        self.current_mission_type: Optional[str] = None  # "MOVE_TO_AREA", "SCAN_AREA", "TELEOP"
        self.active_area: Optional[Dict[str, Any]] = None
        self.scan_pattern: str = "zigzag_horizontal" # rect: "zigzag_horizontal", "zigzag_vertical" | circle: "zigzag_outer_inner", "zigzag_inner_outer"
        self.scan_waypoints: List[List[float]] = []
        self.current_waypoint_idx: int = 0
        self.breadcrumbs: List[Dict[str, Any]] = []

        # Custom Area Dual Entry & Obstacle Rerouting
        self.primary_entry: Optional[List[float]] = None
        self.alt_entry: Optional[List[float]] = None
        self.primary_entry_name: str = "Primary Entry"
        self.alt_entry_name: str = "Alt Entry"
        self.active_entry_type: str = "PRIMARY"  # "PRIMARY" or "ALTERNATIVE"
        self.entry_status: str = "CLEAR"        # "CLEAR", "OBSTRUCTED_REROUTING", "ENTERED"
        self.entry_detour_waypoints: List[List[float]] = []
        self.entry_detour_idx: int = 0

        # Avoidance & Odometry
        self.lane_base_yaw = self.yaw
        self.sweep_direction_y = 1.0
        self.lateral_detour_ledger = 0.0
        self.detour_side = 1.0
        self.active_obstacle_category = "UNKNOWN"
        self.yield_start_time = 0.0

        # Sensors & Perception
        self.closest_obstacle_dist = float('inf')
        self.depth_left = float('inf')
        self.depth_center = float('inf')
        self.depth_right = float('inf')

        self.surface_defect_extent_m = 0.0
        self.surface_defect_fraction = 0.0
        self.surface_defect_severity = 0.0
        self.surface_streak = 0

        # VLM Feed
        self.vlm_status = "IDLE"
        self.vlm_category = "CLEAR"
        self.vlm_risk = 0
        self.vlm_command = "resume_path"
        self.vlm_reasoning = "Path is clear for scanning."
        self.stress_level = 0.0
        self.stress_band = "LOW"
        self.potential = 0.0
        self.latest_camera_frame = ""
        self.event_log: List[Dict[str, Any]] = []

        # Ground Mapper
        safe_min_x = self.terrain["min_x"] + self.terrain["safety_margin"]
        safe_max_x = self.terrain["max_x"] - self.terrain["safety_margin"]
        safe_min_y = self.terrain["min_y"] + self.terrain["safety_margin"]
        safe_max_y = self.terrain["max_y"] - self.terrain["safety_margin"]
        self.ground_mapper = GroundMapper(safe_min_x, safe_max_x, safe_min_y, safe_max_y)

        # Synthetic Environment Features (Defects and Obstacles)
        self._init_synthetic_environment()

        # Step Pause state tracking
        self.step_start_pos = list(self.pos)
        self.pause_start_time = 0.0
        self.last_update_time = time.time()
        self.running = True
        # Battery & Autonomous Charging System (🔋)
        self.charging_station = dict(DEFAULT_CHARGING_STATION)
        self.battery_pct = 96.4
        self.battery_voltage = 25.2
        self.battery_current = 2.4
        self.battery_temp = 31.8
        self.battery_health = 99.0
        self.battery_low_threshold = 25.0  # Auto-return to charging station threshold (%)
        self.battery_target_charge = 95.0  # Recharge completion threshold before resuming (%)
        self.saved_scan_mission: Optional[Dict[str, Any]] = None  # Interrupted mission state memory

        self.total_distance = 18.4
        self.total_steps = 142
        self.boot_time = time.time()
        self.roll = 0.4
        self.pitch = -0.8
        self.joint_temps = [36.2, 38.4, 37.1, 39.0, 36.8, 37.5]

        self.log_event("SYSTEM", "Hexapod Engine Initialized in Digital Twin Mode.")

    def _init_synthetic_environment(self):
        """Initializes simulated potholes, cracks, and obstacles in the terrain."""
        self.synthetic_defects = [
            {"x": -240.0, "y": -185.0, "radius": 0.50, "intensity": 1.3, "name": "West Gate Pothole Alpha"},
            {"x": -218.0, "y": -186.0, "radius": 0.55, "intensity": 1.4, "name": "East Apron Surface Fissure"},
            {"x": -229.0, "y": -189.0, "radius": 0.45, "intensity": 1.2, "name": "Courtyard Subsidence Pit"},
            {"x": -234.0, "y": -195.0, "radius": 0.60, "intensity": 1.5, "name": "South Driveway Erosion Pit"},
            {"x": -214.0, "y": -193.0, "radius": 0.40, "intensity": 1.1, "name": "East Perimeter Defect Beta"},
            {"x": -244.0, "y": -192.0, "radius": 0.52, "intensity": 1.3, "name": "West Perimeter Concrete Fissure"},
            {"x": -223.0, "y": -198.0, "radius": 0.48, "intensity": 1.0, "name": "South Driveway Aggregate Hole"},
            {"x": -226.0, "y": -181.0, "radius": 0.35, "intensity": 0.9, "name": "North Entrance Defect Gamma"},
        ]
        self.synthetic_obstacles = [
            {"x": -233.0, "y": -186.0, "radius": 0.40, "type": "STATIC_BAR", "name": "Barrier Beam"},
            {"x": -222.0, "y": -194.0, "radius": 0.45, "type": "BIOLOGICAL", "name": "Wildlife / Moving Obstacle"},
            {"x": -242.0, "y": -196.0, "radius": 0.35, "type": "STATIC_BAR", "name": "Warning Pylon"},
        ]

    def log_event(self, category: str, message: str):
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "category": category,
            "message": message
        }
        self.event_log.append(entry)
        if len(self.event_log) > 100:
            self.event_log.pop(0)

    # --------------------------------------------------------
    # Commands & Mission Dispatching
    # --------------------------------------------------------
    def dispatch_area_scan(self, area_id: str, pattern: str = "zigzag"):
        with self.lock:
            target_area = next((a for a in self.areas if a["id"] == area_id), None)
            if not target_area:
                return False, f"Area {area_id} not found"

            self.active_area = target_area
            self.scan_pattern = pattern
            self.primary_entry, self.alt_entry, self.primary_entry_name, self.alt_entry_name = self._compute_custom_area_entries(target_area)
            self.active_entry_type = "PRIMARY"
            self.entry_status = "CLEAR"
            self.target_destination = list(self.primary_entry) if self.primary_entry else [target_area["x"], target_area["y"]]
            self.current_mission_type = "MOVE_TO_AREA"
            self.state = "NAVIGATING_TO_AREA"
            self.status_text = f"Navigating to {target_area['name']}..."
            self.step_start_pos = list(self.pos)
            self.lane_base_yaw = self._get_angle_to(self.target_destination[0], self.target_destination[1])
            self.log_event("MISSION", f"Dispatched to {target_area['name']} ({target_area['x']}, {target_area['y']}) for {pattern} scan.")
            return True, f"Navigating to {target_area['name']}"

    def _compute_custom_area_entries(self, area: Dict[str, Any]):
        """Computes Primary and Alternative Entry points along the perimeter of the custom area."""
        shape = area.get("shape", "rectangle")
        cx, cy = area["x"], area["y"]
        cur_x, cur_y = self.pos[0], self.pos[1]

        if shape == "rectangle":
            min_x = area.get("min_x", cx - area.get("width", 4.0) / 2.0)
            max_x = area.get("max_x", cx + area.get("width", 4.0) / 2.0)
            min_y = area.get("min_y", cy - area.get("height", 3.0) / 2.0)
            max_y = area.get("max_y", cy + area.get("height", 3.0) / 2.0)

            # Candidate perimeter entry gates
            candidates = [
                {"name": "South Entry", "pos": [round((min_x + max_x) / 2.0, 2), round(min_y, 2)], "edge": "S"},
                {"name": "North Entry", "pos": [round((min_x + max_x) / 2.0, 2), round(max_y, 2)], "edge": "N"},
                {"name": "West Entry",  "pos": [round(min_x, 2), round((min_y + max_y) / 2.0, 2)], "edge": "W"},
                {"name": "East Entry",  "pos": [round(max_x, 2), round((min_y + max_y) / 2.0, 2)], "edge": "E"},
            ]
            candidates.sort(key=lambda c: math.hypot(c["pos"][0] - cur_x, c["pos"][1] - cur_y))
            primary = candidates[0]
            opposite_map = {"S": "N", "N": "S", "W": "E", "E": "W"}
            alt_matches = [c for c in candidates if c["edge"] == opposite_map.get(primary["edge"])]
            alt = alt_matches[0] if alt_matches else candidates[-1]

            return primary["pos"], alt["pos"], primary["name"], alt["name"]
        else:
            r = area.get("radius", 2.0)
            angle = math.atan2(cur_y - cy, cur_x - cx)
            px = round(cx + r * math.cos(angle), 2)
            py = round(cy + r * math.sin(angle), 2)
            ax = round(cx - r * math.cos(angle), 2)
            ay = round(cy - r * math.sin(angle), 2)
            return [px, py], [ax, ay], "Perimeter Entry A", "Perimeter Entry B (Opposite)"

    def _generate_alt_entry_detour(self) -> List[List[float]]:
        """Generates perimeter clearance waypoints around the custom area to reach the alternative entry without colliding."""
        if not self.active_area or not self.alt_entry:
            return [self.alt_entry] if self.alt_entry else []

        shape = self.active_area.get("shape", "rectangle")
        if shape != "rectangle":
            return [list(self.alt_entry)]

        min_x = self.active_area.get("min_x", self.active_area["x"] - 2.0)
        max_x = self.active_area.get("max_x", self.active_area["x"] + 2.0)
        min_y = self.active_area.get("min_y", self.active_area["y"] - 1.5)
        max_y = self.active_area.get("max_y", self.active_area["y"] + 1.5)

        margin = 1.4
        cur_x, cur_y = self.pos[0], self.pos[1]
        dest_x, dest_y = self.alt_entry[0], self.alt_entry[1]

        t_min_x = self.terrain["min_x"] + self.terrain["safety_margin"] + 0.1
        t_max_x = self.terrain["max_x"] - self.terrain["safety_margin"] - 0.1
        t_min_y = self.terrain["min_y"] + self.terrain["safety_margin"] + 0.1
        t_max_y = self.terrain["max_y"] - self.terrain["safety_margin"] - 0.1

        # Perimeter corner clearance points outside the bounds, bounded within reachable terrain
        corners = [
            [round(max(t_min_x, min(t_max_x, min_x - margin)), 2), round(max(t_min_y, min(t_max_y, min_y - margin)), 2)], # SW
            [round(max(t_min_x, min(t_max_x, min_x - margin)), 2), round(max(t_min_y, min(t_max_y, max_y + margin)), 2)], # NW
            [round(max(t_min_x, min(t_max_x, max_x + margin)), 2), round(max(t_min_y, min(t_max_y, max_y + margin)), 2)], # NE
            [round(max(t_min_x, min(t_max_x, max_x + margin)), 2), round(max(t_min_y, min(t_max_y, min_y - margin)), 2)], # SE
        ]

        closest_start_corner = min(corners, key=lambda c: math.hypot(c[0] - cur_x, c[1] - cur_y))
        closest_end_corner = min(corners, key=lambda c: math.hypot(c[0] - dest_x, c[1] - dest_y))

        detour = []
        if math.hypot(closest_start_corner[0] - cur_x, closest_start_corner[1] - cur_y) > 0.5:
            detour.append(closest_start_corner)
        if closest_end_corner != closest_start_corner:
            detour.append(closest_end_corner)
        detour.append(list(self.alt_entry))
        return detour

    def reroute_to_alternative_entry(self, reason: str = "Obstacle detected"):
        with self.lock:
            if not self.alt_entry:
                return False, "No alternative entry point available."

            self.active_entry_type = "ALTERNATIVE"
            self.entry_status = "OBSTRUCTED_REROUTING"
            self.entry_detour_waypoints = self._generate_alt_entry_detour()
            self.entry_detour_idx = 0

            self.target_destination = list(self.entry_detour_waypoints[0]) if self.entry_detour_waypoints else list(self.alt_entry)
            self.state = "NAVIGATING_TO_AREA"
            self.current_mission_type = "MOVE_TO_AREA"
            self.status_text = f"⚠️ Entry blocked! Rerouting to Alternative Entry at ({self.alt_entry[0]:.1f}, {self.alt_entry[1]:.1f})..."
            self.log_event("AVOID", f"{reason}! Disengaging primary route; executing perimeter flank to Alternative Entry ({self.alt_entry[0]:.2f}, {self.alt_entry[1]:.2f}).")
            return True, "Rerouted to alternative entry"

    def switch_entry_point(self, entry_type: str = "alt"):
        with self.lock:
            if not self.active_area:
                return False, "No active custom area to switch entries."
            if entry_type.lower() in ("alt", "alternative"):
                if not self.alt_entry:
                    return False, "No alternative entry defined."
                return self.reroute_to_alternative_entry("User manually commanded Alternative Entry")
            else:
                if not self.primary_entry:
                    return False, "No primary entry defined."
                self.active_entry_type = "PRIMARY"
                self.entry_status = "CLEAR"
                self.entry_detour_waypoints = []
                self.entry_detour_idx = 0
                self.target_destination = list(self.primary_entry)
                self.state = "NAVIGATING_TO_AREA"
                self.current_mission_type = "MOVE_TO_AREA"
                self.status_text = f"Switched to Primary Entry ({self.primary_entry[0]:.1f}, {self.primary_entry[1]:.1f})."
                self.log_event("MISSION", f"Switched target to Primary Entry ({self.primary_entry[0]:.2f}, {self.primary_entry[1]:.2f}).")
                return True, "Switched to primary entry"

    def simulate_entry_obstacle(self):
        with self.lock:
            if not self.primary_entry:
                if self.active_area:
                    self._compute_custom_area_entries(self.active_area)
                else:
                    return False, "Select a custom area first."
            px, py = self.primary_entry
            obs_x = round(px, 2)
            obs_y = round(py - 0.8, 2) if py <= (self.terrain["min_y"] + self.terrain["max_y"]) / 2 else round(py + 0.8, 2)
            obs = {
                "x": obs_x,
                "y": obs_y,
                "radius": 0.50,
                "type": "STATIC_BAR",
                "name": "Simulated Entry Obstacle Barrier"
            }
            self.synthetic_obstacles = [o for o in self.synthetic_obstacles if o.get("name") != obs["name"]]
            self.synthetic_obstacles.append(obs)
            self.log_event("SIM", f"Simulated obstacle barrier placed at Primary Entry ({obs_x}, {obs_y}).")

            if self.state == "NAVIGATING_TO_AREA" and self.active_entry_type == "PRIMARY":
                self.reroute_to_alternative_entry("Obstacle barrier placed at Primary Entry")
            return True, f"Obstacle placed at Primary Entry ({obs_x}, {obs_y})"

    def set_custom_target_and_scan(self, x: float, y: float, auto_scan: bool = True, pattern: str = "zigzag_horizontal", radius: float = 1.5, shape: str = "rectangle", width: float = 4.0, height: float = 3.0, min_x: float = None, max_x: float = None, min_y: float = None, max_y: float = None, entry_mode: str = "auto"):
        with self.lock:
            shape = "circle" if shape == "circle" else "rectangle"
            valid_patterns = {
                "rectangle": {"zigzag_horizontal", "zigzag_vertical"},
                "circle": {"zigzag_outer_inner", "zigzag_inner_outer"},
            }
            default_pattern = "zigzag_horizontal" if shape == "rectangle" else "zigzag_outer_inner"
            pattern = pattern if pattern in valid_patterns[shape] else default_pattern

            c_min_x = min_x if min_x is not None else x - width / 2.0
            c_max_x = max_x if max_x is not None else x + width / 2.0
            c_min_y = min_y if min_y is not None else y - height / 2.0
            c_max_y = max_y if max_y is not None else y + height / 2.0
            if shape == "rectangle":
                if (c_max_x - c_min_x) < 2.0:
                    c_min_x = x - 1.5
                    c_max_x = x + 1.5
                if (c_max_y - c_min_y) < 1.5:
                    c_min_y = y - 1.0
                    c_max_y = y + 1.0
                width = c_max_x - c_min_x
                height = c_max_y - c_min_y
            else:
                width = max(1.0, c_max_x - c_min_x)
                height = max(1.0, c_max_y - c_min_y)
            radius = max(0.5, radius)
            w = width
            h = height

            custom_area = {
                "id": "CUSTOM",
                "name": f"⚠️ Custom {shape.capitalize()} ({x:.1f}, {y:.1f})",
                "shape": shape,
                "x": x,
                "y": y,
                "radius": radius,
                "width": w,
                "height": h,
                "min_x": c_min_x,
                "max_x": c_max_x,
                "min_y": c_min_y,
                "max_y": c_max_y,
                "color": "#ec4899" if shape == "circle" else "#38bdf8",
                "description": "User click-and-drag designated scan area"
            }
            self.active_area = custom_area
            self.scan_pattern = pattern

            # Compute dual entry points (Primary and Alternative)
            p_pos, a_pos, p_name, a_name = self._compute_custom_area_entries(custom_area)
            self.primary_entry = p_pos
            self.alt_entry = a_pos
            self.primary_entry_name = p_name
            self.alt_entry_name = a_name
            custom_area["primary_entry"] = p_pos
            custom_area["alt_entry"] = a_pos
            custom_area["primary_entry_name"] = p_name
            custom_area["alt_entry_name"] = a_name

            if entry_mode.lower() in ("alt", "alternative"):
                self.active_entry_type = "ALTERNATIVE"
                entry_target = self.alt_entry
            else:
                self.active_entry_type = "PRIMARY"
                entry_target = self.primary_entry

            self.entry_status = "CLEAR"
            self.entry_detour_waypoints = []
            self.entry_detour_idx = 0

            # Boost forward speed for scanning
            is_whole_area = (w >= 20.0 or h >= 10.0)
            if is_whole_area:
                self.forward_speed = max(self.forward_speed, 1.5)
            else:
                self.forward_speed = max(self.forward_speed, 1.0)

            # Check if robot is already inside scan area
            is_inside = (c_min_x - 0.5 <= self.pos[0] <= c_max_x + 0.5) and (c_min_y - 0.5 <= self.pos[1] <= c_max_y + 0.5)

            # Seed dynamic synthetic defects inside custom area if none exist in bounds
            defects_in_bounds = [
                df for df in self.synthetic_defects
                if c_min_x <= df["x"] <= c_max_x and c_min_y <= df["y"] <= c_max_y
            ]
            if len(defects_in_bounds) < 2:
                candidates = [
                    {"x": round(x + 0.15 * w, 2), "y": round(y + 0.1 * h, 2), "radius": 0.40, "intensity": 1.2, "name": "Custom Area Pothole Alpha"},
                    {"x": round(x - 0.2 * w, 2), "y": round(y - 0.15 * h, 2), "radius": 0.55, "intensity": 1.4, "name": "Custom Asphalt Crack Beta"},
                    {"x": round(x + 0.05 * w, 2), "y": round(y - 0.2 * h, 2), "radius": 0.65, "intensity": 1.6, "name": "Custom Depression Pit Gamma"}
                ]
                for cd in candidates:
                    if c_min_x <= cd["x"] <= c_max_x and c_min_y <= cd["y"] <= c_max_y:
                        self.synthetic_defects.append(cd)

            self.step_start_pos = list(self.pos)
            if auto_scan and is_inside:
                # Start scanning immediately
                self.scan_waypoints = self._generate_scan_waypoints(custom_area, pattern)
                self.current_waypoint_idx = 0
                self.target_destination = list(self.scan_waypoints[0]) if self.scan_waypoints else [x, y]
                self.current_mission_type = "SCAN_AREA"
                self.state = "SCANNING_AREA"
                self.entry_status = "ENTERED"
                self.status_text = f"⚡ Zig-Zag Scanning {custom_area['name']}..."
                self.log_event("MISSION", f"Already inside area. Commenced {pattern} scan ({w:.1f}m x {h:.1f}m).")
            else:
                # Robot is outside: walk to designated entry point
                self.current_mission_type = "MOVE_TO_AREA" if auto_scan else "MOVE_TO_TARGET"
                self.state = "NAVIGATING_TO_AREA"
                self.target_destination = list(entry_target)
                self.status_text = f"Navigating to {self.active_entry_type.capitalize()} Entry ({entry_target[0]:.1f}, {entry_target[1]:.1f}) of {shape} area..."
                self.lane_base_yaw = self._get_angle_to(entry_target[0], entry_target[1])
                self.log_event("MISSION", f"Approaching {self.active_entry_type.capitalize()} Entry at ({entry_target[0]:.2f}, {entry_target[1]:.2f}). Alt Entry ready at ({self.alt_entry[0]:.2f}, {self.alt_entry[1]:.2f}).")
            return True, "Target set successfully"

    def teleop(self, action: str, val: float = 1.0):
        with self.lock:
            rad = math.radians(self.yaw)
            fwd_x, fwd_y = math.cos(rad), math.sin(rad)
            left_x, left_y = -math.sin(rad), math.cos(rad)

            if action == "forward_step":
                self.pos[0] += fwd_x * self.step_distance
                self.pos[1] += fwd_y * self.step_distance
                self.add_breadcrumb([0.0, 1.0, 0.0])
                self.state = "PAUSED"
                self.pause_start_time = time.time()
                self.log_event("TELEOP", f"Forward step ({self.step_distance}m)")

            elif action == "reverse_step":
                self.pos[0] -= fwd_x * self.step_distance
                self.pos[1] -= fwd_y * self.step_distance
                self.add_breadcrumb([1.0, 0.0, 0.0])
                self.state = "PAUSED"
                self.pause_start_time = time.time()
                self.log_event("TELEOP", f"Reverse step ({self.step_distance}m)")

            elif action == "turn_left":
                angle = 45.0 * val
                self.yaw = (self.yaw + angle) % 360.0
                self.log_event("TELEOP", f"Turn left {angle:.0f}° -> {self.yaw:.1f}°")

            elif action == "turn_right":
                angle = 45.0 * val
                self.yaw = (self.yaw - angle) % 360.0
                self.log_event("TELEOP", f"Turn right {angle:.0f}° -> {self.yaw:.1f}°")

            elif action == "lane_shift_left":
                self.pos[0] += left_x * self.lane_shift
                self.pos[1] += left_y * self.lane_shift
                self.add_breadcrumb([0.0, 0.5, 1.0])
                self.log_event("TELEOP", f"Lane shift left ({self.lane_shift}m)")

            elif action == "lane_shift_right":
                self.pos[0] -= left_x * self.lane_shift
                self.pos[1] -= left_y * self.lane_shift
                self.add_breadcrumb([0.0, 0.5, 1.0])
                self.log_event("TELEOP", f"Lane shift right ({self.lane_shift}m)")

            elif action == "scan_direction":
                heading_map = {"north": 90.0, "south": 270.0, "east": 0.0, "west": 180.0}
                direction = str(val).lower()
                target_heading = heading_map.get(direction, 0.0)
                self.yaw = target_heading
                self.state = "SCANNING_DIRECTION"
                self.status_text = f"Scanning {direction.upper()}..."
                self.log_event("SCAN", f"Initiated directional scan facing {direction.upper()} ({target_heading}°)")

            elif action == "panoramic_scan":
                self.state = "PANORAMIC_SCANNING"
                self.status_text = "Executing 360° Panoramic Scan..."
                self.log_event("SCAN", "360° Panoramic scan initiated.")

            self._clamp_to_bounds()
            return True, f"Action {action} performed"

    def control_command(self, cmd: str):
        with self.lock:
            if cmd == "pause":
                self.state = "USER_PAUSED"
                self.status_text = "Paused by user"
                self.log_event("CONTROL", "Robot paused.")
            elif cmd == "resume":
                self.state = "IDLE" if self.state == "USER_PAUSED" else self.state
                self.status_text = "Resumed"
                self.log_event("CONTROL", "Robot resumed.")
            elif cmd == "stop":
                self.state = "STOPPED"
                self.status_text = "Emergency Stop Triggered"
                self.log_event("CONTROL", "EMERGENCY STOP.")
            elif cmd == "reset":
                self.pos = list(self.terrain["spawn_pos"])
                self.yaw = self.terrain["spawn_yaw"]
                self.state = "IDLE"
                self.sub_phase = "IDLE"
                self.status_text = "Reset to spawn position"
                self.target_destination = None
                self.scan_waypoints = []
                self.breadcrumbs = []
                self.lateral_detour_ledger = 0.0
                self.ground_mapper.reset()
                self.log_event("CONTROL", "Robot and Ground Mapper reset.")
            elif cmd == "clear_map":
                self.ground_mapper.reset()
                self.breadcrumbs = []
                self.log_event("MAP", "Ground map accumulator cleared.")
            return True, f"Command {cmd} executed"

    def update_settings(self, settings: Dict[str, Any]):
        with self.lock:
            if "forward_speed" in settings:
                self.forward_speed = float(settings["forward_speed"])
            if "turn_speed" in settings:
                self.turn_speed = float(settings["turn_speed"])
            if "step_distance" in settings:
                self.step_distance = float(settings["step_distance"])
            if "pause_duration" in settings:
                self.pause_duration = float(settings["pause_duration"])
            if "lane_shift" in settings:
                self.lane_shift = float(settings["lane_shift"])
            if "obstacle_threshold" in settings:
                self.obstacle_threshold = float(settings["obstacle_threshold"])
            self.log_event("CONFIG", "Robot motion and perception settings updated.")
            return True, "Settings updated"

    def update_terrain(self, terrain_data: Dict[str, Any]):
        with self.lock:
            for k in ["min_x", "max_x", "min_y", "max_y", "safety_margin", "name"]:
                if k in terrain_data:
                    self.terrain[k] = terrain_data[k]
            if "areas" in terrain_data:
                self.areas = terrain_data["areas"]
            if "spawn_pos" in terrain_data:
                self.terrain["spawn_pos"] = terrain_data["spawn_pos"]

            safe_min_x = self.terrain["min_x"] + self.terrain["safety_margin"]
            safe_max_x = self.terrain["max_x"] - self.terrain["safety_margin"]
            safe_min_y = self.terrain["min_y"] + self.terrain["safety_margin"]
            safe_max_y = self.terrain["max_y"] - self.terrain["safety_margin"]
            self.ground_mapper.update_bounds(safe_min_x, safe_max_x, safe_min_y, safe_max_y)
            self.log_event("TERRAIN", f"Terrain updated: {self.terrain['name']} [{self.terrain['min_x']} to {self.terrain['max_x']}]")
            return True, "Terrain updated"

    # --------------------------------------------------------
    # Isaac Sim Live Bridge Data Ingestion
    # --------------------------------------------------------
    def handle_isaac_sim_telemetry(self, packet: Dict[str, Any]):
        with self.lock:
            self.is_isaac_sim_connected = True
            self.last_isaac_packet_time = time.time()
            if "pos" in packet:
                prev_pos = list(self.pos)
                self.pos = packet["pos"]
                dx = self.pos[0] - prev_pos[0]
                dy = self.pos[1] - prev_pos[1]
                if math.hypot(dx, dy) >= 0.15:
                    self.add_breadcrumb([0.0, 1.0, 0.0])
            if "yaw" in packet:
                self.yaw = packet["yaw"]
            if "state" in packet:
                self.state = packet["state"]
            if "sub_phase" in packet:
                self.sub_phase = packet["sub_phase"]
            if "obstacle_dist" in packet:
                self.closest_obstacle_dist = packet["obstacle_dist"]
            if "sectors" in packet:
                s = packet["sectors"]
                self.depth_left = s.get("left", float('inf'))
                self.depth_center = s.get("center", float('inf'))
                self.depth_right = s.get("right", float('inf'))
            if "surface" in packet:
                sf = packet["surface"]
                self.surface_defect_extent_m = sf.get("extent", 0.0)
                self.surface_defect_fraction = sf.get("fraction", 0.0)
                self.surface_defect_severity = sf.get("severity", 0.0)
            if "vlm" in packet:
                v = packet["vlm"]
                self.vlm_category = v.get("category", self.vlm_category)
                self.vlm_risk = v.get("risk", self.vlm_risk)
                self.vlm_command = v.get("command", self.vlm_command)
                self.vlm_reasoning = v.get("reasoning", self.vlm_reasoning)
            if "stress" in packet:
                self.stress_level = float(packet["stress"])
            if "stress_band" in packet:
                self.stress_band = str(packet["stress_band"])
            if "potential" in packet:
                self.potential = float(packet["potential"])
            if "frame" in packet and packet["frame"]:
                self.latest_camera_frame = packet["frame"]
            if "pts" in packet and len(packet["pts"]) > 0:
                xs = [p[0] for p in packet["pts"]]
                ys = [p[1] for p in packet["pts"]]
                self.ground_mapper.integrate(xs, ys)

    # --------------------------------------------------------
    # Autonomous Area Scan Path Generation
    # --------------------------------------------------------
    def _generate_scan_waypoints(self, area: Dict[str, Any], pattern: str = "zigzag_horizontal"):
        shape = area.get("shape", "circle")
        cx, cy = area["x"], area["y"]
        waypoints = []

        if shape == "rectangle":
            half_w = max(1.0, area.get("width", 4.0) / 2.0)
            half_h = max(1.0, area.get("height", 3.0) / 2.0)
            raw_min_x = area.get("min_x", cx - half_w)
            raw_max_x = area.get("max_x", cx + half_w)
            raw_min_y = area.get("min_y", cy - half_h)
            raw_max_y = area.get("max_y", cy + half_h)

            x_min = max(self.terrain["min_x"] + self.terrain["safety_margin"] + 0.05, min(raw_min_x, raw_max_x) + 0.08)
            x_max = min(self.terrain["max_x"] - self.terrain["safety_margin"] - 0.05, max(raw_min_x, raw_max_x) - 0.08)
            y_min = max(self.terrain["min_y"] + self.terrain["safety_margin"] + 0.05, min(raw_min_y, raw_max_y) + 0.08)
            y_max = min(self.terrain["max_y"] - self.terrain["safety_margin"] - 0.05, max(raw_min_y, raw_max_y) - 0.08)

            if x_min >= x_max:
                x_min, x_max = cx - 0.5, cx + 0.5
            if y_min >= y_max:
                y_min, y_max = cy - 0.5, cy + 0.5

            is_large = (x_max - x_min) >= 20.0 or (y_max - y_min) >= 10.0

            if pattern in ("zigzag_horizontal", "zigzag"):
                # Sweep row-by-row along Y, zigzag in X direction across the entire area
                lane_h = y_max - y_min
                step_size = max(1.2, min(1.8, lane_h / 10.0)) if is_large else min(max(0.3, self.lane_shift), max(0.4, lane_h / 5.0))
                y_steps = list(np.arange(y_min, y_max, step_size))
                if len(y_steps) == 0 or y_steps[-1] < (y_max - 0.1):
                    y_steps.append(y_max)

                cur_y = self.pos[1]
                cur_x = self.pos[0]
                if (y_min - 1.0 <= cur_y <= y_max + 1.0) and (x_min - 1.0 <= cur_x <= x_max + 1.0):
                    closest_idx = int(np.argmin([abs(y - cur_y) for y in y_steps]))
                    ordered_lane_indices = list(range(closest_idx, len(y_steps))) + list(range(closest_idx - 1, -1, -1))
                else:
                    ordered_lane_indices = list(range(len(y_steps)))

                last_x = None
                for order_i, lane_idx in enumerate(ordered_lane_indices):
                    y_val = round(float(y_steps[lane_idx]), 3)
                    if order_i == 0 and (x_min - 1.0 <= cur_x <= x_max + 1.0):
                        sweep_to_max = (cur_x < (x_min + x_max) / 2.0)
                        target_end = x_max if sweep_to_max else x_min
                        waypoints.append([round(float(target_end), 3), y_val])
                        last_x = target_end
                    else:
                        if last_x is None:
                            last_x = x_min
                        next_x = x_min if last_x > (x_min + x_max) / 2.0 else x_max
                        waypoints.append([round(float(last_x), 3), y_val])
                        waypoints.append([round(float(next_x), 3), y_val])
                        last_x = next_x

            elif pattern == "zigzag_vertical":
                # Sweep column-by-column along X, zigzag in Y direction across the entire area
                lane_w = x_max - x_min
                step_size = max(1.2, min(1.8, lane_w / 10.0)) if is_large else min(max(0.3, self.lane_shift), max(0.4, lane_w / 5.0))
                x_steps = list(np.arange(x_min, x_max, step_size))
                if len(x_steps) == 0 or x_steps[-1] < (x_max - 0.1):
                    x_steps.append(x_max)

                cur_x = self.pos[0]
                cur_y = self.pos[1]
                if (x_min - 1.0 <= cur_x <= x_max + 1.0) and (y_min - 1.0 <= cur_y <= y_max + 1.0):
                    closest_idx = int(np.argmin([abs(x - cur_x) for x in x_steps]))
                    ordered_lane_indices = list(range(closest_idx, len(x_steps))) + list(range(closest_idx - 1, -1, -1))
                else:
                    ordered_lane_indices = list(range(len(x_steps)))

                last_y = None
                for order_i, lane_idx in enumerate(ordered_lane_indices):
                    x_val = round(float(x_steps[lane_idx]), 3)
                    if order_i == 0 and (y_min - 1.0 <= cur_y <= y_max + 1.0):
                        sweep_to_max = (cur_y < (y_min + y_max) / 2.0)
                        target_end = y_max if sweep_to_max else y_min
                        waypoints.append([x_val, round(float(target_end), 3)])
                        last_y = target_end
                    else:
                        if last_y is None:
                            last_y = y_min
                        next_y = y_min if last_y > (y_min + y_max) / 2.0 else y_max
                        waypoints.append([x_val, round(float(last_y), 3)])
                        waypoints.append([x_val, round(float(next_y), 3)])
                        last_y = next_y
            else:
                waypoints = [[cx, cy]]
        else:
            # Circle scans follow one continuous Archimedean spiral. The radial
            # pitch is the lane spacing, so neighboring loops cover the area.
            safe_min_x = self.terrain["min_x"] + self.terrain["safety_margin"]
            safe_max_x = self.terrain["max_x"] - self.terrain["safety_margin"]
            safe_min_y = self.terrain["min_y"] + self.terrain["safety_margin"]
            safe_max_y = self.terrain["max_y"] - self.terrain["safety_margin"]
            r = max(0.2, area.get("radius", 1.5) - 0.35)
            r = min(r, cx - safe_min_x, safe_max_x - cx, cy - safe_min_y, safe_max_y - cy)
            r = max(0.2, r)

            pitch = max(0.2, self.lane_shift)
            sample_spacing = max(0.2, min(0.5, pitch * 0.5))
            theta = 0.0
            inward = pattern == "zigzag_outer_inner"
            max_theta = (r / pitch) * (2.0 * math.pi)

            while theta <= max_theta:
                radius = max(0.0, r - (pitch * theta / (2.0 * math.pi))) if inward else min(r, pitch * theta / (2.0 * math.pi))
                angle = theta - math.pi / 2.0
                waypoints.append([
                    cx + radius * math.cos(angle),
                    cy + radius * math.sin(angle),
                ])

                # Keep adjacent targets close enough for the robot to trace a
                # smooth curve without creating an excessive waypoint list.
                theta += min(0.1, sample_spacing / max(radius, sample_spacing))

            if not inward and (not waypoints or math.hypot(waypoints[-1][0] - cx, waypoints[-1][1] - cy) < r * 0.95):
                waypoints.append([cx + r, cy])
        return waypoints


    # --------------------------------------------------------
    # Physics & Simulation Update Loop (Digital Twin)
    # --------------------------------------------------------
    def update(self):
        now = time.time()
        dt = min(now - self.last_update_time, 0.1)
        self.last_update_time = now

        if self.is_isaac_sim_connected and (now - self.last_isaac_packet_time) < 1.0:
            return

        self.is_isaac_sim_connected = False

        if self.state in ["STOPPED", "USER_PAUSED"]:
            return

        with self.lock:
            self._update_synthetic_sensors()
            self._update_battery_and_power(dt)

            if self.state == "RETURNING_TO_CHARGER":
                self._handle_navigation_to_charger(dt)

            elif self.state == "CHARGING":
                self._handle_charging_dock(dt)

            elif self.state == "NAVIGATING_TO_AREA":
                self._handle_navigation_to_target(dt)

            elif self.state == "SCANNING_AREA":
                self._handle_scanning_area(dt)

            elif self.state == "SCANNING_DIRECTION":
                self._handle_scanning_direction(dt)

            elif self.state == "PANORAMIC_SCANNING":
                self._handle_panoramic_scanning(dt)

            elif self.state == "PAUSED":
                if (now - self.pause_start_time) >= self.pause_duration:
                    self.state = "NAVIGATING_TO_AREA" if self.current_mission_type == "MOVE_TO_AREA" else "IDLE"
                    self.step_start_pos = list(self.pos)

            elif self.state == "AVOID_INITIAL_TURN":
                self._handle_avoidance(dt)

            elif self.state == "EMERGENCY_BACKUP":
                self._handle_emergency_backup(dt)

            elif self.state == "YIELD_WAITING":
                if (now - self.yield_start_time) > 4.0:
                    self.vlm_command = "standard_bypass"
                    self.state = "AVOID_INITIAL_TURN"
                    self.log_event("VLM", "Biological entity did not clear in time. Switching to bypass.")

            self._clamp_to_bounds()

    def _update_synthetic_sensors(self):
        rad = math.radians(self.yaw)
        fwd_x, fwd_y = math.cos(rad), math.sin(rad)
        right_x, right_y = math.sin(rad), -math.cos(rad)

        min_c, min_l, min_r = float('inf'), float('inf'), float('inf')
        for obs in self.synthetic_obstacles:
            dx = obs["x"] - self.pos[0]
            dy = obs["y"] - self.pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 4.0:
                fwd_proj = dx * fwd_x + dy * fwd_y
                lat_proj = dx * right_x + dy * right_y
                if fwd_proj > 0.05:
                    angle = math.degrees(math.atan2(lat_proj, fwd_proj))
                    if abs(angle) < 45.0:
                        eff_dist = max(0.1, dist - obs["radius"])
                        if abs(angle) < 15.0:
                            min_c = min(min_c, eff_dist)
                        elif angle < 0:
                            min_l = min(min_l, eff_dist)
                        else:
                            min_r = min(min_r, eff_dist)

        self.depth_center = round(min_c, 2)
        self.depth_left = round(min_l, 2)
        self.depth_right = round(min_r, 2)
        self.closest_obstacle_dist = min(self.depth_center, self.depth_left, self.depth_right)

        detected_points_x, detected_points_y = [], []
        current_streak = 0
        max_sev = 0.0
        tot_ext = 0.0

        for df in self.synthetic_defects:
            dx = df["x"] - self.pos[0]
            dy = df["y"] - self.pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            fwd_proj = dx * fwd_x + dy * fwd_y
            lat_proj = dx * right_x + dy * right_y

            if 0.2 < fwd_proj < 1.3 and abs(lat_proj) < (fwd_proj * 0.75 + df["radius"]):
                num_pts = int(df["radius"] * 12)
                for _ in range(num_pts):
                    px = df["x"] + random.uniform(-df["radius"], df["radius"]) * 0.7
                    py = df["y"] + random.uniform(-df["radius"], df["radius"]) * 0.7
                    detected_points_x.append(px)
                    detected_points_y.append(py)

                current_streak += 1
                max_sev = max(max_sev, df["intensity"] * 0.6)
                tot_ext += df["radius"] * 2.0

        if detected_points_x:
            self.ground_mapper.integrate(detected_points_x, detected_points_y)
            self.surface_defect_fraction = min(1.0, len(detected_points_x) / 40.0)
            self.surface_defect_severity = min(1.0, max_sev)
            self.surface_defect_extent_m = round(tot_ext, 2)
            self.surface_streak = current_streak
        else:
            self.surface_defect_fraction = 0.0
            self.surface_defect_severity = 0.0
            self.surface_defect_extent_m = 0.0
            self.surface_streak = 0

    def _handle_navigation_to_target(self, dt: float):
        if not self.target_destination:
            self.state = "IDLE"
            return

        tx, ty = self.target_destination
        dx = tx - self.pos[0]
        dy = ty - self.pos[1]
        dist_to_target = math.sqrt(dx*dx + dy*dy)

        # Dynamic obstacle check: If heading towards Primary Entry and obstacle is getting in the way
        if self.active_entry_type == "PRIMARY" and self.alt_entry and not self.entry_detour_waypoints:
            is_blocked = False
            if self.closest_obstacle_dist <= self.obstacle_threshold:
                is_blocked = True
            for obs in self.synthetic_obstacles:
                obs_d = math.hypot(obs["x"] - self.pos[0], obs["y"] - self.pos[1])
                obs_to_target = math.hypot(obs["x"] - tx, obs["y"] - ty)
                if obs_d <= 2.2 and obs_to_target <= 2.5:
                    is_blocked = True
                    break

            if is_blocked:
                self.reroute_to_alternative_entry("Obstacle detected obstructing path to Primary Entry")
                return

        # If walking perimeter detour to Alternative Entry
        if self.entry_detour_waypoints:
            if dist_to_target <= max(TARGET_RADIUS, 0.60):
                if self.entry_detour_idx + 1 < len(self.entry_detour_waypoints):
                    self.entry_detour_idx += 1
                    self.target_destination = list(self.entry_detour_waypoints[self.entry_detour_idx])
                    self.log_event("NAV", f"Perimeter detour point {self.entry_detour_idx} reached; advancing toward Alternative Entry.")
                    return
                else:
                    self.entry_detour_waypoints = []
                    self.target_destination = list(self.alt_entry)
                    tx, ty = self.target_destination
                    dx = tx - self.pos[0]
                    dy = ty - self.pos[1]
                    dist_to_target = math.sqrt(dx*dx + dy*dy)

        # Arrived at Target / Entry Gate
        if dist_to_target <= TARGET_RADIUS:
            if self.current_mission_type == "MOVE_TO_AREA" and self.active_area:
                self.entry_status = "ENTERED"
                entry_label = self.active_entry_type.capitalize()
                self.log_event("MISSION", f"Successfully arrived via {entry_label} Entry at ({self.pos[0]:.2f}, {self.pos[1]:.2f})! Entering {self.active_area['name']} and commencing {self.scan_pattern} scan.")
                self.scan_waypoints = self._generate_scan_waypoints(self.active_area, self.scan_pattern)
                self.current_waypoint_idx = 0
                self.step_start_pos = list(self.pos)
                self.state = "SCANNING_AREA"
                self.status_text = f"Entered via {entry_label} Entry. Scanning {self.active_area['name']} ({self.scan_pattern})..."
            else:
                self.state = "IDLE"
                self.status_text = f"Arrived at Destination ({tx:.2f}, {ty:.2f})"
                self.log_event("MISSION", f"Arrived at target ({tx:.2f}, {ty:.2f}). Mission complete.")
            return

        target_heading = self._get_angle_to(tx, ty)
        heading_diff = self._angle_diff(target_heading, self.yaw)

        if abs(heading_diff) > 10.0:
            turn_step = math.copysign(min(abs(heading_diff), self.turn_speed * dt), heading_diff)
            self.yaw = (self.yaw + turn_step) % 360.0
        else:
            rad = math.radians(self.yaw)
            fwd_x, fwd_y = math.cos(rad), math.sin(rad)
            step_d = self.forward_speed * dt
            self.pos[0] += fwd_x * step_d
            self.pos[1] += fwd_y * step_d

            if math.sqrt((self.pos[0]-self.step_start_pos[0])**2 + (self.pos[1]-self.step_start_pos[1])**2) >= self.step_distance:
                self.step_start_pos = list(self.pos)
                self.add_breadcrumb([0.0, 1.0, 0.0])

    def _handle_scanning_area(self, dt: float):
        if self.current_waypoint_idx >= len(self.scan_waypoints):
            self.state = "IDLE"
            self.status_text = f"Scan of {self.active_area['name'] if self.active_area else 'Area'} Complete!"
            defects = self.ground_mapper.extract_defects()
            self.log_event("MISSION", f"Area scanning complete! Found {len(defects)} defects in region.")
            return

        wp = self.scan_waypoints[self.current_waypoint_idx]
        dx = wp[0] - self.pos[0]
        dy = wp[1] - self.pos[1]
        dist = math.sqrt(dx*dx + dy*dy)

        waypoint_tolerance = min(0.15, max(0.08, self.lane_shift * 0.3))
        if dist <= waypoint_tolerance:
            self.current_waypoint_idx += 1
            self.add_breadcrumb([0.2, 0.8, 1.0])
            return

        target_heading = self._get_angle_to(wp[0], wp[1])
        heading_diff = self._angle_diff(target_heading, self.yaw)

        if abs(heading_diff) > 4.0:
            turn_step = math.copysign(min(abs(heading_diff), self.turn_speed * dt), heading_diff)
            self.yaw = (self.yaw + turn_step) % 360.0
        else:
            step_d = min(self.forward_speed * dt, dist)
            self.pos[0] += (dx / dist) * step_d
            self.pos[1] += (dy / dist) * step_d
            self.yaw = target_heading

            if math.sqrt((self.pos[0]-self.step_start_pos[0])**2 + (self.pos[1]-self.step_start_pos[1])**2) >= self.step_distance:
                self.step_start_pos = list(self.pos)
                self.add_breadcrumb([1.0, 0.7, 0.0])

    def _handle_scanning_direction(self, dt: float):
        rad = math.radians(self.yaw)
        step_d = self.forward_speed * 0.6 * dt
        self.pos[0] += math.cos(rad) * step_d
        self.pos[1] += math.sin(rad) * step_d

        if math.sqrt((self.pos[0]-self.step_start_pos[0])**2 + (self.pos[1]-self.step_start_pos[1])**2) >= 1.5:
            self.state = "IDLE"
            self.status_text = "Directional Scan Completed."
            self.log_event("SCAN", "Directional scan sweep finished.")

    def _handle_panoramic_scanning(self, dt: float):
        turn_d = 45.0 * dt
        self.yaw = (self.yaw + turn_d) % 360.0
        if not hasattr(self, "_pano_accum"):
            self._pano_accum = 0.0
        self._pano_accum += turn_d
        if self._pano_accum >= 360.0:
            self._pano_accum = 0.0
            self.state = "IDLE"
            self.status_text = "Panoramic 360° Scan Complete."
            self.log_event("SCAN", "360° panoramic scan completed.")

    def _handle_avoidance(self, dt: float):
        rad = math.radians(self.yaw + 90.0 * self.detour_side)
        step_d = self.forward_speed * 0.5 * dt
        self.pos[0] += math.cos(rad) * step_d
        self.pos[1] += math.sin(rad) * step_d
        self.lateral_detour_ledger += step_d

        if self.lateral_detour_ledger >= 0.7:
            self.lateral_detour_ledger = 0.0
            self.state = "NAVIGATING_TO_AREA" if self.current_mission_type == "MOVE_TO_AREA" else "IDLE"
            self.status_text = "Flank complete, resuming course."
            self.log_event("AVOID", "Obstacle successfully bypassed.")

    def _handle_emergency_backup(self, dt: float):
        rad = math.radians(self.yaw)
        step_d = 0.4 * dt
        self.pos[0] -= math.cos(rad) * step_d
        self.pos[1] -= math.sin(rad) * step_d

        if self.closest_obstacle_dist >= 0.65:
            self.state = "AVOID_INITIAL_TURN"
            self.detour_side = 1.0 if self.depth_left > self.depth_right else -1.0
            self.status_text = "Clearance regained, executing bypass."

    # --------------------------------------------------------
    # Battery Simulation, Autonomous Charging & Mission Resume
    # --------------------------------------------------------
    def _update_battery_and_power(self, dt: float):
        if self.state == "CHARGING":
            # Fast charge rate when docked at charging station (~18s for full recharge cycle)
            charge_rate = 4.0  # % per second
            self.battery_pct = min(100.0, self.battery_pct + charge_rate * dt)
            self.battery_voltage = round(24.0 + (self.battery_pct / 100.0) * 2.2, 2)
            self.battery_current = -12.5  # Negative current = charging into pack
            self.battery_temp = max(28.0, self.battery_temp - 0.15 * dt)
            return

        # Power consumption depending on robot activity
        if self.state in ("SCANNING_AREA", "SCANNING_DIRECTION", "PANORAMIC_SCANNING"):
            drain_rate = 0.35  # % per second (intensive zigzag motion + camera + compute load)
            self.battery_current = 6.8
        elif self.state in ("NAVIGATING_TO_AREA", "RETURNING_TO_CHARGER", "AVOID_INITIAL_TURN", "EMERGENCY_BACKUP"):
            drain_rate = 0.20  # % per second (continuous walking)
            self.battery_current = 4.5
        elif self.state == "TELEOP":
            drain_rate = 0.25  # % per second (manual walk)
            self.battery_current = 5.2
        else:  # IDLE, USER_PAUSED, PAUSED, STOPPED
            drain_rate = 0.02  # % per second (idle baseline draw)
            self.battery_current = 1.2

        self.battery_pct = max(0.0, self.battery_pct - drain_rate * dt)
        self.battery_voltage = round(21.0 + (self.battery_pct / 100.0) * 4.2, 2)
        if self.battery_current > 3.0:
            self.battery_temp = min(42.0, self.battery_temp + 0.04 * dt)
        else:
            self.battery_temp = max(30.0, self.battery_temp - 0.03 * dt)

        # Autonomous Low Battery Check & Mission Suspension
        if self.battery_pct <= self.battery_low_threshold and self.state not in ("RETURNING_TO_CHARGER", "CHARGING", "STOPPED"):
            self.return_to_charger_for_low_battery()

    def return_to_charger_for_low_battery(self, reason: str = "Low battery threshold reached"):
        with self.lock:
            # Save active scan mission state so robot can resume after recharging
            if self.state == "SCANNING_AREA" and self.active_area:
                self.saved_scan_mission = {
                    "active_area": dict(self.active_area),
                    "scan_pattern": self.scan_pattern,
                    "scan_waypoints": [list(w) for w in self.scan_waypoints],
                    "current_waypoint_idx": self.current_waypoint_idx,
                    "target_destination": list(self.target_destination) if self.target_destination else None,
                    "entry_status": self.entry_status,
                    "active_entry_type": self.active_entry_type
                }
                wp_curr = self.current_waypoint_idx + 1
                wp_total = len(self.scan_waypoints)
                self.log_event("POWER", f"🔋 Low Battery ({self.battery_pct:.1f}% <= {self.battery_low_threshold:.1f}%)! Suspending scan of {self.active_area['name']} at waypoint {wp_curr}/{wp_total}. Navigating to Charging Station 🔋.")
            elif self.state == "NAVIGATING_TO_AREA" and self.active_area:
                self.saved_scan_mission = {
                    "active_area": dict(self.active_area),
                    "scan_pattern": self.scan_pattern,
                    "scan_waypoints": [list(w) for w in self.scan_waypoints] if self.scan_waypoints else [],
                    "current_waypoint_idx": self.current_waypoint_idx,
                    "target_destination": list(self.target_destination) if self.target_destination else None,
                    "entry_status": self.entry_status,
                    "active_entry_type": self.active_entry_type
                }
                self.log_event("POWER", f"🔋 Low Battery ({self.battery_pct:.1f}%)! Interrupted transit to {self.active_area['name']}. Navigating to Charging Station 🔋.")
            else:
                self.saved_scan_mission = None
                self.log_event("POWER", f"🔋 Low Battery ({self.battery_pct:.1f}%)! Navigating to Charging Station 🔋.")

            cx = self.charging_station["x"]
            cy = self.charging_station["y"]
            self.target_destination = [cx, cy]
            self.current_mission_type = "RETURN_TO_CHARGER"
            self.state = "RETURNING_TO_CHARGER"
            self.status_text = f"🔋 Low Battery ({self.battery_pct:.1f}%)! Returning to Charging Station at ({cx:.1f}, {cy:.1f})..."
            return True, "Returning to charging station"

    def _handle_navigation_to_charger(self, dt: float):
        cx = self.charging_station["x"]
        cy = self.charging_station["y"]
        dx = cx - self.pos[0]
        dy = cy - self.pos[1]
        dist_to_station = math.sqrt(dx*dx + dy*dy)

        # Arrived at charging station dock
        if dist_to_station <= 0.45:
            self.state = "CHARGING"
            self.current_mission_type = "CHARGING"
            self.target_destination = None
            self.status_text = f"⚡ Docked at 🔋 Charging Station. Fast charging in progress ({self.battery_pct:.1f}%)..."
            self.log_event("POWER", f"⚡ Successfully docked at Charging Station 🔋 ({cx:.1f}, {cy:.1f}). Commencing high-rate recharge.")
            return

        # Simple obstacle avoidance while navigating to charger
        if self.closest_obstacle_dist <= self.obstacle_threshold:
            turn_deg = 35.0 if self.depth_left > self.depth_right else -35.0
            self.yaw = (self.yaw + turn_deg) % 360.0
            rad = math.radians(self.yaw)
            self.pos[0] += math.cos(rad) * self.forward_speed * dt * 0.5
            self.pos[1] += math.sin(rad) * self.forward_speed * dt * 0.5
            return

        target_heading = self._get_angle_to(cx, cy)
        heading_diff = self._angle_diff(target_heading, self.yaw)

        if abs(heading_diff) > 10.0:
            turn_step = math.copysign(min(abs(heading_diff), self.turn_speed * dt), heading_diff)
            self.yaw = (self.yaw + turn_step) % 360.0
        else:
            rad = math.radians(self.yaw)
            fwd_x, fwd_y = math.cos(rad), math.sin(rad)
            step_d = self.forward_speed * dt
            self.pos[0] += fwd_x * step_d
            self.pos[1] += fwd_y * step_d

            if math.sqrt((self.pos[0]-self.step_start_pos[0])**2 + (self.pos[1]-self.step_start_pos[1])**2) >= self.step_distance:
                self.step_start_pos = list(self.pos)
                self.add_breadcrumb([0.0, 0.8, 1.0])

    def _handle_charging_dock(self, dt: float):
        if self.battery_pct >= self.battery_target_charge:
            self.battery_pct = min(100.0, self.battery_pct)
            self.log_event("POWER", f"✅ Battery recharge complete ({self.battery_pct:.1f}%). Exiting dock.")

            if self.saved_scan_mission:
                # Resume interrupted mission!
                saved = self.saved_scan_mission
                self.active_area = saved["active_area"]
                self.scan_pattern = saved["scan_pattern"]
                self.scan_waypoints = saved["scan_waypoints"]
                self.current_waypoint_idx = saved["current_waypoint_idx"]
                self.entry_status = saved.get("entry_status", "ENTERED")
                self.active_entry_type = saved.get("active_entry_type", "PRIMARY")
                self.saved_scan_mission = None

                if self.scan_waypoints and self.current_waypoint_idx < len(self.scan_waypoints):
                    self.target_destination = list(self.scan_waypoints[self.current_waypoint_idx])
                    self.current_mission_type = "SCAN_AREA"
                    self.state = "SCANNING_AREA"
                    wp_curr = self.current_waypoint_idx + 1
                    wp_tot = len(self.scan_waypoints)
                    self.status_text = f"🔋 Recharged ({self.battery_pct:.1f}%)! Resuming {self.active_area['name']} scan at waypoint {wp_curr}/{wp_tot}..."
                    self.log_event("MISSION", f"Resumed interrupted scan of {self.active_area['name']} from waypoint {wp_curr}/{wp_tot}.")
                elif self.active_area:
                    # Transit was interrupted before entering area: resume navigation to entry gate!
                    if self.active_entry_type == "PRIMARY" and self.primary_entry:
                        entry_dest = self.primary_entry
                    elif self.alt_entry:
                        entry_dest = self.alt_entry
                    elif saved.get("target_destination"):
                        entry_dest = saved["target_destination"]
                    else:
                        entry_dest = [self.active_area["x"], self.active_area["y"]]

                    self.target_destination = list(entry_dest)
                    self.current_mission_type = "MOVE_TO_AREA"
                    self.state = "NAVIGATING_TO_AREA"
                    self.step_start_pos = list(self.pos)
                    self.status_text = f"🔋 Recharged ({self.battery_pct:.1f}%)! Resuming transit to {self.active_area['name']}..."
                    self.log_event("MISSION", f"Resumed navigation to {self.active_area['name']} after battery recharge.")
                else:
                    self.state = "IDLE"
                    self.current_mission_type = None
                    self.target_destination = None
                    self.status_text = f"✅ Recharged ({self.battery_pct:.1f}%). Mission already completed."
            else:
                self.state = "IDLE"
                self.current_mission_type = None
                self.target_destination = None
                self.status_text = f"✅ Recharged ({self.battery_pct:.1f}%). System Ready."
        else:
            self.status_text = f"⚡ Fast-Charging: {self.battery_pct:.1f}% / {self.battery_target_charge:.1f}% (Docked at 🔋 Station)..."

    def set_battery_percentage(self, pct: float):
        with self.lock:
            self.battery_pct = max(0.0, min(100.0, float(pct)))
            self.battery_voltage = round(21.0 + (self.battery_pct / 100.0) * 4.2, 2)
            self.log_event("POWER", f"Battery level set to {self.battery_pct:.1f}%.")
            if self.battery_pct <= self.battery_low_threshold and self.state not in ("RETURNING_TO_CHARGER", "CHARGING", "STOPPED"):
                self.return_to_charger_for_low_battery("Low battery test override")
            return True, f"Battery set to {self.battery_pct:.1f}%"

    def set_battery_threshold(self, threshold: float):
        with self.lock:
            self.battery_low_threshold = max(5.0, min(50.0, float(threshold)))
            self.log_event("CONFIG", f"Low battery auto-return threshold set to {self.battery_low_threshold:.1f}%.")
            return True, f"Low battery threshold set to {self.battery_low_threshold:.1f}%"

    def _clamp_to_bounds(self):
        m = self.terrain["safety_margin"]
        self.pos[0] = max(self.terrain["min_x"] + m, min(self.terrain["max_x"] - m, self.pos[0]))
        self.pos[1] = max(self.terrain["min_y"] + m, min(self.terrain["max_y"] - m, self.pos[1]))

        # Enforce strict boundary clamping when an active scan area exists and robot is scanning or inside the area
        if self.active_area and self.state not in ("RETURNING_TO_CHARGER", "CHARGING"):
            area = self.active_area
            shape = area.get("shape", "circle")
            
            enforce_area_clamp = False
            scan_active_states = [
                "SCANNING_AREA", "AVOID_INITIAL_TURN", "EMERGENCY_BACKUP", 
                "YIELD_WAITING", "SCANNING_DIRECTION", "PANORAMIC_SCANNING", "PAUSED"
            ]
            if self.state in scan_active_states:
                enforce_area_clamp = True
            elif self.state == "NAVIGATING_TO_AREA":
                # Check if robot has reached inside the area bounds
                if shape == "rectangle":
                    half_w = area.get("width", 4.0) / 2.0
                    half_h = area.get("height", 3.0) / 2.0
                    min_x = area.get("min_x", area["x"] - half_w)
                    max_x = area.get("max_x", area["x"] + half_w)
                    min_y = area.get("min_y", area["y"] - half_h)
                    max_y = area.get("max_y", area["y"] + half_h)
                    if min_x <= self.pos[0] <= max_x and min_y <= self.pos[1] <= max_y:
                        enforce_area_clamp = True
                else:
                    cx, cy, r = area["x"], area["y"], area.get("radius", 1.5)
                    dx = self.pos[0] - cx
                    dy = self.pos[1] - cy
                    if math.sqrt(dx * dx + dy * dy) <= r:
                        enforce_area_clamp = True

            if enforce_area_clamp:
                margin = 0.05  # Leeway buffer so perimeter waypoints are easily reached
                if shape == "rectangle":
                    half_w = area.get("width", 4.0) / 2.0
                    half_h = area.get("height", 3.0) / 2.0
                    min_x = area.get("min_x", area["x"] - half_w) - margin
                    max_x = area.get("max_x", area["x"] + half_w) + margin
                    min_y = area.get("min_y", area["y"] - half_h) - margin
                    max_y = area.get("max_y", area["y"] + half_h) + margin
                    if min_x < max_x:
                        self.pos[0] = max(min_x, min(max_x, self.pos[0]))
                    if min_y < max_y:
                        self.pos[1] = max(min_y, min(max_y, self.pos[1]))
                else:
                    cx, cy, r = area["x"], area["y"], area.get("radius", 1.5)
                    eff_r = max(0.1, r - margin)
                    dx = self.pos[0] - cx
                    dy = self.pos[1] - cy
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist > eff_r and dist > 0.001:
                        angle = math.atan2(dy, dx)
                        self.pos[0] = cx + eff_r * math.cos(angle)
                        self.pos[1] = cy + eff_r * math.sin(angle)

    def add_breadcrumb(self, color):
        self.breadcrumbs.append({
            "x": round(self.pos[0], 3),
            "y": round(self.pos[1], 3),
            "color": color
        })
        if len(self.breadcrumbs) > 400:
            self.breadcrumbs.pop(0)

    def _get_angle_to(self, tx: float, ty: float) -> float:
        dx = tx - self.pos[0]
        dy = ty - self.pos[1]
        return math.degrees(math.atan2(dy, dx)) % 360.0

    def _angle_diff(self, target: float, current: float) -> float:
        diff = (target - current) % 360.0
        return diff - 360.0 if diff > 180.0 else diff

    def get_state_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            defects = self.ground_mapper.extract_defects()
            return {
                "timestamp": time.time(),
                "is_isaac_sim": self.is_isaac_sim_connected,
                "robot": {
                    "x": round(self.pos[0], 3),
                    "y": round(self.pos[1], 3),
                    "z": round(self.z, 3),
                    "yaw": round(self.yaw, 1),
                    "state": self.state,
                    "sub_phase": self.sub_phase,
                    "status_text": self.status_text,
                    "speed": self.forward_speed,
                    "turn_speed": self.turn_speed,
                    "step_distance": self.step_distance,
                    "pause_duration": self.pause_duration,
                    "lane_shift": self.lane_shift,
                },
                "mission": {
                    "active_area": self.active_area,
                    "target_destination": self.target_destination,
                    "scan_pattern": self.scan_pattern,
                    "current_waypoint": self.current_waypoint_idx,
                    "total_waypoints": len(self.scan_waypoints),
                    "mission_type": self.current_mission_type,
                    "primary_entry": self.primary_entry,
                    "alt_entry": self.alt_entry,
                    "primary_entry_name": self.primary_entry_name,
                    "alt_entry_name": self.alt_entry_name,
                    "active_entry_type": self.active_entry_type,
                    "entry_status": self.entry_status,
                    "detour_waypoints": self.entry_detour_waypoints,
                    "charging_station": self.charging_station,
                    "is_docked": self.state == "CHARGING",
                    "has_interrupted_mission": self.saved_scan_mission is not None,
                },
                "sensors": {
                    "closest_dist": self.closest_obstacle_dist if self.closest_obstacle_dist != float('inf') else 99.0,
                    "left_dist": self.depth_left if self.depth_left != float('inf') else 99.0,
                    "center_dist": self.depth_center if self.depth_center != float('inf') else 99.0,
                    "right_dist": self.depth_right if self.depth_right != float('inf') else 99.0,
                    "surface_extent": self.surface_defect_extent_m,
                    "surface_fraction": self.surface_defect_fraction,
                    "surface_severity": self.surface_defect_severity,
                    "surface_streak": self.surface_streak,
                },
                "vlm": {
                    "status": self.vlm_status,
                    "category": self.vlm_category,
                    "risk_score": self.vlm_risk,
                    "command": self.vlm_command,
                    "reasoning": self.vlm_reasoning
                },
                "stress": {
                    "level": getattr(self, "stress_level", 0.0),
                    "band": getattr(self, "stress_band", "LOW"),
                    "potential": getattr(self, "potential", 0.0)
                },
                "camera_frame": getattr(self, "latest_camera_frame", ""),
                "defects": defects,
                "defect_count": len(defects),
                "map_stats": {
                    "frames_integrated": self.ground_mapper.frame_count,
                    "total_points": self.ground_mapper.total_points,
                },
                "heatmap_points": self.ground_mapper.get_sparse_heatmap(),
                "charging_station": self.charging_station,
                "battery": {
                    "percentage": round(self.battery_pct, 1),
                    "voltage": round(self.battery_voltage, 2),
                    "current": round(self.battery_current, 2),
                    "temperature": round(self.battery_temp, 1),
                    "health": round(self.battery_health, 1),
                    "runtime_min": int(max(10, self.battery_pct * 1.3)),
                    "is_charging": self.state == "CHARGING",
                    "is_returning_to_charger": self.state == "RETURNING_TO_CHARGER",
                    "low_threshold": self.battery_low_threshold,
                    "target_charge": self.battery_target_charge,
                    "has_interrupted_mission": self.saved_scan_mission is not None,
                    "charging_station": self.charging_station
                },
                "diagnostics": {
                    "uptime_sec": int(time.time() - self.boot_time),
                    "total_distance_m": round(self.total_distance, 2),
                    "total_steps": self.total_steps,
                    "roll": round(self.roll, 1),
                    "pitch": round(self.pitch, 1),
                    "joint_temperatures": self.joint_temps,
                    "cpu_percent": 24.5,
                    "memory_percent": 38.2,
                    "link_latency_ms": 12 if self.is_isaac_sim_connected else 6
                },
                "breadcrumbs": self.breadcrumbs[-150:],
                "terrain": self.terrain,
                "areas": self.areas,
                "obstacles": self.synthetic_obstacles,
                "event_log": self.event_log[-20:]
            }


# Global Robot Engine instance
robot_engine = RobotEngine()


# ============================================================
# TORNADO WEB & WEBSOCKET HANDLERS
# ============================================================
connected_clients: List[tornado.websocket.WebSocketHandler] = []


class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        source = self.get_query_argument("source", "browser")
        self.is_isaac_sim_client = source == "isaac_sim"
        connected_clients.append(self)
        if source == "isaac_sim":
            robot_engine.is_isaac_sim_connected = True
            robot_engine.log_event("BRIDGE", "Isaac Sim Omniverse Client Connected via WebSocket.")
        _safe_write_message(self, json.dumps(robot_engine.get_state_snapshot()))

    def on_message(self, message):
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "dispatch_area":
                area_id = data.get("area_id", "A")
                pattern = data.get("pattern", "zigzag")
                robot_engine.dispatch_area_scan(area_id, pattern)

            elif msg_type == "set_custom_target":
                x = float(data.get("x", 0.0))
                y = float(data.get("y", 0.0))
                auto_scan = bool(data.get("auto_scan", True))
                pattern = data.get("pattern", "zigzag")
                radius = float(data.get("radius", 1.5))
                shape = data.get("shape", "rectangle")
                width = float(data.get("width", 4.0))
                height = float(data.get("height", 3.0))
                min_x = float(data["min_x"]) if "min_x" in data and data["min_x"] is not None else None
                max_x = float(data["max_x"]) if "max_x" in data and data["max_x"] is not None else None
                min_y = float(data["min_y"]) if "min_y" in data and data["min_y"] is not None else None
                max_y = float(data["max_y"]) if "max_y" in data and data["max_y"] is not None else None
                entry_mode = str(data.get("entry_mode", "auto"))
                robot_engine.set_custom_target_and_scan(
                    x=x, y=y, auto_scan=auto_scan, pattern=pattern, radius=radius,
                    shape=shape, width=width, height=height,
                    min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
                    entry_mode=entry_mode
                )

            elif msg_type == "switch_entry":
                entry_type = data.get("entry_type", "alt")
                robot_engine.switch_entry_point(entry_type)

            elif msg_type == "simulate_entry_obstacle":
                robot_engine.simulate_entry_obstacle()

            elif msg_type in ("dock_charger", "return_to_charger"):
                robot_engine.return_to_charger_for_low_battery("User commanded dock at charging station")

            elif msg_type == "set_battery":
                robot_engine.set_battery_percentage(float(data.get("percentage", 20.0)))

            elif msg_type == "set_battery_threshold":
                robot_engine.set_battery_threshold(float(data.get("threshold", 25.0)))

            elif msg_type == "teleop":
                action = data.get("action", "")
                val = float(data.get("val", 1.0))
                robot_engine.teleop(action, val)

            elif msg_type == "control":
                cmd = data.get("command", "")
                robot_engine.control_command(cmd)

            elif msg_type == "update_settings":
                settings = data.get("settings", {})
                robot_engine.update_settings(settings)

            elif msg_type == "update_terrain":
                terrain = data.get("terrain", {})
                robot_engine.update_terrain(terrain)

            elif msg_type == "isaac_sim_telemetry":
                robot_engine.handle_isaac_sim_telemetry(data.get("telemetry", {}))

            if not self.is_isaac_sim_client and msg_type in {
                "dispatch_area", "set_custom_target", "teleop", "control", "update_settings", "update_terrain"
            }:
                forward_command_to_isaac_sim(data)

            broadcast_state()

        except Exception as e:
            print(f"[WS Error]: {e}")

    def on_close(self):
        if self in connected_clients:
            connected_clients.remove(self)
        if getattr(self, "is_isaac_sim_client", False):
            robot_engine.is_isaac_sim_connected = False
            robot_engine.log_event("BRIDGE", "Isaac Sim Omniverse Client Disconnected.")


def _safe_write_message(client, msg):
    """Safely send a WebSocket message, suppressing unhandled disconnect warnings on page reload."""
    if client.ws_connection is None:
        if client in connected_clients:
            connected_clients.remove(client)
        return
    try:
        future = client.write_message(msg)
        if future is not None:
            def _on_done(f):
                try:
                    f.result()
                except (tornado.websocket.WebSocketClosedError, tornado.iostream.StreamClosedError):
                    if client in connected_clients:
                        connected_clients.remove(client)
                except Exception:
                    pass
            future.add_done_callback(_on_done)
    except (tornado.websocket.WebSocketClosedError, tornado.iostream.StreamClosedError):
        if client in connected_clients:
            connected_clients.remove(client)
    except Exception:
        pass


def forward_command_to_isaac_sim(command: Dict[str, Any]):
    message = json.dumps(command)
    for client in connected_clients[:]:
        if getattr(client, "is_isaac_sim_client", False):
            _safe_write_message(client, message)


def broadcast_state():
    if not connected_clients:
        return
    snapshot = json.dumps(robot_engine.get_state_snapshot())
    for client in connected_clients[:]:
        _safe_write_message(client, snapshot)


class StatusApiHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(robot_engine.get_state_snapshot()))


class CommandApiHandler(tornado.web.RequestHandler):
    def post(self):
        try:
            body = json.loads(self.request.body)
            action = body.get("action", "") or body.get("type", "")
            if action == "dispatch_area":
                ok, msg = robot_engine.dispatch_area_scan(body.get("area_id", "A"), body.get("pattern", "zigzag"))
            elif action in ("set_target", "set_custom_target"):
                ok, msg = robot_engine.set_custom_target_and_scan(
                    x=float(body.get("x", 0.0)),
                    y=float(body.get("y", 0.0)),
                    auto_scan=bool(body.get("auto_scan", True)),
                    pattern=body.get("pattern", "zigzag_horizontal"),
                    radius=float(body.get("radius", 1.5)),
                    shape=body.get("shape", "rectangle"),
                    width=float(body.get("width", 4.0)),
                    height=float(body.get("height", 3.0)),
                    min_x=float(body["min_x"]) if "min_x" in body and body["min_x"] is not None else None,
                    max_x=float(body["max_x"]) if "max_x" in body and body["max_x"] is not None else None,
                    min_y=float(body["min_y"]) if "min_y" in body and body["min_y"] is not None else None,
                    max_y=float(body["max_y"]) if "max_y" in body and body["max_y"] is not None else None,
                    entry_mode=str(body.get("entry_mode", "auto"))
                )
            elif action == "switch_entry":
                ok, msg = robot_engine.switch_entry_point(body.get("entry_type", "alt"))
            elif action == "simulate_entry_obstacle":
                ok, msg = robot_engine.simulate_entry_obstacle()
            elif action in ("dock_charger", "return_to_charger"):
                ok, msg = robot_engine.return_to_charger_for_low_battery("API commanded return to charging dock")
            elif action == "set_battery":
                ok, msg = robot_engine.set_battery_percentage(float(body.get("percentage", 20.0)))
            elif action == "set_battery_threshold":
                ok, msg = robot_engine.set_battery_threshold(float(body.get("threshold", 25.0)))
            elif action == "teleop":
                ok, msg = robot_engine.teleop(body.get("teleop_action", "") or body.get("action", ""), float(body.get("val", 1.0)))
            elif action == "control":
                ok, msg = robot_engine.control_command(body.get("command", ""))
            elif action == "update_settings":
                settings = body.get("settings", body)
                ok, msg = robot_engine.update_settings(settings)
            elif action == "update_terrain":
                terrain = body.get("terrain", body)
                ok, msg = robot_engine.update_terrain(terrain)
            else:
                ok, msg = False, f"Unknown action: {action}"
            self.write({"status": "ok" if ok else "error", "message": msg})
        except Exception as e:
            self.set_status(400)
            self.write({"status": "error", "error": str(e)})


class DefectsExportJsonHandler(tornado.web.RequestHandler):
    def get(self):
        defects = robot_engine.ground_mapper.extract_defects()
        export_data = {
            "generated_at": time.time(),
            "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "terrain": robot_engine.terrain["name"],
            "frames_integrated": robot_engine.ground_mapper.frame_count,
            "total_points": robot_engine.ground_mapper.total_points,
            "defects_detected": len(defects),
            "defects": defects,
        }
        self.set_header("Content-Type", "application/json")
        self.set_header("Content-Disposition", 'attachment; filename="phantomx_defects.json"')
        self.write(json.dumps(export_data, indent=2))


class DefectMapPngHandler(tornado.web.RequestHandler):
    def get(self):
        if not _has_pil:
            self.set_status(500)
            self.write("PIL not installed")
            return

        mapper = robot_engine.ground_mapper
        acc = mapper.accumulator.copy()
        h, w = acc.shape
        img = PILImage.new("RGB", (max(100, w * 4), max(100, h * 4)), color=(20, 24, 33))
        draw = PILImageDraw.Draw(img)

        max_val = np.max(acc) if np.max(acc) > 0 else 1.0
        scale_x = img.width / float(w)
        scale_y = img.height / float(h)

        ys, xs = np.where(acc > 0)
        for y, x in zip(ys, xs):
            intensity = min(1.0, float(acc[y, x]) / max_val)
            r = int(255 * intensity)
            g = int(120 * (1.0 - intensity))
            b = int(40 * (1.0 - intensity))
            x0 = int(x * scale_x)
            y0 = int((h - 1 - y) * scale_y)
            draw.rectangle([x0, y0, x0 + int(scale_x) + 1, y0 + int(scale_y) + 1], fill=(r, g, b))

        defects = mapper.extract_defects()
        for d in defects:
            px = int((d["world_x"] - mapper.min_x) / mapper.res * scale_x)
            py = int((mapper.max_y - d["world_y"]) / mapper.res * scale_y)
            draw.ellipse([px - 6, py - 6, px + 6, py + 6], outline=(255, 0, 0), width=2)
            draw.text((px + 8, py - 6), f"{d['id']} ({d['area_m2']}m²)", fill=(255, 255, 255))

        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="PNG")
        self.set_header("Content-Type", "image/png")
        self.set_header("Content-Disposition", 'attachment; filename="phantomx_defect_map.png"')
        self.write(buf.getvalue())


def make_app():
    os.makedirs(STATIC_DIR, exist_ok=True)
    return tornado.web.Application([
        (r"/ws", WebSocketHandler),
        (r"/api/status", StatusApiHandler),
        (r"/api/command", CommandApiHandler),
        (r"/api/export/json", DefectsExportJsonHandler),
        (r"/api/export/png", DefectMapPngHandler),
        (r"/(.*)", tornado.web.StaticFileHandler, {
            "path": STATIC_DIR,
            "default_filename": "index.html"
        }),
    ], debug=False)


def run_physics_loop():
    """Background simulation thread running at 30Hz."""
    while robot_engine.running:
        try:
            robot_engine.update()
        except Exception as e:
            print(f"[Engine Error]: {e}")
        time.sleep(0.033)


def ensure_port_available(port: int):
    """Checks if the port is in use and safely frees it from stale server.py instances."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
        sock.close()
        return True
    except OSError:
        pass
    finally:
        try:
            sock.close()
        except Exception:
            pass

    # If running on Windows, attempt to terminate any stale background server.py instances
    if sys.platform.startswith("win"):
        current_pid = os.getpid()
        try:
            import subprocess
            cmd = f'powershell -Command "Get-CimInstance Win32_Process -Filter \\"Name like \'python%\'\\" | Where-Object {{ $_.CommandLine -like \'*server.py*\' -and $_.ProcessId -ne {current_pid} }} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}"'
            subprocess.run(cmd, shell=True, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.0)
        except Exception:
            pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hexapod Mission Control Server")
    parser.add_argument("port_pos", nargs="?", type=int, default=None)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args, _ = parser.parse_known_args()

    port = args.port_pos if args.port_pos is not None else args.port
    ensure_port_available(port)

    sim_thread = threading.Thread(target=run_physics_loop, daemon=True)
    sim_thread.start()

    app = make_app()
    try:
        app.listen(port, address="0.0.0.0")
    except OSError as e:
        if "10048" in str(e) or getattr(e, "winerror", None) == 10048:
            print(f"\n[ERROR] Port {port} is occupied by another process.")
            print(f"You can start on another port using: python server.py --port {port + 1}\n")
            sys.exit(1)
        raise

    print("=" * 65)
    print(f"Hexapod Mission Control Server running at http://127.0.0.1:{port}")
    print(f"WebSocket Endpoint: ws://127.0.0.1:{port}/ws")
    print(f"Connect Isaac Sim with source=isaac_sim")
    print("=" * 65)

    periodic = tornado.ioloop.PeriodicCallback(broadcast_state, 66)
    periodic.start()

    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
