"""Pixel-preserving, corridor-aware A* for static grayscale occupancy maps."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt


class PlanningError(ValueError):
    """Invalid configuration, unsafe waypoint, unreachable route, or work limit."""


@dataclass(frozen=True)
class Settings:
    white_threshold: int = 220
    wall_threshold: int = 80
    allow_gray: bool = False
    robot_radius: float = 5.0  # Original-image pixels, not display pixels.
    safety_margin: float = 2.0
    clearance_weight: float = 3.0
    gray_weight: float = 4.0
    resolution: float = 0.05  # metres / original-image pixel
    speed: float = 0.7  # metres / second

    def __post_init__(self):
        for name in ("white_threshold", "wall_threshold"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise PlanningError(f"{name} must be an integer.")
        if not 1 <= self.wall_threshold < self.white_threshold <= 255:
            raise PlanningError("Thresholds must satisfy 1 <= wall < white <= 255.")
        if not isinstance(self.allow_gray, bool):
            raise PlanningError("allow_gray must be boolean.")
        limits = {"robot_radius": (0, 200), "safety_margin": (0, 200),
                  "clearance_weight": (0, 20), "gray_weight": (0, 20),
                  "resolution": (0.0001, 10), "speed": (0.001, 100)}
        for name, (lo, hi) in limits.items():
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not lo <= value <= hi):
                raise PlanningError(f"{name} must be a finite number in [{lo}, {hi}].")


class Costmap:
    """Black/unknown cells are hard obstacles; white cells are preferred.

    EDT measures distance to *cell centres*. Subtract sqrt(2) conservatively
    to cover both the obstacle's half diagonal and motion between grid centres.
    This sacrifices up to 1.42 px of narrow passage width on each side, rather
    than promising clearance that a continuous circular robot cannot maintain.
    The one-cell blocked border also makes space outside the image occupied.
    """

    def __init__(self, grayscale: np.ndarray, settings: Settings | None = None):
        self.settings = settings or Settings()
        image = np.asarray(grayscale)
        if image.ndim != 2 or min(image.shape) < 3 or image.size > 4_000_000:
            raise PlanningError("Use a grayscale map of at least 3x3 and at most 4 million pixels.")
        if image.dtype != np.uint8:
            raise PlanningError("Map must contain uint8 grayscale pixels (0..255).")
        self.image = image.copy()
        self.height, self.width = image.shape
        s = self.settings
        cutoff = s.wall_threshold if s.allow_gray else s.white_threshold
        self.occupied = image < cutoff
        distance = distance_transform_edt(np.pad(~self.occupied, 1, constant_values=False))[1:-1, 1:-1]
        self.clearance = np.maximum(distance - math.sqrt(2), 0)
        self.blocked = self.occupied | (self.clearance <= s.robot_radius + s.safety_margin)
        # The same per-cell field is used in search, objective reporting, and matrices.
        wall_penalty = s.clearance_weight * np.exp(-self.clearance / max(4.0, s.robot_radius + s.safety_margin))
        gray_penalty = s.gray_weight * np.clip((s.white_threshold - image.astype(float)) /
                                              (s.white_threshold - s.wall_threshold), 0, 1)
        self.cost = 1.0 + wall_penalty + gray_penalty

    def point(self, value: Sequence[int], label: str = "Waypoint") -> tuple[int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise PlanningError(f"{label}: expected [x, y].")
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in value):
            raise PlanningError(f"{label}: coordinates must be integer image pixels.")
        x, y = map(int, value)
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise PlanningError(f"{label}: outside the map.")
        if self.occupied[y, x]:
            raise PlanningError(f"{label}: on a wall or a non-traversable gray area. Select a white corridor.")
        if self.blocked[y, x]:
            raise PlanningError(f"{label}: too close to a wall for the robot radius and safety margin.")
        return x, y

    def neighbors(self, x: int, y: int):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height) or self.blocked[ny, nx]:
                continue
            # No diagonal squeezing between touching blocked cells.
            if dx and dy and (self.blocked[y, nx] or self.blocked[ny, x]):
                continue
            yield nx, ny, math.sqrt(2) if dx and dy else 1.0

    def astar(self, start, goal, max_expansions: int = 400_000) -> dict:
        start, goal = self.point(start, "Start"), self.point(goal, "Goal")
        if start == goal:
            return {"path": [list(start)], "distance_px": 0.0, "cost": 0.0, "expanded": 0}

        def heuristic(p):
            dx, dy = abs(goal[0] - p[0]), abs(goal[1] - p[1])
            return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

        queue = [(heuristic(start), 0.0, start)]
        best = {start: 0.0}
        parent = {}
        expanded = 0
        while queue:
            _, g, p = heapq.heappop(queue)
            if g > best.get(p, math.inf):
                continue
            if p == goal:
                route = [p]
                while p != start:
                    p = parent[p]
                    route.append(p)
                route.reverse()
                length = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
                return {"path": [list(p) for p in route], "distance_px": length,
                        "cost": float(g), "expanded": expanded}
            expanded += 1
            if expanded > max_expansions:
                raise PlanningError("Search limit reached; use fewer waypoints or a smaller map crop. No unsafe fallback was drawn.")
            x, y = p
            for nx, ny, step in self.neighbors(x, y):
                q = nx, ny
                new_g = g + step * float(self.cost[y, x] + self.cost[ny, nx]) / 2
                if new_g < best.get(q, math.inf):
                    best[q], parent[q] = new_g, p
                    heapq.heappush(queue, (new_g + heuristic(q), new_g, q))
        raise PlanningError("No collision-free route. A wall, closed doorway, or clearance constraint separates the nodes.")

    def plan(self, nodes: list, max_expansions: int = 400_000) -> dict:
        if not isinstance(nodes, list) or not 2 <= len(nodes) <= 16:
            raise PlanningError("Choose between 2 and 16 waypoints, in visit order.")
        points = [self.point(p, f"Node {i + 1}") for i, p in enumerate(nodes)]
        route, segments = [], []
        n = len(points)
        # Only traversed neighbor pairs are computed. None explicitly means NOT
        # COMPUTED, not unreachable; do not present Euclidean shortcuts as A* edges.
        distances = [[0.0 if i == j else None for j in range(n)] for i in range(n)]
        costs = [[0.0 if i == j else None for j in range(n)] for i in range(n)]
        times = [[0.0 if i == j else None for j in range(n)] for i in range(n)]
        for i, (a, b) in enumerate(zip(points, points[1:])):
            try:
                result = self.astar(a, b, max_expansions)
            except PlanningError as exc:
                raise PlanningError(f"Leg {i + 1} ({i + 1} -> {i + 2}): {exc}") from exc
            route.extend(result["path"] if i == 0 else result["path"][1:])
            metres = result["distance_px"] * self.settings.resolution
            segments.append({"from": i, "to": i + 1, "distance_m": metres,
                             "travel_time_s": metres / self.settings.speed,
                             "cost": result["cost"], "expanded": result["expanded"]})
            for matrix, value in ((distances, metres), (costs, result["cost"]),
                                  (times, metres / self.settings.speed)):
                matrix[i][i + 1] = matrix[i + 1][i] = value
        distance_m = sum(s["distance_m"] for s in segments)
        centre_clearance = min(float(self.clearance[y, x]) for x, y in route)
        return {"nodes": [list(p) for p in points], "path": route, "segments": segments,
                "distance_m": distance_m, "travel_time_s": distance_m / self.settings.speed,
                "cost": sum(s["cost"] for s in segments),
                "expanded": sum(s["expanded"] for s in segments),
                "min_clearance_m": (centre_clearance - self.settings.robot_radius) * self.settings.resolution,
                "distance_matrix": distances, "time_matrix": times, "cost_matrix": costs,
                "matrix_note": "Only consecutive route pairs are evaluated; null = not computed.",
                "coordinate_system": "original image pixels; x right, y down; origin top-left",
                "clearance_note": "Conservative lower bound from robot footprint to blocked cells, including image boundary."}
