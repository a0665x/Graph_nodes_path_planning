import math
from dataclasses import replace

import numpy as np
import pytest

from app import demo_map
from costmap import Costmap, PlanningError, Settings


def config(**changes):
    return replace(Settings(robot_radius=0, safety_margin=0, clearance_weight=0), **changes)


def safe_path(grid, path):
    for x, y in path:
        assert not grid.blocked[y, x]
    for (x, y), (nx, ny) in zip(path, path[1:]):
        assert abs(nx - x) <= 1 and abs(ny - y) <= 1
        assert (nx, ny) != (x, y)
        if nx != x and ny != y:
            assert not grid.blocked[y, nx] and not grid.blocked[ny, x]


def test_detours_around_black_wall():
    image = np.full((64, 90), 255, np.uint8)
    image[:45, 43:46] = 0
    grid = Costmap(image, config())
    route = grid.astar([15, 15], [75, 15])
    safe_path(grid, route['path'])
    assert max(y for _, y in route['path']) >= 46
    assert route['distance_px'] > 60


def test_one_pixel_wall_is_not_lost_to_resizing():
    image = np.full((40, 70), 255, np.uint8)
    image[:, 35] = 0
    with pytest.raises(PlanningError, match='No collision-free'):
        Costmap(image, config()).astar([10, 20], [60, 20])


def test_door_closes_when_robot_grows():
    image = np.full((60, 80), 255, np.uint8)
    image[:, 39:41] = 0
    image[25:36, 39:41] = 255
    grid = Costmap(image, config(robot_radius=1))
    safe_path(grid, grid.astar([10, 30], [70, 30])['path'])
    with pytest.raises(PlanningError, match='No collision-free'):
        Costmap(image, config(robot_radius=6)).astar([10, 30], [70, 30])


def test_forbids_diagonal_corner_cutting():
    grid = Costmap(np.full((15, 15), 255, np.uint8), config())
    grid.blocked[:] = True
    grid.blocked[6, 6] = grid.blocked[7, 7] = False
    with pytest.raises(PlanningError, match='No collision-free'):
        grid.astar([6, 6], [7, 7])


def test_white_corridor_default_avoids_gray_shortcut():
    image = np.full((60, 90), 255, np.uint8)
    image[18:42, 30:60] = 183
    white = Costmap(image, config())
    route = white.astar([10, 30], [80, 30])
    assert all(image[y, x] >= 220 for x, y in route['path'])
    gray = Costmap(image, config(allow_gray=True, gray_weight=0))
    shorter = gray.astar([10, 30], [80, 30])
    assert any(image[y, x] == 183 for x, y in shorter['path'])
    assert shorter['distance_px'] < route['distance_px']


def test_gray_cost_prefers_white_even_when_gray_is_allowed():
    image = np.full((60, 90), 255, np.uint8)
    image[18:42, 30:60] = 183
    grid = Costmap(image, config(allow_gray=True, gray_weight=20))
    result = grid.astar([10, 30], [80, 30])
    assert all(image[y, x] >= 220 for x, y in result['path'])


def test_black_is_still_blocked_in_gray_mode():
    image = np.full((30, 50), 183, np.uint8)
    image[:, 25] = 0
    with pytest.raises(PlanningError, match='No collision-free'):
        Costmap(image, config(allow_gray=True)).astar([10, 15], [40, 15])


@pytest.mark.parametrize('point, match', [([0, 0], 'close'), ([-1, 5], 'outside'), ([35, 5], 'outside'),
                                        ([5.5, 5], 'integer'), ([True, 5], 'integer'), ([5], 'expected')])
def test_rejects_unsafe_or_invalid_waypoints(point, match):
    with pytest.raises(PlanningError, match=match):
        Costmap(np.full((30, 35), 255, np.uint8), config()).point(point)


def test_rejects_node_inside_wall_without_snapping():
    image = np.full((40, 40), 255, np.uint8)
    image[20, 20] = 0
    with pytest.raises(PlanningError, match='wall'):
        Costmap(image, config()).astar([20, 20], [30, 30])


def test_same_start_and_goal_and_repeated_nodes():
    grid = Costmap(np.full((40, 40), 255, np.uint8), config())
    result = grid.plan([[10, 10], [10, 10], [20, 20]])
    assert result['segments'][0]['distance_m'] == 0
    safe_path(grid, result['path'])
    assert grid.astar([10, 10], [10, 10])['path'] == [[10, 10]]


def test_order_metrics_and_partial_matrices_are_consistent():
    grid = Costmap(np.full((50, 60), 255, np.uint8), config(resolution=.1, speed=.5))
    result = grid.plan([[10, 10], [40, 10], [40, 30]])
    assert result['distance_m'] == pytest.approx(5)
    assert result['travel_time_s'] == pytest.approx(10)
    assert result['cost'] == pytest.approx(50)
    assert result['distance_matrix'][0][1] == pytest.approx(3)
    assert result['distance_matrix'][0][2] is None
    assert result['time_matrix'][1][2] == pytest.approx(4)
    assert result['path'].index([40, 10]) < result['path'].index([40, 30])


def test_search_limit_does_not_return_a_straight_line():
    grid = Costmap(np.full((30, 60), 255, np.uint8), config())
    with pytest.raises(PlanningError, match='Search limit'):
        grid.astar([5, 5], [50, 25], max_expansions=1)


@pytest.mark.parametrize('nodes', [[], [[5, 5]], [[5, 5]] * 17, None])
def test_rejects_invalid_node_counts(nodes):
    with pytest.raises(PlanningError):
        Costmap(np.full((20, 20), 255, np.uint8), config()).plan(nodes)


@pytest.mark.parametrize('change', [{'speed': 0}, {'resolution': 0}, {'robot_radius': -1},
                                    {'clearance_weight': float('nan')}, {'gray_weight': float('inf')},
                                    {'allow_gray': 'yes'}, {'wall_threshold': 230},
                                    {'white_threshold': 0}, {'white_threshold': 220.5},
                                    {'speed': True}, {'robot_radius': None}])
def test_invalid_settings(change):
    with pytest.raises(PlanningError):
        Settings(**change)


@pytest.mark.parametrize('image', [np.zeros((3, 3, 3), np.uint8), np.zeros((2, 3), np.uint8),
                                  np.zeros((5, 5), float)])
def test_invalid_grayscale_arrays(image):
    with pytest.raises(PlanningError):
        Costmap(image)


def test_wall_clearance_for_the_entire_swept_robot_disk():
    image = np.full((48, 64), 255, np.uint8)
    image[6:31, 30:33] = 0
    grid = Costmap(image, config(robot_radius=2, safety_margin=1, clearance_weight=3))
    result = grid.plan([[10, 10], [53, 10]])
    safe_path(grid, result['path'])
    # Independently measure distances from continuous samples to occupied pixel
    # SQUARES (not their centres). Include the outside image as blocked pixels.
    forbidden = np.pad(image == 0, 1, constant_values=True)
    ys, xs = np.where(forbidden)
    obstacles = np.column_stack((xs - 1, ys - 1))
    minimum = math.inf
    for a, b in zip(result['path'], result['path'][1:]):
        for t in np.linspace(0, 1, 9):
            p = np.array(a) * (1 - t) + np.array(b) * t
            delta = np.maximum(np.abs(obstacles - p) - .5, 0)
            minimum = min(minimum, float(np.linalg.norm(delta, axis=1).min()))
    assert minimum > 3  # 2 px radius + 1 px safety margin
    assert result['min_clearance_m'] <= (minimum - 2) * grid.settings.resolution


def test_astar_matches_independent_dijkstra_reference():
    import heapq
    rng = np.random.default_rng(42)
    for _ in range(8):
        image = np.full((25, 35), 255, np.uint8)
        for x, y in rng.integers([8, 8], [26, 17], size=(4, 2)):
            image[y:y + 2, x:x + 2] = 0
        grid = Costmap(image, config(clearance_weight=2))
        start, end = (5, 12), (29, 12)
        queue, visited, answer = [(0.0, start)], set(), None
        while queue:
            distance, p = heapq.heappop(queue)
            if p in visited:
                continue
            visited.add(p)
            if p == end:
                answer = distance
                break
            x, y = p
            # Independent enumeration of legal neighbors and cost integration.
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = nx, ny = x + dx, y + dy
                    if not (dx or dy) or not (0 <= nx < 35 and 0 <= ny < 25) or grid.blocked[ny, nx]:
                        continue
                    if dx and dy and (grid.blocked[y, nx] or grid.blocked[ny, x]):
                        continue
                    edge = math.hypot(dx, dy) * (grid.cost[y, x] + grid.cost[ny, nx]) / 2
                    if q not in visited:
                        heapq.heappush(queue, (distance + edge, q))
        assert answer is not None
        assert grid.astar(list(start), list(end))['cost'] == pytest.approx(answer)


@pytest.mark.parametrize('name', ['mall', 'office'])
def test_full_resolution_demo(name):
    image, nodes = demo_map(name)
    grid = Costmap(image)
    result = grid.plan(nodes)
    assert image.shape == (720, 1280)
    safe_path(grid, result['path'])
    assert all(image[y, x] >= 220 for x, y in result['path'])
    assert result['min_clearance_m'] > grid.settings.safety_margin * grid.settings.resolution
