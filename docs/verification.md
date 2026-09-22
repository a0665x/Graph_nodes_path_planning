# Verification report — 2026-09-22

## Scope and baseline

Inspected the connected GitHub repository `a0665x/Graph_nodes_path_planning`,
`main` commit `43f23413ca6d5cb0fc426bfc98ec6d05a5329987`.
The legacy UI reads graph time/weight tables and draws straight node-to-node
edges; it does not inspect costmap pixels along an edge. The uploaded node
file's name is also used to reopen a server-local path instead of reading its
uploaded bytes. The new `app.py` workspace bypasses those legacy execution
paths, preserving the old scripts, matrices and maps for research use.

## Executed results

| Check | Result |
| --- | --- |
| `python -m pytest tests/test_costmap.py tests/test_api.py -q` | **59 passed**, 5.59 s |
| Chromium UI tests: desktop, wall clicks, settings/matrices, node editing/import, zoom | **5 passed**, 15.63 s |
| Chromium UI tests: stale response, mask/edit race, uploaded-map return, downloads | **4 passed**, 19.90 s |
| `node --check web/app.js` | Passed |
| `python -m py_compile app.py costmap.py` | Passed |
| Desktop and mobile browser screenshots | Rendered and inspected |

Total: **68 executed tests passed**, with the browser tests executed in two
batches. A prior all-browser batch hit the sandbox command timeout during
completion; all nine cases were rerun in the two successful batches above.

The browser test batches used `UI_HTTP_BRIDGE=1` because direct Chromium HTTP
navigation is administratively blocked in this sandbox. The real UI HTML,
CSS and JavaScript were served by the local HTTP server, rendered in Chromium,
and exercised against the real HTTP API through an explicit urllib bridge.
This verifies DOM/canvas behavior and API integration, **not** native browser
network navigation or enforcement of the deployed CSP. Direct browser mode is
included in the test suite for execution in a normal development environment.

## Coverage

The planner tests cover black-wall detours, one-pixel walls, a narrow doorway
closing as the robot grows, forbidden diagonal corner-cutting, white-only
routing, optional gray access, gray-area penalties, wall/clearance/outside
waypoint rejection, invalid settings, repeated waypoints, ordered multi-leg
routes, distance/time/cost matrix consistency and explicit search-limit errors.

An independent Dijkstra implementation checks A* objective values on eight
deterministic obstacle layouts. Another test samples the continuous swept
robot path against **occupied pixel squares**, including the image boundary,
to independently check the requested robot radius and margin. Both generated
1280 × 720 maps are tested at original resolution with every returned pixel
checked against the blocked mask.

API checks include malformed images and payloads, transparent unknown pixels,
PGM decoding, legacy TXT / CSV import without `eval`, disallowed origins and
host headers, static-asset/path allowlisting, map selection from a temporary
image directory, and independence from the process working directory.

UI checks include 1512, 1024, 768 and 390 px layouts without horizontal overflow,
collapsed diagnostics, rejection of wall clicks, white-corridor placement,
zoom-correct original coordinates, reordering/deletion/import, settings
invalidation, visible collision mask, metric matrices, stale-response
suppression, editing during mask preparation, switching back to an uploaded
map, and actual JSON/TXT downloads.

## Environment

Python 3.13.5; NumPy 2.3.5; SciPy 1.17.0; Pillow 12.3.0; pytest 9.0.2;
Playwright 1.57.0; system Chromium, headless. The declared dependency ranges
are not a claim that every supported Python/package combination was tested.

## Limitations

No live robot, ROS/Nav2 integration, moving obstacles, multi-floor/elevator
routing, GA/TSP task ordering, or production deployment was tested or added.
Old Streamlit code was inspected but not runtime-certified: Streamlit is not
installed here and package downloads/ordinary git clone were unavailable.
Repository-owned image files were preserved on GitHub rather than downloaded
into the sandbox; the map-picker integration test uses an explicit temporary
fixture. A built-in synthetic costmap is therefore shown in the screenshots,
not a claimed scan of a real building. No GitHub Actions run is claimed.
