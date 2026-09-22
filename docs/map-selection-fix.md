# Original costmap selection restored — 2026-09-22

## Correction

The previous Route Studio startup selected a synthetic mall demo instead of an
original costmap. Its map catalog also scanned only the top-level imgs/
directory, omitting nested images such as imgs/map_7/map_7.png.

- The default is now the last successfully selected original repository map,
  or imgs/map_1.png on first use. If map_1.png is absent, another cataloged
  original is selected. An empty directory stays empty: no demo fallback.
- The read-only catalog recursively lists supported images under imgs/, using
  complete relative paths so duplicate basenames remain distinct. Root images
  appear first, in natural numeric order. External symlinks are excluded.
- Synthetic demos are in a separate, explicitly labeled test group and load
  only by manual selection. Temporary uploads do not overwrite the remembered
  original-map preference.
- Switching clears the previous canvas, nodes, metrics and route. A failed
  map load reports the error without showing the previous or a synthetic map.
- The A* planner, collision thresholds and original image files are unchanged.

## Executed checks

Starting point: main commit 3cf5949b233b9b57a4762e4a7bbef7fc74596a0f.
Input source files extracted from the previous update archive were verified
against their connected GitHub blob SHAs before applying this correction.

| Command / cases | Result |
| --- | --- |
| python -m pytest tests/test_costmap.py tests/test_api.py -q | 63 passed, 5.73 s |
| Chromium: original startup, root/nested switching, preference reinitialization, empty directory, load failure | 5 passed, 3.55 s |
| Chromium: layout, wall clicks, matrices/settings, node edits/import, zoom coordinates | 5 passed, 16.85 s |
| Chromium: stale plan, mask/edit race, upload return, route/node downloads | 4 passed, 19.38 s |
| node --check web/app.js; python -m py_compile app.py | Passed |

Total: 77 executed tests passed (63 core/API and 14 browser cases).
Four new API cases cover recursive cataloging, duplicate basenames, unchanged
original pixels and files, external symlink exclusion, empty directories and
invalid map errors. Five new browser cases cover the corrected selection flow.

Browser cases used UI_HTTP_BRIDGE=1 with Chromium rendering the real UI and
requests forwarded to a real local HTTP server. Test image files are temporary
deterministic fixtures, not purported copies of the user's scans. In bridge
mode the preference test uses an explicit in-memory storage adapter and
reinitializes the picker; native reload/storage persistence is not claimed.
The implementation uses browser localStorage when permitted and gracefully
falls back to selecting an original when storage is unavailable.

The connected repository's original imgs/ tree is
4ddc89df7e19e2495e3fe962dbf913591f7becec. This correction does not submit any
changes under imgs/, points/, df_tables/ or to costmap.py. Original repository
images were inspected by path/tree metadata, not downloaded and individually
run through the planner in this environment.
