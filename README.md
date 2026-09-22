# Graph_nodes_path_planning · Route Studio

A local, map-first workspace for **collision-aware static costmap planning**. Place a start, ordered waypoints and an end directly on the map. Routes follow the original image pixels instead of drawing straight lines between graph nodes.

## Run the new interface

Python 3.10 or newer. Use a new virtual environment, separate from the legacy Streamlit dependencies.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:8501**. An alternative port is `python app.py --port 8502`.

**The new entry point is `python app.py`, not `streamlit run steamlit_route_run.py`.** No Node build step, browser extension, CDN, cloud account or Streamlit installation is needed. The server only listens on loopback and is a local development tool, not a production Internet service.

## 操作方式

1. 選擇內建 1280 × 720 商場／辦公區範例，或匯入 PNG、JPEG、PGM。原有 `imgs/` 根目錄地圖也會出現在選單。
2. 在工具列選擇「起點」「途經點」「終點」，再點選白色走廊。使用左側箭頭調整造訪順序，或匯入原有 `(x, y)` TXT / `x,y` CSV。上限 16 點。
3. 設定機器人半徑、安全邊距與正確的地圖解析度，再按「計算安全路徑」。路段、矩陣及公式收合在下方；節點及完整路徑可匯出。

滾輪縮放、右鍵拖曳、觸控拖曳與「適合」按鈕不改變原圖座標。重新選圖、修改節點或變更設定後，舊路徑和舊數字立即失效；過期的計算回應不會恢復舊路線。

### 地圖與安全規則

| 區域 | 預設行為 |
| --- | --- |
| 白色，灰階 ≥ 220 | 走廊，可通行，但仍須滿足半徑和安全邊距 |
| 灰色，80–219 | 預設禁行；關閉「只走白色走廊」才允許進入，並增加通行成本 |
| 黑色，灰階 < 80 | 牆壁／障礙物，不可通行 |
| 透明像素、圖外範圍 | 視為未知／禁行 |

門檻可在進階設定調整。**灰色代表未知空間而非可通行店家時，請保持「只走白色走廊」開啟。** 單靠灰階不能推斷地圖語意。

原圖不縮放、不旋轉、不模糊後再規劃，因此不會因縮圖而抹去一像素薄牆。最多 400 萬像素，HTTP 請求最多 16 MiB。上傳的影像與節點只在本機記憶體處理，不以使用者提供的檔名寫入伺服器路徑。

預設解析度 **0.05 m/px 只是範例假設**：0.25 m 半徑對應 5 px，0.10 m 安全邊距對應 2 px。匯入實際地圖時必須自行填入正確解析度。

## What the planner actually optimizes

`costmap.py` implements 8-connected A* with an octile lower-bound heuristic. Cardinal edges have length 1 px; diagonal edges have length √2 px. A diagonal is legal only when both adjacent cardinal cells are also traversable.

For an occupied-cell-centre Euclidean distance transform `EDT`, define:

```text
d(p)      = max(EDT(p) - sqrt(2), 0)
free(p)   = original pixel passes threshold AND d(p) > radius + margin
sigma     = max(4 px, radius + margin)
gray(p)   = clip((white_threshold - intensity(p)) /
                 (white_threshold - wall_threshold), 0, 1)
c(p)      = 1 + clearance_weight * exp(-d(p)/sigma) + gray_weight * gray(p)
edge(p,q) = EuclideanStepLength(p,q) * (c(p) + c(q)) / 2
J(path)   = sum(edge(p,q))
```

Subtracting √2 conservatively accounts for the occupied pixel's half diagonal and motion between neighboring cell centres. The planner can therefore reject very narrow passages even for a point robot; it deliberately favors conservative clearance over fitting every geometrically possible passage. Reported minimum clearance is a lower bound from the circular robot's outer edge to **all blocked areas**, not just black walls. The UI renders the actual discrete A* polyline, with no unchecked smoothing or shortcutting.

Distance is the sum of the returned pixel-step lengths multiplied by map resolution. Travel time is distance / speed and **does not include task service or waiting time**. The cost objective is distinct from metres and seconds. Higher gray/clearance penalties can select a longer physical route.

Waypoints are visited **in the list order**: this version does not claim TSP/GA task-order optimization. Distance, time and cost matrices contain only the consecutive pairs that were actually planned; `null` / `—` means **not computed**, not unreachable. The static spatial cost field is symmetric, so evaluated entries are reflected across the diagonal. A blocked waypoint, disconnected route or search-budget limit is an explicit error, never a straight-line fallback. The per-leg work limit is 400,000 expanded states.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_costmap.py tests/test_api.py -q
python -m playwright install chromium
python -m pytest tests/test_ui.py -q
```

`CHROMIUM_PATH` can point to an existing Chromium executable. In an environment that prohibits browser HTTP navigation, `UI_HTTP_BRIDGE=1` runs the same UI in Chromium and forwards fetches to the real local HTTP API via the test harness. That mode does **not** validate live browser-network/CSP deployment behavior.

See [the verification report](docs/verification.md) for the exact executed checks and limitations.

## Layout and legacy tools

```text
app.py                  Local HTTP API, image validation and demo maps
costmap.py              Grid classification, clearance, A*, route metrics
web/                    Browser UI; no external assets or build dependencies
tests/                  Planner, HTTP API and Chromium interaction tests
imgs/, points/          Existing maps and node coordinates (preserved)
df_tables/              Existing research matrices (preserved)
utils_graph/            Original graph research helpers (preserved)
```

The original `steamlit_route_run.py`, `streamlit_label_tool_run.py` and `utils_graph/Route_funs.py` remain available for historical graph/matrix research. Their original dependencies are recorded in `requirements-legacy.txt`; they are old and were not reinstalled or runtime-certified in this update. **Those legacy graph visualizers do not provide the new pixel-level collision guarantee.** Existing time/weight CSVs are not treated as proof of traversability or silently reused as the new spatial cost field.

This update covers **single-floor static global planning**. It does not connect floors/elevators, process live LiDAR, avoid moving people, or provide a certified robot controller. Real-robot use still requires correct calibration, localization, dynamic/local obstacle handling and physical validation.
