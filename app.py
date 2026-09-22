"""Local-only web workspace. Run: python app.py [--port 8501]."""
from __future__ import annotations

import argparse
import base64
import binascii
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import time
from urllib.parse import parse_qs, urlsplit
import warnings

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

from costmap import Costmap, PlanningError, Settings

ROOT = Path(__file__).resolve().parent
MAX_BODY = 16 * 1024 * 1024
MAX_PIXELS = 4_000_000


def png64(image: Image.Image | np.ndarray) -> str:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def decode_image(encoded: str) -> np.ndarray:
    if not isinstance(encoded, str) or len(encoded) > MAX_BODY:
        raise PlanningError("Invalid or oversized map image.")
    try:
        raw = base64.b64decode(encoded.split(",", 1)[-1], validate=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as im:
                if im.width * im.height > MAX_PIXELS or min(im.size) < 3:
                    raise PlanningError("Map must be 3x3 or larger, and at most 4 million pixels. The map is never resized.")
                # Transparent pixels are unknown/blocked, never silently free.
                rgba = im.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
                return np.array(Image.alpha_composite(background, rgba).convert("L"))
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError,
            Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise PlanningError(f"Cannot decode map image: {exc}") from exc


def demo_map(name="mall") -> tuple[np.ndarray, list]:
    image = Image.new("L", (1280, 720), 22)
    d = ImageDraw.Draw(image)
    d.rectangle((24, 24, 1255, 695), fill=250)
    if name == "office":
        rooms = [(170, 160, 550, 295), (680, 160, 1110, 295),
                 (170, 420, 550, 560), (680, 420, 1110, 560)]
        nodes = [[100, 100], [1180, 100], [100, 630]]
    else:
        rooms = [(165, 165, 400, 295), (490, 165, 810, 295), (915, 165, 1115, 295),
                 (165, 420, 400, 555), (490, 420, 810, 555), (915, 420, 1115, 555)]
        nodes = [[100, 100], [1180, 355], [280, 625]]
    for i, (x0, y0, x1, y1) in enumerate(rooms):
        d.rectangle((x0, y0, x1, y1), fill=183, outline=18, width=6)
        door_x = (x0 + x1) // 2
        door_y = y1 if i < len(rooms) // 2 else y0
        d.rectangle((door_x - 23, door_y - 7, door_x + 23, door_y + 7), fill=250)
        # Furniture remains an obstacle even with gray-area access enabled.
        d.rectangle((x0 + 30, y0 + 35, x0 + 65, y0 + 75), fill=30)
    d.rectangle((585, 62, 625, 115), fill=18)
    d.rectangle((855, 590, 888, 640), fill=18)
    return np.array(image), nodes


def parse_points(text: str) -> list[list[int]]:
    if not isinstance(text, str) or len(text) > 10_000:
        raise PlanningError("Invalid points text.")
    points = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().replace(" ", "") == "x,y":
            continue
        match = re.fullmatch(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", line)
        if not match:
            raise PlanningError("Use one (x, y) or x,y integer coordinate per line.")
        points.append([int(match[1]), int(match[2])])
    if not 1 <= len(points) <= 16:
        raise PlanningError("Import between 1 and 16 nodes.")
    return points


def build_map(payload: dict) -> Costmap:
    if not isinstance(payload, dict):
        raise PlanningError("Expected a JSON object.")
    config = payload.get("settings", {})
    if not isinstance(config, dict):
        raise PlanningError("settings must be an object.")
    try:
        settings = Settings(**config)
    except TypeError as exc:
        raise PlanningError("Unknown or malformed planner setting.") from exc
    return Costmap(decode_image(payload.get("image", "")), settings)


class Handler(BaseHTTPRequestHandler):
    server_version = "RouteStudio/1.0"

    def reply(self, data, status=200, content_type="application/json; charset=utf-8"):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(data)

    def safe_request(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        return host in hosts and (not origin or origin == f"http://{host}")

    def do_GET(self):
        if not self.safe_request():
            return self.reply({"error": "Only same-origin localhost requests are accepted."}, 403)
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        if url.path == "/api/health":
            return self.reply({"status": "ok"})
        if url.path == "/api/maps":
            maps = sorted(p.name for p in (ROOT / "imgs").glob("*")
                          if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".pgm"))
            return self.reply({"maps": maps})
        if url.path in ("/api/demo", "/api/map"):
            try:
                if url.path == "/api/demo":
                    image, nodes = demo_map(query.get("name", ["mall"])[0])
                else:
                    name = query.get("name", [""])[0]
                    maps = {p.name: p for p in (ROOT / "imgs").glob("*")
                            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".pgm")}
                    if name not in maps:
                        return self.reply({"error": "Map not found."}, 404)
                    image = decode_image(base64.b64encode(maps[name].read_bytes()).decode("ascii"))
                    nodes = []
                return self.reply({"image": png64(image), "nodes": nodes,
                                   "width": image.shape[1], "height": image.shape[0]})
            except (PlanningError, OSError) as exc:
                return self.reply({"error": str(exc)}, 400)
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8")}
        if url.path not in assets:
            return self.reply({"error": "Not found."}, 404)
        name, content_type = assets[url.path]
        self.reply((ROOT / "web" / name).read_bytes(), content_type=content_type)

    def do_POST(self):
        if not self.safe_request():
            return self.reply({"error": "Only same-origin localhost requests are accepted."}, 403)
        if self.path not in ("/api/prepare", "/api/plan", "/api/points"):
            return self.reply({"error": "Not found."}, 404)
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            return self.reply({"error": "Expected application/json."}, 415)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                return self.reply({"error": "Request is empty or exceeds 16 MiB."}, 413)
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise PlanningError("Expected a JSON object.")
            if self.path == "/api/points":
                return self.reply({"nodes": parse_points(payload.get("text", ""))})
            started = time.perf_counter()
            grid = build_map(payload)
            if self.path == "/api/prepare":
                return self.reply({"image": png64(grid.image),
                                   "mask": png64((~grid.blocked).astype(np.uint8) * 255),
                                   "width": grid.width, "height": grid.height,
                                   "free_percent": round(float((~grid.blocked).mean()) * 100, 1)})
            result = grid.plan(payload.get("nodes", []))
            result["settings"] = asdict(grid.settings)
            result["compute_ms"] = round((time.perf_counter() - started) * 1000, 1)
            self.reply(result)
        except (PlanningError, ValueError, TypeError, OverflowError) as exc:
            self.reply({"error": str(exc)}, 400)
        except Exception:
            # No tracebacks or filesystem paths are exposed to the browser.
            import traceback
            traceback.print_exc()
            self.reply({"error": "Unexpected planner error; check the local server log."}, 500)


def make_server(port=8501):
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    with make_server(args.port) as server:
        print(f"Route Studio: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
