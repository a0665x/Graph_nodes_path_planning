import base64
from dataclasses import asdict
import io
import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import pytest

import app
from costmap import PlanningError, Settings


@pytest.fixture(scope='module')
def server():
    httpd = app.make_server(0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{httpd.server_port}'
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=3)


def request(server, path, payload=None, extra_headers=None):
    headers = {'Content-Type': 'application/json', **(extra_headers or {})}
    req = Request(server + path, data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
    try:
        response = urlopen(req, timeout=20)
    except HTTPError as exc:
        response = exc
    return response.status, response.read(), response.headers


def sample_payload():
    image = np.full((48, 64), 255, np.uint8)
    image[:30, 30:33] = 0
    return {'image': app.png64(image), 'nodes': [[10, 10], [54, 10]],
            'settings': asdict(Settings(robot_radius=1, safety_margin=1))}


def test_real_http_plan_and_prepare(server):
    payload = sample_payload()
    code, body, _ = request(server, '/api/prepare', payload)
    prepared = json.loads(body)
    assert code == 200 and prepared['width'] == 64 and prepared['height'] == 48
    mask = app.decode_image(prepared['mask'])
    code, body, _ = request(server, '/api/plan', payload)
    result = json.loads(body)
    assert code == 200
    assert all(mask[y, x] == 255 for x, y in result['path'])
    assert result['path'][0] == [10, 10] and result['path'][-1] == [54, 10]
    assert result['compute_ms'] >= 0


def test_invalid_node_reports_error_not_http500(server):
    payload = sample_payload()
    payload['nodes'][0] = [31, 10]
    code, body, _ = request(server, '/api/plan', payload)
    assert code == 400 and 'wall' in json.loads(body)['error']
    assert 'path' not in json.loads(body)


@pytest.mark.parametrize('image', ['', 'not base64!', base64.b64encode(b'not an image').decode()])
def test_bad_images_are_rejected(image):
    with pytest.raises(PlanningError):
        app.decode_image(image)


def test_transparent_unknown_pixels_are_obstacles():
    image = Image.new('RGBA', (10, 8), (255, 255, 255, 0))
    image.putpixel((4, 4), (255, 255, 255, 255))
    decoded = app.decode_image(app.png64(image))
    assert decoded[0, 0] == 0 and decoded[4, 4] == 255
    assert decoded.shape == (8, 10)


def test_pgm_is_supported_without_browser_image_decoding():
    buffer = io.BytesIO()
    image = Image.new('L', (10, 8), 240)
    image.save(buffer, format='PPM')
    result = app.decode_image(base64.b64encode(buffer.getvalue()).decode())
    assert result.shape == (8, 10) and (result == 240).all()


@pytest.mark.parametrize('text, expected', [('(10, 20)\n(30, 40)', [[10, 20], [30, 40]]),
                                          ('x,y\n10,20\n#comment\n30,40', [[10, 20], [30, 40]])])
def test_import_legacy_points(text, expected):
    assert app.parse_points(text) == expected


@pytest.mark.parametrize('text', ['__import__("os").system("echo nope")', '1.5,3', '', '1,2,3'])
def test_bad_points_are_not_evaluated(text):
    with pytest.raises(PlanningError):
        app.parse_points(text)


def test_points_api(server):
    code, body, _ = request(server, '/api/points', {'text': '(10, 20)\n(30, 40)'})
    assert code == 200 and json.loads(body)['nodes'] == [[10, 20], [30, 40]]


@pytest.mark.parametrize('headers', [{'Origin': 'http://evil.example'}, {'Host': 'evil.example'}])
def test_rejects_cross_origin_or_rebinding_requests(server, headers):
    code, body, _ = request(server, '/api/plan', sample_payload(), headers)
    assert code == 403


def test_assets_are_allowlisted(server):
    for path, content_type in [('/', 'text/html'), ('/style.css', 'text/css'), ('/app.js', 'text/javascript')]:
        code, body, headers = request(server, path)
        assert code == 200 and content_type in headers['Content-Type'] and body
        assert "frame-ancestors 'none'" in headers['Content-Security-Policy']
    for path in ['/app.py', '/../app.py', '/.git/config', '/api/map?name=../app.py']:
        code, _, _ = request(server, path)
        assert code == 404


def test_repository_map_picker_reads_only_allowed_images(server, tmp_path, monkeypatch):
    monkeypatch.setattr(app, 'ROOT', tmp_path)
    (tmp_path / 'imgs').mkdir()
    Image.new('L', (20, 30), 255).save(tmp_path / 'imgs' / 'test.png')
    (tmp_path / 'imgs' / 'secret.txt').write_text('never served')
    code, body, _ = request(server, '/api/maps')
    assert json.loads(body)['maps'] == ['test.png']
    code, body, _ = request(server, '/api/map?name=test.png')
    assert code == 200 and json.loads(body)['width'] == 20


def test_unknown_setting_and_non_object_json(server):
    payload = sample_payload()
    payload['settings']['unexpected'] = 1
    assert request(server, '/api/plan', payload)[0] == 400
    assert request(server, '/api/plan', [1, 2])[0] == 400


def test_no_process_working_directory_dependency(server, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert request(server, '/')[0] == 200
    assert request(server, '/api/health')[0] == 200


def test_catalog_includes_nested_originals_without_basename_collisions(server, tmp_path, monkeypatch):
    monkeypatch.setattr(app, 'ROOT', tmp_path)
    images = tmp_path / 'imgs'
    for name, size, value in [('map_1.png', (40, 30), 235), ('map_2.png', (50, 35), 240),
                              ('map_10.jpg', (60, 40), 245), ('map_7/map_7.png', (70, 45), 250),
                              ('map_1/map_1.png', (80, 50), 255)]:
        path = images / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('L', size, value).save(path)
    before = {p.relative_to(images).as_posix(): p.read_bytes() for p in images.rglob('*') if p.is_file()}
    code, body, _ = request(server, '/api/maps')
    assert code == 200
    assert json.loads(body)['maps'] == ['map_1.png', 'map_2.png', 'map_10.jpg',
                                        'map_1/map_1.png', 'map_7/map_7.png']
    for name, size in [('map_1.png', (40, 30)), ('map_1/map_1.png', (80, 50)), ('map_7/map_7.png', (70, 45))]:
        from urllib.parse import quote
        code, body, _ = request(server, '/api/map?name=' + quote(name, safe=''))
        result = json.loads(body)
        assert code == 200
        assert (result['width'], result['height']) == size
        assert result['nodes'] == []  # No synthetic demo waypoints on an original map.
        original = np.array(Image.open(images / name).convert('L'))
        assert np.array_equal(app.decode_image(result['image']), original)
    after = {p.relative_to(images).as_posix(): p.read_bytes() for p in images.rglob('*') if p.is_file()}
    assert after == before  # Catalog and loading never write to imgs/.


def test_recursive_catalog_does_not_expose_external_symlinks(server, tmp_path, monkeypatch):
    monkeypatch.setattr(app, 'ROOT', tmp_path)
    images = tmp_path / 'imgs'
    images.mkdir()
    outside = tmp_path / 'private.png'
    Image.new('L', (20, 20), 255).save(outside)
    (images / 'leaked.png').symlink_to(outside)
    code, body, _ = request(server, '/api/maps')
    assert code == 200 and json.loads(body)['maps'] == []
    for name in ['leaked.png', '..%2Fprivate.png', '%2Fprivate.png']:
        assert request(server, '/api/map?name=' + name)[0] == 404


def test_missing_image_directory_does_not_substitute_demos(server, tmp_path, monkeypatch):
    monkeypatch.setattr(app, 'ROOT', tmp_path)
    code, body, _ = request(server, '/api/maps')
    assert code == 200 and json.loads(body)['maps'] == []
    code, body, _ = request(server, '/api/map?name=map_1.png')
    assert code == 404 and 'image' not in json.loads(body)


def test_invalid_original_image_is_not_silently_replaced(server, tmp_path, monkeypatch):
    monkeypatch.setattr(app, 'ROOT', tmp_path)
    (tmp_path / 'imgs').mkdir()
    (tmp_path / 'imgs' / 'broken.png').write_bytes(b'not an image')
    code, body, _ = request(server, '/api/map?name=broken.png')
    assert code == 400 and 'image' not in json.loads(body)
