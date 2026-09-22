"""Browser interaction tests. UI_HTTP_BRIDGE=1 is for network-restricted sandboxes.

Bridge mode renders the actual HTML/CSS/JS in Chromium and forwards fetch calls
through urllib to a real local HTTP server. It is NOT a live browser-network /
CSP deployment test; default mode navigates directly to the local server.
"""
import json
import os
from pathlib import Path
import shutil
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import pytest

import app

playwright = pytest.importorskip('playwright.sync_api')


@pytest.fixture(scope='module')
def browser_server():
    server = app.make_server(0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


@pytest.fixture(scope='module')
def browser():
    with playwright.sync_playwright() as p:
        executable = os.environ.get('CHROMIUM_PATH') or shutil.which('chromium')
        browser = p.chromium.launch(executable_path=executable, headless=True,
                                     args=['--no-sandbox'])
        yield browser
        browser.close()


@pytest.fixture
def page(browser, browser_server):
    context = browser.new_context(viewport={'width': 1512, 'height': 1100}, accept_downloads=True)
    page = context.new_page()
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    if os.environ.get('UI_HTTP_BRIDGE') == '1':
        def bridge(path, options=None):
            options = options or {}
            headers = {**options.get('headers', {}), 'Origin': browser_server}
            request = Request(browser_server + path, headers=headers,
                              data=options.get('body', '').encode() if options.get('method') == 'POST' else None)
            try:
                response = urlopen(request, timeout=30)
            except HTTPError as error:
                response = error
            return {'status': response.status, 'body': response.read().decode()}

        html = urlopen(browser_server + '/').read().decode()
        html = html.replace('<link rel="stylesheet" href="/style.css">', '').replace('<script defer src="/app.js"></script>', '')
        page.set_content(html)
        page.add_style_tag(content=urlopen(browser_server + '/style.css').read().decode())
        page.expose_function('__httpBridge', bridge)
        page.evaluate('''() => {window.fetch = async(path, options) => {
            const r = await window.__httpBridge(path, options);
            return new Response(r.body, {status:r.status,headers:{'Content-Type':'application/json'}});
        };}''')
        page.add_script_tag(content=urlopen(browser_server + '/app.js').read().decode())
    else:
        page.goto(browser_server, wait_until='domcontentloaded')
    page.wait_for_function("state.result !== null", timeout=30000)
    yield page
    assert errors == []
    context.close()


def click_map(page, x, y):
    position = page.evaluate('''([x,y]) => {
        const t = transform(), r = document.querySelector('#map').getBoundingClientRect();
        return {x:r.left+t.x+x*t.scale,y:r.top+t.y+y*t.scale};
    }''', [x, y])
    page.mouse.click(position['x'], position['y'])


def test_desktop_mobile_and_collapsed_diagnostics(page, tmp_path):
    assert page.locator('#route-status').inner_text() == '路徑可通行'
    assert page.locator('#node-list .node').count() == 3
    assert not page.locator('#diagnostics').evaluate('(e) => e.open')
    assert not page.locator('#advanced').evaluate('(e) => e.open')
    for width, height in [(1512, 1100), (1024, 768), (768, 1024), (390, 844)]:
        page.set_viewport_size({'width': width, 'height': height})
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
    page.screenshot(path=str(tmp_path / 'mobile.png'), full_page=True)


def test_wall_click_is_rejected_and_white_click_adds_waypoint(page):
    click_map(page, 165, 200)
    assert page.locator('#node-list .node').count() == 3
    assert '牆壁' in page.locator('#message').inner_text()
    click_map(page, 450, 355)
    assert page.locator('#node-list .node').count() == 4
    assert page.evaluate('state.result') is None
    page.locator('#plan').click()
    page.wait_for_function('state.result !== null', timeout=30000)
    assert page.evaluate('state.result.nodes[2]') == [450, 355]


def test_thresholds_safety_overlay_and_matrix(page):
    page.locator('#show-clearance').check()
    assert page.evaluate('state.overlay !== null')
    page.locator('#diagnostics summary').click()
    assert page.locator('#matrix tbody tr').count() == 3
    assert '—' in page.locator('#matrix').inner_text()
    page.locator('#matrix-type').select_option('cost_matrix')
    assert page.locator('#matrix tbody tr').count() == 3
    page.locator('#corridor-only').uncheck()
    page.wait_for_function('state.ready && !state.result', timeout=10000)
    assert page.locator('#distance').inner_text() == '—'
    assert page.locator('#export-route').is_disabled()
    assert '允許灰區' in page.locator('#canvas-state').inner_text()


def test_node_reorder_remove_and_import_validation(page):
    page.get_by_role('button', name='節點 2 上移', exact=True).click()
    assert page.evaluate('state.nodes[0]') == [1180, 355]
    page.get_by_role('button', name='刪除節點 2', exact=True).click()
    assert page.locator('#node-list .node').count() == 2
    page.locator('#node-file').set_input_files({'name': 'bad.txt', 'mimeType': 'text/plain', 'buffer': b'(165, 200)\n(100, 100)'})
    page.wait_for_function("document.querySelector('#message').textContent.includes('未套用')")
    assert page.locator('#node-list .node').count() == 2
    page.locator('#node-file').set_input_files({'name': 'points.csv', 'mimeType': 'text/csv', 'buffer': b'x,y\n100,100\n1180,355\n280,625'})
    page.wait_for_function('state.nodes.length === 3')
    assert page.evaluate('state.nodes[0]') == [100, 100]


def test_original_pixel_coordinates_survive_zoom(page):
    page.locator('#zoom-in').click()
    page.locator('[data-mode="start"]').click()
    click_map(page, 445, 355)
    assert page.evaluate('state.nodes[0]') == [445, 355]
    page.locator('#fit').click()
    assert page.locator('#zoom-label').inner_text() == '100%'


def test_old_plan_response_cannot_restore_deleted_route(page):
    page.evaluate('''() => {const fetchNow = window.fetch; window.fetch = async(path, options) => {
      const response = await fetchNow(path, options);
      if (path === '/api/plan') await new Promise(r=>setTimeout(r,1000));
      return response;
    };}''')
    page.locator('#plan').click()
    page.locator('#clear-nodes').click()
    page.wait_for_timeout(1500)
    assert page.evaluate('state.result') is None
    assert page.locator('#node-list .node').count() == 0
    assert page.locator('#distance').inner_text() == '—'
    assert page.locator('#export-route').is_disabled()


def test_editing_nodes_during_mask_prepare_does_not_deadlock(page):
    page.evaluate('''() => {const fetchNow = window.fetch; window.fetch = async(path, options) => {
      const response = await fetchNow(path, options);
      if (path === '/api/prepare') await new Promise(r=>setTimeout(r,700));
      return response;
    };}''')
    page.locator('#radius').fill('0.30')
    page.wait_for_timeout(300)
    page.locator('#clear-nodes').click()
    page.wait_for_function('state.ready', timeout=10000)
    assert page.evaluate('state.nodes.length') == 0


def test_upload_switch_away_and_back_retains_original_map(page, tmp_path):
    path = tmp_path / 'uploaded.png'
    Image.fromarray(np.full((80, 120), 250, np.uint8)).save(path)
    page.locator('#map-file').set_input_files(str(path))
    page.wait_for_function('state.ready && state.image.width === 120')
    assert page.locator('#map-source').input_value() == 'upload'
    assert page.evaluate('state.nodes.length') == 0
    page.locator('#map-source').select_option('demo:office')
    page.wait_for_function('state.result !== null', timeout=30000)
    page.locator('#map-source').select_option('upload')
    page.wait_for_function('state.ready && state.image.width === 120')
    assert page.locator('#map-title').inner_text() == 'uploaded.png'


def test_download_contains_real_planned_pixels(page):
    with page.expect_download() as event:
        page.locator('#export-route').click()
    download = event.value
    result = json.loads(Path(download.path()).read_text())
    assert result['image_width'] == 1280 and result['image_height'] == 720
    assert result['path'][0] == [100, 100]
    assert result['path'][-1] == [280, 625]
    assert result['settings']['allow_gray'] is False
    assert result['distance_matrix'][0][2] is None
    with page.expect_download() as event:
        page.locator('#export-nodes').click()
    assert '(100, 100)' in Path(event.value.path()).read_text()
