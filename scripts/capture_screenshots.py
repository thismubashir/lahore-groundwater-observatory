"""Optional: screenshot each dashboard tab with headless Google Chrome.

Local view (per-well data; images are git-ignored):
      streamlit run app/dashboard.py          (in another terminal)
      python scripts/capture_screenshots.py [--url http://localhost:8501] [--width 1400]
      -> docs/screenshots/<tab>.png
      Also clicks one well on the "Explore a Well" map and reports whether the
      selected well changed, as a check of map selection in a real browser.

Public view (aggregates and a synthetic example only; images may be published after review):
      python scripts/capture_screenshots.py --public [--width 1400]
      -> starts its own dashboard with LGO_PUBLIC_MODE=1 on port 8502 and writes
         docs/screenshots/public/<tab>.png
"""

import argparse
import asyncio
import base64
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from tornado.websocket import websocket_connect   # installed with streamlit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TABS = ["Overview", "Explore a Well", "Model Results", "Data Quality"]
PORT = 9333
MAP_CENTER, MAP_ZOOM = (31.45, 74.33), 9.1          # must match app/dashboard.py


class Page:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def call(self, method, **params):
        self.n += 1
        await self.ws.write_message(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            m = json.loads(await self.ws.read_message())
            if m.get("id") == self.n:
                return m.get("result", {})

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    async def wait_text(self, text, timeout=60):
        for _ in range(timeout * 2):
            if text in (await self.js("document.body.innerText") or ""):
                return True
            await asyncio.sleep(0.5)
        return False

    async def click(self, x, y):
        for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
            await self.call("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1)
            await asyncio.sleep(0.1)

    async def click_tab(self, i):
        x, y = await self.js(f"(() => {{ const r = document.querySelectorAll('[role=tab]')[{i}]"
                             ".getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; })()")
        await self.click(x, y)

    async def settle(self, seconds=4):
        await asyncio.sleep(1)
        for _ in range(60):   # wait while Streamlit shows its "running" indicator
            if not await self.js("!!document.querySelector('[data-testid=\"stStatusWidget\"]')"):
                break
            await asyncio.sleep(0.5)
        await asyncio.sleep(seconds)

    async def shot(self, path, width):
        h = await self.js("Math.max(...[...document.querySelectorAll('[data-testid=\"stMainBlockContainer\"]')]"
                          ".map(e => e.getBoundingClientRect().bottom + 80), 900)")
        await self.call("Emulation.setDeviceMetricsOverride", width=width, height=int(h), deviceScaleFactor=1,
                        mobile=width < 700)
        await asyncio.sleep(2.5)
        r = await self.call("Page.captureScreenshot", format="png")
        path.write_bytes(base64.b64decode(r["data"]))


def map_pixel(lat, lon, w, h):
    """deck.gl Web Mercator: offset of (lat, lon) from the map centre, in CSS pixels."""
    scale = 512 * 2 ** MAP_ZOOM / (2 * math.pi)

    def proj(la, lo):
        return scale * math.radians(lo), -scale * math.log(math.tan(math.pi / 4 + math.radians(la) / 2))
    cx, cy = proj(*MAP_CENTER)
    x, y = proj(lat, lon)
    return w / 2 + (x - cx), h / 2 + (y - cy)


async def run(url, width, out, map_check):
    out.mkdir(parents=True, exist_ok=True)
    for _ in range(40):
        try:
            tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
            break
        except OSError:
            await asyncio.sleep(0.5)
    ws = await websocket_connect([t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"],
                                 max_message_size=100_000_000)
    p = Page(ws)
    await p.call("Page.enable")
    await p.call("Emulation.setDeviceMetricsOverride", width=width, height=1000, deviceScaleFactor=1,
                 mobile=width < 700)
    await p.call("Page.navigate", url=url)
    if not await p.wait_text("Data Quality"):
        raise SystemExit("dashboard did not load; is `streamlit run app/dashboard.py` running?")
    await p.settle(6)
    suffix = "" if width >= 700 else "_mobile"
    report = {}
    for i, name in enumerate(TABS):
        await p.click_tab(i)
        await p.settle()
        errors = await p.js("[...document.querySelectorAll('[data-testid=\"stException\"]')].length")
        path = out / f"{i + 1}_{name.lower().replace(' ', '_')}{suffix}.png"
        await p.shot(path, width)
        report[name] = {"file": str(path.relative_to(ROOT)), "errors_on_page": errors}
        await p.call("Emulation.setDeviceMetricsOverride", width=width, height=1000, deviceScaleFactor=1,
                     mobile=width < 700)
    if width >= 700 and map_check:
        report["map_click"] = await click_map(p)
    print(json.dumps(report, indent=1))


async def click_map(p, target="LHR-155"):
    """Click a well on the Explore map and read which well the page shows afterwards."""
    import pandas as pd
    wells = pd.read_csv(ROOT / "data" / "processed" / "lahore_wells.csv").set_index("well_id")
    await p.click_tab(1)
    await p.settle()
    before = await p.js("[...document.querySelectorAll('h4')].filter(e => e.offsetParent)[0].innerText")
    box = await p.js("(() => { const els = [...document.querySelectorAll('[data-testid=\"stDeckGlJsonChart\"]')]"
                     ".filter(e => e.getBoundingClientRect().width > 0); const r = els[0].getBoundingClientRect();"
                     " return [r.left, r.top, r.width, r.height]; })()")
    x, y = map_pixel(wells.loc[target, "lat"], wells.loc[target, "lon"], box[2], box[3])
    await p.click(box[0] + x, box[1] + y)
    await p.settle(5)
    after = await p.js("[...document.querySelectorAll('h4')].filter(e => e.offsetParent)[0].innerText")
    return {"clicked": target, "heading_before": before.strip(), "heading_after": after.strip(),
            "selection_worked": after.startswith(target)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--public", action="store_true", help="screenshot the public aggregate-only mode")
    a = ap.parse_args()
    server = None
    if a.public:
        a.url = "http://localhost:8502"
        server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "dashboard.py"),
                                   "--server.headless", "true", "--server.port", "8502"],
                                  env={**os.environ, "LGO_PUBLIC_MODE": "1"}, cwd=ROOT,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            try:
                urllib.request.urlopen(a.url + "/_stcore/health")
                break
            except OSError:
                time.sleep(0.5)
    out = OUT / "public" if a.public else OUT
    with tempfile.TemporaryDirectory() as profile:
        chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
                                   f"--user-data-dir={profile}", "--use-angle=swiftshader",
                                   "--enable-unsafe-swiftshader", "--hide-scrollbars", "about:blank"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            asyncio.run(run(a.url, a.width, out, map_check=not a.public))
        finally:
            chrome.terminate()
            if server:
                server.terminate()
            time.sleep(1)


if __name__ == "__main__":
    main()
