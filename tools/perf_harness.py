"""Rendering performance harness: full-loop frame times.

Each measured frame is one full trip around the app's real
loop, driven the way the app is actually driven now that
repaints are event-driven:

    synthetic gesture (changes the view) -> update() ->
    Qt delivers paintGL -> glFinish

The start and end timestamps are captured on the lines
immediately before and after that work -- nothing of the
harness's own bookkeeping sits inside the timed region (the
statistics are computed from the recorded durations afterwards).
Because every frame's gesture moves the viewpoint, every render
is of a changed scene.  FPS is reported alongside milliseconds:
frames / total loop time.

Configurations toggle paintGL passes via the widget's
``disabled_passes`` instrumentation (sky / blocks / bedrock /
sea / terrain), one at a time, so a pass's cost is its delta
from the matching baseline.  Run under the same
Xvfb/software-GL environment as the smoke render:

    xvfb-run -a python tools/perf_harness.py [--samples 4]

Absolute numbers under software GL are CPU-rasterisation
numbers; the *relative* pass costs are the signal, and the same
harness on a real GPU gives real ones.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
from OpenGL import GL as gl  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent, QSurfaceFormat  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from rhombic_game.app import MainWindow  # noqa: E402

SCENARIOS = {
    # name: (position, yaw_deg, pitch_deg)
    "ground": (None, None, None),  # the app's default spawn view
    "altitude": ((0.0, 400.0, 20.0), 0.0, -3.0),
    "steep-down": ((0.0, 7.0, 20.0), 0.0, -80.0),
}

CONFIGS = {
    # name: (disabled_passes, smooth)
    "baseline": (frozenset(), 1.0),
    "no-sky": (frozenset({"sky"}), 1.0),
    "no-blocks": (frozenset({"blocks"}), 1.0),
    "no-bedrock": (frozenset({"bedrock"}), 1.0),
    "no-sea": (frozenset({"sea"}), 1.0),
    "no-terrain": (frozenset({"terrain"}), 1.0),
    "flat-shading": (frozenset(), 0.0),
    "bare": (frozenset({"sky", "blocks", "bedrock", "sea", "terrain"}), 1.0),
}


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))]


def run_cell(glw, app, frames: int, warmup: int) -> list[float]:
    """Full-loop times (ms) for a drag gesture that keeps the
    view rotating: a slow circle, so no pitch clamp engages and
    every frame's scene differs from the last."""
    cx, cy, radius = 640.0, 400.0, 260.0
    total_steps = warmup + frames
    durations: list[float] = []
    prev_frame = glw._frame
    glw._last_mouse = QPointF(cx + radius, cy)
    for step in range(total_steps):
        ang = 2.0 * math.pi * step / 240.0
        pos = QPointF(cx + radius * math.cos(ang), cy + radius * math.sin(ang))
        event = QMouseEvent(
            QEvent.Type.MouseMove,
            pos,
            glw.mapToGlobal(pos.toPoint()).toPointF(),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        t0 = time.perf_counter()
        glw.mouseMoveEvent(event)
        app.processEvents()
        glw.makeCurrent()
        gl.glFinish()
        glw.doneCurrent()
        t1 = time.perf_counter()
        if step >= warmup:
            durations.append((t1 - t0) * 1000.0)
    glw._last_mouse = None
    rendered = glw._frame - prev_frame
    if rendered < frames:
        print(f"    warning: only {rendered}/{frames} frames were painted")
    return durations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=4, help="MSAA samples")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(args.samples)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    app.processEvents()
    glw = window._gl
    cam = glw.camera
    spawn = (cam.position.copy(), cam.yaw, cam.pitch)

    print(f"MSAA samples: {args.samples}; full-loop frames/cell: {args.frames} "
          f"(+{args.warmup} warmup); window 1280x800")
    for scen_name, (pos, yaw, pitch) in SCENARIOS.items():
        print(f"\nscenario: {scen_name}")
        base_mean = None
        for cfg_name, (disabled, smooth) in CONFIGS.items():
            if pos is None:
                cam.position = spawn[0].copy()
                cam.yaw, cam.pitch = spawn[1], spawn[2]
            else:
                cam.position = np.array(pos, dtype=float)
                cam.yaw, cam.pitch = math.radians(yaw), math.radians(pitch)
            glw.disabled_passes = disabled
            glw._smooth = smooth
            times = run_cell(glw, app, args.frames, args.warmup)
            mean = statistics.fmean(times)
            med = statistics.median(times)
            p95 = pct(times, 0.95)
            fps = len(times) * 1000.0 / sum(times)
            if base_mean is None:
                base_mean = mean
                delta = ""
            else:
                delta = f"  delta {base_mean - mean:+7.2f} ms vs baseline"
            print(f"  {cfg_name:12s} mean {mean:7.2f} ms  median {med:7.2f}  "
                  f"p95 {p95:7.2f}  ({fps:6.1f} fps){delta}")
    glw.disabled_passes = frozenset()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
