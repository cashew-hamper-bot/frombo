"""Rendering performance harness.

Times ``paintGL`` directly (no reliance on the app's repaint
timer) across a matrix of camera scenarios x feature toggles,
using the widget's ``disabled_passes`` instrumentation to turn
the sky / blocks / bedrock / sea passes off one by one.  Each
cell reports mean / median / p95 frame milliseconds and the
implied frame rate; the cost of a pass is the difference from
the matching baseline.

Timing is wall clock around ``paintGL`` + ``glFinish``, measured
from outside the app -- the harness adds no timers to the app.

Run under the same Xvfb/software-GL environment as the smoke
render:

    xvfb-run -a python tools/perf_harness.py [--samples 4]
    xvfb-run -a python tools/perf_harness.py --samples 0

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
from PySide6.QtGui import QSurfaceFormat  # noqa: E402
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
    "flat-shading": (frozenset(), 0.0),
    "bare": (frozenset({"sky", "blocks", "bedrock", "sea"}), 1.0),
}


def time_frames(gl_widget, frames: int) -> list[float]:
    out = []
    gl_widget.makeCurrent()
    for _ in range(frames):
        t0 = time.perf_counter()
        gl_widget.paintGL()
        gl.glFinish()
        out.append((time.perf_counter() - t0) * 1000.0)
    gl_widget.doneCurrent()
    return out


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))]


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

    print(f"MSAA samples: {args.samples}; frames/cell: {args.frames} "
          f"(+{args.warmup} warmup); window 1280x800")
    for scen_name, (pos, yaw, pitch) in SCENARIOS.items():
        if pos is None:
            cam.position = spawn[0].copy()
            cam.yaw, cam.pitch = spawn[1], spawn[2]
        else:
            cam.position = np.array(pos, dtype=float)
            cam.yaw, cam.pitch = math.radians(yaw), math.radians(pitch)
        print(f"\nscenario: {scen_name}")
        base_mean = None
        for cfg_name, (disabled, smooth) in CONFIGS.items():
            glw.disabled_passes = disabled
            glw._smooth = smooth
            time_frames(glw, args.warmup)
            times = time_frames(glw, args.frames)
            mean = statistics.fmean(times)
            med = statistics.median(times)
            p95 = pct(times, 0.95)
            if base_mean is None:
                base_mean = mean
                delta = ""
            else:
                delta = f"  delta {base_mean - mean:+7.2f} ms vs baseline"
            print(f"  {cfg_name:12s} mean {mean:7.2f} ms  median {med:7.2f}  "
                  f"p95 {p95:7.2f}  ({1000.0 / mean:6.1f} fps){delta}")
    glw.disabled_passes = frozenset()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
