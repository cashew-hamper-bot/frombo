"""Viewpoint-interaction performance harness.

Drives the real RDWidget with synthetic mouse-event streams and
reports the numbers that "rotation feels jerky" has to answer to:

- per-frame rotation increments during a constant-velocity
  captured-look stream: mean / std / min / max (a smooth stream
  has std near zero; steps and dropouts show up immediately);
- total rotation vs the analytic expectation;
- glued-point drift over a constant-velocity grab-the-world
  drag, and the per-event step sizes along it.

Run under the same Xvfb/software-GL environment as the smoke
render (it needs a real window for cursor warping):

    xvfb-run -a python tools/look_harness.py
"""

from __future__ import annotations

import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent, QSurfaceFormat  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from rhombic_game.app import FOV_Y, TAN_HALF_FOV, MainWindow  # noqa: E402
from rhombic_game.math3d import ray_direction  # noqa: E402


def _move_event(gl_pos: QPointF, local_pos: QPointF, buttons) -> QMouseEvent:
    return QMouseEvent(
        QEvent.Type.MouseMove,
        local_pos,
        gl_pos,
        Qt.MouseButton.NoButton,
        buttons,
        Qt.KeyboardModifier.NoModifier,
    )


def _report(name: str, steps: list[float]) -> None:
    mean = statistics.fmean(steps)
    std = statistics.pstdev(steps) if len(steps) > 1 else 0.0
    print(
        f"  {name}: n={len(steps)} mean={math.degrees(mean):8.4f} deg "
        f"std={math.degrees(std):8.4f} min={math.degrees(min(steps)):8.4f} "
        f"max={math.degrees(max(steps)):8.4f} cv={std / mean if mean else 0.0:6.3f}"
    )


def scenario_capture(gl, app) -> None:
    """Constant-velocity capture stream, events batched per frame."""
    print("capture: 480 events x 6 px right, delivered 8 per frame x 60 frames")
    gl.camera.yaw, gl.camera.pitch = 0.0, 0.0
    gl._set_captured(True)
    app.processEvents()
    center = QPointF(gl.mapToGlobal(gl.rect().center()))
    per_frame: list[float] = []
    prev_yaw = gl.camera.yaw
    total_dx = 0.0
    for _frame in range(60):
        for _ in range(8):
            # The widget warps the pointer back to centre after
            # every event, so each real event arrives at centre +
            # one step of fresh motion -- model exactly that.
            pos = QPointF(center.x() + 6.0, center.y())
            total_dx += 6.0
            gl.mouseMoveEvent(
                _move_event(pos, gl.mapFromGlobal(pos.toPoint()).toPointF(),
                            Qt.MouseButton.NoButton)
            )
        gl.makeCurrent()
        gl.paintGL()
        gl.doneCurrent()
        per_frame.append(abs(gl.camera.yaw - prev_yaw))
        prev_yaw = gl.camera.yaw
    gl._set_captured(False)
    k = 2.0 * TAN_HALF_FOV / gl.height()
    expected = k * total_dx  # pitch 0: dyaw total = k * total dx
    got = abs(gl.camera.yaw)
    print(f"  total yaw: got {math.degrees(got):.3f} deg, "
          f"analytic {math.degrees(expected):.3f} deg, "
          f"ratio {got / expected:.4f}")
    _report("per-frame |dyaw|", per_frame)


def scenario_drag(gl, app) -> None:
    """Straight-line grab-the-world drag; measure glued drift."""
    print("drag: (200, 300) -> (824, 460) in 2 px steps, left button held")
    gl.camera.yaw, gl.camera.pitch = 0.7, -0.3
    w, h = float(gl.width()), float(gl.height())

    def ndc(p: QPointF) -> tuple[float, float]:
        return (2.0 * p.x() / w - 1.0, 1.0 - 2.0 * p.y() / h)

    def direction_at(p: QPointF) -> np.ndarray:
        inv = np.linalg.inv(gl.camera.view_projection(FOV_Y, w / h, 0.1))
        return ray_direction(inv, *ndc(p))

    start = QPointF(200.0, 300.0)
    end = QPointF(824.0, 460.0)
    target = direction_at(start)
    gl._last_mouse = QPointF(start)
    steps = 312
    per_event: list[float] = []
    prev = (gl.camera.yaw, gl.camera.pitch)
    for s in range(1, steps + 1):
        f = s / steps
        p = QPointF(
            start.x() + (end.x() - start.x()) * f,
            start.y() + (end.y() - start.y()) * f,
        )
        gl.mouseMoveEvent(
            _move_event(gl.mapToGlobal(p.toPoint()).toPointF(), p,
                        Qt.MouseButton.LeftButton)
        )
        per_event.append(
            abs(gl.camera.yaw - prev[0]) + abs(gl.camera.pitch - prev[1])
        )
        prev = (gl.camera.yaw, gl.camera.pitch)
    got = direction_at(end)
    drift = math.degrees(math.acos(float(np.clip(got @ target, -1, 1))))
    print(f"  glued-point drift over the drag: {drift:.4f} deg")
    _report("per-event |dyaw|+|dpitch|", per_event)


def main() -> int:
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1280, 800)
    window.show()
    app.processEvents()
    gl = window._gl
    gl.makeCurrent()
    gl.paintGL()
    gl.doneCurrent()

    scenario_capture(gl, app)
    scenario_drag(gl, app)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
