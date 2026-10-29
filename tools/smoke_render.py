"""Render a few frames offscreen and save a PNG.

Useful for eyeballing renderer changes without a display:

    QT_QPA_PLATFORM=offscreen python tools/smoke_render.py out.png \
        [--pos X Y Z] [--yaw DEG] [--pitch DEG]

Requires PySide6 + a (possibly software) OpenGL implementation.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtGui import QSurfaceFormat  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from rhombic_game.app import MainWindow  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pos", nargs=3, type=float, default=None)
    ap.add_argument("--yaw", type=float, default=None)
    ap.add_argument("--pitch", type=float, default=None)
    args = ap.parse_args()

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)  # match the app's MSAA, so renders judge aliasing fairly
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = MainWindow()
    cam = window._gl.camera
    if args.pos is not None:
        import numpy as np

        cam.position = np.array(args.pos, dtype=float)
    if args.yaw is not None:
        import math

        cam.yaw = math.radians(args.yaw)
    if args.pitch is not None:
        import math

        cam.pitch = math.radians(args.pitch)
    window.show()

    def grab() -> None:
        image = window._gl.grabFramebuffer()
        image.save(args.out)
        print(f"saved {args.out} ({image.width()}x{image.height()})")
        app.quit()

    QTimer.singleShot(900, grab)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
