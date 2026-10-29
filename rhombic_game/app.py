"""PySide6 + OpenGL application: the first playable slice.

Scene, in draw order:

1. Static daytime skybox at infinity (direction vertices, w = 0),
   depth-write off, depth function LEQUAL -- it lands exactly on the
   far plane of the infinite projection.
2. One layer of rhombic dodecahedron cells, roughly 30 m x 30 m,
   static and deliberately unoptimised (one merged VBO).
3. The complete infinite bedrock plane at y = -100 (4-vertex strip).
4. The infinite sea plane at y = -10 with a ~1000 m hole around the
   player (rim/horizon strip), blended.

Navigation: WASD or IJKL (both live at once, IJKL being the
right-hand set for left-mousers) move in the current horizontal
plane, Space/C for up/down, grab-the-world dragging with the primary
mouse button to look, wheel adjusts fly speed, G toggles flat vs
Gouraud shading on the ground patch.
"""

from __future__ import annotations

import math
import sys

import numpy as np
from OpenGL import GL as gl
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow

from . import planes, shaders
from .lattice import GroundPatch, GroundPatch111
from .math3d import Camera, ray_direction

FOV_Y = math.radians(70.0)
NEAR = 0.1


def _compile(kind: int, source: str) -> int:
    shader = gl.glCreateShader(kind)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)
    if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
        raise RuntimeError(gl.glGetShaderInfoLog(shader).decode())
    return shader


def _program(vert: str, frag: str) -> int:
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, _compile(gl.GL_VERTEX_SHADER, vert))
    gl.glAttachShader(prog, _compile(gl.GL_FRAGMENT_SHADER, frag))
    gl.glLinkProgram(prog)
    if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
        raise RuntimeError(gl.glGetProgramInfoLog(prog).decode())
    return prog


def _sun() -> np.ndarray:
    v = np.array(shaders.SUN_DIR, dtype=np.float32)
    return v / np.linalg.norm(v)


class _DynamicStrip:
    """A VAO/VBO pair re-uploaded each frame (a handful of vertices)."""

    def __init__(self) -> None:
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        self.count = 0
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glBindVertexArray(0)

    def upload(self, verts: np.ndarray) -> None:
        self.count = len(verts)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_DYNAMIC_DRAW)

    def draw(self) -> None:
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, self.count)
        gl.glBindVertexArray(0)


class RDWidget(QOpenGLWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.camera = Camera()
        self._smooth = 1.0  # Gouraud shading on the ground patch (G toggles)
        self._keys: set[int] = set()
        self._last_mouse = None
        self._clock = QElapsedTimer()
        self.on_hud = None  # callable(str), set by the window
        self._frame = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(16)

    # -- GL setup -----------------------------------------------------

    def initializeGL(self) -> None:
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(*shaders.HAZE_COLOR, 1.0)
        self._sky_prog = _program(shaders.SKY_VERT, shaders.SKY_FRAG)
        self._bedrock_prog = _program(shaders.PLANE_VERT, shaders.PLANE_FRAGS["bedrock"])
        self._sea_prog = _program(shaders.PLANE_VERT, shaders.PLANE_FRAGS["sea"])
        self._block_prog = _program(shaders.BLOCK_VERT, shaders.BLOCK_FRAG)

        # Skybox cube (directions).
        corners, idx = planes.skybox_cube()
        self._sky_count = len(idx)
        self._sky_vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)
        ebo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self._sky_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, corners.nbytes, corners, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glBindVertexArray(0)

        # Ground patches (static, unoptimised merged meshes): the
        # {100} cleavage at the origin and the {111} cleavage 75 m
        # ahead, for side-by-side comparison.
        self._patches = [
            self._make_patch_vao(GroundPatch()),
            self._make_patch_vao(GroundPatch111()),
        ]

        self._bedrock = _DynamicStrip()
        self._sea = _DynamicStrip()
        self._clock.start()

    @staticmethod
    def _make_patch_vao(patch):
        verts, idx = patch.build_mesh()
        vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)
        ebo = gl.glGenBuffers(1)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, gl.GL_STATIC_DRAW)
        stride = 12 * 4
        for loc, offset in ((0, 0), (1, 12), (2, 24), (3, 36)):
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(
                loc, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.GLvoidp(offset)
            )
        gl.glBindVertexArray(0)
        return vao, len(idx)

    # -- per-frame ----------------------------------------------------

    def _pressed(self, *keys: int) -> bool:
        return any(k in self._keys for k in keys)

    def _move(self, dt: float) -> None:
        cam = self.camera
        fwd = back = left = right = up = down = 0.0
        K = Qt.Key
        if self._pressed(K.Key_W, K.Key_I):
            fwd = 1.0
        if self._pressed(K.Key_S, K.Key_K):
            back = 1.0
        if self._pressed(K.Key_A, K.Key_J):
            left = 1.0
        if self._pressed(K.Key_D, K.Key_L):
            right = 1.0
        if self._pressed(K.Key_Space, K.Key_E, K.Key_U):
            up = 1.0
        if self._pressed(K.Key_C, K.Key_Q, K.Key_O):
            down = 1.0
        # Movement stays in the current horizontal plane: forward and
        # strafe follow the view azimuth only, never the pitch, so
        # looking down and pressing W does not descend.
        fxz = cam.forward_xz()
        forward3 = np.array([fxz[0], 0.0, fxz[1]])
        right3 = np.array([-fxz[1], 0.0, fxz[0]])
        move = (
            forward3 * (fwd - back)
            + right3 * (right - left)
            + np.array([0.0, 1.0, 0.0]) * (up - down)
        )
        if np.any(move):
            cam.position = cam.position + move * (cam.move_speed * dt)

    def paintGL(self) -> None:
        dt = min(self._clock.restart() / 1000.0, 0.05)
        self._move(dt)
        cam = self.camera
        dpr = self.devicePixelRatioF()
        w, h = max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr))
        gl.glViewport(0, 0, w, h)
        vp = cam.view_projection(FOV_Y, w / h, NEAR).astype(np.float32)
        sun = _sun()
        haze = np.array(shaders.HAZE_COLOR, dtype=np.float32)
        cam_pos = cam.position.astype(np.float32)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glDisable(gl.GL_CULL_FACE)

        # 1. Sky at infinity: depth exactly 1.0, drawn first, no depth write.
        gl.glDepthMask(gl.GL_FALSE)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glUseProgram(self._sky_prog)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._sky_prog, "uVP"), 1, gl.GL_TRUE, vp)
        gl.glUniform3fv(gl.glGetUniformLocation(self._sky_prog, "uSunDir"), 1, sun)
        gl.glBindVertexArray(self._sky_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self._sky_count, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDepthFunc(gl.GL_LESS)

        # 2. Local RD ground.
        gl.glUseProgram(self._block_prog)
        gl.glUniform1f(gl.glGetUniformLocation(self._block_prog, "uSmooth"), self._smooth)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._block_prog, "uVP"), 1, gl.GL_TRUE, vp)
        gl.glUniform3fv(gl.glGetUniformLocation(self._block_prog, "uSunDir"), 1, sun)
        gl.glUniform3fv(gl.glGetUniformLocation(self._block_prog, "uCamPos"), 1, cam_pos)
        gl.glUniform3fv(gl.glGetUniformLocation(self._block_prog, "uHazeColor"), 1, haze)
        for vao, count in self._patches:
            gl.glBindVertexArray(vao)
            gl.glDrawElements(gl.GL_TRIANGLES, count, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        # 3. Complete infinite bedrock plane (4 vertices).
        self._bedrock.upload(
            planes.full_plane_strip(
                cam.position[0],
                cam.position[2],
                planes.BEDROCK_Y,
                cam.forward_xz(),
                height=float(cam.position[1] - planes.BEDROCK_Y),
            )
        )
        gl.glUseProgram(self._bedrock_prog)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._bedrock_prog, "uVP"), 1, gl.GL_TRUE, vp)
        gl.glUniform3fv(gl.glGetUniformLocation(self._bedrock_prog, "uCamPos"), 1, cam_pos)
        gl.glUniform3fv(gl.glGetUniformLocation(self._bedrock_prog, "uHazeColor"), 1, haze)
        self._bedrock.draw()

        # 4. Infinite sea plane with the hole around the player.
        self._sea.upload(
            planes.hole_plane_strip(cam.position[0], cam.position[2], planes.SEA_Y)
        )
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glUseProgram(self._sea_prog)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._sea_prog, "uVP"), 1, gl.GL_TRUE, vp)
        gl.glUniform3fv(gl.glGetUniformLocation(self._sea_prog, "uCamPos"), 1, cam_pos)
        gl.glUniform3fv(gl.glGetUniformLocation(self._sea_prog, "uSunDir"), 1, sun)
        gl.glUniform3fv(gl.glGetUniformLocation(self._sea_prog, "uHazeColor"), 1, haze)
        self._sea.draw()
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)

        self._frame += 1
        if self.on_hud is not None and self._frame % 6 == 0:
            p = cam.position
            self.on_hud(
                f"pos ({p[0]:7.1f}, {p[1]:6.1f}, {p[2]:7.1f})   "
                f"speed {cam.move_speed:5.1f} m/s   "
                f"yaw {math.degrees(cam.yaw):6.1f} deg   pitch {math.degrees(cam.pitch):6.1f} deg"
            )

    # -- input ----------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.window().close()
            return
        if event.key() == Qt.Key.Key_G and not event.isAutoRepeat():
            self._smooth = 0.0 if self._smooth else 1.0
            return
        if not event.isAutoRepeat():
            self._keys.add(event.key())
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if not event.isAutoRepeat():
            self._keys.discard(event.key())
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event) -> None:
        self._last_mouse = event.position()
        self.setFocus()
        super().mousePressEvent(event)

    def _drag_glue(self, new_pos) -> None:
        """Grab-the-world dragging with exact 1:1 tracking at infinity.

        The scene follows the cursor, as in panorama viewers and map
        apps (the opposite sense from FPS mouse-look).  Rather than a
        fixed radians-per-pixel rate, each motion event rotates the
        camera by the exact rotation that carries the world direction
        currently under the cursor onto the direction that was under
        the cursor at the previous event -- so an object at infinity
        stays glued to the pointer at any screen position, centre or
        edge.  Two passes: the second mops up the sub-pixel residue
        of projecting the rotation back onto yaw/pitch.
        """
        w = max(1.0, float(self.width()))
        h = max(1.0, float(self.height()))

        def ndc(p):
            return (2.0 * p.x() / w - 1.0, 1.0 - 2.0 * p.y() / h)

        cam = self.camera
        for _ in range(2):
            inv = np.linalg.inv(cam.view_projection(FOV_Y, w / h, NEAR))
            a = ray_direction(inv, *ndc(new_pos))
            b = ray_direction(inv, *ndc(self._last_mouse))
            axis = np.cross(a, b)
            s = float(np.linalg.norm(axis))
            if s < 1e-12:
                break
            axis = axis / s
            c = float(np.clip(a @ b, -1.0, 1.0))
            f = cam.forward()
            f2 = f * c + np.cross(axis, f) * s + axis * float(axis @ f) * (1.0 - c)
            cam.yaw = float(np.arctan2(-f2[0], -f2[2]))
            cam.pitch = float(
                np.clip(np.arcsin(np.clip(f2[1], -1.0, 1.0)), -1.55, 1.55)
            )

    def mouseMoveEvent(self, event) -> None:
        if self._last_mouse is not None and (
            event.buttons() & Qt.MouseButton.LeftButton
        ):
            self._drag_glue(event.position())
        self._last_mouse = event.position()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._last_mouse = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        self.camera.move_speed = float(
            np.clip(self.camera.move_speed * (1.25**steps), 1.0, 240.0)
        )
        super().wheelEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Rhombic Dodecahedron World -- first slice")
        self.resize(1280, 800)
        self._gl = RDWidget(self)
        self.setCentralWidget(self._gl)
        self._hud = QLabel(self)
        self._hud.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._hud.setStyleSheet(
            "background: rgba(10, 14, 20, 150); color: #dfe9f2; "
            "padding: 6px 10px; font-family: monospace; font-size: 12px;"
        )
        self._hud.setText(
            "WASD / IJKL move (horizontal plane) - Space up, C down - "
            "drag world to look - wheel: speed - G: flat/Gouraud - Esc: quit"
        )
        self._hud.move(10, 10)
        self._hud.adjustSize()
        self._status = QLabel(self)
        self._status.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._status.setStyleSheet(self._hud.styleSheet())
        self._status.move(10, 44)
        self._gl.on_hud = self._set_status

    def _set_status(self, text: str) -> None:
        self._status.setText(text)
        self._status.adjustSize()


def main(argv: list[str] | None = None) -> int:
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(argv if argv is not None else sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
