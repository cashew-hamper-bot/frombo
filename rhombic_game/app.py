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
from PySide6.QtGui import QCursor, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow

from . import planes, shaders
from .lattice import (
    DEFAULT_S,
    PATCH111_OFFSET,
    PATCH111_ROTATION,
    build_cells_mesh,
    island_cells_100,
    island_cells_111,
)
from .math3d import Camera, drag_glue

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
        self._boost = 1.0  # Shift-held speed multiplier (set in _move)
        self._captured = False  # Minecraft-like mouse capture (FPS look)
        self._skip_move = False  # swallow the recenter synthetic move
        self._press_pos = None  # for click-vs-drag detection
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
        self._checker_tex = self._make_pattern_texture()
        self._noise_tex = self._make_noise_texture()

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

        # Two islands, one merged mesh each (patch + walls + column),
        # so smooth normals and ambient occlusion see across the
        # joins: the {100} cleavage at the origin and the {111}
        # cleavage 75 m ahead, for side-by-side comparison.
        self._patches = [
            self._vao_from_mesh(
                *build_cells_mesh(island_cells_100(), DEFAULT_S, np.eye(3), (0.0, 0.0, 0.0))
            ),
            self._vao_from_mesh(
                *build_cells_mesh(
                    island_cells_111(), DEFAULT_S, PATCH111_ROTATION, PATCH111_OFFSET
                )
            ),
        ]

        self._bedrock = _DynamicStrip()
        self._sea = _DynamicStrip()
        self._clock.start()

    @staticmethod
    def _upload_texture(rgba: np.ndarray) -> int:
        """Upload an RGBA float image as a mipmapped, aniso texture."""
        n_z, n_x = rgba.shape[:2]
        data = np.ascontiguousarray((np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8))
        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, n_x, n_z, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        # EXT_texture_filter_anisotropic (0x84FE / 0x84FF); near-
        # universally available, but degrade to plain mipmaps if not.
        try:
            max_aniso = float(gl.glGetFloatv(0x84FF))
            gl.glTexParameterf(gl.GL_TEXTURE_2D, 0x84FE, min(16.0, max_aniso))
        except Exception:
            pass
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return tex

    @classmethod
    def _make_pattern_texture(cls) -> int:
        """The planes' pattern texture: a rectangular tile (42 m x
        24.25 m) with hexagonal symmetry.

        RGB bakes the bedrock look: flat-top small hexagons (points
        east/west, matching the sun), 3-coloured with a per-hex tint,
        grouped into big hexagons whose centre spacing is 7 small
        spacings -- 49 small hexes by area; a count of the lattice
        gives 50 small centres per big hex, ~38 fully inside and 12
        exactly bisected by the parent edge.  Alpha holds the bare
        2x2 checker factor for the sea.  Hexagonal tilings have no
        square period, hence the rectangular tile; its edges were
        verified to match exactly.
        """
        a_hex = 2.0
        sq3 = math.sqrt(3.0)
        tile_x, tile_z = 42.0, 7.0 * sq3 * a_hex
        nx, nz = 1344, 776

        def cube_round(q, r):
            s = -q - r
            qi, ri, si = np.round(q), np.round(r), np.round(s)
            dq, dr, ds = np.abs(qi - q), np.abs(ri - r), np.abs(si - s)
            fq = (dq > dr) & (dq > ds)
            fr = (~fq) & (dr > ds)
            qi = np.where(fq, -ri - si, qi)
            ri = np.where(fr, -qi - si, ri)
            return qi, ri

        def hex_sdf(vx, vz, apothem):
            return np.maximum(
                np.abs(vz),
                np.maximum(
                    np.abs(0.8660254 * vx + 0.5 * vz),
                    np.abs(0.8660254 * vx - 0.5 * vz),
                ),
            ) - apothem

        xs = (np.arange(nx) + 0.5) * (tile_x / nx)
        zs = (np.arange(nz) + 0.5) * (tile_z / nz)
        x, z = np.meshgrid(xs, zs)
        q = (2.0 / 3.0 * x) / a_hex
        r = (-1.0 / 3.0 * x + sq3 / 3.0 * z) / a_hex
        qi, ri = cube_round(q, r)
        cx = a_hex * 1.5 * qi
        cz = a_hex * sq3 * (ri + qi / 2.0)
        sdf_s = hex_sdf(x - cx, z - cz, a_hex * sq3 / 2.0)
        palette = np.array(
            [[0.44, 0.42, 0.40], [0.50, 0.48, 0.45], [0.39, 0.37, 0.35]]
        )
        rgb = palette[(qi - ri).astype(int) % 3]
        hsh = ((qi.astype(np.int64) * 73856093) ^ (ri.astype(np.int64) * 19349663)) & 0xFFFF
        rgb = rgb * (0.94 + 0.12 * (hsh / 0xFFFF))[..., None]
        edge = np.clip((sdf_s + 0.14) / 0.14, 0.0, 1.0)
        rgb = rgb * (1 - edge[..., None] * 0.45) + np.array([0.20, 0.19, 0.18]) * (edge[..., None] * 0.45)
        big_q, big_r = cube_round((q - 0.5) / 7.0, r / 7.0)
        q_c, r_c = 7.0 * big_q + 0.5, 7.0 * big_r
        bcx = a_hex * 1.5 * q_c
        bcz = a_hex * sq3 * (r_c + q_c / 2.0)
        sdf_b = hex_sdf(x - bcx, z - bcz, 7.0 * sq3 * a_hex / 2.0)
        bedge = np.clip((sdf_b + 0.45) / 0.45, 0.0, 1.0)
        rgb = rgb * (1 - bedge[..., None] * 0.62) + np.array([0.13, 0.12, 0.11]) * (bedge[..., None] * 0.62)
        alpha = ((np.floor(2 * x / tile_x) + np.floor(2 * z / tile_z)) % 2)
        rgba = np.concatenate([rgb, alpha[..., None]], axis=2)
        return cls._upload_texture(rgba)

    @classmethod
    def _make_noise_texture(cls) -> int:
        """Tileable value noise (two octaves), 256 px, grey in RGB.

        Sampled at a 500 m period by both plane shaders for
        large-scale colour variation toward the horizon.
        """
        rng = np.random.default_rng(20261029)
        n = 256
        out = np.zeros((n, n))
        for cells, weight in ((16, 0.65), (32, 0.35)):
            grid = rng.random((cells, cells))
            fy = np.arange(n) * cells / n
            iy = np.floor(fy).astype(int)
            ty = fy - iy
            ty = ty * ty * (3 - 2 * ty)
            fx = np.arange(n) * cells / n
            ix = np.floor(fx).astype(int)
            tx = fx - ix
            tx = tx * tx * (3 - 2 * tx)
            v00 = grid[np.ix_(iy % cells, ix % cells)]
            v01 = grid[np.ix_(iy % cells, (ix + 1) % cells)]
            v10 = grid[np.ix_((iy + 1) % cells, ix % cells)]
            v11 = grid[np.ix_((iy + 1) % cells, (ix + 1) % cells)]
            top = v00 * (1 - tx)[None, :] + v01 * tx[None, :]
            bot = v10 * (1 - tx)[None, :] + v11 * tx[None, :]
            out += weight * (top * (1 - ty)[:, None] + bot * ty[:, None])
        rgba = np.stack([out, out, out, np.ones_like(out)], axis=2)
        return cls._upload_texture(rgba)

    @classmethod
    def _make_patch_vao(cls, patch):
        return cls._vao_from_mesh(*patch.build_mesh())

    @staticmethod
    def _vao_from_mesh(verts, idx):
        vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)
        ebo = gl.glGenBuffers(1)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, gl.GL_STATIC_DRAW)
        stride = 13 * 4
        for loc, offset, size in ((0, 0, 3), (1, 12, 3), (2, 24, 3), (3, 36, 3), (4, 48, 1)):
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(
                loc, size, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.GLvoidp(offset)
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
        # Hold Shift for a 10x speed boost -- the travel gesture for
        # crossing the gaps between islands (and, later, real maps).
        self._boost = 10.0 if self._pressed(Qt.Key.Key_Shift) else 1.0
        if np.any(move):
            cam.position = cam.position + move * (cam.move_speed * self._boost * dt)

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
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._checker_tex)
        gl.glUniform1i(gl.glGetUniformLocation(self._bedrock_prog, "uTex"), 0)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._noise_tex)
        gl.glUniform1i(gl.glGetUniformLocation(self._bedrock_prog, "uNoise"), 1)
        gl.glActiveTexture(gl.GL_TEXTURE0)
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
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._checker_tex)
        gl.glUniform1i(gl.glGetUniformLocation(self._sea_prog, "uTex"), 0)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._noise_tex)
        gl.glUniform1i(gl.glGetUniformLocation(self._sea_prog, "uNoise"), 1)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        self._sea.draw()
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)

        self._frame += 1
        if self.on_hud is not None and self._frame % 6 == 0:
            p = cam.position
            boost = "  x10 BOOST" if self._boost > 1.0 else ""
            mode = "  MOUSE CAPTURED (Esc to release)" if self._captured else ""
            self.on_hud(
                f"pos ({p[0]:7.1f}, {p[1]:6.1f}, {p[2]:7.1f})   "
                f"speed {cam.move_speed * self._boost:5.1f} m/s{boost}   "
                f"yaw {math.degrees(cam.yaw):6.1f} deg   pitch {math.degrees(cam.pitch):6.1f} deg{mode}"
            )

    # -- input ----------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            # Esc always leaves mouse capture first; only a second
            # Esc (while uncaptured) quits.
            if self._captured:
                self._set_captured(False)
            else:
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

    def _set_captured(self, on: bool) -> None:
        self._captured = on
        if on:
            self.grabMouse()
            self.setCursor(Qt.CursorShape.BlankCursor)
            center = self.mapToGlobal(self.rect().center())
            self._skip_move = True
            QCursor.setPos(center)
        else:
            self.releaseMouse()
            self.unsetCursor()

    def mousePressEvent(self, event) -> None:
        if self._captured:
            # A click while captured releases back to grab-the-world.
            self._set_captured(False)
            super().mousePressEvent(event)
            return
        self._last_mouse = event.position()
        self._press_pos = event.position()
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

        drag_glue(self.camera, FOV_Y, w / h, NEAR, ndc(self._last_mouse), ndc(new_pos))

    def mouseMoveEvent(self, event) -> None:
        if self._captured:
            if self._skip_move:
                self._skip_move = False
                super().mouseMoveEvent(event)
                return
            center = self.mapToGlobal(self.rect().center())
            delta = event.globalPosition() - center
            if delta.x() != 0.0 or delta.y() != 0.0:
                # Same angular scale as grab-the-world drag at the
                # screen centre, FPS sense (Minecraft-like).
                w = max(1.0, float(self.width()))
                h = max(1.0, float(self.height()))
                tan_y = math.tan(FOV_Y / 2.0)
                self.camera.yaw -= float(delta.x()) * (2.0 * tan_y * (w / h) / w)
                self.camera.pitch = float(
                    np.clip(
                        self.camera.pitch - float(delta.y()) * (2.0 * tan_y / h),
                        -1.55,
                        1.55,
                    )
                )
                self._skip_move = True
                QCursor.setPos(center)
            super().mouseMoveEvent(event)
            return
        if self._last_mouse is not None and (
            event.buttons() & Qt.MouseButton.LeftButton
        ):
            self._drag_glue(event.position())
        self._last_mouse = event.position()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        # A click (press + release without dragging) toggles the
        # Minecraft-like mouse capture.
        if (
            not self._captured
            and self._press_pos is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            moved = event.position() - self._press_pos
            if moved.manhattanLength() < 5.0:
                self._set_captured(True)
        self._press_pos = None
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
