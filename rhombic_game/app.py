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
from PySide6.QtCore import QElapsedTimer, QPointF, Qt
from PySide6.QtGui import QCursor, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
)

from . import planes, shaders, terrain
from .lattice import (
    DEFAULT_S,
    PATCH111_OFFSET,
    PATCH111_ROTATION,
    build_cells_mesh,
    island_cells_100,
    island_cells_111,
)
from .math3d import Camera, look_delta

FOV_Y = math.radians(70.0)
# The projection never changes, so its one derived constant for
# the look Jacobian is computed once, here (docs/look-jacobian.md).
TAN_HALF_FOV = math.tan(FOV_Y / 2.0)
NEAR_PLANE = 0.03  # near depth pass: 3 cm
DEPTH_SPLIT = 300.0  # metres: nominal near/far depth-pass boundary
SPLIT_OVERLAP = 30.0  # each projection reaches this far past the split
CLIP_SPLIT = DEPTH_SPLIT + SPLIT_OVERLAP / 2.0  # shared manual clip plane, 315 m
CLIP_HYSTERESIS = 0.5  # both passes draw within this band of the clip plane


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
        # Mouse tracking ON: without it, Qt only delivers move events
        # while a button is held -- which silently starved the
        # mouse-capture look (no button is held in capture mode).
        self.setMouseTracking(True)
        self.camera = Camera()
        self._smooth = 1.0  # Gouraud shading on the ground patch (G toggles)
        self._show_terrain = True  # distant terrain + horizon ring (T toggles)
        self._show_sea = True  # ocean surface (Y toggles)
        self._show_bedrock = True  # bedrock layer (options menu)
        self._show_far = True  # far depth pass, split..infinity (F toggles)
        self._show_near = True  # near depth pass, 3 cm..split (N toggles)
        self._show_boundary = False  # split-plane intersection lines (B toggles)
        self._haze_col = np.array(shaders.HAZE_COLOR, dtype=np.float32)
        self._boost = 1.0  # Shift-held speed multiplier (set in _move)
        self._look_size = None  # (w, h) cached by resizeGL for look math
        # Instrumentation for tools/perf_harness.py: names of
        # paintGL passes to skip ("sky", "blocks", "bedrock",
        # "sea", "terrain").  Empty in normal use; costs one
        # membership test per pass per frame.
        self.disabled_passes: frozenset = frozenset()
        self._captured = False  # Minecraft-like mouse capture (FPS look)
        self._last_global = None  # last pointer global pos (capture)
        self._look_pending = [0.0, 0.0]  # captured deltas awaiting a frame
        # (Minecraft mode: no drag-look state -- the mouse is
        # either captured for looking or free in the menu.)
        self._keys: set[int] = set()
        self._clock = QElapsedTimer()
        self.on_hud = None  # callable(str), set by the window
        self.on_escape_menu = None  # callable(), set by the window
        self._frame = 0
        # No repaint timer: frames are event-driven.  update() is
        # called at the moment something changes the view -- a key
        # or mouse event -- and paintGL re-requests one more frame
        # only while movement keys remain held.  A metronome would
        # insert its own jitter into that flow.  (The QElapsedTimer
        # above schedules nothing; paintGL reads it for dt.)

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

        # Distant hex-lattice terrain (opaque annulus, drawn with
        # the block program) and its statistical horizon ring
        # (drawn last, blended over the sea).
        self._terrain_vao = self._vao_from_mesh(*terrain.build_distant_terrain())
        # The sea's horizon opacity: sea fraction of the disc x
        # the sea's base opacity (0.86); see the sea shader.
        self._sea_horizon_alpha = terrain.terrain_stats()["sea_fraction"] * 0.86
        self._horizon_prog = _program(shaders.HORIZON_VERT, shaders.HORIZON_FRAG)
        hverts, hidx = terrain.build_horizon_fan()
        self._horizon_count = len(hidx)
        self._horizon_vao = gl.glGenVertexArrays(1)
        hvbo = gl.glGenBuffers(1)
        hebo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self._horizon_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, hvbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, hverts.nbytes, hverts, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, hebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, hidx.nbytes, hidx, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 32, None)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 32, gl.GLvoidp(16))
        gl.glBindVertexArray(0)
        # Debug boundary cylinder (split-radius overlay, key B):
        # same 8-float layout as the horizon ring, dynamic.
        self._debug_vao = gl.glGenVertexArrays(1)
        self._debug_vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self._debug_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._debug_vbo)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 32, None)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 32, gl.GLvoidp(16))
        gl.glBindVertexArray(0)
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
        exactly bisected by the parent edge.  Alpha holds the same
        hexagons as a shade factor -- the sea's pattern.  Hexagonal tilings have no
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
        c3 = (qi - ri).astype(int) % 3
        rgb = palette[c3]
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
        # Alpha is the SEA's hex factor: same hexagons, as a shade
        # value (3-colour step + per-hex jitter, edges darkened).
        alpha = 0.30 + 0.22 * c3 + (hsh / 0xFFFF - 0.5) * 0.10
        alpha = alpha * (1 - edge * 0.35) * (1 - bedge * 0.25)
        rgba = np.concatenate([rgb, alpha[..., None]], axis=2)
        return cls._upload_texture(rgba)

    @classmethod
    def _make_noise_texture(cls) -> int:
        """Tileable value noise (two octaves), 256 px, grey in RGB.

        Sampled at a 500 m period by both plane shaders for
        large-scale colour variation toward the horizon.
        """
        n = 256
        out = np.zeros((n, n))
        for cells, weight in ((16, 0.65), (32, 0.35)):
            # Deterministic hash grid (splitmix64) instead of
            # np.random: the noise is fixed either way, and this
            # keeps the app from importing numpy.random at all --
            # that submodule fails to load in at least one real
            # environment (Christopher's Python 3.10, 2026-10-29),
            # while core numpy works fine there.
            ii, jj = np.meshgrid(
                np.arange(cells), np.arange(cells), indexing="ij"
            )
            h = (
                ii.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
                ^ jj.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
                ^ np.uint64(0x165667B19E3779F9)
            )
            h ^= h >> np.uint64(29)
            h *= np.uint64(0xBF58476D1CE4E5B9)
            h ^= h >> np.uint64(32)
            grid = (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)
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

    def _look_params(self) -> tuple[float, float, float]:
        """(width, height, pixel scale k) for the look Jacobian.

        Cached by resizeGL; falls back to the live widget size if
        no resize has been delivered yet.
        """
        if self._look_size is not None:
            w, h = self._look_size
        else:
            w, h = max(1.0, float(self.width())), max(1.0, float(self.height()))
        return w, h, 2.0 * TAN_HALF_FOV / h

    def resizeGL(self, w: int, h: int) -> None:
        # Everything the look math needs that depends on the
        # screen size is precomputed here, once per resize --
        # never per mouse event (docs/look-jacobian.md).
        self._look_size = (max(1.0, float(w)), max(1.0, float(h)))
        super().resizeGL(w, h)

    def _apply_pending_look(self) -> None:
        """Apply the frame's accumulated captured-mouse rotation.

        The look Jacobian at the screen centre (where the warped
        cursor lives), FPS sense, integrated once per frame like
        WASD translation.
        """
        dx, dy = self._look_pending
        if dx == 0.0 and dy == 0.0:
            return
        self._look_pending = [0.0, 0.0]
        _, _, k = self._look_params()
        dyaw, dpitch = look_delta(self.camera.pitch, 0.0, 0.0, dx, dy, k)
        self.camera.yaw -= dyaw
        self.camera.pitch = float(
            np.clip(self.camera.pitch - dpitch, -1.55, 1.55)
        )

    def _draw_blocks(self, vp, cam_pos, sun, cam_fwd, clip_near=0.0, clip_far=0.0) -> None:
        """The local RD islands (block program), one merged mesh
        per island.  Drawn in both depth passes."""
        gl.glUseProgram(self._block_prog)
        gl.glUniform1f(gl.glGetUniformLocation(self._block_prog, "uSmooth"), self._smooth)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._block_prog, "uVP"), 1, gl.GL_TRUE, vp)
        gl.glUniform3fv(gl.glGetUniformLocation(self._block_prog, "uSunDir"), 1, sun)
        gl.glUniform3fv(gl.glGetUniformLocation(self._block_prog, "uCamPos"), 1, cam_pos)
        gl.glUniform3fv(gl.glGetUniformLocation(self._block_prog, "uCamFwd"), 1, cam_fwd)
        gl.glUniform1f(gl.glGetUniformLocation(self._block_prog, "uClipNear"), clip_near)
        gl.glUniform1f(gl.glGetUniformLocation(self._block_prog, "uClipFar"), clip_far)
        gl.glUniform3fv(gl.glGetUniformLocation(self._block_prog, "uHazeColor"), 1, self._haze_col)
        for vao, count in self._patches:
            gl.glBindVertexArray(vao)
            gl.glDrawElements(gl.GL_TRIANGLES, count, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

    def _draw_bedrock(self, vp, cam_pos, cam_fwd, clip_near=0.0, clip_far=0.0) -> None:
        """The complete infinite bedrock plane (4 vertices).

        An ordinary depth-writing surface, in both passes: the
        depth buffer decides visibility, and draw order is only
        the tie breaker (LEQUAL) for the asymptotic band at the
        horizon where several surfaces' depths converge on 1.0
        within a few depth quanta of each other.  (An earlier
        no-depth-write protocol for the "infinity family" dated
        from the single-pass era, when depth error at 10 km was
        ~62 m and writing this plane's depth meant writing
        noise; the two-pass split made its depth honest, and
        Christopher's ruling -- painter order is a tie breaker,
        not a replacement for the depth buffer -- retired it.)
        """
        cam = self.camera
        self._bedrock.upload(
            planes.full_plane_strip(
                cam.position[0],
                cam.position[2],
                planes.BEDROCK_Y,
                cam.forward_xz(),
                height=float(cam.position[1] - planes.BEDROCK_Y),
            )
        )
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glUseProgram(self._bedrock_prog)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._bedrock_prog, "uVP"), 1, gl.GL_TRUE, vp)
        gl.glUniform3fv(gl.glGetUniformLocation(self._bedrock_prog, "uCamPos"), 1, cam_pos)
        gl.glUniform1f(gl.glGetUniformLocation(self._bedrock_prog, "uPlaneY"), planes.BEDROCK_Y)
        gl.glUniform3fv(gl.glGetUniformLocation(self._bedrock_prog, "uCamFwd"), 1, cam_fwd)
        gl.glUniform1f(gl.glGetUniformLocation(self._bedrock_prog, "uClipNear"), clip_near)
        gl.glUniform1f(gl.glGetUniformLocation(self._bedrock_prog, "uClipFar"), clip_far)
        gl.glUniform3fv(gl.glGetUniformLocation(self._bedrock_prog, "uHazeColor"), 1, self._haze_col)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._checker_tex)
        gl.glUniform1i(gl.glGetUniformLocation(self._bedrock_prog, "uTex"), 0)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._noise_tex)
        gl.glUniform1i(gl.glGetUniformLocation(self._bedrock_prog, "uNoise"), 1)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        self._bedrock.draw()
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDepthFunc(gl.GL_LESS)

    def paintGL(self) -> None:
        dt = min(self._clock.restart() / 1000.0, 0.05)
        self._apply_pending_look()
        self._move(dt)
        cam = self.camera
        dpr = self.devicePixelRatioF()
        w, h = max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr))
        gl.glViewport(0, 0, w, h)
        # Two depth passes share the frame: the FAR pass (split
        # .. infinity, infinite projection with near = split) and
        # the NEAR pass (3 cm .. split, finite projection).  With
        # near = 0.1 and far = infinity, depth error at 10 km was
        # ~60 m -- the distant coastlines fought the sea.  At
        # near = 300 the same error is ~0.02 m.
        # The two projections OVERLAP by SPLIT_OVERLAP metres
        # around the split, and the shaders discard against a
        # shared manual clip plane inside the overlap (see the
        # block fragment shader): the hardware clips no longer
        # define the boundary, so no coverage gap can open
        # along it.  If this proves out, two separate depth
        # buffers become a simplicity option again.
        vp_far = cam.view_projection(FOV_Y, w / h, DEPTH_SPLIT - SPLIT_OVERLAP).astype(np.float32)
        vp_near = cam.view_projection(FOV_Y, w / h, NEAR_PLANE, DEPTH_SPLIT + SPLIT_OVERLAP).astype(np.float32)
        cam_fwd = (-cam.view_matrix()[2, :3]).astype(np.float32)
        clip_near = float(CLIP_SPLIT + CLIP_HYSTERESIS)
        clip_far = float(CLIP_SPLIT - CLIP_HYSTERESIS)
        vp = vp_far
        sun = _sun()
        # Below bedrock the sky palette turns nether-red (the sky
        # shader's uUnder fade).  There is no haze anywhere to tint.
        # The bedrock plane is the boundary between worlds: the
        # nether palette is essentially fully in force a metre
        # below it (a 12 m fade read as "blue, reddening as you
        # descend" from just underneath, which is wrong -- below
        # is below).
        under = float(np.clip((planes.BEDROCK_Y - cam.position[1]) / 1.0, 0.0, 1.0))
        # The haze colour is the sky's current horizon colour:
        # day colour above bedrock, nether colour below it.
        self._haze_col = np.array(
            [(1.0 - under) * c + under * n
             for c, n in zip(shaders.HAZE_COLOR, shaders.NETHER_HAZE_COLOR)],
            dtype=np.float32,
        )
        cam_pos = cam.position.astype(np.float32)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glDisable(gl.GL_CULL_FACE)

        if "sky" not in self.disabled_passes and self._show_far:
            # 1. Sky at infinity: depth exactly 1.0, drawn first
            # INTO a buffer just cleared to 1.0, and it WRITES
            # that depth (LEQUAL admits it against the clear).
            # From here on, every horizon tie is a tie against a
            # really-written depth, broken by painter order --
            # the tie breaker exercised, not assumed.  (Far
            # depth range, set here because the sky precedes the
            # near pass.)
            gl.glDepthRange(0.5, 1.0)
            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glUseProgram(self._sky_prog)
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._sky_prog, "uVP"), 1, gl.GL_TRUE, vp)
            gl.glUniform3fv(gl.glGetUniformLocation(self._sky_prog, "uSunDir"), 1, sun)
            # (The matching `under` factor is computed with the haze
            # colour above; the sky palette fades over the same range.)
            gl.glUniform1f(gl.glGetUniformLocation(self._sky_prog, "uUnder"), under)
            gl.glBindVertexArray(self._sky_vao)
            gl.glDrawElements(gl.GL_TRIANGLES, self._sky_count, gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)
            gl.glDepthMask(gl.GL_TRUE)
            gl.glDepthFunc(gl.GL_LESS)


        # One depth buffer, two ranges: the near pass owns
        # [0, 0.5), the far pass [0.5, 1].  Near geometry is
        # drawn first; every far depth is larger than every near
        # depth, so cross-pass visibility is exact with no
        # mid-frame depth clear (the clear left a shimmery seam
        # where the bedrock was drawn twice into two buffers'
        # worth of state).
        if self._show_near:
            # 3. Near pass: blocks and bedrock, 3 cm .. split.
            gl.glDepthRange(0.0, 0.5)
            if "blocks" not in self.disabled_passes:
                self._draw_blocks(vp_near, cam_pos, sun, cam_fwd, clip_near=clip_near)
            if "bedrock" not in self.disabled_passes and self._show_bedrock:
                self._draw_bedrock(vp_near, cam_pos, cam_fwd, clip_near=clip_near)
        gl.glDepthRange(0.5, 1.0)

        if "blocks" not in self.disabled_passes and self._show_far:
            # 4. Local RD ground (far pass; the near pass redraws
            # it with its own projection below).
            self._draw_blocks(vp, cam_pos, sun, cam_fwd, clip_far=clip_far)

        if "bedrock" not in self.disabled_passes and self._show_far and self._show_bedrock:
            # 5. Complete infinite bedrock plane (4 vertices).
            self._draw_bedrock(vp, cam_pos, cam_fwd, clip_far=clip_far)

        if "terrain" not in self.disabled_passes and self._show_terrain and self._show_far:
            # 6. Distant terrain annulus (block program; uniforms
            # set here so the pass stands alone when "blocks" is
            # disabled by the perf harness).  Far pass only: the
            # annulus starts at 1.5 km, beyond the split.
            gl.glUseProgram(self._block_prog)
            gl.glUniform1f(gl.glGetUniformLocation(self._block_prog, "uSmooth"), self._smooth)
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._block_prog, "uVP"), 1, gl.GL_TRUE, vp)
            gl.glUniform3fv(gl.glGetUniformLocation(self._block_prog, "uSunDir"), 1, sun)
            gl.glUniform3fv(gl.glGetUniformLocation(self._block_prog, "uCamPos"), 1, cam_pos)
            gl.glUniform3fv(gl.glGetUniformLocation(self._block_prog, "uCamFwd"), 1, cam_fwd)
            gl.glUniform1f(gl.glGetUniformLocation(self._block_prog, "uClipNear"), 0.0)
            gl.glUniform1f(gl.glGetUniformLocation(self._block_prog, "uClipFar"), clip_far)
            gl.glUniform3fv(gl.glGetUniformLocation(self._block_prog, "uHazeColor"), 1, self._haze_col)
            tvao, tcount = self._terrain_vao
            gl.glBindVertexArray(tvao)
            gl.glDrawElements(gl.GL_TRIANGLES, tcount, gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)

        if "terrain" not in self.disabled_passes and self._show_terrain and self._show_far:
            # 7. The horizon fan: triangles from the mesh's own
            # outer edge loop (its true boundary vertices) to 180
            # horizon vertices at infinity.  Finite, depth-
            # writing geometry in the ordinary way -- its inner
            # edge is watertight with the mesh because it IS the
            # mesh edge -- drawn after the mesh, before the sea,
            # whose horizon blend paints over its outer sliver
            # on the LEQUAL tie at depth 1.0.
            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glUseProgram(self._horizon_prog)
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._horizon_prog, "uVP"), 1, gl.GL_TRUE, vp)
            gl.glUniform3fv(gl.glGetUniformLocation(self._horizon_prog, "uSunDir"), 1, sun)
            gl.glUniform3fv(gl.glGetUniformLocation(self._horizon_prog, "uCamPos"), 1, cam_pos)
            gl.glUniform3fv(gl.glGetUniformLocation(self._horizon_prog, "uHazeColor"), 1, self._haze_col)
            gl.glUniform1f(gl.glGetUniformLocation(self._horizon_prog, "uDebug"), 0.0)
            gl.glBindVertexArray(self._horizon_vao)
            gl.glDrawElements(gl.GL_TRIANGLES, self._horizon_count, gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)
            gl.glDepthFunc(gl.GL_LESS)

        if "sea" not in self.disabled_passes and self._show_sea and self._show_far:
            # 8. The sea, drawn last because it is transparent
            # (blending needs the surfaces behind it down first),
            # not because visibility needs the order: it writes
            # depth and tests LEQUAL, so the depth buffer decides
            # where it shows, and LEQUAL merely breaks the
            # horizon tie in its favour as the later surface.
            self._sea.upload(
                planes.hole_plane_strip(cam.position[0], cam.position[2], planes.SEA_Y)
            )
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glUseProgram(self._sea_prog)
            gl.glUniformMatrix4fv(gl.glGetUniformLocation(self._sea_prog, "uVP"), 1, gl.GL_TRUE, vp)
            gl.glUniform3fv(gl.glGetUniformLocation(self._sea_prog, "uCamPos"), 1, cam_pos)
            gl.glUniform1f(gl.glGetUniformLocation(self._sea_prog, "uPlaneY"), planes.SEA_Y)
            gl.glUniform3fv(gl.glGetUniformLocation(self._sea_prog, "uCamFwd"), 1, cam_fwd)
            gl.glUniform1f(gl.glGetUniformLocation(self._sea_prog, "uClipNear"), 0.0)
            gl.glUniform1f(gl.glGetUniformLocation(self._sea_prog, "uClipFar"), clip_far)
            gl.glUniform3fv(gl.glGetUniformLocation(self._sea_prog, "uSunDir"), 1, sun)
            gl.glUniform1f(gl.glGetUniformLocation(self._sea_prog, "uHorizonAlpha"), self._sea_horizon_alpha)
            gl.glUniform3fv(gl.glGetUniformLocation(self._sea_prog, "uHazeColor"), 1, self._haze_col)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._checker_tex)
            gl.glUniform1i(gl.glGetUniformLocation(self._sea_prog, "uTex"), 0)
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._noise_tex)
            gl.glUniform1i(gl.glGetUniformLocation(self._sea_prog, "uNoise"), 1)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            self._sea.draw()
            gl.glDepthMask(gl.GL_TRUE)
            gl.glDepthFunc(gl.GL_LESS)
            gl.glDisable(gl.GL_BLEND)

        # (No below-bedrock bedrock redraw anymore: it was a
        # painter-era device for when the bedrock wrote no
        # depth and the later-drawn sea could paint over the
        # ceiling.  With depth written everywhere, the buffer
        # already keeps the sea behind the ceiling from below.)

        if self._show_boundary:
            # 9. Debug overlay: where the split plane meets the
            # world.  The boundary is the view plane at the
            # manual clip plane inside the overlap; a translucent quad on that plane
            # showed nothing useful (it fills the view and, with
            # depth test off, floats over everything).  What
            # locates the boundary is its INTERSECTION with the
            # surfaces -- here, magenta ribbons lying in the
            # bedrock and sea planes along the line where the
            # view plane crosses them, depth-tested against the
            # scene and drawn in both depth ranges with each
            # pass's own projection, exactly like the bedrock
            # itself.  The overlay is the scene's one depth
            # non-writer: it is a guide, not a surface.
            view = cam.view_matrix()
            upv, back = view[1, :3], view[2, :3]
            fwd = -back
            centre = cam.position + fwd * CLIP_SPLIT
            d = np.array([-fwd[2], 0.0, fwd[0]])
            dn = float(np.linalg.norm(d))
            rows = []
            if dn > 1e-6 and abs(float(upv[1])) > 1e-3:
                d = d / dn
                perp = np.array([d[2], 0.0, -d[0]]) * 0.6
                for plane_y in (planes.BEDROCK_Y, planes.SEA_Y):
                    t = (plane_y + 0.25 - centre[1]) / float(upv[1])
                    x0 = centre + upv * t
                    e0, e1 = x0 - d * 1400.0, x0 + d * 1400.0
                    for ex, sx in ((e0, -1.0), (e0, 1.0), (e1, -1.0), (e1, 1.0)):
                        p = ex + perp * sx
                        rows.append((p[0], plane_y + 0.25, p[2], 1.0,
                                     1.0, 0.0, 1.0, 0.85))
            if rows:
                data = np.array(rows, dtype=np.float32)
                gl.glBindVertexArray(self._debug_vao)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._debug_vbo)
                gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
                gl.glDepthMask(gl.GL_FALSE)
                gl.glUseProgram(self._horizon_prog)
                gl.glUniform1f(gl.glGetUniformLocation(self._horizon_prog, "uDebug"), 1.0)
                loc_vp = gl.glGetUniformLocation(self._horizon_prog, "uVP")
                for rng, vpx in ((0.0, vp_near), (0.5, vp_far)):
                    gl.glDepthRange(rng, rng + 0.5)
                    gl.glUniformMatrix4fv(loc_vp, 1, gl.GL_TRUE, vpx)
                    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
                    if len(rows) > 4:
                        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 4, 4)
                gl.glBindVertexArray(0)
                gl.glDepthRange(0.0, 1.0)
                gl.glDepthMask(gl.GL_TRUE)
                gl.glDisable(gl.GL_BLEND)

        self._frame += 1
        if self.on_hud is not None and self._frame % 6 == 0:
            p = cam.position
            boost = "  x10 BOOST" if self._boost > 1.0 else ""
            mode = "  MOUSE CAPTURED (Esc for menu)" if self._captured else ""
            if not self._show_terrain:
                mode += "  [terrain off]"
            if not self._show_sea:
                mode += "  [sea off]"
            if not self._show_far:
                mode += "  [far off]"
            if not self._show_near:
                mode += "  [near off]"
            if self._show_boundary:
                mode += "  [split shown]"
            self.on_hud(
                f"pos ({p[0]:7.1f}, {p[1]:6.1f}, {p[2]:7.1f})   "
                f"speed {cam.move_speed * self._boost:5.1f} m/s{boost}   "
                f"yaw {math.degrees(cam.yaw):6.1f} deg   pitch {math.degrees(cam.pitch):6.1f} deg{mode}"
            )

        # Event-driven loop: while movement keys are held the
        # viewpoint keeps changing, so request the next frame;
        # otherwise stop here and wait for the next event.
        if self._keys:
            self.update()

    # -- input ----------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            # Minecraft mode: Esc while captured leaves the
            # world for the options menu (releasing the mouse);
            # Esc in the menu returns to the world and
            # recaptures -- handled by the window, which owns
            # the menu.
            if self._captured:
                self._set_captured(False)
            if self.on_escape_menu is not None:
                self.on_escape_menu()
            else:
                self.window().close()
            return
        if event.key() == Qt.Key.Key_G and not event.isAutoRepeat():
            self._smooth = 0.0 if self._smooth else 1.0
        if event.key() == Qt.Key.Key_T and not event.isAutoRepeat():
            self._show_terrain = not self._show_terrain
            self.update()
        if event.key() == Qt.Key.Key_Y and not event.isAutoRepeat():
            self._show_sea = not self._show_sea
            self.update()
        if event.key() == Qt.Key.Key_F and not event.isAutoRepeat():
            self._show_far = not self._show_far
            self.update()
        if event.key() == Qt.Key.Key_N and not event.isAutoRepeat():
            self._show_near = not self._show_near
            self.update()
        if event.key() == Qt.Key.Key_B and not event.isAutoRepeat():
            self._show_boundary = not self._show_boundary
            self.update()
            self.update()
            return
        if not event.isAutoRepeat():
            self._keys.add(event.key())
            self.update()
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
            self._last_global = QPointF(center)
            self._look_pending = [0.0, 0.0]
            QCursor.setPos(center)
        else:
            self.releaseMouse()
            self.unsetCursor()
            self._last_global = None
        self.update()

    def mousePressEvent(self, event) -> None:
        # Minecraft mode: a click in the world captures the
        # mouse for looking.  There is no drag-look and no
        # click-to-release: Esc is the way out (to the menu).
        if not self._captured:
            self._set_captured(True)
        self.setFocus()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._captured:
            # Accumulate deltas; the rotation is applied once per
            # frame in the tick, exactly like WASD translation --
            # applying per-event made the view step at the mouse's
            # event rate and fought the cursor-recentering warps.
            # Deltas are measured against the last known global
            # position (not an assumed centre), so a warp event that
            # arrives merged with real motion is still measured
            # correctly, and the warp's own event measures ~zero.
            pos = event.globalPosition()
            if self._last_global is not None:
                d = pos - self._last_global
                self._look_pending[0] += float(d.x())
                self._look_pending[1] += float(d.y())
            center_q = self.mapToGlobal(self.rect().center())
            center = QPointF(center_q)
            if pos != center:
                QCursor.setPos(center_q)
                self._last_global = center
            else:
                self._last_global = pos
            if self._look_pending[0] != 0.0 or self._look_pending[1] != 0.0:
                self.update()
            super().mouseMoveEvent(event)
            return
    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        self.camera.move_speed = float(
            np.clip(self.camera.move_speed * (1.25**steps), 1.0, 240.0)
        )
        self.update()  # the HUD speed readout changed
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
            "click to capture mouse - wheel: speed - G: flat/Gouraud - Esc: menu"
        )
        self._hud.move(10, 10)
        self._hud.adjustSize()
        self._status = QLabel(self)
        self._status.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._status.setStyleSheet(self._hud.styleSheet())
        self._status.move(10, 44)
        self._gl.on_hud = self._set_status
        self._gl.on_escape_menu = self._toggle_menu
        self._menu = self._build_menu()
        self._menu.hide()

    def _build_menu(self) -> QFrame:
        """The Esc options panel: layer toggles mirroring the
        keyboard toggles (plus the bedrock layer, which has no
        key), the debug toggles, and Resume / Quit buttons."""
        gl = self._gl
        panel = QFrame(self)
        panel.setStyleSheet(
            "QFrame { background: rgba(10, 14, 20, 225); "
            "border: 1px solid #4a6076; } "
            "QLabel, QCheckBox { color: #dfe9f2; font-size: 13px; } "
            "QPushButton { padding: 4px 14px; font-size: 13px; }"
        )
        layout = QVBoxLayout(panel)
        title = QLabel("Options")
        title.setStyleSheet("font-size: 15px; font-weight: bold;")
        layout.addWidget(title)
        self._menu_checks: dict[str, QCheckBox] = {}

        def add_check(label: str, attr: str) -> None:
            box = QCheckBox(label)
            box.setChecked(bool(getattr(gl, attr)))
            box.toggled.connect(
                lambda on, a=attr: (setattr(gl, a, on), gl.update()))
            layout.addWidget(box)
            self._menu_checks[attr] = box

        add_check("Distant terrain + horizon fan  (T)", "_show_terrain")
        add_check("Ocean surface  (Y)", "_show_sea")
        add_check("Bedrock layer", "_show_bedrock")
        add_check("Far depth pass, 300 m - infinity  (F)", "_show_far")
        add_check("Near depth pass, 3 cm - 300 m  (N)", "_show_near")
        add_check("Split-plane intersection lines  (B)", "_show_boundary")
        smooth = QCheckBox("Smooth (Gouraud) shading  (G)")
        smooth.setChecked(bool(gl._smooth))
        smooth.toggled.connect(
            lambda on: (setattr(gl, "_smooth", 1.0 if on else 0.0),
                        gl.update()))
        layout.addWidget(smooth)
        self._menu_checks["_smooth"] = smooth
        row = QHBoxLayout()
        resume = QPushButton("Resume")
        resume.clicked.connect(self._close_menu)
        quit_btn = QPushButton("Quit")
        quit_btn.clicked.connect(self.close)
        row.addWidget(resume)
        row.addWidget(quit_btn)
        layout.addLayout(row)
        panel.adjustSize()
        return panel

    def _toggle_menu(self) -> None:
        # Esc is a toggle: in the world it opens the menu;
        # with the menu up it acts as Resume (close and
        # recapture), whichever widget holds focus.
        if self._menu.isVisible():
            self._close_menu()
        else:
            self._open_menu()

    def _close_menu(self) -> None:
        # Back to the world, Minecraft-style: the mouse is
        # recaptured for looking.
        self._menu.hide()
        self._gl.setFocus()
        self._gl._set_captured(True)

    def _open_menu(self) -> None:
        if self._menu.isVisible():
            return
        # Sync the checkboxes from the live flags (keyboard
        # toggles may have changed them since the last opening).
        gl = self._gl
        for attr, box in self._menu_checks.items():
            value = bool(getattr(gl, attr))
            if box.isChecked() != value:
                box.blockSignals(True)
                box.setChecked(value)
                box.blockSignals(False)
        self._menu.adjustSize()
        self._menu.move(
            (self.width() - self._menu.width()) // 2,
            (self.height() - self._menu.height()) // 2,
        )
        self._menu.show()
        self._menu.raise_()

    def keyPressEvent(self, event) -> None:
        # Esc closes the menu when a menu widget holds focus
        # (otherwise the GL widget's own Esc handling runs).
        if event.key() == Qt.Key.Key_Escape and self._menu.isVisible():
            self._close_menu()
            return
        super().keyPressEvent(event)

    def _set_status(self, text: str) -> None:
        self._status.setText(text)
        self._status.adjustSize()


def main(argv: list[str] | None = None) -> int:
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)
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
