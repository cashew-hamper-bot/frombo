"""Small 3D math module, built around one idea: the far plane is at infinity.

Conventions
-----------
- Right-handed world, +Y up (Minecraft-style).
- Column vectors: ``clip = M @ world``.  Matrices are numpy arrays in
  mathematical (row-major) order; the GL upload uses ``GL_TRUE`` transpose.
- The camera looks down its local -Z, as usual in OpenGL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def perspective_infinite(fov_y_rad: float, aspect: float, near: float) -> np.ndarray:
    """Perspective projection with the far plane at infinity.

    This is the ordinary finite perspective matrix taken to the limit
    ``far -> inf``::

        a = (far + near) / (near - far)  ->  -1
        b = 2 * far * near / (near - far) -> -2 * near

    so the third row becomes ``[0, 0, -1, -2*near]``.

    Consequences worth remembering:

    - A finite point at the near plane maps to NDC z = -1; a finite point
      arbitrarily far away maps to NDC z approaching +1 from below.
    - A *direction* (homogeneous w = 0) straight ahead maps to NDC z
      exactly +1.  Points at infinity land exactly on the far plane,
      which is why the skybox and the plane horizons in this project
      can be honest w = 0 geometry instead of a huge finite stand-in.
    """
    f = 1.0 / math.tan(fov_y_rad / 2.0)
    return np.array(
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, -1.0, -2.0 * near],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )


def ray_direction(vp_inverse: np.ndarray, ndc_x: float, ndc_y: float) -> np.ndarray:
    """World-space direction of the viewing ray through an NDC point.

    With the infinite projection, NDC z = +1 is the plane at infinity,
    so its inverse image is a homogeneous direction (w = 0): exactly
    the ray direction, no near-point subtraction needed.
    """
    h = vp_inverse @ np.array([ndc_x, ndc_y, 1.0, 1.0])
    d = h[:3]
    return d / np.linalg.norm(d)


def look_delta(
    pitch: float,
    x: float,
    y: float,
    dx: float,
    dy: float,
    k: float,
) -> tuple[float, float]:
    """One grab-the-world look step, from the analytic Jacobian.

    ``x``, ``y`` are the cursor's camera-space ray coordinates
    (``X = aspect * tan(fovY/2) * ndc_x``, ``Y = tan(fovY/2) *
    ndc_y``), ``dx``, ``dy`` the cursor motion in pixels, and
    ``k = 2 * tan(fovY/2) / height_px`` the shared pixel scale
    (precomputed in resizeGL).  Returns ``(dyaw, dpitch)`` in
    grab sense: the rotation that keeps the world direction
    under the cursor glued to it.  Derivation:
    docs/look-jacobian.md.

    The only singularity is ``sigma = cos(pitch) - sin(pitch)*Y
    = 0`` -- the cursor ray parallel to the world Y axis, where
    yaw cannot move the grabbed point; sigma is clamped to
    ``|sigma| >= 0.10`` so gains stay bounded there.
    """
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    lam2 = 1.0 + x * x + y * y
    sigma = cp - sp * y
    if abs(sigma) < 0.10:
        sigma = 0.10 if sigma >= 0.0 else -0.10
    denom = lam2 * sigma
    dyaw = k * ((1.0 + y * y) * dx + x * y * dy) / denom
    dpitch = k * (
        x * (sp + cp * y) * dx + (cp * (1.0 + x * x) - sp * y) * dy
    ) / denom
    return dyaw, dpitch


def rotation_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    """World-from-camera rotation R = Ry(yaw) @ Rx(pitch)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64
    )
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64
    )
    return ry @ rx


@dataclass
class Camera:
    """A fly camera.  yaw = pitch = 0 looks along world -Z."""

    position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 7.0, 20.0])
    )
    yaw: float = 0.0
    pitch: float = -0.30
    move_speed: float = 12.0  # metres per second, wheel-adjustable

    def rotation(self) -> np.ndarray:
        return rotation_yaw_pitch(self.yaw, self.pitch)

    def forward(self) -> np.ndarray:
        return self.rotation() @ np.array([0.0, 0.0, -1.0])

    def right(self) -> np.ndarray:
        return self.rotation() @ np.array([1.0, 0.0, 0.0])

    def up(self) -> np.ndarray:
        return self.rotation() @ np.array([0.0, 1.0, 0.0])

    def forward_xz(self) -> np.ndarray:
        """Horizontal forward direction (unit length, in the XZ plane).

        Used to orient the infinite-plane strips.  Falls back to -Z when
        looking essentially straight up or down.
        """
        f = self.forward()
        fx, fz = float(f[0]), float(f[2])
        n = math.hypot(fx, fz)
        if n < 1e-6:
            return np.array([0.0, -1.0])
        return np.array([fx / n, fz / n])

    def view_matrix(self) -> np.ndarray:
        """World-to-eye 4x4.

        Note what happens to a homogeneous direction (w = 0): the
        translation column is multiplied by 0, so directions are only
        rotated.  That is not a special case in the code -- it is just
        homogeneous coordinates doing their job.
        """
        r = self.rotation()
        right, up, back = r[:, 0], r[:, 1], r[:, 2]  # back = -forward
        p = self.position
        m = np.identity(4)
        m[0, :3] = right
        m[0, 3] = -float(right @ p)
        m[1, :3] = up
        m[1, 3] = -float(up @ p)
        m[2, :3] = back  # z_eye = back . (point - position)
        m[2, 3] = -float(back @ p)
        return m

    def view_projection(self, fov_y_rad: float, aspect: float, near: float) -> np.ndarray:
        return perspective_infinite(fov_y_rad, aspect, near) @ self.view_matrix()
