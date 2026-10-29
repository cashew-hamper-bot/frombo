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


def drag_glue(
    camera: "Camera",
    fov_y_rad: float,
    aspect: float,
    near: float,
    ndc_old: tuple[float, float],
    ndc_new: tuple[float, float],
) -> None:
    """Rotate the camera so the grabbed world direction stays grabbed.

    Closed form, no iteration.  Write the new cursor ray in camera
    space as v = (ndc.x * tan(fovx/2), ndc.y * tan(fovy/2), -1); the
    camera must satisfy Ry(yaw') Rx(pitch') v = |v| * target.

    - Rx preserves the x component, so with w = Ry(-yaw') target we
      need w.x = v.x / |v|.  w.x as a function of yaw' is a sinusoid
      of amplitude rho = the target's horizontal magnitude, giving
      yaw' = delta +/- acos(vx_hat / rho); of the two solutions, take
      the one nearer the current (yaw, pitch).
    - Then pitch' rotates v's (y, z) components onto w's: pitch' is
      the difference of their atan2 angles.

    Exact (to ~1e-6 degrees, stress-tested) whenever a solution
    exists.  The one impossible geometry for a yaw/pitch camera: a
    near-vertical target (|vx_hat| > rho -- looking almost straight
    up or down with the cursor off to the side) would need roll;
    there the acos clamps and the formula returns the closest yaw.

    History: v1 recomputed the target after rotating (applied the
    step twice, ~2x drag); v2 fixed that but refined with Newton
    iterations; this closed form replaces the loop.
    """
    inv = np.linalg.inv(camera.view_projection(fov_y_rad, aspect, near))
    target = ray_direction(inv, *ndc_old)
    ty = math.tan(fov_y_rad / 2.0)
    v = np.array([ndc_new[0] * ty * aspect, ndc_new[1] * ty, -1.0])
    vx_hat = float(v[0] / np.linalg.norm(v))
    dx, dz = float(target[0]), float(target[2])
    rho = math.hypot(dx, dz)
    solutions: list[tuple[float, float]] = []
    if rho < 1e-9:
        yaws = [camera.yaw]
    else:
        delta = math.atan2(-dz, dx)
        off = math.acos(max(-1.0, min(1.0, vx_hat / rho)))
        yaws = [delta + off, delta - off]
    for yaw_new in yaws:
        cy, sy = math.cos(yaw_new), math.sin(yaw_new)
        w = np.array(
            [cy * dx - sy * dz, float(target[1]), sy * dx + cy * dz]
        )
        pitch_new = math.atan2(float(w[2]), float(w[1])) - math.atan2(-1.0, float(v[1]))
        pitch_new = (pitch_new + math.pi) % (2.0 * math.pi) - math.pi
        solutions.append((yaw_new, pitch_new))

    def _dist(sol: tuple[float, float]) -> float:
        dyaw = (sol[0] - camera.yaw + math.pi) % (2.0 * math.pi) - math.pi
        return abs(dyaw) + abs(sol[1] - camera.pitch)

    yaw_new, pitch_new = min(solutions, key=_dist)
    camera.yaw = float(yaw_new)
    camera.pitch = float(np.clip(pitch_new, -1.55, 1.55))


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
