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

    The target is the world direction under the cursor at drag start
    (ndc_old), computed ONCE, in world space.

    Two stages: a Rodrigues rotation carrying the current cursor ray
    onto the target, applied to the forward vector, gives a yaw/pitch
    within ~a degree (a yaw/pitch camera cannot realise an arbitrary
    rotation exactly -- it discards the roll about the forward axis);
    then Newton iterations on (yaw, pitch) drive the cursor ray onto
    the target to machine precision.

    An earlier version recomputed the target from ndc_old after
    rotating -- re-aiming at a moving target, which applied the step
    twice and made the world drag ~2x faster than the mouse.
    """
    inv = np.linalg.inv(camera.view_projection(fov_y_rad, aspect, near))
    target = ray_direction(inv, *ndc_old)

    def current_ray() -> np.ndarray:
        inv_now = np.linalg.inv(camera.view_projection(fov_y_rad, aspect, near))
        return ray_direction(inv_now, *ndc_new)

    # Stage 1: Rodrigues initial guess.
    a = current_ray()
    axis = np.cross(a, target)
    s = float(np.linalg.norm(axis))
    if s > 1e-12:
        axis = axis / s
        c = float(np.clip(a @ target, -1.0, 1.0))
        f = camera.forward()
        f2 = f * c + np.cross(axis, f) * s + axis * float(axis @ f) * (1.0 - c)
        camera.yaw = float(np.arctan2(-f2[0], -f2[2]))
        camera.pitch = float(
            np.clip(np.arcsin(np.clip(f2[1], -1.0, 1.0)), -1.55, 1.55)
        )

    # Stage 2: Newton refinement on (yaw, pitch).
    seed = np.array([1.0, 0.0, 0.0]) if abs(target[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = np.cross(target, seed)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(target, e1)
    eps = 1e-6
    for _ in range(6):
        r = current_ray()
        if float(r @ target) > 1.0 - 1e-14:
            break
        res = np.array([(r - target) @ e1, (r - target) @ e2])
        jac = np.zeros((2, 2))
        for col, attr in enumerate(("yaw", "pitch")):
            saved = getattr(camera, attr)
            setattr(camera, attr, saved + eps)
            rp = current_ray()
            setattr(camera, attr, saved)
            jac[:, col] = np.array([(rp - r) @ e1, (rp - r) @ e2]) / eps
        try:
            delta = np.linalg.solve(jac, -res)
        except np.linalg.LinAlgError:
            break
        camera.yaw += float(delta[0])
        camera.pitch = float(np.clip(camera.pitch + float(delta[1]), -1.55, 1.55))


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
