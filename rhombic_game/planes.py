"""Infinite planes built from homogeneous vertices.

Both planes are triangle strips whose vertices are *homogeneous world
positions* (vec4).  Finite vertices carry w = 1; horizon vertices carry
w = 0 and a purely horizontal direction, i.e. they are points at
infinity lying in the plane.  The vertex shader passes the homogeneous
position through as a varying, and the fragment shader recovers the
true world point as ``v.xyz / v.w`` -- projective interpolation of the
homogeneous coordinates gives exactly the point where the view ray
through that fragment pierces the plane, so texturing stays correct
all the way out to the horizon with no giant finite quad anywhere.

Full plane -- 4 vertices
------------------------
The strip is a single quad: a finite grounding segment (two w = 1
vertices) set behind the point below the camera -- looking steeply
down, the bottom of the screen shows ground a little way *behind*
that point -- and stretched wide sideways, opening out to two w = 0
horizon vertices at wide forward-left and forward-right angles.  The
strip is re-anchored each frame to the camera's ground position,
heading, and height above the plane; the segment's exact placement
is unimportant because shading comes from the recovered world
position, not from the geometry's extent.

Plane with a hole -- a strip of pairs
-------------------------------------
The sea plane has a circular hole of radius ``HOLE_RADIUS`` around the
player.  Its strip walks once around the hole: 12 finite vertices on
the hole's rim (a dodecagon), each followed by the horizon vertex in
the same azimuth, plus a repeated first pair to close the loop:
13 finite + 13 horizon vertices, alternating, w = 1, 0, 1, 0, ...
The rim follows the player's XZ position, so the hole travels with
the player and the plane still reaches the horizon in every direction.
"""

from __future__ import annotations

import math

import numpy as np

BEDROCK_Y = -100.0  # ~100 m below the terrain layer (centres at y = 0)
SEA_Y = -10.0  # ~10 m below the terrain layer
HOLE_RADIUS = 500.0  # the hole is ~1000 m wide
HOLE_SEGMENTS = 12
GROUNDING_HALF_WIDTH = 64.0
# The horizon vertices sit in the directions forward +- right * SPREAD
# (about 80 degrees off-axis, not 45): the strip's sides must open
# wider than the widest azimuth a screen corner can ever show, at
# every range -- looking down from altitude, near ground at the
# bottom corners sits at far wider azimuths than the horizon does.
# Overshooting costs nothing: shading comes from the recovered world
# position, not from the geometry's extent.
HORIZON_SPREAD = 6.0


def full_plane_strip(
    center_x: float,
    center_z: float,
    y: float,
    forward_xz: np.ndarray,
    height: float = 0.0,
    half_width: float = GROUNDING_HALF_WIDTH,
) -> np.ndarray:
    """4 homogeneous vertices (triangle strip) for a complete plane.

    ``(center_x, center_z)`` is the point below the camera and
    ``height`` the camera's height above the plane.  The grounding
    segment sits *behind* that point: looking steeply down, the
    bottom of the screen shows ground behind the camera's ground
    point (up to ~0.7 * height behind, looking straight down), and
    the visible near-ground region widens with height, so both the
    back-off and the width scale with ``height``.  All of this is
    coverage only -- shading is analytic in the world position.
    """
    fx, fz = float(forward_xz[0]), float(forward_xz[1])
    rx, rz = fz, -fx  # horizontal right, perpendicular to forward
    s = HORIZON_SPREAD
    back = 1.25 * height + 10.0
    hw = max(half_width, 1.2 * height + 64.0)
    ex, ez = center_x - fx * back, center_z - fz * back
    return np.array(
        [
            [ex - rx * hw, y, ez - rz * hw, 1.0],
            [ex + rx * hw, y, ez + rz * hw, 1.0],
            [fx - rx * s, 0.0, fz - rz * s, 0.0],
            [fx + rx * s, 0.0, fz + rz * s, 0.0],
        ],
        dtype=np.float32,
    )


def hole_plane_strip(
    center_x: float,
    center_z: float,
    y: float,
    radius: float = HOLE_RADIUS,
    segments: int = HOLE_SEGMENTS,
) -> np.ndarray:
    """Alternating rim/horizon vertex pairs, as a triangle strip."""
    verts: list[list[float]] = []
    for m in range(segments + 1):
        phi = 2.0 * math.pi * (m % segments) / segments
        dx, dz = math.cos(phi), math.sin(phi)
        verts.append([center_x + radius * dx, y, center_z + radius * dz, 1.0])
        verts.append([dx, 0.0, dz, 0.0])
    return np.array(verts, dtype=np.float32)


def skybox_cube() -> tuple[np.ndarray, np.ndarray]:
    """A cube whose vertices are pure directions (used with w = 0).

    The positions are direction vectors, not places: in the shader they
    are submitted as ``vec4(dir, 0.0)``, so the view matrix's translation
    drops out by itself and the infinite projection puts every fragment
    at NDC z = +1 exactly -- the sky *is* at infinity here, not pasted
    on a big box.
    """
    corners = np.array(
        [
            [x, y, z]
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
        dtype=np.float32,
    )

    def idx(x: int, y: int, z: int) -> int:
        return ((x + 1) // 2) * 4 + ((y + 1) // 2) * 2 + ((z + 1) // 2)

    quads = [
        (idx(-1, -1, -1), idx(1, -1, -1), idx(1, 1, -1), idx(-1, 1, -1)),  # back
        (idx(1, -1, 1), idx(-1, -1, 1), idx(-1, 1, 1), idx(1, 1, 1)),  # front
        (idx(1, -1, -1), idx(1, -1, 1), idx(1, 1, 1), idx(1, 1, -1)),  # right
        (idx(-1, -1, 1), idx(-1, -1, -1), idx(-1, 1, -1), idx(-1, 1, 1)),  # left
        (idx(-1, 1, -1), idx(1, 1, -1), idx(1, 1, 1), idx(-1, 1, 1)),  # top
        (idx(-1, -1, 1), idx(1, -1, 1), idx(1, -1, -1), idx(-1, -1, -1)),  # bottom
    ]
    indices: list[int] = []
    for a, b, c, d in quads:
        indices += [a, b, c, a, c, d]
    return corners, np.array(indices, dtype=np.uint32)
