"""Distant terrain: a coarse hexagonal-lattice surface to 10 km,
plus a statistical horizon ring at infinity.

The local islands end a few hundred metres out and the sea and
bedrock planes bridge from there to infinity, but both are flat.
This layer puts actual elevation in the far field:

- Hexagonal cells (circumradius ``HEX_A``) in an annulus from
  ``HOLE_RADIUS`` (the near field is not this layer's job, same
  as the sea's hole) out to ``OUTER_RADIUS`` = 10 km -- past
  Distant Horizons' 256-chunk default (~4 km), well short of
  ten times it.
- Heights come from value noise whose nodes sit on the
  triangular lattice -- the vertex lattice of the hexagonal
  tiling -- at three octave spacings, so the noise is generated
  on the hex lattice itself.  The same lattice at finer scale
  is the planned hierarchical subdivision basis (the 49-hexagon
  subdivision used by the bedrock pattern).
- Beyond 10 km, a horizon ring at infinity (w = 0 vertices):
  its far edge carries the *mean* terrain colour, with opacity
  equal to the sampled land fraction -- at the horizon, land
  and sea mix at sub-pixel scale in exactly that ratio.  It is
  drawn after the sea so it blends over the sea's horizon.
  (If the mean elevation were below sea level the roles would
  swap; here the mean is above, by construction.)
"""

from __future__ import annotations

import math

import numpy as np

from .planes import SEA_Y

HEX_A = 500.0  # hex circumradius (m); feature scale ~1/20 of range
HOLE_RADIUS = 1500.0
OUTER_RADIUS = 10000.0
TARGET_MEAN_ELEVATION = 20.0  # metres; above SEA_Y, by design
HEIGHT_SPAN = 400.0
HORIZON_SEGMENTS = 180
_OCTAVES = ((3000.0, 0.60, 11), (1500.0, 0.27, 23), (750.0, 0.13, 37))

_SQRT3 = math.sqrt(3.0)


def _hash01(i: int, j: int, seed: int) -> float:
    """splitmix64-style integer hash to [0, 1).  (No numpy.random:
    see the standing environment landmine.)"""
    x = (i * 0x9E3779B97F4A7C15 + j * 0xC2B2AE3D27D4EB4F
         + seed * 0x165667B19E3779F9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x = x ^ (x >> 31)
    return (x >> 11) / float(1 << 53)


def _lattice_noise(x: float, z: float, spacing: float, seed: int) -> float:
    """Value noise on the triangular lattice, barycentric in the
    containing triangle of the lattice cell."""
    v = z / (spacing * _SQRT3 / 2.0)
    u = x / spacing - v / 2.0
    i, j = math.floor(u), math.floor(v)
    fu, fv = u - i, v - j
    if fu + fv <= 1.0:
        c00 = _hash01(i, j, seed)
        c10 = _hash01(i + 1, j, seed)
        c01 = _hash01(i, j + 1, seed)
        return c00 * (1.0 - fu - fv) + c10 * fu + c01 * fv
    c11 = _hash01(i + 1, j + 1, seed)
    c10 = _hash01(i + 1, j, seed)
    c01 = _hash01(i, j + 1, seed)
    return c11 * (fu + fv - 1.0) + c10 * (1.0 - fv) + c01 * (1.0 - fu)


def _noise01(x: float, z: float) -> float:
    return sum(
        amp * _lattice_noise(x, z, spacing, seed)
        for spacing, amp, seed in _OCTAVES
    )


def _height_raw(x: float, z: float) -> float:
    return (_noise01(x, z) - 0.5) * HEIGHT_SPAN


_STATS: dict | None = None


def terrain_stats() -> dict:
    """Sampled statistics over the disc out to OUTER_RADIUS:
    mean elevation (pinned to TARGET_MEAN_ELEVATION by the
    height offset), sea fraction, and mean terrain colour."""
    global _STATS
    if _STATS is not None:
        return _STATS
    heights = []
    colours = []
    rings, sectors = 40, 90
    for ri in range(1, rings + 1):
        r = OUTER_RADIUS * ri / rings
        for si in range(sectors):
            th = 2.0 * math.pi * si / sectors
            x, z = r * math.cos(th), r * math.sin(th)
            h = _height_raw(x, z)
            heights.append(h)
            colours.append(_base_colour(h))
    mean_raw = float(np.mean(heights))
    offset = TARGET_MEAN_ELEVATION - mean_raw
    heights = [h + offset for h in heights]
    sea_fraction = float(np.mean([1.0 if h < SEA_Y else 0.0 for h in heights]))
    mean_colour = tuple(float(np.mean([c[k] for c in colours])) for k in range(3))
    _STATS = {
        "offset": offset,
        "mean_elevation": TARGET_MEAN_ELEVATION,
        "sea_fraction": sea_fraction,
        "land_fraction": 1.0 - sea_fraction,
        "mean_colour": mean_colour,
    }
    return _STATS


def height(x: float, z: float) -> float:
    """Terrain elevation (world Y) at (x, z)."""
    return _height_raw(x, z) + terrain_stats()["offset"]


def _base_colour(h: float) -> tuple[float, float, float]:
    """Terrain colour by elevation, as lerped stops."""
    stops = [
        (-60.0, (0.34, 0.31, 0.22)),   # deep sea floor
        (SEA_Y, (0.76, 0.65, 0.46)),   # shore sand (the block colour)
        (25.0, (0.42, 0.55, 0.30)),    # lowland sage
        (90.0, (0.45, 0.43, 0.40)),    # highland rock
        (130.0, (0.74, 0.73, 0.71)),   # pale peaks
    ]
    if h <= stops[0][0]:
        return stops[0][1]
    for (h0, c0), (h1, c1) in zip(stops, stops[1:]):
        if h <= h1:
            f = (h - h0) / (h1 - h0)
            return tuple(c0[k] + (c1[k] - c0[k]) * f for k in range(3))
    return stops[-1][1]


def colour_at(x: float, z: float) -> tuple[float, float, float]:
    h = height(x, z)
    base = _base_colour(h)
    tint = 0.96 + 0.08 * _hash01(int(x // HEX_A), int(z // HEX_A), 91)
    return tuple(min(1.0, c * tint) for c in base)


def _hex_centres():
    """Flat-top hex centres (points east/west, like the bedrock
    pattern) whose centres lie in the annulus."""
    reach = int(OUTER_RADIUS / (1.5 * HEX_A)) + 2
    for q in range(-reach, reach + 1):
        for r in range(-reach, reach + 1):
            cx = HEX_A * 1.5 * q
            cz = HEX_A * _SQRT3 * (r + q / 2.0)
            rad = math.hypot(cx, cz)
            if HOLE_RADIUS <= rad <= OUTER_RADIUS:
                yield cx, cz


def build_distant_terrain() -> tuple[np.ndarray, np.ndarray]:
    """Interleaved 13-float mesh (the block format: position,
    flat normal, smooth normal, colour, AO = 1), non-indexed
    triangles fanned per hexagon."""
    # Unique points (hex corners are shared; centres are not).
    points: dict[tuple[int, int], tuple[float, float, float]] = {}

    def point(x: float, z: float) -> tuple[float, float, float]:
        key = (round(x / 0.05), round(z / 0.05))
        if key not in points:
            points[key] = (x, height(x, z), z)
        return points[key]

    tris: list[tuple[tuple, tuple, tuple]] = []
    corner_offsets = [
        (HEX_A * math.cos(math.radians(60 * k)),
         HEX_A * math.sin(math.radians(60 * k)))
        for k in range(6)
    ]
    for cx, cz in _hex_centres():
        centre = point(cx, cz)
        corners = [point(cx + dx, cz + dz) for dx, dz in corner_offsets]
        for k in range(6):
            tris.append((centre, corners[k], corners[(k + 1) % 6]))

    # Smooth normals: accumulate face normals per position.
    smooth: dict[tuple, np.ndarray] = {}
    face_normals = []
    for a, b, c in tris:
        n = np.cross(np.array(b) - np.array(a), np.array(c) - np.array(a))
        face_normals.append(n)
        for p in (a, b, c):
            smooth[p] = smooth.get(p, np.zeros(3)) + n
    for p, n in smooth.items():
        ln = float(np.linalg.norm(n))
        smooth[p] = n / ln if ln > 0 else np.array([0.0, 1.0, 0.0])

    verts = np.empty((len(tris) * 3, 13), dtype=np.float32)
    colours: dict[tuple, tuple] = {}
    row = 0
    for (a, b, c), fn in zip(tris, face_normals):
        ln = float(np.linalg.norm(fn))
        fn = fn / ln if ln > 0 else np.array([0.0, 1.0, 0.0])
        for p in (a, b, c):
            if p not in colours:
                colours[p] = colour_at(p[0], p[2])
            verts[row, 0:3] = p
            verts[row, 3:6] = fn
            verts[row, 6:9] = smooth[p]
            verts[row, 9:12] = colours[p]
            verts[row, 12] = 1.0
            row += 1
    idx = np.arange(len(tris) * 3, dtype=np.uint32)
    return verts, idx


def build_horizon_ring() -> tuple[np.ndarray, np.ndarray]:
    """Strip from the OUTER_RADIUS circle to infinity.

    Vertex format: (aHomog vec4, aColor vec4) -- 8 floats.
    Inner vertices are finite (w = 1) with the terrain's own
    colour and alpha 1; outer vertices are directions (w = 0)
    carrying the mean terrain colour with opacity equal to the
    land fraction, to be blended over the sea after it is drawn.
    """
    stats = terrain_stats()
    mean = stats["mean_colour"]
    land = stats["land_fraction"]
    rows = []
    for i in range(HORIZON_SEGMENTS + 1):
        th = 2.0 * math.pi * i / HORIZON_SEGMENTS
        x, z = OUTER_RADIUS * math.cos(th), OUTER_RADIUS * math.sin(th)
        h = height(x, z)
        c = colour_at(x, z)
        rows.append((x, h, z, 1.0, c[0], c[1], c[2], 1.0))
        rows.append((x, h, z, 0.0, mean[0], mean[1], mean[2], land))
    verts = np.array(rows, dtype=np.float32)
    idx = []
    for i in range(HORIZON_SEGMENTS):
        a0, b0 = 2 * i, 2 * i + 1
        a1, b1 = 2 * i + 2, 2 * i + 3
        idx += [a0, a1, b1, a0, b1, b0]
    return verts, np.array(idx, dtype=np.uint32)
