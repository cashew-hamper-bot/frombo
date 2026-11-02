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
- Beyond the mesh, a horizon FAN: the mesh's own outer edge
  vertices (its true boundary loop, hex zigzag and all) joined
  by triangles to 180 evenly spaced horizon vertices at
  infinity (w = 0).  A horizon vertex carries one uniform
  colour -- a constant medium dark green/brown for now --
  because a horizon point represents the whole disc.  The disc is ocean-dominated
  (land fraction 0.40, median below sea level), so the sea is
  the transparent partner at the horizon: the fan is opaque,
  drawn before the sea, and the sea's alpha fades at the
  horizon to sea_fraction x its base opacity.
"""

from __future__ import annotations

import math

import numpy as np

from .planes import SEA_Y

HEX_A = 500.0  # hex circumradius (m); feature scale ~1/20 of range
HOLE_RADIUS = 1500.0
OUTER_RADIUS = 10000.0
TARGET_LAND_FRACTION = 0.40  # most of the disc is ocean, like Earth
HEIGHT_SPAN = 400.0
HORIZON_VERTICES = 180
TERRAIN_COLOUR = (0.38, 0.42, 0.28)  # THE terrain colour for now: one uniform albedo everywhere
HORIZON_COLOUR = TERRAIN_COLOUR  # the horizon points carry the same colour
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
    """Sampled statistics over the disc out to OUTER_RADIUS.

    The height offset is chosen so the sampled land fraction
    (samples at or above SEA_Y) equals TARGET_LAND_FRACTION --
    which puts the median elevation below sea level, like
    Earth's ocean-dominated surface.  Also computes the mean
    terrain colour (the horizon ring's far albedo).
    """
    global _STATS
    if _STATS is not None:
        return _STATS
    raw = []
    rings, sectors = 40, 90
    for ri in range(1, rings + 1):
        r = OUTER_RADIUS * ri / rings
        for si in range(sectors):
            th = 2.0 * math.pi * si / sectors
            raw.append(_height_raw(r * math.cos(th), r * math.sin(th)))
    raw_sorted = sorted(raw)
    # Offset so that the (1 - land_fraction) quantile sits at SEA_Y.
    q = raw_sorted[min(len(raw_sorted) - 1,
                       int((1.0 - TARGET_LAND_FRACTION) * len(raw_sorted)))]
    offset = SEA_Y - q
    heights = [h + offset for h in raw]
    sea_fraction = float(np.mean([1.0 if h < SEA_Y else 0.0 for h in heights]))
    colours = [_base_colour(h) for h in heights]
    mean_colour = tuple(float(np.mean([c[k] for c in colours])) for k in range(3))
    _STATS = {
        "offset": offset,
        "mean_elevation": float(np.mean(heights)),
        "median_elevation": float(np.median(heights)),
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
    # A single uniform colour for ALL distant terrain for the
    # time being (Christopher, 2026-11-02): with one albedo
    # everywhere -- mesh, fan edge, horizon -- colour blending
    # across the fan cannot be an issue at all, whatever the
    # interpolation does.  The elevation stops in _base_colour
    # are parked, not deleted: they return when the bigger
    # geometry is flawless.
    return TERRAIN_COLOUR


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
    global _EDGE_LOOP
    _EDGE_LOOP = _outer_edge_loop(tris)
    return verts, idx


_EDGE_LOOP: list | None = None


def _outer_edge_loop(tris) -> list:
    """The mesh's outer boundary as an ordered loop of points.

    Boundary edges belong to exactly one triangle; the outer
    loop is the boundary component far from the origin (the
    other component rings the 1.5 km hole).  Walked edge to
    edge into a single ordered cycle.
    """
    from collections import Counter

    counts: Counter = Counter()
    for a, b, c in tris:
        for e in ((a, b), (b, c), (c, a)):
            counts[tuple(sorted(e))] += 1
    adjacency: dict[tuple, list] = {}
    for (a, b), n in counts.items():
        if n == 1 and math.hypot(a[0], a[2]) > 6000.0:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
    start = next(iter(adjacency))
    loop = [start]
    prev, cur = None, start
    while True:
        nxt = adjacency[cur][0] if adjacency[cur][1] == prev else adjacency[cur][1]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
    return loop


def build_horizon_fan() -> tuple[np.ndarray, np.ndarray]:
    """Triangles from the mesh's outer edge loop to 180 horizon
    vertices at infinity.

    Vertex format: (aHomog vec4, aColor vec4) -- 8 floats.
    Edge vertices are the mesh's own boundary points (same
    positions, same colour_at colours), so the join is
    watertight by construction; horizon vertices are purely
    horizontal directions (w = 0, y = 0), exactly on the
    horizon line, carrying a constant medium dark green/brown
    (HORIZON_COLOUR) as albedo -- deliberately not fancy until
    the bigger geometry is flawless.  Lighting happens in the horizon shader, which
    integrates the diffuse response over a fan of normals
    tilted from straight up to 45 degrees toward the viewer.
    Each loop segment fans to the horizon vertex of its
    azimuth sector; at sector boundaries a bridging triangle
    (edge point, old horizon vertex, new horizon vertex)
    keeps the annulus between loop and horizon gapless.
    """
    if _EDGE_LOOP is None:
        build_distant_terrain()
    loop = _EDGE_LOOP
    rows = []
    for x, h, z in loop:
        c = colour_at(x, z)
        rows.append((x, h, z, 1.0, c[0], c[1], c[2], 1.0))
    horizon_base = len(rows)
    for k in range(HORIZON_VERTICES):
        th = 2.0 * math.pi * k / HORIZON_VERTICES
        rows.append((math.cos(th), 0.0, math.sin(th), 0.0,
                     HORIZON_COLOUR[0], HORIZON_COLOUR[1],
                     HORIZON_COLOUR[2], 1.0))
    verts = np.array(rows, dtype=np.float32)

    def sector(x: float, z: float) -> int:
        frac = (math.atan2(z, x) % (2.0 * math.pi)) / (2.0 * math.pi)
        return int(frac * HORIZON_VERTICES) % HORIZON_VERTICES

    idx: list[int] = []
    n = len(loop)
    for i in range(n):
        j = (i + 1) % n
        a, b = loop[i], loop[j]
        k = sector((a[0] + b[0]) / 2.0, (a[2] + b[2]) / 2.0)
        idx += [i, j, horizon_base + k]
        c = loop[(j + 1) % n]
        k_next = sector((b[0] + c[0]) / 2.0, (b[2] + c[2]) / 2.0)
        # Bridge every sector boundary the segment spans (a
        # ~500 m edge segment covers ~3 degrees, more than one
        # 2-degree sector at 180 horizon vertices).
        m = k
        while m != k_next:
            m2 = (m + 1) % HORIZON_VERTICES
            idx += [j, horizon_base + m, horizon_base + m2]
            m = m2
    return verts, np.array(idx, dtype=np.uint32)
