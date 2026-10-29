"""Rhombic dodecahedron cells on their native lattice.

A rhombic dodecahedron (RD) is the Voronoi cell of the face-centred
cubic (FCC) lattice, so "blocks" in this world live on FCC points, not
on a rectangular grid.

Coordinates
-----------
A cell centre is ``s * (i, j, k)`` with integers ``i, j, k`` and the
parity invariant ``i + j + k`` even.  The scale ``s`` is the distance
from a centre to its six 4-valent vertices (the ones on the coordinate
axes), so with the default ``s = 1`` a cell reaches 1 m above and below
its centre, and nearest neighbours are ``s * sqrt(2)`` metres apart.

The cell itself has 14 vertices:

- 6 "axis" vertices at ``(+-s, 0, 0)`` and permutations (4 faces meet),
- 8 "cube" vertices at ``(+-s/2, +-s/2, +-s/2)`` (3 faces meet),

and 12 rhombic faces, one toward each of the 12 FCC nearest neighbours
(the offsets with exactly one zero component and the rest +-1).  The
face toward neighbour ``n`` lies in the plane ``v . n = s``.

Orientation choice: a 4-valent vertex points straight up (+Y), so a
one-cell-thick layer is *not* flat -- its top is a field of 1 m peaks
over the centres, joined by rhombic slopes through saddle vertices at
height ``s/2``.  That bumpiness is intrinsic to the medium, not a bug.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

DEFAULT_S = 1.0

# The 12 nearest-neighbour offsets of the FCC lattice: exactly one zero
# component, the other two +-1.  Each satisfies the parity invariant.
NEIGHBOR_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    sorted(
        off
        for off in itertools.product((-1, 0, 1), repeat=3)
        if sum(c == 0 for c in off) == 1 and sum(abs(c) for c in off) == 2
    )
)


@dataclass(frozen=True)
class FccCoord:
    """Integer coordinates of one cell centre on the FCC lattice."""

    i: int
    j: int
    k: int

    def __post_init__(self) -> None:
        if (self.i + self.j + self.k) % 2 != 0:
            raise ValueError(
                f"({self.i}, {self.j}, {self.k}) is not an FCC point: "
                "i + j + k must be even"
            )

    def to_world(self, s: float = DEFAULT_S) -> np.ndarray:
        return s * np.array([self.i, self.j, self.k], dtype=np.float64)

    def neighbors(self) -> tuple["FccCoord", ...]:
        return tuple(
            FccCoord(self.i + di, self.j + dj, self.k + dk)
            for di, dj, dk in NEIGHBOR_OFFSETS
        )


def _vertex_table(s: float) -> tuple[np.ndarray, dict[tuple[float, float, float], int]]:
    """The 14 RD vertices (times s), plus a lookup from position to index."""
    verts: list[tuple[float, float, float]] = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            v = [0.0, 0.0, 0.0]
            v[axis] = sign * s
            verts.append(tuple(v))
    for signs in itertools.product((-0.5, 0.5), repeat=3):
        verts.append((signs[0] * s, signs[1] * s, signs[2] * s))
    index = {v: n for n, v in enumerate(verts)}
    return np.array(verts, dtype=np.float64), index


def rd_faces(s: float = DEFAULT_S):
    """The 12 rhombic faces.

    Returns a list of ``(vertex_indices, outward_normal, neighbor_offset)``
    where ``vertex_indices`` are 4 indices into :func:`_vertex_table`,
    wound counter-clockwise seen from outside.

    Construction: for the face toward neighbour ``n`` (nonzero components
    on axes ``p`` and ``q`` with signs ``sp`` and ``sq``, third axis ``r``),
    the rhombus runs axis-vertex(p) -> cube vertex with r = -s/2 ->
    axis-vertex(q) -> cube vertex with r = +s/2.
    """
    all_verts, index = _vertex_table(s)

    def key(x: float, y: float, z: float) -> int:
        return index[(x, y, z)]

    faces = []
    for r in range(3):
        p, q = (a for a in range(3) if a != r)
        for sp in (-1.0, 1.0):
            for sq in (-1.0, 1.0):
                axis_p = [0.0, 0.0, 0.0]
                axis_p[p] = sp * s
                axis_q = [0.0, 0.0, 0.0]
                axis_q[q] = sq * s

                def cube(r_sign: float) -> tuple[float, float, float]:
                    v = [0.0, 0.0, 0.0]
                    v[p] = sp * 0.5 * s
                    v[q] = sq * 0.5 * s
                    v[r] = r_sign * 0.5 * s
                    return tuple(v)

                quad = (
                    key(*axis_p),
                    key(*cube(-1.0)),
                    key(*axis_q),
                    key(*cube(+1.0)),
                )
                n = np.zeros(3)
                n[p], n[q] = sp, sq
                # The (p, q) axis pair is cyclically ordered for only
                # half the choices of r; check the winding against the
                # outward normal and reverse the quad if it is inward.
                pts = all_verts[list(quad)]
                if float(np.cross(pts[1] - pts[0], pts[2] - pts[0]) @ n) < 0.0:
                    quad = (quad[0], quad[3], quad[2], quad[1])
                faces.append((quad, n / np.linalg.norm(n), tuple(int(c) for c in n)))
    return faces


def cell_mesh(s: float = DEFAULT_S):
    """Flat-shaded mesh of one cell centred at the origin.

    Returns ``(positions, normals, indices)`` with vertices duplicated
    per face so every rhombus has a single exact face normal.
    """
    verts, _ = _vertex_table(s)
    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    indices: list[int] = []
    for quad, normal, _ in rd_faces(s):
        base = len(positions)
        for vi in quad:
            positions.append(verts[vi])
            normals.append(normal)
        indices += [base, base + 1, base + 2, base, base + 2, base + 3]
    return (
        np.array(positions, dtype=np.float64),
        np.array(normals, dtype=np.float64),
        np.array(indices, dtype=np.uint32),
    )


def _tint(coord: FccCoord) -> float:
    """Per-cell brightness variation, +/-3%.

    A splitmix-style integer hash: multiply-xorshift mixing decorre-
    lates the low bits, unlike the earlier structured hash whose low
    bits mirrored at i = 0 / k = 0 and printed quadrant seams and
    stripes onto the terrain (at +/-10%, also too strong).  This one
    is deliberately subtle -- texture, not pattern.
    """
    h = ((coord.i & 0xFFFFFFFF) * 0x9E3779B1) & 0xFFFFFFFF
    h ^= ((coord.j & 0xFFFFFFFF) * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= ((coord.k & 0xFFFFFFFF) * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0x27D4EB2F) & 0xFFFFFFFF
    h ^= h >> 15
    return 0.97 + 0.06 * ((h & 0xFFFF) / 0xFFFF)


def _face_color(normal: np.ndarray, coord: FccCoord) -> np.ndarray:
    """Face colour: base by orientation class, times the cell tint."""
    ny = float(normal[1])
    if ny > 0.35:
        base = np.array([0.36, 0.60, 0.27])  # grassy upper rhombi
    elif ny < -0.35:
        base = np.array([0.23, 0.18, 0.14])  # shadowed underside
    else:
        base = np.array([0.52, 0.40, 0.29])  # earthy sides
    return base * _tint(coord)


def rotation_y_to(direction) -> np.ndarray:
    """Rotation matrix carrying +Y onto the given direction (Rodrigues)."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(y, d)
    s = float(np.linalg.norm(v))
    c = float(y @ d)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


class GroundPatch:
    """A one-cell-thick layer of cells, about 30 m across.

    This is the {100} cleavage: centres sit at j = 0 with i, k in
    [-half, +half] and i + k even (the parity invariant with j = 0),
    so a 4-valent cell vertex points straight up.  The layer's peaks
    reach y = +s and its bottom points reach y = -s.

    An optional rotation reorients the whole lattice (cell geometry
    and centre positions alike) and an optional offset translates the
    patch in the world; GroundPatch111 uses both.
    """

    def __init__(
        self,
        half: int = 15,
        s: float = DEFAULT_S,
        rotation: np.ndarray | None = None,
        offset=(0.0, 0.0, 0.0),
    ):
        self.s = s
        self.rotation = np.eye(3) if rotation is None else rotation
        self.offset = np.asarray(offset, dtype=np.float64)
        self.cells: list[FccCoord] = self._lattice_coords(half)

    def _lattice_coords(self, half: int) -> list[FccCoord]:
        return [
            FccCoord(i, 0, k)
            for i in range(-half, half + 1)
            for k in range(-half, half + 1)
            if (i + k) % 2 == 0
        ]

    def center_world(self, coord: FccCoord) -> np.ndarray:
        return self.offset + self.rotation @ coord.to_world(self.s)

    def build_mesh(self):
        return build_cells_mesh(self.cells, self.s, self.rotation, self.offset)


def build_cells_mesh(cells, s: float, rotation: np.ndarray, offset) -> tuple:
    """Merge a set of cells into one static mesh.

    Interleaved per vertex: position, flat (per-face) normal, smooth
    (Gouraud) normal, color -- 12 floats.  The smooth normal at a
    surface point is the normalised sum of the flat normals of every
    face meeting there, across the whole set; faces shared between
    two cells are interior, come in opposite pairs, and cancel out of
    the sum by themselves.  (Verified: at interior top vertices of
    both patch cleavages the sum is exactly vertical.)
    """
    pos, nrm, idx = cell_mesh(s)
    rot = rotation
    off = np.asarray(offset, dtype=np.float64)
    # First pass: world positions and flat normals.
    placed: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for coord in cells:
        origin = off + rot @ coord.to_world(s)
        for n in range(len(pos)):
            world_n = rot @ nrm[n]
            placed.append(
                (rot @ pos[n] + origin, world_n, _face_color(world_n, coord))
            )
    # Smooth normals: group vertices by world position.
    sums: dict[tuple[float, float, float], np.ndarray] = {}
    for p, n, _ in placed:
        key = (round(float(p[0]), 6), round(float(p[1]), 6), round(float(p[2]), 6))
        sums[key] = sums.get(key, np.zeros(3)) + n
    verts_out: list[np.ndarray] = []
    idx_out: list[int] = []
    for cell_no in range(len(cells)):
        base = cell_no * len(pos)
        for n in range(len(pos)):
            p, nrm_flat, color = placed[base + n]
            key = (round(float(p[0]), 6), round(float(p[1]), 6), round(float(p[2]), 6))
            smooth = sums[key]
            norm = np.linalg.norm(smooth)
            smooth = smooth / norm if norm > 0 else nrm_flat
            verts_out.append(np.concatenate([p, nrm_flat, smooth, color]))
        idx_out.extend(base + int(t) for t in idx)
    return (
        np.array(verts_out, dtype=np.float32),
        np.array(idx_out, dtype=np.uint32),
        )


# The {111} patch's frame, shared by the patch and its wall.
PATCH111_ROTATION = rotation_y_to((1.0, 1.0, 1.0)).T
PATCH111_OFFSET = (0.0, 0.0, -75.0)


class GroundPatch111(GroundPatch):
    """The other cleavage: a {111} layer of the same lattice.

    Centres are the FCC points with i + j + k = 0 -- a triangular
    lattice in the (111) plane -- and the whole lattice is rotated so
    that the (1, 1, 1) direction points straight up.  A 3-valent
    (cube) vertex of each cell then tops the layer, at height
    s * sqrt(3) / 2 ~= 0.866 s instead of the {100} layer's 1.0 s,
    and the surface pattern is threefold rather than fourfold --
    built so the two cleavages can be compared side by side.
    """

    def __init__(
        self,
        radius: float = 16.0,
        s: float = DEFAULT_S,
        offset=PATCH111_OFFSET,
    ):
        self.radius = radius
        super().__init__(
            half=0,
            s=s,
            # PATCH111_ROTATION is the transpose of "y onto (1,1,1)",
            # i.e. "(1,1,1) onto y": centres with i+j+k = 0 then land
            # in the world y = 0 plane, and each cell's (1,1,1)
            # vertex points up.
            rotation=PATCH111_ROTATION,
            offset=offset,
        )

    def _lattice_coords(self, half: int) -> list[FccCoord]:
        out: list[FccCoord] = []
        lim = int(self.radius / self.s) + 2
        for i in range(-lim, lim + 1):
            for j in range(-lim, lim + 1):
                coord = FccCoord(i, j, -(i + j))
                p = self.rotation @ coord.to_world(self.s)
                if math.hypot(float(p[0]), float(p[2])) <= self.radius:
                    out.append(coord)
        return out


def wall_cells_100() -> list[FccCoord]:
    """Wall segment for the {100} patch: ~4 courses high, ~5 blocks long.

    A one-cell-thick slab in the k = 0 plane, sitting on the ground
    layer (j = 0 is the patch itself).  Vertical stacking in this
    frame is staggered by the FCC parity: course j holds the cells
    with i + j even, each nestled between the two below it.
    """
    return [
        FccCoord(i, j, 0)
        for i in range(-5, 6)
        for j in range(1, 5)
        if (i + j) % 2 == 0
    ]


# Course offsets for stacking along (1, 1, 1): the steps cycle
# (1,1,0) -> (1,0,1) -> (0,1,1), the three nestling directions.
# After three courses the cumulative offset is (2,2,2) -- pure
# (1,1,1), zero sideways shift -- so the wall zigzags by at most
# ~0.82 m but rises straight on average instead of leaning forever.
_COURSE_STEPS = [(1, 1, 0), (1, 0, 1), (0, 1, 1)]


def _course_offset(course: int) -> tuple[int, int, int]:
    x = y = z = 0
    for c in range(course):
        sx, sy, sz = _COURSE_STEPS[c % 3]
        x, y, z = x + sx, y + sy, z + sz
    return x, y, z


def wall_cells_111() -> list[FccCoord]:
    """Wall segment for the {111} patch: 4 courses, 5 cells per course.

    Course c lives in the layer i+j+k = 2c, shifted by the cycling
    course offset (see _COURSE_STEPS).  Course 0 would duplicate
    patch cells, so courses run 1..4 on top of the patch.
    """
    cells = []
    for c in range(1, 5):
        ox, oy, oz = _course_offset(c)
        cells += [FccCoord(m + ox, -m + oy, oz) for m in range(5)]
    return cells


def wall_cells_111_b() -> list[FccCoord]:
    """Second {111} wall, adjoining the first at 90 degrees.

    The {111} plane's lattice directions meet at 60/120 degrees, so
    no lattice row runs at 90 degrees to wall A's (1,-1,0).  This
    wall follows the (1,1,-2) direction -- exactly perpendicular to
    (1,-1,0) within the plane -- by staircasing: steps alternate
    (1,0,-1) and (0,1,-1), each a nearest-neighbour step, zigzagging
    about the 90-degree line with ~0.71 m amplitude.  Same cycling
    course offsets as wall A, so it also rises straight on average.
    It starts from wall A's corner cell and runs ~10 m.
    """
    base = [(0, 0, 0)]
    steps = [(1, 0, -1), (0, 1, -1)]
    for n in range(8):
        sx, sy, sz = steps[n % 2]
        px, py, pz = base[-1]
        base.append((px + sx, py + sy, pz + sz))
    cells = []
    for c in range(1, 5):
        ox, oy, oz = _course_offset(c)
        for px, py, pz in base[1:]:
            cells.append(FccCoord(px + ox, py + oy, pz + oz))
    return cells


def wall_cells_100_b() -> list[FccCoord]:
    """Second {100} wall, adjoining wall A at 90 degrees.

    In the cubic frame 90 degrees is a natural direction (unlike
    {111}): this wall runs along k at wall A's end column (i = 5),
    with the same staggered courses.
    """
    return [
        FccCoord(5, j, k)
        for j in range(1, 5)
        for k in range(0, 11)
        if (5 + j + k) % 2 == 0
    ]


def column_cells_100() -> list[FccCoord]:
    """The best single column on the {100} patch: 8 courses up.

    A cell directly overhead is not a lattice cell, so the column
    alternates (1,1,0) and (-1,1,0) steps -- each course face-bonds
    to the one below and the sideways wobble cancels every two
    courses.  Stands on the patch cell (10, 0, 10).
    """
    return [FccCoord(10 + (j % 2), j, 10) for j in range(1, 9)]


def column_cells_111() -> list[FccCoord]:
    """The best single column on the {111} patch: 8 courses up.

    Uses the cycling course offsets (ABC stacking), so after every
    three courses the column is exactly over its base again; the
    sideways excursion never exceeds ~0.82 m.  Stands on the patch
    cell (-6, 3, 3).
    """
    bx, by, bz = -6, 3, 3
    cells = []
    for c in range(1, 9):
        ox, oy, oz = _course_offset(c)
        cells.append(FccCoord(bx + ox, by + oy, bz + oz))
    return cells
