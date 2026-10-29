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
    """Deterministic per-cell brightness variation in [0.90, 1.10]."""
    h = (coord.i * 73856093) ^ (coord.j * 19349663) ^ (coord.k * 83492791)
    return 0.90 + 0.20 * ((h & 0xFFFF) / 0xFFFF)


def _face_color(normal: np.ndarray, coord: FccCoord) -> np.ndarray:
    ny = float(normal[1])
    if ny > 0.35:
        base = np.array([0.36, 0.60, 0.27])  # grassy upper rhombi
    elif ny < -0.35:
        base = np.array([0.23, 0.18, 0.14])  # shadowed underside
    else:
        base = np.array([0.52, 0.40, 0.29])  # earthy sides
    return base * _tint(coord)


class GroundPatch:
    """A one-cell-thick layer of cells, about 30 m across.

    Centres sit at j = 0 with i, k in [-half, +half] and i + k even
    (the parity invariant with j = 0).  The layer's peaks reach y = +s
    and its bottom points reach y = -s.
    """

    def __init__(self, half: int = 15, s: float = DEFAULT_S):
        self.s = s
        self.cells: list[FccCoord] = [
            FccCoord(i, 0, k)
            for i in range(-half, half + 1)
            for k in range(-half, half + 1)
            if (i + k) % 2 == 0
        ]

    def build_mesh(self):
        """Merged static mesh.

        Interleaved per vertex: position, flat (per-face) normal,
        smooth (Gouraud) normal, color -- 12 floats.  The smooth
        normal at a surface point is the normalised sum of the flat
        normals of every face meeting there, across the whole patch;
        faces shared between two cells are interior, come in opposite
        pairs, and cancel out of the sum by themselves.
        """
        pos, nrm, idx = cell_mesh(self.s)
        # First pass: world positions and flat normals.
        placed: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for coord in self.cells:
            origin = coord.to_world(self.s)
            for n in range(len(pos)):
                placed.append((pos[n] + origin, nrm[n], _face_color(nrm[n], coord)))
        # Smooth normals: group vertices by world position.
        sums: dict[tuple[float, float, float], np.ndarray] = {}
        for p, n, _ in placed:
            key = (round(float(p[0]), 6), round(float(p[1]), 6), round(float(p[2]), 6))
            sums[key] = sums.get(key, np.zeros(3)) + n
        verts_out: list[np.ndarray] = []
        idx_out: list[int] = []
        for cell_no, coord in enumerate(self.cells):
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
