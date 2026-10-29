"""Rhombic-dodecahedron block world prototype.

Infinity-first rendering: an infinite-far-plane projection matrix,
a skybox made of direction (w = 0) vertices, infinite ground planes
whose horizon vertices also carry w = 0, and a first local patch of
rhombic dodecahedron cells on their native FCC lattice.
"""

__all__ = ["math3d", "lattice", "planes"]
