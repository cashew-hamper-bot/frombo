# Voxel shapes: spheres, waists, and context

Design note (exploration, 2026-11-02). The rhombic dodecahedron
is the *logical* shape of a voxel here — it is the Voronoi cell
of the FCC lattice, the tiling with no gaps. But the *rendered*
shape need not be the cell. Minecraft already renders context-
dependent voxel shapes (stairs, slabs); this note explores the
analogous move for this lattice: render the idea of the local
voxel configuration, compromised toward the lattice.

## Cases

- **One voxel surrounded by air: a sphere.** The sphere is the
  idealised voxel; the dodecahedron is its Voronoi truth. Render
  the sphere.
- **Two connected voxels: two spheres and a waist.** Joined by a
  neck *narrower* than a capsule's. A fragment of a torus makes
  the prettiest waist but is not a quadric, which makes it hard
  to render with imposters. The quadric with a waist is the
  **hyperboloid of one sheet** — a true quadric (one quadratic
  equation), ruled, with a tunable neck radius and flare. So:
  spheres + hyperboloid necks, all quadrics.
- **Voxels on flat ground: flatten the contact.** Where the
  configuration rests on a plane, the rendered shape should meet
  it with a flat face in that direction — the sphere squashes /
  is cut by the support plane.
- **A pole (vertical run): spheres on hyperboloid necks, nudged
  toward the pole's axis.** Not all the way — a compromise
  parameter (0 = pure lattice positions, 1 = fully centred on
  the local medial axis) keeps the shape honest about the
  lattice underneath while shedding the staircase look.

## Rendering route (sketch)

Imposters: one bounding volume per voxel (or per small cluster),
with the fragment shader solving the ray/quadric intersection
analytically — spheres and hyperboloids share the same quadratic
solve, and exact normals come from the quadric gradient. The
open problem is the union: neighbouring quadrics must blend or
meet cleanly (smooth-minimum blends along the necks are the
obvious candidate), and that is where the research risk sits.

## Status

Exploration only. The practice pole and walls still render as
plain lattice cells; this note is the direction, not a plan with
dates.
