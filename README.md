# Rhombic Dodecahedron World — first slice

A Minecraft-like voxel world, except the blocks are **rhombic
dodecahedra** on their native FCC lattice, written in Python with
PySide6 + OpenGL — and built **infinity-first**: nothing in the scene
pretends to be far away by being very large and finite. Things that
are at infinity are submitted to the GPU *at* infinity, as homogeneous
vertices with w = 0, under a projection matrix whose far plane is at
infinity.

## The medium-term vision (the sketch this slice serves)

- Infinity projection matrix; daytime skybox at infinity.
- One **complete** infinite plane at bedrock level, textured well
  enough that fast flight over it reads as motion.
- One **partial** infinite plane at sea level, with a hole roughly
  1000 m wide around the player.
- Lazily updated spherical billboards at 1000 m and 10000 m.
- Eventually, distant-horizons-style 3D LODs for distant terrain.
- Local terrain made of rhombic dodecahedron cells on a rhombic
  dodecahedral (FCC) lattice — *not* a rectangular grid, and not flat:
  a one-cell layer has 1 m peaks over each centre joined by rhombic
  slopes through saddles at half height.

This first implementation covers: static skybox, both infinite
planes, and a static, deliberately unoptimised one-layer ground patch
about 30 m × 30 m. Billboards and LODs are future slices.

## Run it

```sh
pip install -r requirements.txt
python -m rhombic_game
python -m unittest discover -s tests -v   # headless math tests
python tools/smoke_render.py out.png      # offscreen one-frame render
```

Python ≥ 3.10. Controls:

| input | action |
|---|---|
| W A S D | move (left hand), in the current horizontal plane |
| I J K L | move (right hand — for left-mousers), same plane |
| Space / E / U | up |
| C / Q / O | down |
| drag, primary mouse button | look, grab-the-world style, exact 1:1 tracking at infinity |
| wheel | fly speed |
| G | toggle flat / Gouraud shading on the ground patch |
| Esc | quit |

Movement uses the view azimuth only — looking down and pressing W
does not descend. Dragging uses the grab-the-world convention of
panorama viewers and map apps, the opposite sense from FPS
mouse-look: drag right pulls the world right, so the view turns
left. It is also *exact*: each drag step rotates the camera by the
rotation that carries the direction under the cursor onto the
direction previously under it, so an object at infinity stays glued
to the pointer at any screen position (no fixed radians-per-pixel
rate — the same drag travels further at wide FOV edges than at the
centre, as it should).

Running from an IDE: the canonical entry is `python -m rhombic_game`
from the project root. To run a file directly (e.g. PyCharm's Run
button), run `run.py` at the project root, or make a run
configuration whose "Module name" is `rhombic_game` — the package's
own modules use relative imports and cannot be run as scripts.

## The infinity math

**Infinite projection** (`math3d.perspective_infinite`) is the finite
perspective matrix at the limit far → ∞: the third row becomes
`[0, 0, -1, -2·near]`. A finite point's NDC z approaches +1 as it
recedes but never reaches it; a direction (w = 0) straight ahead maps
to NDC z *exactly* +1.

**Skybox** (`planes.skybox_cube`, `shaders.SKY_*`): the cube's
vertices are direction vectors, submitted as `vec4(dir, 0.0)`. The
view matrix's translation is multiplied by that 0, so no
"rotation-only view matrix" special case exists anywhere in the code;
the sky is coloured from direction alone (gradient, sun disc, corona)
and drawn first with depth-write off and `LEQUAL`, since it lands
exactly on the far plane.

**Infinite planes** (`planes.py`, `shaders.PLANE_*`): both planes are
triangle strips of *homogeneous world positions*. Finite vertices
carry w = 1; horizon vertices carry w = 0 with a horizontal direction
(the limit of a plane point receding to infinity: its y/w → the plane
height's vanishing line, direction purely horizontal). The vertex
shader passes the homogeneous position through as a varying; the
fragment shader recovers the true world point as `xyz / w`, which is
exact at every distance — texturing never distorts and never runs
out, and there is no giant finite quad to clip or z-fight.

- The **bedrock plane** (y = −100, about 100 m below the terrain
  layer) is complete, so 4 vertices suffice: the strip is one quad —
  a finite grounding segment (two w = 1 vertices) set behind and wide
  around the point below the camera, opening to two w = 0 horizon
  vertices at wide (~80°) forward-left and forward-right angles.
  Re-anchored each frame to the camera's ground position, heading,
  and height above the plane (looking steeply down, the bottom of the
  screen shows ground a little way behind the camera's ground point,
  and from altitude the visible near-ground region is wide — the
  segment's back-off and width scale with height to cover both; this
  is coverage only, since shading is analytic). The 8 m checkerboard
  (plus 1 m / 8 m grid lines) is the texture that makes fast flight
  legible; the checker is box-filtered over each fragment's actual
  footprint (from the pattern coordinates' screen-space derivatives),
  so at grazing angles the across-view cells survive as stripes
  instead of averaging to grey — a procedural answer to anisotropic
  filtering. The grid lines are gated by footprint too: a line
  pattern sampled below its own period aliases into moiré bands no
  matter how it is faded by distance, so lines now vanish once their
  cells go sub-pixel. Both planes haze out to the *same* colour the
  skybox uses at the horizon, so plane and sky meet without a seam
  band.
- The **sea plane** (y = −10, about 10 m below the terrain layer)
  has the hole: its strip walks once
  around the player — 12 finite rim vertices on a 500 m-radius
  dodecagon, each followed by the horizon vertex in the same azimuth,
  plus a repeated first pair to close the loop (13 + 13 vertices,
  alternating w = 1, 0, 1, 0, …). The rim follows the player's XZ, so
  the hole travels with the player while the plane still reaches the
  horizon in every direction.

## The local ground

`lattice.py` is the seed of the object model — deliberately small,
but with the real nouns in place:

- `FccCoord(i, j, k)` — a cell centre at `s·(i, j, k)`, with the FCC
  parity invariant `i + j + k` even enforced in the constructor.
  Centres have exactly 12 nearest neighbours at `s·√2`, at the offsets
  with exactly one zero component.
- The cell geometry is derived, not tabulated: 14 vertices (6 axis
  vertices at ±s, 8 cube vertices at ±s/2) and 12 rhombic faces
  generated from the neighbour offsets — the face toward neighbour
  `n` lies in the plane `v · n = s`. Each rhombus has diagonals in
  ratio √2 : 1. Tests pin all of this down, including the cell volume
  (2 m³ at s = 1, the FCC primitive-cell volume).
- `GroundPatch` — one layer (`j = 0`), centres spanning ±15 m,
  481 cells, merged into a single static VBO with per-cell tint
  variation and *two* normals per vertex: the flat per-face normal,
  and a smooth (Gouraud) normal averaged over every face meeting at
  that point (interior faces shared between cells cancel in pairs).
  The G key cross-fades between them at runtime — flat shows the true
  rhombic facets; Gouraud softens the field of peaks considerably.
  No instancing, no culling: optimisation is a later slice, on purpose.
- `GroundPatch111` — a second patch of the same lattice with the
  other cleavage: centres on the (111) lattice plane (`i+j+k = 0`),
  the whole lattice rotated so (1, 1, 1) points up. A 3-valent vertex
  tops each cell at 0.866 m instead of 1.0 m, in a threefold pattern.
  It sits 75 m ahead of the first patch so the two surfaces can be
  compared in one flight.

Orientation choice, made once here: a 4-valent (axis) vertex points
straight up, so the layer's top is peaks at y = +1 m over the centres
with saddles at y = +0.5 m. The alternative (a 3-valent vertex up)
exists; nothing else in the code depends on the choice.

## Layout

```
rhombic_game/
  math3d.py    infinite projection, camera, view matrix
  lattice.py   FccCoord, rhombic dodecahedron geometry, GroundPatch
  planes.py    infinite-plane strips + skybox cube (homogeneous verts)
  shaders.py   GLSL: sky / bedrock / sea / blocks
  app.py       QOpenGLWidget renderer, input, main window
tests/test_core.py   headless tests (numpy only)
tools/smoke_render.py  offscreen render to PNG, for eyeballing changes
```

## Open questions, deliberately not settled yet

- **Nomenclature.** `FccCoord`, "cell", "rim/horizon vertex" are
  seedlings. A richer object model (regions/chunks on an FCC lattice,
  what a "layer" means vertically given the parity invariant) should
  grow from use, not be designed up front.
- **Levels in cell terms.** Sea (−10 m) and bedrock (−100 m) are
  currently round metric offsets from the layer's centre plane;
  whether they should be defined *in cell terms* is open.
- **The grounding segment's width** for the bedrock strip is a
  coverage detail only (shading comes from the recovered world
  position), but its interaction with extreme pitch/FOV corners could
  stand a second look.
- **Billboards at 1000 m / 10000 m**, and how "lazily updated" is
  scheduled, are untouched.
- **Distant LODs** in the distant-horizons sense, on an FCC lattice,
  are the big open design problem of the project.
