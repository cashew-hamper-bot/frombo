# The look Jacobian: grab-the-world dragging without a solver

Both view-rotation interactions — grab-the-world drag and the
Minecraft-like captured look — are driven by one closed-form
*derivative*: the Jacobian of the glued-cursor condition. This
note derives it, states exactly what is precomputed where, and
records the verification numbers. It replaces an earlier exact
per-event solver (`drag_glue`: a 4x4 matrix inversion, an `acos`
branch chosen between two candidate solutions, and an `atan2`
difference, per mouse event), whose cost and branch behaviour
made rotation feel jerky next to the frame-integrated WASD
translation.

## Setup

Camera rotation `R = Ry(yaw) @ Rx(pitch)`, looking down `-Z`.
For a cursor at pixel `(px, py)` in a `W x H` widget:

```
t = tan(fovY / 2),        a = W / H
u = 2*px/W - 1,           v = 1 - 2*py/H        (NDC)
X = a*t*u,                Y = t*v
```

The camera-space ray through the cursor is `c = (X, Y, -1)`.

## Derivation

Grab-the-world means the world direction under the cursor does
not move as the camera responds: differentiating `d = R c` and
requiring `dd = 0` gives, in camera space,

```
(e x c) * dyaw + (xh x c) * dpitch + dc  is parallel to  c
```

where `e = (0, cos p, -sin p)` is the yaw axis expressed in
camera space and `dc = (dX, dY, 0)`. Projecting onto
`(1, 0, X)` and `(0, 1, Y)` — both perpendicular to `c` —
removes the unknown length-change component along `c` and
leaves a 2x2 linear system whose determinant collapses:

```
det = (1 + X^2 + Y^2) * sigma,     sigma = cos(p) - sin(p)*Y
```

Inverting (constant factors only — no matrix work at run time):

```
dyaw   = [ (1 + Y^2) * dX - X*Y * dY ] / (lam^2 * sigma)
dpitch = [ X*(sin p + cos p*Y) * dX
           - (cos p*(1 + X^2) - sin p*Y) * dY ] / (lam^2 * sigma)
lam^2  = 1 + X^2 + Y^2
```

In pixel units one more simplification falls out: `dX =
(2*a*t/W) dx` and `dY = -(2*t/H) dy`, and since `a/W = 1/H`
both axes share a *single* pixel scale `k = 2*t/H`:

```
dyaw   = k * [ (1 + Y^2) * dx + X*Y * dy ] / (lam^2 * sigma)
dpitch = k * [ X*(sin p + cos p*Y) * dx
               + (cos p*(1 + X^2) - sin p*Y) * dy ] / (lam^2 * sigma)
```

Checks: at screen centre (`X = Y = 0`) this reduces to
`dyaw = k*dx / cos(pitch)`, `dpitch = k*dy` — the familiar FPS
rates, including the `1/cos(pitch)` yaw gain when pitched.
Mouse right gives yaw+, mouse down gives pitch+ (grab sense);
captured look is the same Jacobian negated, evaluated at the
centre, where the warped cursor lives.

## The singularity

`sigma = 0` exactly when the ray through the cursor is
parallel to the world Y axis — grabbing a point at the zenith
or nadir through an off-centre cursor — where yaw genuinely
cannot move the grabbed point. That is the only special case:
`sigma` is clamped to `|sigma| >= 0.10` with its sign
preserved, gains stay bounded, and the error is confined to
that cone. Pitch itself is clamped to +-1.55 rad as before.

## Precompute map

- Startup (the projection never changes; `fovY` is a module
  constant): `t = tan(fovY / 2)`.
- `resizeGL`: `W`, `H`, `a`, and the pixel scale `k = 2*t/H`.
- Per mouse event: `X`, `Y` from the cursor position;
  `lam^2`, `sigma`; one sin/cos pair for the current pitch
  (recomputed only because pitch itself just changed).
  ~20 flops. No matrices, no inversions, no candidate
  comparisons, no iteration.

## The trade, quantified

The old solver was exact per event; the Jacobian is exact in
the limit, with second-order error per step. Measured against
the old solver as ground truth (~2,000 random drags outside
the singular band, `|sigma| >= 0.30` for the step study):

| step size | worst glue error per event |
|-----------|----------------------------|
| 0.6 px    | 0.0001 deg                 |
| 6 px      | 0.015 deg                  |

The error shrinks quadratically with step size, and unlike
the branch-picking solver the Jacobian cannot jump solutions.

## Measuring smoothness

`tools/look_harness.py` drives the real widget with synthetic
event streams (constant-velocity drag paths and capture
streams) and reports the per-frame rotation increments:
mean, standard deviation, and max/min — the numbers that
"feels jerky" has to answer to — plus total rotation against
the analytic expectation and glued-point drift over a drag.
Run it under the same Xvfb/software-GL environment as
`tools/smoke_render.py`.
