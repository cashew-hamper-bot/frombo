# Sparse terrain: critical points and Hermite reconstruction

Design note (exploration, 2026-11-01). The distant terrain is
currently a dense hexagonal mesh sampled from a closed-form
height function. The long-term idea is the opposite: a *sparse*
representation from which a perfectly smooth surface can be
reconstructed — no dense grid, no jagged facets.

## Two separate problems

**Extraction (sparsification)** and **reconstruction** are
independent tasks with independent quality measures; conflating
them muddies both.

- Extraction: find the terrain's most significant critical
  points — local maxima, minima, and saddle points of the
  height field. "Most significant" has a principled meaning via
  persistence: in a flood-fill of the height field, critical
  points are born and annihilate in pairs; a pair that persists
  over a large elevation range is a landmark, a short-lived
  pair is noise. Because our height function is closed-form,
  critical points can be located by coarse sampling plus
  Newton refinement against the analytic gradient, not merely
  estimated from a grid.
- Reconstruction: build a smooth surface that honours the
  sparse samples.

## Reconstruction tiers

The touchstone is 1D Hermite interpolation: given value and
slope at each sample point, a cubic reproduces a smooth curve
far better than value-only (linear or spline) reconstruction at
the same density. The direct 2.5D analog gives each terrain
sample a value plus a gradient:

1. **Elevation only** — points with height. (Baseline tier;
   smooth interpolants exist, but they invent the slopes.)
2. **Elevation + normal** — points with height and surface
   normal (equivalently, the height gradient). The Hermite
   tier: per-point value+gradient constraints, e.g.
   Clough–Tocher / HCT-style macro-elements or Hermite radial
   basis functions; on the GPU, plausibly a tessellation stage
   driven by the sparse control data.
3. **Elevation + normal + curvature** — add the Hessian
   (curvature tensor) per point. Possibly overengineering;
   decide empirically, not by taste.

## The study that decides

For each tier, plot rendering quality (beauty — judged on
renders, with height-error metrics as a sanity check) against
data density (points per km²). The interesting output is the
knee of each curve: how few points carry the terrain at each
tier, and whether curvature earns its bytes.

## Status

Not started. The dense hex mesh remains the working baseline;
the first cheap spike is extraction alone: compute the current
height field's critical points over the disc, rank by
persistence, and count how many survive at various thresholds.
