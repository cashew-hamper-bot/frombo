"""Distant hex-lattice terrain: geometry bounds and statistics."""

import math
import unittest

import numpy as np

from rhombic_game import terrain
from rhombic_game.planes import SEA_Y


class TerrainTests(unittest.TestCase):
    def test_height_is_deterministic_and_finite(self):
        for x, z in [(0.0, 0.0), (3000.0, -4500.0), (-9999.0, 123.0)]:
            h = terrain.height(x, z)
            self.assertTrue(math.isfinite(h))
            self.assertEqual(h, terrain.height(x, z))

    def test_stats(self):
        s = terrain.terrain_stats()
        # Mean elevation is pinned above sea level by construction.
        self.assertAlmostEqual(s["mean_elevation"], terrain.TARGET_MEAN_ELEVATION)
        self.assertGreater(s["mean_elevation"], SEA_Y)
        # A real coastline: some sea, mostly land.
        self.assertGreater(s["sea_fraction"], 0.05)
        self.assertLess(s["sea_fraction"], 0.60)
        self.assertAlmostEqual(s["land_fraction"], 1.0 - s["sea_fraction"])

    def test_mesh_annulus_bounds(self):
        verts, idx = terrain.build_distant_terrain()
        self.assertEqual(verts.shape[1], 13)  # block vertex format
        self.assertEqual(len(idx), len(verts))  # non-indexed fan
        r = np.hypot(verts[:, 0], verts[:, 2])
        # Corners of the innermost hexes may reach one radius
        # inside the hole; nothing starts inside that.
        self.assertGreaterEqual(float(r.min()), terrain.HOLE_RADIUS - terrain.HEX_A - 1.0)
        self.assertLessEqual(float(r.max()), terrain.OUTER_RADIUS + terrain.HEX_A + 1.0)
        self.assertTrue(np.all(np.isfinite(verts)))
        # AO is 1 everywhere: no occlusion in the far field.
        self.assertTrue(np.all(verts[:, 12] == 1.0))

    def test_horizon_ring(self):
        verts, idx = terrain.build_horizon_ring()
        self.assertEqual(verts.shape[1], 8)  # aHomog vec4 + aColor vec4
        inner = verts[0::2]
        outer = verts[1::2]
        self.assertTrue(np.all(inner[:, 3] == 1.0))  # finite
        self.assertTrue(np.all(outer[:, 3] == 0.0))  # at infinity
        # Infinity points lie exactly on the horizon: their
        # direction is purely horizontal.
        self.assertTrue(np.all(outer[:, 1] == 0.0))
        land = terrain.terrain_stats()["land_fraction"]
        np.testing.assert_allclose(outer[:, 7], land, atol=1e-6)
        self.assertTrue(np.all(inner[:, 7] == 1.0))
        r = np.hypot(inner[:, 0], inner[:, 2])
        np.testing.assert_allclose(r, terrain.OUTER_RADIUS, atol=1.0)


if __name__ == "__main__":
    unittest.main()
