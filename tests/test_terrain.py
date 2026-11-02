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
        # Ocean-dominated, like Earth: the land fraction is the
        # target, and the median elevation is below sea level.
        self.assertAlmostEqual(
            s["land_fraction"], terrain.TARGET_LAND_FRACTION, delta=0.03)
        self.assertLess(s["median_elevation"], SEA_Y)
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

    def test_horizon_fan(self):
        verts, idx = terrain.build_horizon_fan()
        self.assertEqual(verts.shape[1], 8)  # aHomog vec4 + aColor vec4
        n_h = terrain.HORIZON_VERTICES
        edge, horizon = verts[:-n_h], verts[-n_h:]
        self.assertEqual(len(horizon), terrain.HORIZON_VERTICES)
        self.assertTrue(np.all(edge[:, 3] == 1.0))  # finite
        self.assertTrue(np.all(horizon[:, 3] == 0.0))  # at infinity
        # Horizon points lie exactly on the horizon: purely
        # horizontal directions, evenly spaced in azimuth.
        self.assertTrue(np.all(horizon[:, 1] == 0.0))
        ang = np.sort(np.arctan2(horizon[:, 2], horizon[:, 0]))
        np.testing.assert_allclose(
            np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi])),
            2 * np.pi / terrain.HORIZON_VERTICES, atol=1e-6)
        # Albedo: horizon vertices uniform constant colour;
        # edge vertices carry their local terrain colour.
        self.assertTrue(np.all(verts[:, 7] == 1.0))
        np.testing.assert_allclose(
            horizon[:, 4:7],
            np.tile(np.array(terrain.HORIZON_COLOUR, dtype=np.float32),
                    (terrain.HORIZON_VERTICES, 1)), atol=1e-6)
        np.testing.assert_allclose(
            edge[0, 4:7],
            np.array(terrain.colour_at(edge[0, 0], edge[0, 2]),
                     dtype=np.float32),
            atol=1e-6)
        # Watertight by construction: every edge vertex position
        # is a vertex of the distant-terrain mesh itself.
        mverts, _ = terrain.build_distant_terrain()
        mesh_pos = {
            (round(float(x), 2), round(float(y), 2), round(float(z), 2))
            for x, y, z in mverts[:, 0:3]
        }
        for x, y, z in edge[:, 0:3]:
            self.assertIn(
                (round(float(x), 2), round(float(y), 2), round(float(z), 2)),
                mesh_pos)
        # The edge loop rings the mesh near the outer radius,
        # zigzagging with the hex boundary rather than sitting
        # on the ideal circle.
        r = np.hypot(edge[:, 0], edge[:, 2])
        self.assertGreater(float(r.min()), terrain.OUTER_RADIUS - 1200.0)
        self.assertLess(float(r.max()), terrain.OUTER_RADIUS + terrain.HEX_A + 1.0)
        # Every horizon vertex is used by at least one triangle.
        used = set(int(i) for i in idx)
        for k in range(terrain.HORIZON_VERTICES):
            self.assertIn(len(edge) + k, used)


if __name__ == "__main__":
    unittest.main()
