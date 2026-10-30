"""Headless tests for the math that carries the design.

Run with:  python -m unittest discover -s tests -v
(Only numpy is needed -- no Qt, no GL context.)
"""

import math
import unittest

import numpy as np

from rhombic_game import planes
from rhombic_game.lattice import (
    NEIGHBOR_OFFSETS,
    FccCoord,
    GroundPatch,
    GroundPatch111,
    _vertex_table,
    cell_mesh,
    rd_faces,
)
from rhombic_game.math3d import Camera, look_delta, perspective_infinite, ray_direction


def ndc(m: np.ndarray, p: np.ndarray) -> np.ndarray:
    clip = m @ p
    return clip[:3] / clip[3]


class InfiniteProjectionTests(unittest.TestCase):
    def setUp(self):
        self.m = perspective_infinite(math.radians(70.0), 16 / 9, 0.1)

    def test_near_plane_maps_to_minus_one(self):
        z = ndc(self.m, np.array([0.0, 0.0, -0.1, 1.0]))[2]
        self.assertAlmostEqual(z, -1.0)

    def test_far_finite_points_approach_but_never_reach_one(self):
        for dist in (10.0, 1_000.0, 1_000_000.0):
            z = ndc(self.m, np.array([0.0, 0.0, -dist, 1.0]))[2]
            self.assertLess(z, 1.0)
            self.assertGreater(z, 0.9)

    def test_point_at_infinity_lands_exactly_on_far_plane(self):
        # A direction straight ahead: w = 0.
        z = ndc(self.m, np.array([0.0, 0.0, -1.0, 0.0]))[2]
        self.assertAlmostEqual(z, 1.0)

    def test_view_matrix_places_camera_correctly(self):
        cam = Camera(position=np.array([0.0, 7.0, 20.0]), yaw=0.0, pitch=0.0)
        eye = cam.view_matrix() @ np.array([0.0, 7.0, 19.0, 1.0])
        np.testing.assert_allclose(eye, [0.0, 0.0, -1.0, 1.0], atol=1e-9)
        # A direction is rotated but never translated.
        d = cam.view_matrix() @ np.array([0.0, 0.0, -1.0, 0.0])
        np.testing.assert_allclose(d, [0.0, 0.0, -1.0, 0.0], atol=1e-9)


class LatticeTests(unittest.TestCase):
    def test_parity_invariant_enforced(self):
        with self.assertRaises(ValueError):
            FccCoord(1, 0, 0)
        FccCoord(1, 1, 0)  # must not raise

    def test_twelve_neighbors_all_at_sqrt2(self):
        self.assertEqual(len(NEIGHBOR_OFFSETS), 12)
        origin = FccCoord(0, 0, 0)
        for n in origin.neighbors():
            d = np.linalg.norm(n.to_world() - origin.to_world())
            self.assertAlmostEqual(d, math.sqrt(2.0))

    def test_faces_are_rhombi_in_the_right_planes(self):
        verts, _ = _vertex_table(1.0)
        for quad, normal, offset in rd_faces(1.0):
            pts = verts[list(quad)]
            n = np.array(offset, dtype=float)
            # All four vertices satisfy the face plane v . n = s.
            np.testing.assert_allclose(pts @ n, np.ones(4), atol=1e-9)
            # Equal sides, and diagonals in the ratio sqrt(2) : 1.
            sides = [np.linalg.norm(pts[(a + 1) % 4] - pts[a]) for a in range(4)]
            np.testing.assert_allclose(sides, sides[0], atol=1e-9)
            d_long = np.linalg.norm(pts[2] - pts[0])
            d_short = np.linalg.norm(pts[3] - pts[1])
            self.assertAlmostEqual(max(d_long, d_short) / min(d_long, d_short),
                                   math.sqrt(2.0))
            # Winding gives the outward normal.
            tri_n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            self.assertGreater(float(tri_n @ normal), 0.0)

    def test_cell_volume_is_the_fcc_primitive_volume(self):
        pos, _, idx = cell_mesh(1.0)
        volume = 0.0
        for a, b, c in idx.reshape(-1, 3):
            volume += float(np.dot(pos[a], np.cross(pos[b], pos[c]))) / 6.0
        # One FCC point per 2 m^3 in this parametrisation.
        self.assertAlmostEqual(abs(volume), 2.0, places=6)

    def test_ground_patch_is_one_layer_about_30m(self):
        patch = GroundPatch()
        self.assertTrue(all(c.j == 0 for c in patch.cells))
        self.assertEqual(len(patch.cells), 481)
        xs = [c.to_world()[0] for c in patch.cells]
        self.assertEqual((min(xs), max(xs)), (-15.0, 15.0))
        verts, idx = patch.build_mesh()
        # pos + flat normal + smooth normal + color
        self.assertEqual(verts.shape[1], 13)
        self.assertEqual(len(idx) % 3, 0)
        # Peaks 1 m above the centre plane, bottom points 1 m below.
        self.assertAlmostEqual(float(verts[:, 1].max()), 1.0)
        self.assertAlmostEqual(float(verts[:, 1].min()), -1.0)
        # The smooth normal at a peak (0, 1, 0) points straight up,
        # even though every flat face normal there is diagonal.
        peak = verts[
            (np.abs(verts[:, 0]) < 1e-6)
            & (np.abs(verts[:, 1] - 1.0) < 1e-6)
            & (np.abs(verts[:, 2]) < 1e-6)
        ]
        self.assertGreater(len(peak), 0)
        for row in peak:
            np.testing.assert_allclose(row[6:9], [0.0, 1.0, 0.0], atol=1e-5)
            self.assertLess(abs(float(row[3])), 0.99)  # flat normal is diagonal


class PlaneGeometryTests(unittest.TestCase):
    def test_full_plane_has_4_vertices_two_at_infinity(self):
        strip = planes.full_plane_strip(10.0, -4.0, -1.0, np.array([0.0, -1.0]))
        self.assertEqual(strip.shape, (4, 4))
        np.testing.assert_allclose(strip[:, 3], [1, 1, 0, 0])
        # Finite vertices sit on the plane; horizon vertices are
        # horizontal directions (y component exactly 0).
        np.testing.assert_allclose(strip[:2, 1], [-1.0, -1.0])
        np.testing.assert_allclose(strip[2:, 1], [0.0, 0.0])
        for row in strip[2:]:
            self.assertGreater(math.hypot(row[0], row[2]), 0.0)

    def test_homogeneous_interpolation_recovers_plane_points(self):
        """Simulate the GPU's perspective-correct interpolation of the
        homogeneous varying across the bedrock strip: xyz / w must give
        back the exact world point, near and far.  This is the property
        the plane fragment shaders rely on for texturing."""
        cam = Camera(position=np.array([0.0, 7.0, 20.0]), yaw=0.0, pitch=-0.3)
        vp = cam.view_projection(math.radians(70.0), 16 / 9, 0.1)
        # Centre the strip 10 m ahead of the camera so its grounding
        # segment (set 10 m behind the centre at height 0) lands on the
        # camera's ground point, keeping every vertex in front of the
        # eye plane for this analytic check.  (In production the
        # segment sits behind the camera and the GPU clips the strip
        # at the eye plane; the interpolation property is the same.)
        strip = planes.full_plane_strip(
            0.0, 10.0, planes.BEDROCK_Y, cam.forward_xz()
        ).astype(np.float64)
        clip = np.array([vp @ v for v in strip])
        ndc = clip[:, :3] / clip[:, 3:4]

        def recover(point: np.ndarray, tri: tuple[int, int, int]):
            cp = vp @ np.append(point, 1.0)
            pn = cp[:3] / cp[3]
            a = np.array(
                [
                    [ndc[t, 0] for t in tri],
                    [ndc[t, 1] for t in tri],
                    [1.0, 1.0, 1.0],
                ]
            )
            lam = np.linalg.solve(a, np.array([pn[0], pn[1], 1.0]))
            if (lam < -1e-9).any():
                return None
            num = sum(lam[n] * strip[t] / clip[t, 3] for n, t in enumerate(tri))
            den = sum(lam[n] / clip[t, 3] for n, t in enumerate(tri))
            h = num / den
            return h[:3] / h[3]

        y = planes.BEDROCK_Y
        for point in (
            np.array([5.0, y, -30.0]),
            np.array([-20.0, y, -150.0]),
            np.array([150.0, y, -800.0]),
        ):
            got = recover(point, (0, 1, 2))
            if got is None:
                got = recover(point, (1, 3, 2))
            self.assertIsNotNone(got)
            np.testing.assert_allclose(got, point, atol=1e-6)

    def test_hole_plane_alternates_rim_and_horizon(self):
        strip = planes.hole_plane_strip(100.0, 200.0, 0.5)
        self.assertEqual(strip.shape, (26, 4))  # 13 pairs, loop closed
        np.testing.assert_allclose(strip[::2, 3], np.ones(13))   # rim, w = 1
        np.testing.assert_allclose(strip[1::2, 3], np.zeros(13))  # horizon
        for row in strip[::2]:
            r = math.hypot(row[0] - 100.0, row[2] - 200.0)
            self.assertAlmostEqual(r, planes.HOLE_RADIUS, delta=1e-3)
            self.assertAlmostEqual(row[1], 0.5)
        # Each horizon vertex points in its rim vertex's azimuth.
        for rim, hor in zip(strip[::2], strip[1::2]):
            d = np.array([rim[0] - 100.0, rim[2] - 200.0])
            d /= np.linalg.norm(d)
            np.testing.assert_allclose([hor[0], hor[2]], d, atol=1e-6)


if __name__ == "__main__":
    unittest.main()


class GroundPatch111Tests(unittest.TestCase):
    def test_111_patch_geometry(self):
        patch = GroundPatch111(radius=16.0, offset=(0.0, 0.0, 0.0))
        # Centres lie in the world y = 0 plane (the rotated (111) plane).
        for coord in patch.cells:
            centre = patch.center_world(coord)
            self.assertAlmostEqual(float(centre[1]), 0.0, places=9)
            self.assertLessEqual(
                math.hypot(float(centre[0]), float(centre[2])), 16.0 + 1e-9
            )
        self.assertTrue(300 < len(patch.cells) < 600)
        verts, _ = patch.build_mesh()
        # 3-valent vertex on top: peak height s * sqrt(3) / 2,
        # lower than the {100} layer's 1.0.
        self.assertAlmostEqual(
            float(verts[:, 1].max()), math.sqrt(3.0) / 2.0, places=5
        )
        self.assertAlmostEqual(
            float(verts[:, 1].min()), -math.sqrt(3.0) / 2.0, places=5
        )


class RayDirectionTests(unittest.TestCase):
    def test_center_ray_is_forward(self):
        cam = Camera()
        cam.yaw, cam.pitch = 0.4, -0.25
        inv = np.linalg.inv(cam.view_projection(1.1, 1.6, 0.1))
        np.testing.assert_allclose(
            ray_direction(inv, 0.0, 0.0), cam.forward(), atol=1e-9
        )

    def test_off_center_ray_bends_outward(self):
        cam = Camera()
        inv = np.linalg.inv(cam.view_projection(1.1, 1.6, 0.1))
        right_ray = ray_direction(inv, 0.9, 0.0)
        self.assertGreater(float(right_ray @ cam.right()), 0.5)
        self.assertGreater(float(right_ray @ cam.forward()), 0.5)


class ShadingUniformityTests(unittest.TestCase):
    """Regression pins for the stripes/quadrant-seam bug: a per-cell
    hash tint once made corresponding faces of neighbouring blocks
    differ.  Colour must depend on orientation class only, and the
    smooth normals at interior top vertices must be exactly up."""

    def _check(self, patch, center_xz):
        verts, _ = patch.build_mesh()
        upper = verts[verts[:, 4] > 0.35]  # flat normal ny
        colors = {tuple(np.round(row[9:12], 6)) for row in upper}
        # The per-cell tint may vary colours, but only within +/-3%
        # of the single sandy base colour, and it must actually vary
        # (decorrelated hash, many distinct values).
        self.assertGreater(len(colors), 50)
        base = np.array([0.76, 0.65, 0.46])
        for color in colors:
            ratio = np.array(color) / base
            self.assertTrue(np.all(ratio > 0.965) and np.all(ratio < 1.035),
                            f"upper-face color out of tint bounds: {color}")
        cx, cz = center_xz
        top = verts[
            (np.abs(verts[:, 0] - cx) < 10)
            & (np.abs(verts[:, 2] - cz) < 10)
            & (verts[:, 1] > 0.25)
        ]
        self.assertGreater(len(top), 100)
        for row in top:
            np.testing.assert_allclose(row[6:9], [0.0, 1.0, 0.0], atol=1e-4)

    def test_patch_100_uniform(self):
        self._check(GroundPatch(), (0.0, 0.0))

    def test_patch_111_uniform(self):
        self._check(GroundPatch111(offset=(0.0, 0.0, 0.0)), (0.0, 0.0))


class WallTests(unittest.TestCase):
    def test_wall_100(self):
        from rhombic_game.lattice import wall_cells_100

        cells = wall_cells_100()
        self.assertEqual(len(cells), 22)
        self.assertTrue(all(c.k == 0 and 1 <= c.j <= 4 for c in cells))

    def test_wall_111_courses_climb(self):
        from rhombic_game.lattice import PATCH111_ROTATION, wall_cells_111

        cells = wall_cells_111()
        self.assertEqual(len(cells), 20)
        heights = sorted(
            {round(float((PATCH111_ROTATION @ c.to_world(1.0))[1]), 4) for c in cells}
        )
        expected = [round(2 * course / math.sqrt(3.0), 4) for course in range(1, 5)]
        self.assertEqual(heights, expected)

    def test_wall_111_rises_straight_on_average(self):
        # After 3 courses the cumulative offset is (2,2,2): zero
        # sideways shift.  Check course centroids' sideways drift
        # from the wall's base line stays under 1 m for all courses.
        from rhombic_game.lattice import PATCH111_ROTATION, wall_cells_111

        cells = wall_cells_111()
        rows: dict[int, list] = {}
        for c in cells:
            rows.setdefault(c.i + c.j + c.k, []).append(
                PATCH111_ROTATION @ c.to_world(1.0)
            )
        base_line = np.mean(
            [PATCH111_ROTATION @ np.array([m, -m, 0.0]) for m in range(5)],
            axis=0,
        )
        for pts in rows.values():
            centroid = np.mean(pts, axis=0)
            # Sideways = horizontal (world XZ) part, since the
            # rotation carries lattice (1,1,1) onto world +Y.
            drift = (centroid - base_line) * np.array([1.0, 0.0, 1.0])
            self.assertLess(float(np.linalg.norm(drift)), 1.0)

    def test_wall_111_b_staircases_at_90_degrees(self):
        from rhombic_game.lattice import _course_offset, wall_cells_111_b

        cells = wall_cells_111_b()
        self.assertEqual(len(cells), 32)
        self.assertEqual(
            sorted({c.i + c.j + c.k for c in cells}), [2, 4, 6, 8]
        )
        # Course-1 cells minus the course offset are the staircase
        # base points; each stays within ~0.71 m of the (1,1,-2)
        # line through the corner (the exact 90-degree direction).
        o1 = np.array(_course_offset(1), dtype=float)
        u = np.array([1.0, 1.0, -2.0]) / math.sqrt(6.0)
        base = [c for c in cells if c.i + c.j + c.k == 2]
        self.assertEqual(len(base), 8)
        for c in base:
            q = np.array([c.i, c.j, c.k], dtype=float) - o1
            self.assertAlmostEqual(float(q.sum()), 0.0)  # in the layer plane
            dist = float(np.linalg.norm(q - (q @ u) * u))
            self.assertLessEqual(dist, 0.72)


class LookDeltaTests(unittest.TestCase):
    """The analytic look Jacobian (docs/look-jacobian.md)."""

    FOV = 1.1
    ASPECT = 1.6
    H = 800.0

    def _k(self):
        return 2.0 * math.tan(self.FOV / 2.0) / self.H

    def _xy(self, ndc_x, ndc_y):
        t = math.tan(self.FOV / 2.0)
        return self.ASPECT * t * ndc_x, t * ndc_y

    def _glue_error_deg(self, cam, ndc_old, ndc_new, steps=40):
        """Drag in small steps; return the glued direction's drift."""
        inv = np.linalg.inv(cam.view_projection(self.FOV, self.ASPECT, 0.1))
        target = ray_direction(inv, *ndc_old)
        for s in range(1, steps + 1):
            f = s / steps
            ndc = (
                ndc_old[0] + (ndc_new[0] - ndc_old[0]) * f,
                ndc_old[1] + (ndc_new[1] - ndc_old[1]) * f,
            )
            prev = (
                ndc_old[0] + (ndc_new[0] - ndc_old[0]) * (f - 1.0 / steps),
                ndc_old[1] + (ndc_new[1] - ndc_old[1]) * (f - 1.0 / steps),
            )
            # Pixel motion of this sub-step on an ASPECT x 1 screen
            # of height H: dx = d(ndc_x) * W/2, dy = -d(ndc_y) * H/2.
            dx = (ndc[0] - prev[0]) * (self.ASPECT * self.H) / 2.0
            dy = -(ndc[1] - prev[1]) * self.H / 2.0
            x, y = self._xy(*ndc)
            dyaw, dpitch = look_delta(cam.pitch, x, y, dx, dy, self._k())
            cam.yaw += dyaw
            cam.pitch = float(np.clip(cam.pitch + dpitch, -1.55, 1.55))
        inv2 = np.linalg.inv(cam.view_projection(self.FOV, self.ASPECT, 0.1))
        got = ray_direction(inv2, *ndc_new)
        return math.degrees(math.acos(float(np.clip(got @ target, -1, 1))))

    def test_centre_rates(self):
        k = self._k()
        dyaw, dpitch = look_delta(0.0, 0.0, 0.0, 10.0, 4.0, k)
        self.assertAlmostEqual(dyaw, k * 10.0)
        self.assertAlmostEqual(dpitch, k * 4.0)
        # Pitched: yaw gains 1/cos(pitch) at the centre, pitch rate same.
        dyaw, dpitch = look_delta(0.6, 0.0, 0.0, 10.0, 4.0, k)
        self.assertAlmostEqual(dyaw, k * 10.0 / math.cos(0.6))
        self.assertAlmostEqual(dpitch, k * 4.0)

    def test_grabbed_direction_stays_under_cursor(self):
        cam = Camera()
        cam.yaw, cam.pitch = 0.7, -0.3
        err = self._glue_error_deg(cam, (0.10, 0.05), (0.38, -0.12))
        self.assertLess(err, 0.05)  # second-order residue only

    def test_singular_band_stays_bounded(self):
        # Pitch near vertical, cursor ray near the Y axis:
        # sigma -> 0 is clamped, deltas must stay finite and sane.
        dyaw, dpitch = look_delta(1.50, 0.0, 0.30, 5.0, 5.0, self._k())
        self.assertTrue(math.isfinite(dyaw) and math.isfinite(dpitch))
        self.assertLess(abs(dyaw), 0.5)
        self.assertLess(abs(dpitch), 0.5)


class Wall100BAndColumnTests(unittest.TestCase):
    def test_wall_100_b(self):
        from rhombic_game.lattice import wall_cells_100_b

        cells = wall_cells_100_b()
        self.assertEqual(len(cells), 22)
        self.assertTrue(all(c.i == 5 and 1 <= c.j <= 4 for c in cells))

    def test_column_100(self):
        from rhombic_game.lattice import NEIGHBOR_OFFSETS, column_cells_100

        cells = column_cells_100()
        self.assertEqual(len(cells), 8)
        prev = (10, 0, 10)
        for c in cells:
            step = (c.i - prev[0], c.j - prev[1], c.k - prev[2])
            self.assertIn(step, NEIGHBOR_OFFSETS)  # face-bonded courses
            prev = (c.i, c.j, c.k)
        self.assertEqual((cells[-1].i, cells[-1].k), (10, 10))  # wobble cancels

    def test_column_111(self):
        from rhombic_game.lattice import PATCH111_ROTATION, column_cells_111

        cells = column_cells_111()
        self.assertEqual(len(cells), 8)
        base = PATCH111_ROTATION @ np.array([-6.0, 3.0, 3.0])
        for n, c in enumerate(cells, start=1):
            p = PATCH111_ROTATION @ c.to_world(1.0)
            self.assertAlmostEqual(float(p[1]), 2 * n / math.sqrt(3.0), places=4)
            drift = (p - base) * np.array([1.0, 0.0, 1.0])
            self.assertLessEqual(float(np.linalg.norm(drift)), 0.83)


class SandyAndAOTests(unittest.TestCase):
    def test_all_orientations_sandy(self):
        from rhombic_game.lattice import SANDY, GroundPatch

        verts, _ = GroundPatch().build_mesh()
        ratio = verts[:, 9:12] / SANDY[None, :]
        self.assertTrue(np.all(ratio > 0.965) and np.all(ratio < 1.035))

    def test_peaks_unoccluded_valleys_shaded(self):
        from rhombic_game.lattice import GroundPatch

        verts, _ = GroundPatch().build_mesh()
        peaks = verts[(np.abs(verts[:, 1] - 1.0) < 1e-4) & (verts[:, 4] > 0.35)]
        self.assertGreater(len(peaks), 50)
        self.assertTrue(np.all(peaks[:, 12] == 1.0))
        # Somewhere on the island mesh, neighbour blocks do occlude.
        from rhombic_game.lattice import island_cells_100, build_cells_mesh

        iverts, _ = build_cells_mesh(island_cells_100(), 1.0, np.eye(3), (0.0, 0.0, 0.0))
        self.assertTrue(np.any(iverts[:, 12] < 1.0))
        levels = {round(float(x), 2) for x in np.unique(iverts[:, 12])}
        self.assertTrue(levels <= {1.0, 0.85, 0.72, 0.60})

    def test_island_cell_counts(self):
        from rhombic_game.lattice import island_cells_100, island_cells_111

        self.assertEqual(len(island_cells_100()), 533)
        self.assertEqual(len(island_cells_111()), 523)
