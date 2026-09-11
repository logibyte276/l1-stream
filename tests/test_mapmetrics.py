import numpy as np
import pytest

from l1_stream.mapmetrics import (
    box_mask,
    fit_plane,
    plane_angle_deg,
    voxel_downsample,
)

rng = np.random.default_rng(0)


def wall(n=2000, noise=0.0, x=3.0):
    """A y-z wall at x, optionally with gaussian range noise along x."""
    pts = np.column_stack([
        np.full(n, x) + rng.normal(0, noise, n),
        rng.uniform(-2, 2, n),
        rng.uniform(-1, 1, n),
    ])
    return pts


def test_perfect_plane_fits_with_essentially_zero_rms():
    f = fit_plane(wall(noise=0.0))
    assert f.rms < 1e-9
    assert abs(abs(f.normal[0]) - 1.0) < 1e-6      # normal points along x
    assert f.inlier_fraction == 1.0


def test_rms_recovers_the_noise_that_was_injected():
    # The metric has to be calibrated, not just monotonic: 20 mm of range
    # noise must read back as ~20 mm, or "is this map as flat as the sensor
    # can see" has no answer.
    f = fit_plane(wall(n=20000, noise=0.020))
    assert 0.018 < f.rms < 0.022


def test_ransac_ignores_clutter_that_plain_svd_would_follow():
    # A radiator in the bounding box. Least squares over everything would tilt
    # the plane; RANSAC should reject the blob and still fit the wall.
    w = wall(n=4000, noise=0.005)
    clutter = np.column_stack([
        rng.uniform(2.3, 2.6, 800), rng.uniform(-0.5, 0.5, 800),
        rng.uniform(-1, 0, 800),
    ])
    f = fit_plane(np.vstack([w, clutter]), threshold=0.05)
    assert f.rms < 0.010
    assert abs(abs(f.normal[0]) - 1.0) < 0.01
    assert f.n_inliers == pytest.approx(4000, abs=200)


def test_perpendicular_walls_read_90_degrees():
    a = fit_plane(wall(x=3.0))
    b_pts = np.column_stack([
        rng.uniform(-2, 2, 2000), np.full(2000, 3.0), rng.uniform(-1, 1, 2000),
    ])
    assert plane_angle_deg(a, fit_plane(b_pts)) == pytest.approx(90.0, abs=0.1)


def test_parallel_walls_read_zero_degrees():
    assert plane_angle_deg(fit_plane(wall(x=3.0)), fit_plane(wall(x=-3.0))) < 0.1


def test_angle_is_sign_folded_so_a_corner_never_reports_270():
    # SVD hands back an arbitrary normal sign; without folding the same corner
    # would report 90 or 270 depending on the run.
    n1 = np.array([1.0, 0, 0])
    assert plane_angle_deg(n1, np.array([0, 1.0, 0])) == pytest.approx(90.0)
    assert plane_angle_deg(n1, np.array([0, -1.0, 0])) == pytest.approx(90.0)
    assert plane_angle_deg(n1, np.array([-1.0, 0, 0])) == pytest.approx(0.0)


def test_voxel_downsample_thins_without_moving_points():
    pts = rng.uniform(0, 1, (5000, 3))
    out = voxel_downsample(pts, 0.1)
    assert len(out) < len(pts)
    # Every survivor must be an ORIGINAL point. Centroid averaging would smooth
    # away the very deviation plane-rms is measuring.
    assert np.isin(out, pts).all()


def test_voxel_downsample_is_a_noop_for_zero_or_empty():
    pts = rng.uniform(0, 1, (10, 3))
    assert len(voxel_downsample(pts, 0)) == 10
    assert len(voxel_downsample(np.empty((0, 3)), 0.1)) == 0


def test_box_mask_selects_inclusively():
    pts = np.array([[0, 0, 0], [1, 1, 1], [5, 5, 5]], dtype=float)
    m = box_mask(pts, (0, 0, 0), (1, 1, 1))
    assert m.tolist() == [True, True, False]


def test_fit_plane_rejects_bad_shapes_and_tiny_inputs():
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        fit_plane(np.zeros((10, 2)))
    with pytest.raises(ValueError, match="at least 3"):
        fit_plane(np.zeros((2, 3)))


def test_distance_from_origin_recovers_a_known_wall_range():
    """The case-A/case-B test in one number.

    Parked, the map origin is the sensor, so a fitted wall's distance from the
    origin IS the range the LiDAR reported. Compare against a laser measurement:
    short by ~5% means the ranges are biased; correct means the scale error is
    in registration instead.
    """
    for truth in (2.5, 3.0, 7.0):
        f = fit_plane(wall(n=4000, noise=0.005, x=truth))
        assert f.distance_from_origin == pytest.approx(truth, abs=0.002)


def test_distance_from_origin_is_sign_independent():
    # SVD may hand back either normal direction; a distance must not flip sign.
    assert fit_plane(wall(x=3.0)).distance_from_origin == pytest.approx(3.0, abs=0.002)
    assert fit_plane(wall(x=-3.0)).distance_from_origin == pytest.approx(3.0, abs=0.002)


def test_ceiling_tilt_reads_out_as_degrees_off_horizontal():
    # A level ceiling at z=2.4, then the same ceiling tilted 1.5 deg.
    rng2 = np.random.default_rng(7)
    xy = rng2.uniform(-2, 2, (3000, 2))
    up = np.array([0.0, 0.0, 1.0])
    # A horizontal plane's normal is PARALLEL to up, so the angle IS the tilt.
    flat = np.column_stack([xy, np.full(len(xy), 2.4)])
    assert plane_angle_deg(fit_plane(flat), up) == pytest.approx(0.0, abs=0.05)

    slope = np.tan(np.radians(1.5))
    tilted = np.column_stack([xy, 2.4 + slope * xy[:, 0]])
    assert plane_angle_deg(fit_plane(tilted), up) == pytest.approx(1.5, abs=0.05)

    # And a wall reads 90 against the same reference -- the check that would
    # have caught the inverted formula this test was first written with.
    assert plane_angle_deg(fit_plane(wall(x=3.0)), up) == pytest.approx(90.0, abs=0.1)
