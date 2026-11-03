"""Hardware-free checks for the revision scripts.  pytest revision/test_rev.py"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rev_analysis import hold_decomposition  # noqa: E402
from rev_imu import quat_multiply, strip_yaw, yaw_deg  # noqa: E402
from rev_replays import build_jobs, end_pose_components  # noqa: E402
from rev_wall_check import fit_line, pick_front_wall  # noqa: E402

from l1_stream.rotation import quaternion_to_matrix  # noqa: E402

Rotation = pytest.importorskip("scipy.spatial.transform").Rotation


def test_quat_multiply_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = Rotation.random(random_state=rng), Rotation.random(random_state=rng)
        got = quat_multiply(a.as_quat(), b.as_quat())
        want = (a * b).as_quat()
        assert np.allclose(got, want) or np.allclose(got, -want)


def test_strip_yaw_removes_heading_and_keeps_tilt():
    rng = np.random.default_rng(1)
    for _ in range(50):
        roll, pitch = rng.uniform(-0.3, 0.3, 2)
        yaw = rng.uniform(-math.pi, math.pi)
        q = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
        q0 = strip_yaw(q)
        assert abs(yaw_deg(q0)) < 1e-6
        tilt_only = Rotation.from_euler("ZYX", [0.0, pitch, roll]).as_matrix()
        assert np.allclose(quaternion_to_matrix(q0), tilt_only, atol=1e-9)
        # gravity seen in the sensor frame is unchanged
        z = np.array([0.0, 0.0, 1.0])
        assert np.allclose(quaternion_to_matrix(q).T @ z, quaternion_to_matrix(q0).T @ z)


def test_fit_line_recovers_slope_and_ci_covers_it():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 0.97 * x + 0.18 + np.array([0.4, -0.2, 0.1, -0.3, 0.0]) * 1e-3
    f = fit_line(x, y)
    assert abs(f["slope"] - 0.97) < 1e-3
    assert f["slope_ci"][0] < 0.97 < f["slope_ci"][1]


def test_front_wall_is_picked_by_sign_not_by_slope():
    truths = np.array([1.0, 2.0, 3.0])
    front = np.array([1.0, 0.0, 0.0])
    side = np.array([0.0, 1.0, 0.0])
    back = np.array([-1.0, 0.0, 0.0])
    per_rec = []
    for d in truths:
        per_rec.append([
            {"normal": side, "distance": 2.4, "inliers": 9000, "rms_mm": 8},
            {"normal": back, "distance": 8.0 - d, "inliers": 5000, "rms_mm": 8},
            {"normal": front, "distance": 0.9 * d + 0.2, "inliers": 4000, "rms_mm": 8},
        ])
    chosen, _ = pick_front_wall(per_rec, truths)
    assert all(c["normal"] is front for c in chosen)


def test_end_pose_components_straight_drive_with_jitter():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 1, 120)
    heading = np.array([math.cos(0.4), math.sin(0.4)])
    xy = np.outer(5.0 * t, heading) + rng.normal(0, 0.005, (120, 2))
    xyz = np.column_stack([xy, np.full(120, 0.01)])
    xyz[0] = 0.0
    c = end_pose_components(xyz)
    assert abs(c["end_along_m"] - 5.0) < 0.02
    assert abs(c["end_sideways_m"]) < 0.02
    assert abs(c["end_vertical_m"] - 0.01) < 1e-9


def test_hold_decomposition_adds_up():
    t = np.arange(0, 20, 0.2)
    x = np.clip((t - 4) * 0.5, 0, 5.0)
    xyz = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    xyz[: 10, 0] += 0.003       # some drift while parked at the start
    d = hold_decomposition(t, xyz)
    total = d["start_hold_cm"] / 100 + d["motion_m"] + d["end_hold_cm"] / 100
    assert abs(total - d["total_m"]) < 1e-12
    assert abs(d["motion_m"] - 5.0) < 0.01


def test_job_counts():
    jobs = build_jobs(["baseline", "ablation", "framelen"])
    assert sum(j["group"] == "baseline" for j in jobs) == 35
    assert sum(j["group"] == "ablation" for j in jobs) == 30
    assert sum(j["group"] == "framelen" for j in jobs) == 25
    assert len({j["job"] for j in jobs}) == len(jobs)
