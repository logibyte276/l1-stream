"""Map accuracy, measured against the building.

WHY THIS EXISTS. Every metric this project had until now measured the
TRAJECTORY -- loop closure, net displacement, path length. The research
question is about the MAP, and the two come apart in a way that matters: a
perfect trajectory built on biased ranges produces a map that is internally
consistent and the wrong size. Loop closure cannot see that, by construction.

The trick for measuring a map indoors without a motion-capture rig is that
buildings are already ground truth, for free and to a tolerance better than the
sensor:

    walls are FLAT          -> fit a plane, report RMS deviation
    corners are SQUARE      -> angle between two fitted planes, expect 90 deg
    opposite walls PARALLEL -> expect 0 deg
    floors are LEVEL        -> plane normal vs gravity

None of that needs a tape measure, and all of it is per-map rather than one
endpoint number. Plane RMS in particular is the metric to watch: it responds to
range noise AND to registration error, so it degrades when either does.

    m = accumulate_map("personal/room.l1raw", map_voxel=0.03)
    wall = m[box_mask(m, (-1, 3.0, -0.5), (4, 3.6, 1.5))]
    fit = fit_plane(wall)
    print(fit.rms, fit.n_inliers)

CAVEAT WORTH KEEPING IN VIEW: the L1 sees only the hemisphere ABOVE itself, so
z is weakly observable and floor planes will be sparse or absent. Prefer walls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .offline import replay

__all__ = [
    "PlaneFit",
    "accumulate_map",
    "box_mask",
    "fit_plane",
    "plane_angle_deg",
    "voxel_downsample",
]


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """One point per occupied voxel (the first seen, not the centroid).

    Centroid averaging would smooth exactly the deviation a plane-RMS metric is
    trying to measure, so it is deliberately not done here.
    """
    if voxel <= 0 or len(points) == 0:
        return np.asarray(points, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def accumulate_map(path: str, *, map_voxel: float = 0.03,
                   map_deskew: bool = True, **cfg) -> np.ndarray:
    """Replay a recording and return the accumulated world-frame cloud.

    Points come from ``KissOdometry.last_preprocessed`` -- the frame AS
    KISS-ICP used it, deskewed (when enabled) and cropped to
    [min_range, max_range] -- not the raw frame. That distinction is not
    cosmetic: the raw frame carries ~273 chassis self-hits per frame below
    min_range, and driving they land in fresh voxels at every pose, laying a
    tube of false points along the whole trajectory.

    ``map_deskew`` defaults to True INDEPENDENTLY of the registration
    ``deskew`` setting. Intra-frame smear is speed * frame_duration -- 100 mm
    at 0.5 m/s -- which is below the 0.15 m registration voxel (absorbed, which
    is why deskew=False measured fine on loop closure) but 3.3x the 0.03 m map
    voxel, where it shows up as thickened walls and inflated plane RMS.

    Points are placed with ``last_pose``, which is the transform from that
    cloud into the map -- so this is correct whether or not the frames were
    IMU-rotated first. ``map_voxel`` bounds memory; at 0.03 m a 60 s drive
    lands in the low millions of points.
    """
    run = replay(path, map_voxel=map_voxel, map_deskew=map_deskew, **cfg)
    if run.world_points is None:
        raise RuntimeError("replay returned no map; pass map_voxel > 0")
    return run.world_points


def box_mask(points: np.ndarray, lo, hi) -> np.ndarray:
    """Boolean mask for points inside an axis-aligned box.

    Crude on purpose: picking a wall out of a room is a two-minute job with a
    viewer and a bounding box, and anything cleverer would need a segmentation
    step whose failures you would then have to debug alongside your results.
    """
    pts = np.asarray(points, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    return np.all((pts >= lo) & (pts <= hi), axis=1)


@dataclass
class PlaneFit:
    """A fitted plane and, more importantly, how well it fitted."""

    normal: np.ndarray      # unit normal
    offset: float           # plane is  normal . x + offset = 0
    rms: float              # RMS distance of inliers to the plane, metres
    max_abs: float          # worst inlier deviation
    inliers: np.ndarray     # boolean mask over the input
    n_inliers: int
    n_points: int

    @property
    def inlier_fraction(self) -> float:
        return self.n_inliers / self.n_points if self.n_points else 0.0

    @property
    def distance_from_origin(self) -> float:
        """Perpendicular distance from the map origin to the plane, metres.

        On a STATIONARY recording the origin is the sensor, so this is the
        sensor-to-wall RANGE as the LiDAR reports it. Park at a laser-measured
        distance and compare: if the wall reads ~5% short, the bias is in the
        ranges themselves; if it reads correct, the ranges are fine and the
        scale error lives in registration. Those two need opposite fixes.
        """
        return float(abs(self.offset))

    def distances(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) @ self.normal + self.offset


def _svd_plane(pts: np.ndarray) -> tuple[np.ndarray, float]:
    centroid = pts.mean(axis=0)
    # Smallest singular vector of the centred cloud is the plane normal.
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vt[-1]
    return normal, float(-normal @ centroid)


def fit_plane(
    points: np.ndarray,
    *,
    threshold: float = 0.05,
    iterations: int = 200,
    seed: int | None = 0,
) -> PlaneFit:
    """RANSAC a plane, then least-squares refine on the inliers.

    Plain SVD on the whole selection would be dragged off the wall by whatever
    else is in the bounding box -- a radiator, a chair, the doorway. RANSAC
    finds the dominant plane first; the refit then uses every inlier, so the
    reported RMS is a real fit statistic and not a RANSAC artefact.

    ``threshold`` is the inlier band in metres. The default of 50 mm is
    deliberately looser than the L1's +/-20 mm point accuracy so that a genuinely
    bad map still produces a fit you can look at, instead of failing to find one.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected (N, 3) points, got {pts.shape}")
    if len(pts) < 3:
        raise ValueError(f"need at least 3 points to fit a plane, got {len(pts)}")

    rng = np.random.default_rng(seed)
    best_inliers = None
    best_count = -1

    for _ in range(iterations):
        sample = pts[rng.choice(len(pts), size=3, replace=False)]
        v1, v2 = sample[1] - sample[0], sample[2] - sample[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:      # collinear sample, no plane
            continue
        normal = normal / norm
        offset = -normal @ sample[0]
        inliers = np.abs(pts @ normal + offset) <= threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers

    if best_inliers is None or best_count < 3:
        best_inliers = np.ones(len(pts), dtype=bool)

    normal, offset = _svd_plane(pts[best_inliers])
    d = pts[best_inliers] @ normal + offset
    return PlaneFit(
        normal=normal,
        offset=offset,
        rms=float(np.sqrt(np.mean(d**2))),
        max_abs=float(np.max(np.abs(d))) if len(d) else 0.0,
        inliers=best_inliers,
        n_inliers=int(best_inliers.sum()),
        n_points=len(pts),
    )


def plane_angle_deg(a: PlaneFit | np.ndarray, b: PlaneFit | np.ndarray) -> float:
    """Angle between two planes in degrees, folded to [0, 90].

    Folded because a fitted normal's sign is arbitrary -- SVD may hand back
    either direction, so an unfolded angle would report 90 or 270 at random for
    the same corner.
    """
    na = a.normal if isinstance(a, PlaneFit) else np.asarray(a, dtype=np.float64)
    nb = b.normal if isinstance(b, PlaneFit) else np.asarray(b, dtype=np.float64)
    na = na / np.linalg.norm(na)
    nb = nb / np.linalg.norm(nb)
    cos = float(np.clip(abs(na @ nb), 0.0, 1.0))
    return float(np.degrees(np.arccos(cos)))
