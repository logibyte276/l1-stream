"""KISS-ICP wrapper, pinned to the behaviour of version 1.3.0.

Everything asserted here was checked against the installed package rather than
its documentation, because several defaults differ from what the docs and
older releases describe. The two that will cost you an afternoon:

* ``mapping.voxel_size`` defaults to **None**, which crashes ``VoxelHashMap``.
  It has to be set explicitly.
* ``KissICP.__init__`` builds the preprocessor, the registration, the local map
  and the adaptive threshold immediately. **Mutating the config afterwards has
  no effect** (``voxelize()`` is the lone exception -- it reads the config
  live). So every parameter must be final before construction, which is why
  this class takes them all as constructor arguments and never exposes ``cfg``.

Also worth knowing: ``register_frame`` returns ``(frame, source)`` -- the
processed clouds, *not* the pose. The pose is ``odom.last_pose``.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["KissOdometry"]


class KissOdometry:
    """Frame-in, pose-out. Feed it :class:`~l1_stream.frames.Frame` objects.

    Args:
        voxel_size: Map resolution, metres. KISS-ICP derives its two
            downsampling steps from this (``voxel_size*0.5`` into the local
            map, ``voxel_size*1.5`` into ICP), so it is the single most
            influential parameter.
        max_range / min_range: Range gate, metres. **Measure min_range** --
            it should sit just outside your chassis's self-hit radius, or the
            car's own body registers perfectly against itself every frame and
            pins the solution at zero motion.
        deskew: Undo intra-frame motion using the previous frame delta. Needs
            the per-point timestamps that :class:`FrameAssembler` provides.
        initial_threshold / min_motion_th: Adaptive correspondence threshold
            seed. The L1's 25 m range and centimetre accuracy make the stock
            2.0 m seed far too loose.
    """

    def __init__(
        self,
        *,
        voxel_size: float = 0.25,
        max_range: float = 25.0,
        min_range: float = 0.25,
        deskew: bool = True,
        initial_threshold: float = 0.4,
        min_motion_th: float = 0.02,
        max_points_per_voxel: int = 20,
        max_num_iterations: int = 500,
        convergence_criterion: float = 1e-4,
    ):
        from kiss_icp.config import KISSConfig
        from kiss_icp.kiss_icp import KissICP

        if voxel_size is None or voxel_size <= 0:
            raise ValueError("voxel_size must be a positive float (None crashes VoxelHashMap).")
        if min_range >= max_range:
            raise ValueError("min_range must be < max_range")

        cfg = KISSConfig()
        cfg.data.max_range = float(max_range)
        cfg.data.min_range = float(min_range)
        cfg.data.deskew = bool(deskew)
        cfg.mapping.voxel_size = float(voxel_size)
        cfg.mapping.max_points_per_voxel = int(max_points_per_voxel)
        cfg.adaptive_threshold.initial_threshold = float(initial_threshold)
        cfg.adaptive_threshold.min_motion_th = float(min_motion_th)
        cfg.registration.max_num_iterations = int(max_num_iterations)
        cfg.registration.convergence_criterion = float(convergence_criterion)

        # Nothing above may change after this line. See the module docstring.
        self._odom = KissICP(cfg)
        self.config = cfg

        self.poses: list[np.ndarray] = []
        self.stamps: list[float] = []

    # --- stepping ---

    def register(self, frame) -> np.ndarray:
        """Register one frame and return the 4x4 pose as of ``frame.t_end``.

        The pose timestamp is the END of the frame, not its midpoint: deskew
        was measured to transform points into the last point's reference, so
        that is the instant the pose describes.
        """
        self._odom.register_frame(frame.points, frame.timestamps)
        pose = np.array(self._odom.last_pose, dtype=np.float64, copy=True)
        self.poses.append(pose)
        self.stamps.append(frame.t_end)
        return pose

    # --- read-only views ---

    @property
    def last_pose(self) -> np.ndarray:
        return np.asarray(self._odom.last_pose, dtype=np.float64)

    @property
    def last_delta(self) -> np.ndarray:
        """Frame-to-frame motion. With IMU-rotated frames this should be close
        to a pure translation -- a growing rotation here means the IMU's yaw is
        drifting, which a 6-axis IMU cannot help."""
        return np.asarray(self._odom.last_delta, dtype=np.float64)

    @property
    def threshold(self) -> float:
        return float(self._odom.adaptive_threshold.get_threshold())

    def trajectory(self) -> np.ndarray:
        """(K, 3) array of positions. Directly usable; the rotation part of a
        pose is NOT robot attitude when frames were IMU-rotated."""
        if not self.poses:
            return np.empty((0, 3), dtype=np.float64)
        return np.array([p[:3, 3] for p in self.poses], dtype=np.float64)

    def path_length(self) -> float:
        xyz = self.trajectory()
        if len(xyz) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())

    def loop_closure_error(self) -> float:
        """Distance from the last pose back to the first.

        With no ground truth this is the most honest metric you have: drive a
        closed loop back to the exact start point and this is your accumulated
        drift, best read as a fraction of :meth:`path_length`.
        """
        xyz = self.trajectory()
        if len(xyz) < 2:
            return 0.0
        return float(np.linalg.norm(xyz[-1] - xyz[0]))
