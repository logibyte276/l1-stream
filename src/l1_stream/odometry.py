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

from .config import DEFAULTS, UNTUNED

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
        # Defaults come from l1_stream.config so that constructing this class
        # bare gives the TUNED configuration. They used to be written out here
        # and had drifted three values away from the rest of the pipeline.
        voxel_size: float = DEFAULTS["voxel_size"],
        max_range: float = DEFAULTS["max_range"],
        min_range: float = DEFAULTS["min_range"],
        deskew: bool = DEFAULTS["deskew"],
        map_deskew: bool | None = None,
        initial_threshold: float = DEFAULTS["initial_threshold"],
        min_motion_th: float = UNTUNED["min_motion_th"],
        max_points_per_voxel: int = UNTUNED["max_points_per_voxel"],
        max_num_iterations: int = UNTUNED["max_num_iterations"],
        convergence_criterion: float = UNTUNED["convergence_criterion"],
    ):
        from kiss_icp.config import KISSConfig
        from kiss_icp.kiss_icp import KissICP

        if voxel_size is None or voxel_size <= 0:
            raise ValueError("voxel_size must be a positive float (None crashes VoxelHashMap).")
        if min_range >= max_range:
            raise ValueError("min_range must be < max_range")

        # MAP DESKEW IS A SEPARATE QUESTION FROM REGISTRATION DESKEW.
        #
        # Intra-frame smear is speed * frame_duration -- 100 mm at 0.5 m/s.
        # Registration voxelizes at 0.15 m, so at survey speeds the smear is
        # SMALLER than the voxel and gets absorbed; that is why deskew=False
        # measured fine on loop closure. A map accumulates at 0.03 m, where the
        # same 100 mm is 3.3x the voxel and fully visible as thickened walls.
        #
        # So the map deskews by default regardless of the registration setting.
        # It only stops being worth it when the smear drops below the noise
        # deskew injects (last_delta carries ~25 mm/frame of jitter), i.e.
        # below roughly 0.125 m/s -- essentially parked.
        #
        # Set map_deskew=False to compare, or None to follow `deskew`.
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

        self.map_deskew = deskew if map_deskew is None else bool(map_deskew)
        self._map_pre = None
        if self.map_deskew != bool(deskew):
            from kiss_icp.preprocess import Preprocessor
            self._map_pre = Preprocessor(
                float(max_range), float(min_range), self.map_deskew,
                int(getattr(cfg, "max_num_threads", 0) or 0),
            )

        self.poses: list[np.ndarray] = []
        self.stamps: list[float] = []
        #: The last frame AS KISS-ICP ACTUALLY USED IT -- deskewed (when
        #: enabled) and cropped to [min_range, max_range]. Kept because
        #: register_frame returns it and throwing it away means anything built
        #: downstream (a map, say) would use points the estimator never saw.
        self.last_preprocessed: np.ndarray | None = None
        #: The last frame prepared for MAPPING. Identical to
        #: :attr:`last_preprocessed` unless ``map_deskew`` differs from
        #: ``deskew``; see the note in ``__init__``.
        self.last_map_cloud: np.ndarray | None = None

    # --- stepping ---

    def register(self, frame) -> np.ndarray:
        """Register one frame and return the 4x4 pose as of ``frame.t_end``.

        The pose timestamp is the END of the frame, not its midpoint: deskew
        was measured to transform points into the last point's reference, so
        that is the instant the pose describes.

        Also stashes :attr:`last_preprocessed` -- the frame after KISS-ICP's
        own motion compensation and range cropping. That is the cloud the pose
        actually describes, so it is the only correct thing to place into a
        map. (``register_frame`` returns ``(frame, source)``; ``source`` is
        additionally voxel-downsampled for registration and is too sparse to
        map from.)
        """
        # last_delta as it stands BEFORE registration is what deskew uses, so
        # capture it now -- register_frame overwrites it.
        delta = np.array(self._odom.last_delta, dtype=np.float64, copy=True)
        processed, _source = self._odom.register_frame(frame.points, frame.timestamps)
        self.last_preprocessed = np.asarray(processed, dtype=np.float64)
        self.last_map_cloud = (
            self.last_preprocessed if self._map_pre is None
            else np.asarray(
                self._map_pre.preprocess(frame.points, frame.timestamps, delta),
                dtype=np.float64,
            )
        )
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
