"""Tuned pipeline configuration: one definition, imported everywhere.

``FrameAssembler`` and ``KissOdometry`` take their signature defaults from
here, and :func:`l1_stream.offline.add_args` exposes the same values as
command-line flags, so every entry point runs the same configuration unless
told otherwise. ``tests/test_config.py`` enforces this: it fails if a
signature default or an example script disagrees with this file.

The values were measured on a Unitree L1 mounted on a small ground robot.
``min_range`` in particular depends on your chassis; measure it with
``examples/07_precheck.py``.

This module imports nothing from the rest of the package, so it can be
imported from anywhere without a cycle.
"""

from __future__ import annotations

__all__ = ["DEFAULTS", "UNTUNED"]

#: Measured on hardware. Change a value here and the whole pipeline follows.
DEFAULTS = {
    # frame assembly
    "frame_duration": 0.2,      # 0.05 s and 0.5 s both measured worse
    "rotate_with_imu": True,    # loop closure was 32-126x worse without IMU pre-rotation

    # registration
    "voxel_size": 0.15,         # 0.10 is also real-time on a Jetson Orin Nano; 0.15 keeps headroom
    "max_range": 25.0,          # trimming to 10 m measurably hurt rotation estimates
    "min_range": 0.25,          # just outside the reference robot's 0.19 m chassis self-hit radius
    "deskew": True,             # KISS-ICP's default; results vary by scene, so compare with --no-deskew
    "initial_threshold": 0.4,   # the adaptive threshold settles at 0.32-0.55 m, so this seed is close
}

#: KISS-ICP parameters that :class:`~l1_stream.odometry.KissOdometry` exposes
#: but that have not been tuned for this sensor. Kept apart from
#: :data:`DEFAULTS` so it is clear which values were measured.
UNTUNED = {
    "min_motion_th": 0.02,
    "max_points_per_voxel": 20,
    "max_num_iterations": 500,
    "convergence_criterion": 1e-4,
}
