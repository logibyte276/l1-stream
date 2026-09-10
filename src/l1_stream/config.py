"""The settled pipeline configuration. ONE definition, imported everywhere.

This module exists because the same constants were previously written out in
six places and drifted apart:

    06_odometry_offline   voxel 0.15  min_range 0.25  deskew off
    07_diagnostics        voxel 0.15  min_range 0.40  deskew off  (now 07_precheck)
    09_ablation           voxel 0.25  min_range 0.40  deskew on
    08_odometry_live      voxel 0.25  min_range 0.40  deskew on   <- the ROBOT
    KissOdometry.__init__ voxel 0.25  min_range 0.40  deskew on
    11_vibration          voxel 0.25                             (now 07_precheck)

The last one survived the first consolidation, because it was an argparse
`default=` and the guard skipped those wholesale. It was judging "does this
vibration smear exceed a voxel?" against a voxel 67% larger than the one in
use. tests/test_config.py now checks argparse defaults too.

Each looks plausible alone, and their outputs look comparable when they are
not. The live script and the class signature were both still on values the
tuning had moved away from weeks earlier.

``FrameAssembler`` and ``KissOdometry`` take their signature defaults from
here, so constructing either one bare gives the tuned configuration rather
than a stale one. ``tests/test_config.py`` asserts that they still agree --
without that test this file is a convention, and conventions drift.

This module imports nothing from the rest of the package, so it can be
imported from anywhere without a cycle.

Every value's justification lives in claude/slam-results.md.
"""

from __future__ import annotations

__all__ = ["DEFAULTS", "UNTUNED"]

#: Measured on hardware. Change a value HERE and the whole pipeline follows.
DEFAULTS = {
    # frame assembly
    "frame_duration": 0.2,      # 0.05 and 0.5 both measured worse
    "rotate_with_imu": True,    # ESSENTIAL: 32-126x worse on loop closure without

    # registration
    "voxel_size": 0.15,         # 0.10 also real-time viable; 0.15 keeps margin
    "max_range": 25.0,          # trimming to 10 measurably hurt rotation
    "min_range": 0.25,          # chassis measured at 0.2 m radius, plus margin
    # UNRESOLVED, not settled. Better OFF in the room (4.890 vs 4.870) and on
    # the loop (0.1% vs 0.2%), but better ON in the bare hallway (6.770 vs
    # 6.640). One of those scenes is lying and we do not yet know which. Left
    # OFF because that is where the first two measurements pointed -- treat it
    # as a coin the evidence has not landed, not as a finding.
    "deskew": False,
    "initial_threshold": 0.4,   # adaptive settles at 0.32-0.55, so the seed is right
}

#: Exposed but never swept on this rig. Listed separately so nobody mistakes
#: "it is in DEFAULTS" for "somebody measured it".
UNTUNED = {
    "min_motion_th": 0.02,
    "max_points_per_voxel": 20,
    "max_num_iterations": 500,
    "convergence_criterion": 1e-4,
}
