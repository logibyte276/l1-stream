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
    # DISPUTED, and the current value is the weakest of the three measurements.
    # Straight-line drives, net displacement vs a tape measure:
    #
    #   recording        truth      ON              OFF             margin
    #   room, line       5.0 m      4.870 (-130mm)  4.890 (-110mm)  20 mm  OFF
    #   hallway, line    7.0 m      6.770 (-230mm)  6.640 (-360mm)  130 mm ON
    #   closed loop      --         0.2% of path    0.1% of path    0.1 pp OFF
    #
    # The 20 mm margin is the L1's own single-point accuracy (+/-20 mm), so that
    # row is a TIE read at the noise floor, not a win. Weighted by margin this
    # is one real result (ON) against two ties -- the opposite of how it reads
    # if you just count rows.
    #
    # Worse, the two line drives changed SCENE and DISTANCE together, so neither
    # is isolated. A straight line measures scale, and the shapes differ:
    # ON is -130 -> -230 mm over 5 -> 7 m (1.77x, ~scale-shaped); OFF is
    # -110 -> -360 mm (3.27x, superlinear -- neither a fixed offset nor a scale
    # error). Two points cannot separate offset from scale: the fit returns a
    # physically impossible negative offset for both conditions.
    #
    # Working hypothesis: a bare corridor's walls run PARALLEL to travel, so
    # they constrain lateral and vertical position but barely constrain
    # along-track position. ICP has little to fix how far the car went forward,
    # falls back on the prior, and under-reads -- exactly the -5.14%. Deskew's
    # constant-velocity model supplies precisely that missing along-track
    # information, which predicts its benefit tracks along-track degeneracy:
    # ~0 in a cluttered room (20 mm, seen), large in a corridor (130 mm, seen),
    # and growing with distance in a corridor.
    #
    # THE TEST: 5 m and 7 m in the SAME corridor, both conditions. Four runs.
    # Separates distance from scene and gives open item 1 a second distance.
    #
    # Left at False pending that test, but note KISS-ICP's own default is True
    # and True is the more defensible setting if a result has to be justified.
    # Flipping it is a one-word change on the line below.
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
