"""The settled pipeline configuration. ONE definition, imported everywhere.

This module exists because the same constants were previously written out in
six places and drifted apart:

    06_odometry_offline   voxel 0.15  min_range 0.25  deskew off
    07_diagnostics        voxel 0.15  min_range 0.40  deskew off  (now 07_precheck)
    09_ablation           voxel 0.25  min_range 0.40  deskew on   (deleted)
    08_odometry_live      voxel 0.25  min_range 0.40  deskew on   <- the ROBOT
    KissOdometry.__init__ voxel 0.25  min_range 0.40  deskew on
    11_vibration          voxel 0.25                             (now 07_precheck)

(Those names are history. 10_selfhit and 11_vibration were merged into
07_precheck on 2026-09-09 and 09_ablation was deleted on 2026-09-10; the
numbers 09 and 10 now belong to the map viewers, 09_map_offline and
10_map_live.)

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

__all__ = ["DEFAULTS", "SCALE_FACTOR", "UNTUNED"]

#: Measured on hardware. Change a value HERE and the whole pipeline follows.
DEFAULTS = {
    # frame assembly
    "frame_duration": 0.2,      # 0.05 and 0.5 both measured worse
    "rotate_with_imu": True,    # ESSENTIAL: 32-126x worse on loop closure without

    # registration
    "voxel_size": 0.15,         # 0.10 also real-time viable; 0.15 keeps margin
    "max_range": 25.0,          # trimming to 10 measurably hurt rotation
    "min_range": 0.25,          # clears the MEASURED 0.19 m self-hit radius
    # False on the LOOP evidence. Loop closure is the right basis here because
    # its ground truth is exactly zero -- no tape measure, no judgement about
    # where the car stopped -- which is the failure mode that invalidated three
    # of the five line drives.
    #
    #   recording       metric       deskew ON   deskew OFF   winner
    #   loop_drive_1    loop error   0.030 m     0.007 m      OFF, 4.3x
    #   loop_drive_2    loop error   0.068 m     0.048 m      OFF, 1.4x
    #
    # ON also lengthens the path on both (18.59 vs 17.61; 38.89 vs 36.15), i.e.
    # it is adding jitter, which is the mechanism behind the worse closure.
    #
    # THE KNOWN EXCEPTION -- a featureless hallway, 7 m line, truth 7.0 m:
    #
    #   line_drive_7m   net displ.   6.77 m      6.64 m       ON, by 130 mm
    #
    # That is the largest margin in the whole dataset and it goes the other way.
    # A corridor's walls run PARALLEL to travel: they pin lateral and vertical
    # position and barely constrain ALONG-TRACK position, so ICP has little to
    # fix how far the car went forward and falls back on the prior. Deskew's
    # constant-velocity model supplies exactly that. Note ON made the path
    # SHORTER there (10.27 vs 10.69) while raising net -- straighter, not
    # inflated, so "deskew just adds distance" does not explain it.
    #
    # Loop closure is provably BLIND to scale error (a uniform scale cancels
    # around a symmetric loop), so the loops cannot see this effect at all.
    # Deciding on loops alone means using the metric that structurally cannot
    # detect the corridor result. Both findings can be true at once.
    #
    # THE TEST, and it needs the scale factor first: the LiDAR under-reads
    # distance by ~5.3% (see SCALE_FACTOR below). The 130 mm may be deskew
    # partially compensating for that bias. Apply the correction, then re-run
    # the corridor ablation -- two 06_odometry_offline runs one flag apart. If
    # the margin collapses, OFF wins on every metric and this exception closes.
    #
    # KISS-ICP's own default is True. Flipping is a one-word change below.
    "deskew": False,
    "initial_threshold": 0.4,   # adaptive settles at 0.32-0.55, so the seed is right
}

#: Metric calibration. The LiDAR consistently UNDER-reads distance travelled.
#:
#: Measured on the only two line drives with trustworthy ground truth -- a
#: suitcase at the 0 mark so the car physically stops at zero, with the SAME
#: reference (front wheels) at both ends:
#:
#:     truth 5.0 m -> measured 4.72 m    scale 1.0593
#:     truth 7.0 m -> measured 6.64 m    scale 1.0542
#:     least squares through the origin  scale 1.0559   (under-reads 5.30%)
#:
#: There is NO fixed-offset term, and this is geometry rather than a fitting
#: choice: the LiDAR is rigidly mounted, so on a straight line it travels
#: exactly as far as the front wheels do. An offset could only appear if the
#: start and end references were different parts of the car; they were not.
#:
#: That matters for how much to trust this. Forcing offset to zero leaves ONE
#: free parameter against TWO measurements -- a spare degree of freedom -- and
#: the residuals come out at -16 mm and +11 mm, both inside the L1's own
#: +/-20 mm point accuracy. An earlier offset+scale fit matched exactly only
#: because it had nothing left over to be wrong with.
#:
#: NOT APPLIED ANYWHERE. Trajectories out of KissOdometry are uncorrected.
#: This is a documented constant so the correction is applied consistently and
#: derived once, not re-guessed per script.
#:
#: OUT OF DATE (2026-09-11). The re-recorded room drives read 4.81 m on 5 m
#: (twice, at 0.5 and 2.0 m/s) and about 2.91 m on 3 m, which fit ~1.037,
#: not 1.0559. Nothing reads this constant: while the value is being
#: re-established, try candidates with `06_odometry_offline --scale`, and
#: update this line once one holds on drives it was not fitted to.
#:
#: BEFORE RELYING ON IT: validate at a third distance (10 m), and re-derive
#: after any change to the LiDAR mount -- it came off and was re-glued on
#: 2026-09-08, so this number describes the post-09-08 robot only.
SCALE_FACTOR = 1.0559

#: Exposed but never swept on this rig. Listed separately so nobody mistakes
#: "it is in DEFAULTS" for "somebody measured it".
UNTUNED = {
    "min_motion_th": 0.02,
    "max_points_per_voxel": 20,
    "max_num_iterations": 500,
    "convergence_criterion": 1e-4,
}
