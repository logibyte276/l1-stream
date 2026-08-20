"""Drive the visualiser frame by frame while doing your own processing.

This is the pattern to use when plugging in odometry: the visualiser and your
algorithm read the SAME accumulated cloud, so what you see is what you feed in.
"""

import logging

import numpy as np

from l1_stream import LidarStream, RotatedScanAccumulator
from l1_stream.visualizer import LiveVisualizer

logging.basicConfig(level=logging.INFO, format="%(message)s")

stream = LidarStream.for_history(seconds=2.0)
accumulator = RotatedScanAccumulator(max_scans=120, max_time_gap=0.01)

with LiveVisualizer(stream=stream, accumulator=accumulator) as vis:
    while vis.step():
        points = accumulator.get_points()
        if len(points) < 100:
            continue

        # Replace this with your odometry / obstacle-avoidance step.
        # Note `points` is float64 and C-contiguous already.
        centroid = points.mean(axis=0)
        nearest = np.linalg.norm(points, axis=1).min()
        print(f"{len(points)} pts | centroid={np.round(centroid, 2)} | nearest={nearest:.2f} m")
