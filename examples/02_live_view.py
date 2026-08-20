"""Live Open3D view. Requires the viz extra:  pip install -e '.[viz]'

Equivalent to the `l1-visualize` console command.
"""

import logging

from l1_stream.visualizer import LiveVisualizer

logging.basicConfig(level=logging.INFO, format="%(message)s")

LiveVisualizer(
    max_scans=150,      # ~0.8 s of packets on screen at 180 Hz
    refresh_hz=30.0,
    point_size=2.0,
).run()
