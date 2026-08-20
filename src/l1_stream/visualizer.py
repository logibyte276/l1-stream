"""Live Open3D visualisation of an accumulated, rotation-compensated cloud.

Importing this module has **no side effects**. Nothing opens a socket, starts a
thread, or creates a window until you construct a :class:`LiveVisualizer` and
call :meth:`LiveVisualizer.run` (or use it as a context manager). That keeps the
module importable from test suites, notebooks, and headless processes that only
want the class definition.

Open3D is an **optional** dependency (``pip install l1-stream[viz]``).
It is imported lazily inside :meth:`LiveVisualizer.open` so that the rest of
the package stays usable on a headless Jetson where installing Open3D is a
chore.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

import numpy as np

from .rotation import RotatedScanAccumulator
from .stream import LidarStream

logger = logging.getLogger(__name__)

__all__ = ["LiveVisualizer", "OPEN3D_INSTALL_HINT"]

OPEN3D_INSTALL_HINT = (
    "Open3D is required for visualisation but is not installed. "
    "Install it with:  pip install 'l1-stream[viz]'   "
    "(on Jetson/ARM64 you may need a prebuilt wheel from "
    "https://github.com/isl-org/Open3D/releases)"
)


def _import_open3d():
    try:
        import open3d as o3d  # noqa: WPS433 (deliberate lazy import)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(OPEN3D_INSTALL_HINT) from exc
    return o3d


class LiveVisualizer:
    """Renders a rolling, IMU-rotated point cloud in an Open3D window.

    The visualiser can own its stream or borrow one::

        # owns the stream: starts and stops it for you
        LiveVisualizer().run()

        # borrows an existing stream, so odometry can read the same data
        with LidarStream.for_history(2.0) as lidar:
            vis = LiveVisualizer(stream=lidar)
            vis.run()

    For frame-by-frame control (e.g. interleaving with your own processing),
    drive it manually::

        with LiveVisualizer() as vis:
            while vis.step():
                my_odometry.process(vis.accumulator.get_points())

    Args:
        stream: An existing :class:`~l1_stream.stream.LidarStream`. If
            ``None``, one is created and its lifetime is managed here.
        accumulator: An existing accumulator to share with other consumers.
        window_name / width / height: Open3D window settings.
        max_scans: How many rotated scan packets stay on screen.
        max_time_gap: Maximum scan-to-IMU timestamp mismatch, in seconds.
        refresh_hz: Target render rate. This is a *ceiling*; if a frame takes
            longer than the budget the loop does not sleep at all.
        show_axes: Draw a coordinate frame at the origin.
        stats_interval: Seconds between status log lines. Status goes through
            ``logging`` at a throttled rate rather than ``print`` per frame; at
            display rates an unthrottled write is thousands of flushes a second,
            which over SSH can dominate frame time on its own.
        on_frame: Optional ``callback(points, accumulator)`` invoked once per
            rendered frame. A hook for odometry, recording, or logging without
            forking this class.
    """

    def __init__(
        self,
        stream: Optional[LidarStream] = None,
        accumulator: Optional[RotatedScanAccumulator] = None,
        *,
        window_name: str = "Unitree L1 Point Cloud",
        width: int = 1280,
        height: int = 800,
        max_scans: int = 100,
        max_time_gap: float = 0.01,
        refresh_hz: float = 30.0,
        show_axes: bool = True,
        axes_size: float = 1.0,
        point_size: float = 2.0,
        background_color: Optional[Any] = None,
        stats_interval: float = 1.0,
        on_frame: Optional[Callable[[np.ndarray, RotatedScanAccumulator], None]] = None,
    ):
        if refresh_hz <= 0:
            raise ValueError("refresh_hz must be > 0")

        self._owns_stream = stream is None
        self.stream = stream if stream is not None else LidarStream.for_history(2.0)
        self.accumulator = accumulator or RotatedScanAccumulator(
            max_scans=max_scans, max_time_gap=max_time_gap
        )

        self.window_name = window_name
        self.width = width
        self.height = height
        self.show_axes = show_axes
        self.axes_size = axes_size
        self.point_size = point_size
        self.background_color = background_color
        self.refresh_period = 1.0 / float(refresh_hz)
        self.stats_interval = stats_interval
        self.on_frame = on_frame

        self._o3d = None
        self._vis = None
        self._pcd = None
        self._is_open = False
        self._started_stream = False
        self._next_frame_at = 0.0
        self._last_stats_at = 0.0
        self._frames = 0

    # --- lifecycle ---

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self, wait_for_data: float = 5.0) -> "LiveVisualizer":
        """Start the stream (if owned) and create the window."""
        if self._is_open:
            logger.warning("LiveVisualizer already open; open() ignored.")
            return self

        self._o3d = _import_open3d()

        # Start the stream if it is not already running, whether we created it
        # or it was handed to us. Ownership decides who *stops* it, not who
        # starts it -- otherwise passing in a freshly constructed stream would
        # give you a window that renders nothing forever.
        if not self.stream.is_running:
            self.stream.start()
            self._started_stream = True

        if wait_for_data > 0 and not self.stream.wait_until_ready(wait_for_data):
            # Do not silently show an empty window for the next ten minutes.
            logger.warning(
                "No scan+IMU data within %.1fs. Is the C++ UDP bridge running, and "
                "is it sending to the port this stream is bound to?",
                wait_for_data,
            )

        try:
            self._create_window()
        except BaseException:
            # Window creation fails routinely -- no DISPLAY, broken X11
            # forwarding, no GLX. Without this the reader thread we just
            # started would be left running with nothing draining it.
            if self._started_stream and self.stream.is_running:
                self.stream.stop()
                self._started_stream = False
            raise

        self._is_open = True
        self._next_frame_at = time.monotonic()
        self._last_stats_at = time.monotonic()
        return self

    def _create_window(self) -> None:
        o3d = self._o3d
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(
            window_name=self.window_name, width=self.width, height=self.height
        )

        # Add a geometry with real extent FIRST. Adding an empty point cloud
        # first gives Open3D a degenerate bounding box to fit the camera to,
        # which is where "the view is zoomed to nowhere on startup" comes from.
        if self.show_axes:
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=self.axes_size, origin=[0.0, 0.0, 0.0]
            )
            self._vis.add_geometry(axes)

        self._pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._pcd, reset_bounding_box=not self.show_axes)

        opt = self._vis.get_render_option()
        opt.point_size = self.point_size
        if self.background_color is not None:
            opt.background_color = np.asarray(self.background_color, dtype=np.float64)

        ctr = self._vis.get_view_control()
        ctr.set_lookat([0.0, 0.0, 1.5])
        ctr.set_front([0.0, -1.0, 0.3])
        ctr.set_up([0.0, 0.0, 1.0])
        ctr.set_zoom(2.0)

    def close(self) -> None:
        """Destroy the window and stop the stream if this object owns it."""
        if self._vis is not None:
            try:
                self._vis.destroy_window()
            except Exception:  # pragma: no cover - Open3D teardown is noisy
                logger.debug("destroy_window() raised during close", exc_info=True)
            self._vis = None
        self._pcd = None
        self._is_open = False

        if (self._owns_stream or self._started_stream) and self.stream.is_running:
            self.stream.stop()
        self._started_stream = False

    def __enter__(self) -> "LiveVisualizer":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # --- rendering ---

    def step(self) -> bool:
        """Ingest, render one frame, and sleep to the frame budget.

        Returns ``False`` once the user closes the window, so it drops
        straight into ``while vis.step(): ...``.
        """
        if not self._is_open or self._vis is None:
            raise RuntimeError("Visualizer is not open -- call open() first.")

        self.accumulator.update(self.stream)
        points = self.accumulator.get_points()

        if len(points):
            self._pcd.points = self._o3d.utility.Vector3dVector(points)
            self._vis.update_geometry(self._pcd)

        if not self._vis.poll_events():
            return False
        self._vis.update_renderer()

        self._frames += 1
        if self.on_frame is not None:
            self.on_frame(points, self.accumulator)

        now = time.monotonic()
        if self.stats_interval > 0 and now - self._last_stats_at >= self.stats_interval:
            self._log_stats(now)
            self._last_stats_at = now

        # Deadline-based pacing. `time.sleep(period)` on its own would make the
        # real frame rate (1 / (period + work_time)), i.e. always slower than
        # requested, and it hides the fact that you are over budget.
        self._next_frame_at += self.refresh_period
        remaining = self._next_frame_at - now
        if remaining > 0:
            time.sleep(remaining)
        else:
            # Behind schedule: reset the deadline rather than trying to catch
            # up, which would busy-spin.
            self._next_frame_at = now
        return True

    def run(self, wait_for_data: float = 5.0) -> None:
        """Open (if needed) and render until the window closes or Ctrl+C."""
        opened_here = not self._is_open
        try:
            if opened_here:
                self.open(wait_for_data=wait_for_data)
            while self.step():
                pass
        except KeyboardInterrupt:
            logger.info("Stopped by Ctrl+C.")
        finally:
            if opened_here:
                self.close()

    def _log_stats(self, now: float) -> None:
        acc = self.accumulator.stats()
        stream = self.stream.stats()
        fps = self._frames / max(now - self._last_stats_at, 1e-9)
        self._frames = 0
        logger.info(
            "%.1f fps | %d pts/s | on-screen %d pts (%d scans) | pending %d | unmatched %d | "
            "scans rx %d drop %d | imu rx %d drop %d",
            fps, acc["points_per_second"], acc["points_held"], acc["scans_held"], acc["scans_pending"],
            acc["scans_unmatched"], stream["scans_total"], stream["scans_dropped"],
            stream["imu_total"], stream["imu_dropped"],
        )
