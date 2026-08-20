"""Visualizer tests.

Open3D needs a display, which CI does not have, so a stub is injected in place
of the lazy import. That covers everything except the actual pixels: lifecycle,
stream ownership, the frame callback, and the exit condition.
"""

import types

from l1_stream import LidarStream
from l1_stream import visualizer as V

PORT = 45700


# --- Open3D stub ------------------------------------------------------------

class _FakeViewControl:
    def set_lookat(self, *a): pass
    def set_front(self, *a): pass
    def set_up(self, *a): pass
    def set_zoom(self, *a): pass


class _FakeRenderOption:
    point_size = 1.0
    background_color = None


class _FakeVisualizer:
    def __init__(self):
        self.created = False
        self.destroyed = False
        self.geometries = []
        self.frames = 0
        self.max_frames = 3

    def create_window(self, **kwargs):
        self.created = True

    def add_geometry(self, geom, reset_bounding_box=True):
        self.geometries.append(geom)

    def get_render_option(self):
        return _FakeRenderOption()

    def get_view_control(self):
        return _FakeViewControl()

    def update_geometry(self, geom): pass

    def poll_events(self):
        self.frames += 1
        return self.frames <= self.max_frames  # False => window closed

    def update_renderer(self): pass

    def destroy_window(self):
        self.destroyed = True


class _FakePointCloud:
    def __init__(self):
        self.points = None


def _fake_open3d():
    o3d = types.SimpleNamespace()
    o3d.visualization = types.SimpleNamespace(Visualizer=_FakeVisualizer)
    o3d.geometry = types.SimpleNamespace(
        PointCloud=_FakePointCloud,
        TriangleMesh=types.SimpleNamespace(
            create_coordinate_frame=lambda size, origin: ("axes", size)
        ),
    )
    o3d.utility = types.SimpleNamespace(Vector3dVector=lambda arr: arr)
    return o3d


def patch_open3d():
    original = V._import_open3d
    V._import_open3d = _fake_open3d
    return original


def restore_open3d(original):
    V._import_open3d = original


# --- tests ------------------------------------------------------------------

def test_module_imports_without_open3d_installed():
    # The whole point of the lazy import: `import l1_stream.visualizer` must
    # never fail on a headless Jetson.
    assert hasattr(V, "LiveVisualizer")
    assert "pip install" in V.OPEN3D_INSTALL_HINT


def test_missing_open3d_gives_an_actionable_error():
    def boom():
        raise ImportError(V.OPEN3D_INSTALL_HINT)

    original = V._import_open3d
    V._import_open3d = boom
    try:
        V.LiveVisualizer(stream=LidarStream(port=PORT, timeout=0.1)).open(wait_for_data=0)
    except ImportError as exc:
        assert "pip install" in str(exc)
    else:
        raise AssertionError("expected ImportError")
    finally:
        V._import_open3d = original


def test_bad_refresh_rate_rejected():
    for bad in (0, -5):
        try:
            V.LiveVisualizer(refresh_hz=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for refresh_hz={bad}")


def test_step_returns_false_when_window_closes():
    original = patch_open3d()
    try:
        stream = LidarStream(port=PORT + 1, timeout=0.1)
        vis = V.LiveVisualizer(stream=stream, refresh_hz=1000.0, stats_interval=0)
        vis.open(wait_for_data=0)
        assert vis.step() is True
        assert vis.step() is True
        assert vis.step() is True
        assert vis.step() is False  # stub reports the window closed on frame 4
        vis.close()
    finally:
        restore_open3d(original)


def test_step_before_open_raises():
    try:
        V.LiveVisualizer(stream=LidarStream(port=PORT + 2, timeout=0.1)).step()
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_owned_stream_is_started_and_stopped():
    original = patch_open3d()
    try:
        vis = V.LiveVisualizer(refresh_hz=1000.0, stats_interval=0)
        vis.open(wait_for_data=0)
        assert vis.stream.is_running
        vis.close()
        assert not vis.stream.is_running
    finally:
        restore_open3d(original)


def test_borrowed_but_unstarted_stream_gets_started():
    """Ownership decides who STOPS the stream, not who starts it.

    A caller who constructs a stream and hands it over still expects the window
    to render; requiring them to call start() first would be a silent no-data
    trap.
    """
    original = patch_open3d()
    try:
        stream = LidarStream(port=PORT + 3, timeout=0.1)
        assert not stream.is_running
        vis = V.LiveVisualizer(stream=stream, refresh_hz=1000.0, stats_interval=0)
        vis.open(wait_for_data=0)
        assert stream.is_running
        vis.close()
        assert not stream.is_running  # we started it, so we stop it
    finally:
        restore_open3d(original)


def test_borrowed_already_running_stream_is_left_alone():
    original = patch_open3d()
    try:
        stream = LidarStream(port=PORT + 4, timeout=0.1)
        stream.start()
        try:
            vis = V.LiveVisualizer(stream=stream, refresh_hz=1000.0, stats_interval=0)
            vis.open(wait_for_data=0)
            vis.close()
            assert stream.is_running  # caller's stream, caller's problem
        finally:
            stream.stop()
    finally:
        restore_open3d(original)


def test_window_creation_failure_does_not_leak_the_reader_thread():
    class _ExplodingVisualizer(_FakeVisualizer):
        def create_window(self, **kwargs):
            raise RuntimeError("no DISPLAY")

    def exploding_o3d():
        o3d = _fake_open3d()
        o3d.visualization = types.SimpleNamespace(Visualizer=_ExplodingVisualizer)
        return o3d

    original = V._import_open3d
    V._import_open3d = exploding_o3d
    try:
        stream = LidarStream(port=PORT + 5, timeout=0.1)
        vis = V.LiveVisualizer(stream=stream, stats_interval=0)
        try:
            vis.open(wait_for_data=0)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")
        assert not stream.is_running
    finally:
        V._import_open3d = original


def test_on_frame_callback_fires_once_per_frame():
    original = patch_open3d()
    try:
        seen = []
        stream = LidarStream(port=PORT + 6, timeout=0.1)
        vis = V.LiveVisualizer(
            stream=stream, refresh_hz=1000.0, stats_interval=0,
            on_frame=lambda pts, acc: seen.append(len(pts)),
        )
        vis.open(wait_for_data=0)
        vis.step()
        vis.step()
        vis.close()
        assert len(seen) == 2
    finally:
        restore_open3d(original)


def test_axes_geometry_added_before_the_point_cloud():
    # Adding an empty PointCloud first gives Open3D a degenerate bounding box
    # to fit the camera to, which is the "zoomed to nowhere on startup" bug.
    original = patch_open3d()
    try:
        vis = V.LiveVisualizer(
            stream=LidarStream(port=PORT + 7, timeout=0.1),
            show_axes=True, stats_interval=0,
        )
        vis.open(wait_for_data=0)
        assert vis._vis.geometries[0] == ("axes", 1.0)
        assert isinstance(vis._vis.geometries[1], _FakePointCloud)
        vis.close()
    finally:
        restore_open3d(original)


def test_double_open_is_ignored():
    original = patch_open3d()
    try:
        vis = V.LiveVisualizer(stream=LidarStream(port=PORT + 8, timeout=0.1),
                               stats_interval=0)
        vis.open(wait_for_data=0)
        first = vis._vis
        vis.open(wait_for_data=0)
        assert vis._vis is first
        vis.close()
        vis.close()  # idempotent
    finally:
        restore_open3d(original)
