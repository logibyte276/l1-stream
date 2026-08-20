import threading
import time

from l1_stream import RingBuffer


def test_drops_oldest_when_full():
    rb = RingBuffer(3)
    for i in range(5):
        rb.append(i)
    assert rb.snapshot() == [2, 3, 4]
    assert rb.total_received == 5
    assert rb.total_dropped == 2
    assert len(rb) == 3


def test_latest_and_empty_behaviour():
    rb = RingBuffer(2)
    assert rb.latest() is None
    rb.append("a")
    assert rb.latest() == "a"


def test_latest_n_returns_oldest_first():
    rb = RingBuffer(5)
    for i in range(5):
        rb.append(i)
    assert rb.latest_n(3) == [2, 3, 4]


def test_latest_n_over_capacity_always_raises():
    # Unsatisfiable no matter how long you wait => caller bug => fail loudly.
    rb = RingBuffer(3)
    for i in range(3):
        rb.append(i)
    try:
        rb.latest_n(4)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_latest_n_partial_is_allowed_by_default():
    # Happens legitimately at startup; raising here would crash every launch.
    rb = RingBuffer(10)
    rb.append(1)
    assert rb.latest_n(5) == [1]


def test_latest_n_partial_can_be_forbidden():
    rb = RingBuffer(10)
    rb.append(1)
    try:
        rb.latest_n(5, allow_partial=False)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_bool_is_rejected_as_n():
    # bool subclasses int, so True would otherwise sneak through as n=1.
    rb = RingBuffer(4)
    rb.append(1)
    for bad in (True, 1.5, "2"):
        try:
            rb.latest_n(bad)
        except TypeError:
            continue
        raise AssertionError(f"expected TypeError for {bad!r}")


def test_zero_and_negative_n_rejected():
    rb = RingBuffer(4)
    for bad in (0, -1):
        try:
            rb.latest_n(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_maxlen_must_be_positive_int():
    for bad, exc in ((0, ValueError), (-3, ValueError), (True, TypeError), (2.0, TypeError)):
        try:
            RingBuffer(bad)
        except exc:
            continue
        raise AssertionError(f"expected {exc.__name__} for {bad!r}")


def test_drain_empties_and_returns_everything():
    rb = RingBuffer(5)
    for i in range(4):
        rb.append(i)
    assert rb.drain() == [0, 1, 2, 3]
    assert len(rb) == 0
    assert rb.latest() is None
    assert rb.total_received == 4  # counters survive a drain


def test_snapshot_does_not_consume():
    rb = RingBuffer(5)
    rb.append(1)
    assert rb.snapshot() == [1]
    assert rb.snapshot() == [1]


def test_snapshot_is_a_copy_not_a_live_view():
    rb = RingBuffer(5)
    rb.append(1)
    snap = rb.snapshot()
    rb.append(2)
    assert snap == [1]


def test_wait_for_new_returns_on_producer_append():
    rb = RingBuffer(4)
    result = {}

    def consumer():
        result["value"] = rb.wait_for_new(timeout=2.0)

    t = threading.Thread(target=consumer)
    t.start()
    time.sleep(0.05)
    rb.append("hello")
    t.join(timeout=2.0)
    assert result["value"] == "hello"


def test_wait_for_new_times_out():
    rb = RingBuffer(4)
    start = time.monotonic()
    assert rb.wait_for_new(timeout=0.1) is None
    assert time.monotonic() - start >= 0.09


def test_concurrent_producers_lose_nothing():
    rb = RingBuffer(10_000)
    n_threads, per_thread = 8, 500

    def produce(tid):
        for i in range(per_thread):
            rb.append((tid, i))

    threads = [threading.Thread(target=produce, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert rb.total_received == n_threads * per_thread
    assert rb.total_dropped == 0
    assert len(rb.snapshot()) == n_threads * per_thread


def test_reader_never_sees_a_torn_state():
    """Hammer append() from one thread while another reads, checking invariants."""
    rb = RingBuffer(64)
    stop = threading.Event()
    failures = []

    def writer():
        i = 0
        while not stop.is_set():
            rb.append(i)
            i += 1

    def reader():
        try:
            while not stop.is_set():
                snap = rb.snapshot()
                assert len(snap) <= 64
                assert snap == sorted(snap)  # monotonic: nothing reordered/torn
                rb.latest_n(min(len(snap) or 1, 64))
        except Exception as exc:  # pragma: no cover
            failures.append(exc)

    w, r = threading.Thread(target=writer), threading.Thread(target=reader)
    w.start()
    r.start()
    time.sleep(0.3)
    stop.set()
    w.join()
    r.join()
    assert not failures, failures
