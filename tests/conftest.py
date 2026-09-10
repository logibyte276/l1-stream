"""Shared pytest configuration.

WHY THE SKIP GUARD EXISTS. Two of this suite's dependencies fail *silently* when
absent -- they turn tests into SKIPs, not errors:

    scipy      -> test_matches_scipy skips, and the quaternion scalar-last
                  convention that every downstream result depends on goes
                  unverified. It is the ONLY independent reference in the suite.
    kiss-icp   -> every odometry test skips, and the whole SLAM pipeline goes
                  untested.

`pytest -q` reports "48 passed, 2 skipped" and exits 0. A human skims that and
reads it as green. CI does the same, which is how ci.yml installing only
`.[dev]` produced a passing badge over an untested odometry pipeline.

So: set L1_NO_SKIPS=1 and any skip becomes a non-zero exit.

    L1_NO_SKIPS=1 pytest -rs      # the "did everything actually run" check

Left opt-in rather than always-on because a genuine platform skip (no display,
wrong OS) should not block someone running the suite casually. Turn it on in CI
and before a release.
"""

import os


def pytest_sessionfinish(session, exitstatus):
    if not os.environ.get("L1_NO_SKIPS"):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return
    skipped = reporter.stats.get("skipped", [])
    if skipped:
        print(f"\nL1_NO_SKIPS=1 and {len(skipped)} test(s) skipped -- failing:")
        for report in skipped:
            print(f"  SKIPPED {report.nodeid}")
        print("Install the missing extras:  pip install -e '.[dev,slam]'")
        session.exitstatus = 1
