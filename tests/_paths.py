"""Where a test's scratch artifacts go, and how a test decides a child eval really ran.

Shared by the four script-style tests that drive examples/02_evaluate.py in a subprocess
(test_integrity, test_perturbations_integrity, test_single_task, test_pi0_integration) so the two
rules below are stated once instead of four times, and cannot drift apart.
"""
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()


def scratch_log_root(name):
    """Absolute path for a test's throwaway log tree.

    NOT `PROJECT_ROOT/logs/<name>`, which is what these tests used to use and which cannot work
    inside the container. In this checkout `logs` is a SYMLINK to an absolute host path
    (.../projects/REALM/logs); scripts/clara/interactive/rr binds the checkout at /app and the log
    tree at /logs, and does NOT bind the symlink's target -- so `/app/logs` resolves to nothing and
    the very first `os.makedirs()` dies with

        FileNotFoundError: [Errno 2] No such file or directory: '/app/logs/<name>'

    before a single task is evaluated. Measured 2026-08-16 on this branch. The symlink arrived with
    the OG 3.9.1 port; under the retired `scripts/run_docker.sh` (`-v $(pwd):/app`) `logs` was a
    real directory inside the checkout and `/app/logs` worked, which is why the tests were written
    this way.

    Order: an explicit REALM_TEST_LOG_DIR wins, then the container's bound log tree at /logs, then
    PROJECT_ROOT/logs for a plain host checkout that has a real logs directory.
    """
    override = os.environ.get("REALM_TEST_LOG_DIR")
    if override:
        return os.path.join(override, name)
    if os.path.isdir("/logs"):
        return os.path.join("/logs", name)
    return os.path.join(PROJECT_ROOT, "logs", name)


# Signatures that mean a child eval died even if its exit status says otherwise. Same list as
# tests/test_vector_integrity.py's CRASH_MARKERS -- kept identical on purpose.
CRASH_MARKERS = re.compile(
    r"Traceback \(most recent call last\)|AssertionError|AttributeError|KeyError|TypeError|"
    r"IndexError|RuntimeError|Segmentation fault|CUDA error|out of memory", re.I)
# Isaac tears down with a segfault after all work is done, on passing runs as well as failing ones.
# Counting that as a crash would fail every cell.
TEARDOWN_NOISE = re.compile(r"Fatal Python error: Segmentation fault|"
                            r"srun: error:.*Segmentation fault|core dumped")


def crash_lines(output):
    """Lines in a child's combined output that indicate a real failure, teardown noise removed."""
    return [ln for ln in (output or "").splitlines()
            if CRASH_MARKERS.search(ln) and not TEARDOWN_NOISE.search(ln)]
