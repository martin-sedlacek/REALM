"""Pin the matrix harness's refusal-vs-breakage verdict, on the host, with no GPU.

WHY THIS EXISTS. `tests/test_vector_integrity.py` decides whether a cell CRASHED or deliberately
REFUSED by pattern-matching its log. Four cells of the 160-cell cross product are designed to raise
NotImplementedError (both drawer tasks under VB-MOBJ and under SB-VRB), and before
`EXPECTED_NOT_IMPLEMENTED` existed they were reported identically to a cell that broke -- the matrix
verdict shipped "41 PASS, 4 CRASH" with those refusals inside the 4.

The distinction is worth exactly as much as its false-negative rate: a classifier that calls a real
failure "expected" is worse than the conflation it replaced. So the case that matters most here is
`test_other_error_alongside_not_implemented_still_crashes` -- a genuine breakage in a declared-refusal
cell must NOT be laundered.

Every case runs against a synthetic log string. The real path needs an Isaac boot and ~7 min per
cell, which is why this logic was previously only ever exercised by the runs whose verdicts depend on
it.
"""

import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

# Safe at collection time: test_vector_integrity is a DRIVER. It reads SUPPORTED_TASKS /
# SUPPORTED_PERTURBATIONS out of realm/eval.py with ast.literal_eval precisely so it never imports
# omnigibson -- see its _const_list docstring. If that ever changes, this import starts failing on the
# host and the change is the bug, not this test.
import test_vector_integrity as tvi  # noqa: E402

TRACEBACK = "Traceback (most recent call last):"
FRAME = '  File "/app/realm/environments/perturbations/sb_vrb.py", line 99, in sb_vrb'

SB_VRB_REFUSAL = "\n".join([
    "[INFO] [omnigibson] Loading scene",
    TRACEBACK,
    FRAME,
    "    raise NotImplementedError(",
    "NotImplementedError: SB-VRB does not support task_type 'close_drawer': the drawer configs "
    "declare target_objects: [], so the perturbation would inject a 'receiver' object",
])

# vb_mobj.py:71 raises bare `NotImplementedError()`, which Python renders with NO colon and no
# message. The two spellings are why NOT_IMPL_LINE has to accept an end-of-line as well as a colon.
VB_MOBJ_REFUSAL = "\n".join([
    TRACEBACK,
    '  File "/app/realm/environments/perturbations/vb_mobj.py", line 71, in vb_mobj',
    "    raise NotImplementedError()",
    "NotImplementedError",
])

CLEAN = "\n".join([
    "[INFO] [omnigibson] Loading scene",
    "Saved report to /logs/x/debug/t9_Default/reports/close_drawer_Default.csv",
    "[INFO] done",
])


def test_clean_log_has_no_crash_verdict():
    assert tvi.classify_log(CLEAN, "9:Default", 0) is None


def test_teardown_segfault_alone_is_not_a_crash():
    """The filter that predates this file: Isaac can segfault after all work is done."""
    log = CLEAN + "\nFatal Python error: Segmentation fault\nsrun: error: l40s-03: Segmentation fault"
    assert tvi.classify_log(log, "9:Default", -11) is None


@pytest.mark.parametrize("cid,log", [
    ("8:SB-VRB", SB_VRB_REFUSAL),
    ("9:SB-VRB", SB_VRB_REFUSAL),
    ("8:VB-MOBJ", VB_MOBJ_REFUSAL),
    ("9:VB-MOBJ", VB_MOBJ_REFUSAL),
])
def test_declared_refusals_report_not_impl(cid, log):
    status, detail = tvi.classify_log(log, cid, 1)
    assert status == "NOT_IMPL", f"{cid} should read as an intentional refusal, got {status}: {detail}"
    assert detail.startswith("NotImplementedError"), (
        f"the detail must carry the exception line so a reader can see WHICH refusal fired: {detail!r}")


def test_bare_not_implemented_is_recognised():
    """`raise NotImplementedError()` prints without a colon; matching only 'NotImplementedError:'
    would silently classify vb_mobj's refusal as a CRASH."""
    status, _ = tvi.classify_log(VB_MOBJ_REFUSAL, "8:VB-MOBJ", 1)
    assert status == "NOT_IMPL"


def test_undeclared_refusal_is_a_failure():
    """A refusal from a cell nobody declared means the code and the table disagree. Loud, not quiet."""
    status, detail = tvi.classify_log(SB_VRB_REFUSAL, "0:SB-VRB", 1)
    assert status == "UNDECLARED_NOT_IMPL", (
        f"an undeclared NotImplementedError must not be waved through as expected: {status} {detail}")


def test_other_error_alongside_not_implemented_still_crashes():
    """THE case this file exists for. If a declared-refusal cell breaks for a DIFFERENT reason, the
    verdict has to be CRASH -- otherwise the refusal declaration becomes a blanket amnesty for those
    four cells and a real regression in them can never be observed again."""
    log = SB_VRB_REFUSAL + "\n" + "\n".join([
        TRACEBACK,
        '  File "/app/realm/envs.py", line 12, in build',
        "KeyError: 'close_drawer'",
    ])
    status, detail = tvi.classify_log(log, "9:SB-VRB", 1)
    assert status == "CRASH", (
        f"a KeyError in a declared-refusal cell must still report CRASH, got {status}: {detail}")


@pytest.mark.parametrize("err", ["AssertionError: bad", "RuntimeError: CUDA error", "IndexError: pop"])
def test_ordinary_tracebacks_still_crash(err):
    log = "\n".join([TRACEBACK, '  File "/app/x.py", line 1, in f', err])
    status, _ = tvi.classify_log(log, "9:Default", 1)
    assert status == "CRASH"


def test_expected_not_implemented_keys_are_real_cells():
    """A typo'd or renamed key is a silent amnesty that applies to nothing -- and it would leave the
    cell it was meant to cover reporting UNDECLARED_NOT_IMPL, i.e. a failure, with no hint why."""
    for cid in tvi.EXPECTED_NOT_IMPLEMENTED:
        task_id, _, pert = cid.partition(":")
        assert task_id.isdigit() and int(task_id) < len(tvi.SUPPORTED_TASKS), (
            f"{cid!r} does not name a task index in SUPPORTED_TASKS")
        assert pert in tvi.SUPPORTED_PERTURBATIONS, (
            f"{cid!r} does not name a perturbation in SUPPORTED_PERTURBATIONS "
            f"(valid: {tvi.SUPPORTED_PERTURBATIONS})")
        assert cid == tvi.cell_id(int(task_id), pert), (
            f"{cid!r} is not spelled the way cell_id() spells it, so it can never match")


def test_every_declared_refusal_names_a_real_raise_site():
    """Each reason cites a file, e.g. 'vb_mobj.py:71'. The line number will drift; the requirement
    pinned here is that the named file exists and really does raise NotImplementedError -- so the
    citation cannot point at a module that stopped refusing."""
    perts = PROJECT_ROOT / "realm/environments/perturbations"
    for cid, reason in tvi.EXPECTED_NOT_IMPLEMENTED.items():
        m = re.search(r"(\w+\.py):(\d+)", reason)
        assert m, f"{cid}'s reason must cite its raise site as <file>.py:<line>, got: {reason!r}"
        src = perts / m.group(1)
        assert src.is_file(), f"{cid} cites {m.group(1)}, which does not exist under {perts}"
        assert "NotImplementedError" in src.read_text(), (
            f"{cid} cites {m.group(1)}, but that file no longer raises NotImplementedError -- either "
            f"the refusal moved or the cell should not be declared any more")
