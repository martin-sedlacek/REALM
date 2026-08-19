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

# THE FORMAT ISAAC ACTUALLY EMITS, captured verbatim from
# logs/drawerpert_0819/_runlogs/t8_SBVRB.log lines 905-912 (2026-08-19). omni.kit routes Python's
# stderr through its own logger, so every traceback line arrives stamped with a timestamp, an uptime,
# a level and a channel.
#
# This is the whole reason the fixture looks like this. The first version of these tests hand-wrote
# the tracebacks in the shape *Python* produces, all 24 passed, and the classifier still called both
# refusal cells CRASH on the real run -- the anchored pattern was being applied to a line beginning
# "2026-08-19T11:08:58Z". A fixture invented to match the code proves the code matches the fixture and
# nothing else. Do NOT "simplify" these strings back into bare Python tracebacks.
PFX = "2026-08-19T11:08:58Z [524,717ms] [Error] [omni.kit.app._impl] [py stderr]: "


def isaac(*lines):
    """Wrap plain traceback lines in the omni.kit stderr prefix, as the real logs carry them."""
    return "\n".join(PFX + ln for ln in lines)


SB_VRB_REFUSAL = "\n".join([
    "[0m[00:02:18.326] [INFO] [omnigibson.simulator] -------- Welcome to OmniGibson! --------[0m",
    isaac(
        TRACEBACK,
        '  File "/app/realm/environments/env_vector.py", line 130, in <listcomp>',
        "    obss = [env.apply_perturbations(res[0]) for env, res in zip(self.envs, results)]",
        "             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
        '  File "/app/realm/environments/env_dynamic.py", line 233, in apply_perturbations',
        "    self.supported_pertrubations[p]()",
        FRAME,
        "    raise NotImplementedError(",
        "NotImplementedError: SB-VRB does not support task_type 'open_drawer': the drawer configs "
        "declare target_objects: [], so the perturbation would inject a 'receiver' object that has no "
        "placeable position in these scenes and gets dropped from the air. Deliberate refusal, not an "
        "unimplemented branch -- do not 'fix' this by making it a no-op.",
    ),
    "2026-08-19T11:08:59Z [524,977ms] [Warning] [omni.graph.core.plugin] Could not find category "
    "'Replicator:Annotators' for removal",
])

# The same refusal WITHOUT the Isaac prefix. Kept as its own case because the classifier has to handle
# both: a cell that dies before omni.kit takes over stderr writes a plain Python traceback.
SB_VRB_REFUSAL_UNPREFIXED = "\n".join([
    "[INFO] [omnigibson] Loading scene",
    TRACEBACK,
    FRAME,
    "    raise NotImplementedError(",
    "NotImplementedError: SB-VRB does not support task_type 'close_drawer': the drawer configs "
    "declare target_objects: [], so the perturbation would inject a 'receiver' object",
])

# vb_mobj.py:71 raises bare `NotImplementedError()`, which Python renders with NO colon and no
# message. The two spellings are why NOT_IMPL_LINE has to accept an end-of-line as well as a colon.
VB_MOBJ_REFUSAL = isaac(
    TRACEBACK,
    '  File "/app/realm/environments/perturbations/vb_mobj.py", line 71, in vb_mobj',
    "    raise NotImplementedError()",
    "NotImplementedError",
)

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
    ("8:SB-VRB", SB_VRB_REFUSAL_UNPREFIXED),
    ("8:VB-MOBJ", VB_MOBJ_REFUSAL),
    ("9:VB-MOBJ", VB_MOBJ_REFUSAL),
])
def test_declared_refusals_report_not_impl(cid, log):
    status, detail = tvi.classify_log(log, cid, 1)
    assert status == "NOT_IMPL", f"{cid} should read as an intentional refusal, got {status}: {detail}"
    assert detail.startswith("NotImplementedError"), (
        f"the detail must carry the exception line so a reader can see WHICH refusal fired: {detail!r}")


def test_omni_kit_prefixed_traceback_is_recognised():
    """REGRESSION, drawerpert_0819. The refusal was at line 912 of t8_SBVRB.log and the cell was
    reported CRASH, because NOT_IMPL_LINE was anchored against a line that starts with a timestamp.
    Asserted on the exact prefix rather than via the helper, so a change to LOG_PREFIX that stops
    covering the real format fails here."""
    line = PFX + "NotImplementedError: SB-VRB does not support task_type 'open_drawer'"
    assert tvi.LOG_PREFIX.sub("", line).startswith("NotImplementedError:"), (
        "LOG_PREFIX no longer strips the omni.kit stderr stamp, so the anchored NOT_IMPL_LINE will "
        "match nothing and every intentional refusal will report CRASH again")
    status, _ = tvi.classify_log("\n".join([TRACEBACK, line]), "8:SB-VRB", -11)
    assert status == "NOT_IMPL"


def test_raise_echo_alone_is_not_proof_the_exception_propagated():
    """A traceback echoes the source line that raised -- "    raise NotImplementedError(" -- and that
    echo must NOT satisfy the classifier on its own. This is what the ^ anchor buys, and why the fix
    for the prefix bug strips the prefix rather than switching match() for search()."""
    log = isaac(
        TRACEBACK,
        '  File "/app/realm/environments/perturbations/sb_vrb.py", line 99, in sb_vrb',
        "    raise NotImplementedError(",
        "KeyError: 'close_drawer'",     # what ACTUALLY ended the process
    )
    status, detail = tvi.classify_log(log, "9:SB-VRB", 1)
    assert status == "CRASH", (
        f"only the raise ECHO is present and a KeyError ended the run; calling that an intentional "
        f"refusal would hide a real failure: {status} {detail}")


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


REAL_LOG = Path("/mnt/home_lustre/sedlam56/projects/REALM/logs/drawerpert_0819/_runlogs/t8_SBVRB.log")


@pytest.mark.skipif(not REAL_LOG.is_file(), reason=f"{REAL_LOG} not on this machine")
def test_against_the_real_captured_log():
    """The one case no fixture can fake: classify the actual 8:SB-VRB log off disk.

    Every other test here uses a string I wrote, and a string I wrote is exactly what let the prefix
    bug through. This one reads the 1000-line log Isaac really produced -- Kit banners, teardown
    warnings, the -11 exit and all -- and is the reason to keep drawerpert_0819 around."""
    status, detail = tvi.classify_log(REAL_LOG.read_text(errors="replace"), "8:SB-VRB", -11)
    assert status == "NOT_IMPL", f"real log misclassified as {status}: {detail}"
    assert "does not support task_type 'open_drawer'" in detail, (
        f"classified correctly but reported the wrong line: {detail!r}")


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
