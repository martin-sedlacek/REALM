"""The run log-directory convention, stated once.

Every launcher files a run's artifacts under::

    <log_root>/<experiment_name>/<model_name>[/<run_id>]

and `realm.eval` / `realm.vector_eval` then create ``reports/``, ``qpos/``, ``actions/`` and
``videos/`` inside that. This module exists because the join used to be hand-built at every
launcher and reader (examples 02/03/04 plus four tests), and the copies had already drifted --
different defaults, different missing-run_id behaviour. The PER-LAUNCHER decisions (what the
default log root is, what a missing run_id means) deliberately stay at the call sites, where they
are visible flags; only the layout lives here.

Pure and import-light on purpose: no omnigibson, so tests and host-side tools can import it.
"""
import os


def run_log_dir(log_root, experiment_name, model_name, run_id=None):
    """``<log_root>/<experiment_name>/<model_name>[/<run_id>]``.

    A ``run_id`` of None drops the segment entirely (examples/02_evaluate.py's historical
    behaviour); the vector launcher passes its own ``"vec"`` default instead. Everything is joined
    as given -- no normalisation, no creation.
    """
    parts = [str(log_root), str(experiment_name), str(model_name)]
    if run_id is not None:
        parts.append(str(run_id))
    return os.path.join(*parts)
