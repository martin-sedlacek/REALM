"""Rubric stages that latch only after their condition held N consecutive control steps.

Used by `TaskProgressionMixin.recompute_task_progression` (realm/environments/task_progression.py), which
decides each stage's hold: `--stage_hold_steps` N, or 1 for the stages in `SHORT_STAGES`. Pure Python, no
simulator imports, so it is tested on its own (tests/test_stage_hold.py).
"""


class StageHold:
    """One rubric's latching with one hold per stage.

    `step(flags, evaluate)` takes the rubric's latched flags (a prefix of True values) and `evaluate(i)`,
    stage i's condition now, and walks the stages in order as the original rubric loop does: latched
    stages are skipped; the first unlatched stage is evaluated; a true value extends its streak, and once
    the streak reaches the stage's hold the stage latches and the next stage is evaluated in the same step
    (so several stages can latch in one step, and a stage's streak counts from the step the previous stage
    latched, that step included); a false value resets the streak and ends the walk; an unfinished streak
    ends the walk too. Only the stage under consideration is evaluated, at most once per step -- with every
    hold 1 these are exactly the original loop's checker calls. A latched stage stays latched. A checker
    that raises propagates.
    """

    def __init__(self, stages, holds):
        self.stages = list(stages)
        self.holds = [int(h) for h in holds]
        if len(self.holds) != len(self.stages):
            raise ValueError(f"{len(self.holds)} holds for {len(self.stages)} stages")
        if any(h < 1 for h in self.holds):
            raise ValueError(f"every hold must be >= 1, got {self.holds}")
        self.streaks = [0] * len(self.stages)

    def step(self, flags, evaluate):
        """One control step: returns the new flags (a new list)."""
        flags = [bool(f) for f in flags]
        if len(flags) != len(self.stages):
            raise ValueError(f"{len(flags)} flags for {len(self.stages)} stages")
        for i in range(len(flags)):
            if flags[i]:
                continue
            if evaluate(i):
                self.streaks[i] += 1
                if self.streaks[i] >= self.holds[i]:
                    flags[i] = True
                    continue
            else:
                self.streaks[i] = 0
            break
        return flags
