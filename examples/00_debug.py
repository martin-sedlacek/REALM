import sys
import omnigibson as og
from realm.eval import evaluate


if __name__ == "__main__":
    evaluate(
        task_id=1,
        perturbation_id=0,
        repeats=1,
        max_steps=15,
        model_type="debug",
        port=8000
    )
    og.shutdown()
    sys.exit(0)
