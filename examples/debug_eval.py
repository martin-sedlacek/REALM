from realm.eval import evaluate
import omnigibson as og
import sys

if __name__ == "__main__":
    evaluate(
        task_id=0,
        perturbation_id=0,
        repeats=1,
        max_steps=1,
        model_type="debug",
        port=0, # Not used for debug
        log_dir="/app/logs/debug",
        no_record=True,
    )
    og.shutdown()
    sys.exit(0)
