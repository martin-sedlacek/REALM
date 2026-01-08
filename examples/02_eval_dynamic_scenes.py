import argparse
import sys
import omnigibson as og
from realm.eval import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dynamic sim evals")
    parser.add_argument('--perturbation_id', type=int, required=False, default=0)
    parser.add_argument('--task_id', type=int, required=False, default=0)
    parser.add_argument('--repeats', type=int, required=False, default=5)
    parser.add_argument('--max_steps', type=int, required=False, default=500)
    parser.add_argument('--model', type=str, required=True, default=None)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    assert args.model is not None
    evaluate(
        task_id=args.task_id,
        perturbation_id=args.perturbation_id,
        repeats=args.repeats,
        max_steps=args.max_steps,
        model_type=args.model,
        port=args.port
    )
    og.shutdown()
    sys.exit(0)
