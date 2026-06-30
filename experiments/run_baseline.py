"""Non-private LR baseline experiment (uses real Adult data via main pipeline)."""
import os
import sys
import matplotlib.pyplot as plt

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from experiments.experiment_utils import run_with_overrides
from utils.logger import logger


def run_baseline_experiment():
    logger.info("Running non-private Logistic Regression baseline...")
    os.makedirs("results", exist_ok=True)

    results = run_with_overrides({
        "data": {"split_mode": "non_iid", "force_repartition": True},
        "dp": {"enabled": False},
        "quantization": {"enabled": False},
        "drift": {"enabled": False},
        "selection_strategy": "utility_proportional",
        "experiment": {"compare_models": ["lr"]},
        "rounds": 8,
    })

    history = results["lr"]
    plt.figure(figsize=(8, 5))
    plt.plot(history["round"], history["accuracy"], marker="x", label="LR Baseline")
    plt.title("Baseline: Accuracy vs Communication Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    chart_out = "results/baseline_learning_curve.png"
    plt.savefig(chart_out, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Final accuracy: {history['accuracy'][-1]*100:.2f}%")
    logger.info(f"Chart saved -> {chart_out}")


if __name__ == "__main__":
    run_baseline_experiment()
