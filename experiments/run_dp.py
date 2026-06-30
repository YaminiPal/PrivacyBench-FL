"""DP-enabled MLP experiment (uses Opacus via main pipeline)."""
import os
import sys
import matplotlib.pyplot as plt

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from experiments.experiment_utils import run_with_overrides
from utils.logger import logger


def run_private_experiment():
    logger.info("Running DP-enabled MLP experiment...")
    os.makedirs("results", exist_ok=True)

    results = run_with_overrides({
        "data": {"split_mode": "non_iid", "force_repartition": True},
        "dp": {"enabled": True, "noise_multiplier": 1.2, "max_epsilon": 6.0},
        "quantization": {"enabled": True, "bits": 8},
        "selection_strategy": "privacy_aware",
        "experiment": {"compare_models": ["mlp"]},
        "rounds": 8,
    })

    history = results["mlp"]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(history["round"], history["accuracy"], color="tab:blue", marker="o", label="Accuracy")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(history["round"], history["epsilon"], color="tab:red", linestyle="--", marker="s", label="ε")
    ax2.set_ylabel("Privacy ε", color="tab:red")
    plt.title("Privacy-Utility Trade-off")
    fig.tight_layout()
    chart_out = "results/private_dp_curve.png"
    plt.savefig(chart_out, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Final accuracy: {history['accuracy'][-1]*100:.2f}% | ε: {history['epsilon'][-1]:.4f}")
    logger.info(f"Chart saved -> {chart_out}")


if __name__ == "__main__":
    run_private_experiment()
