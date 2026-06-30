"""Systematic ablation study using the shared pipeline."""
import os
import sys
import matplotlib.pyplot as plt

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from experiments.experiment_utils import run_with_overrides
from utils.logger import logger


def run_ablation_suite():
    logger.info("Running ablation study matrix...")
    os.makedirs("results", exist_ok=True)

    ablation_matrix = {
        "Ablation_A_Baseline": {
            "dp": {"enabled": False},
            "quantization": {"enabled": False},
            "lambda_reg": 1e-4,
        },
        "Ablation_B_Heavy_L2": {
            "dp": {"enabled": False},
            "quantization": {"enabled": False},
            "lambda_reg": 1e-2,
        },
        "Ablation_C_DP_Low_Noise": {
            "dp": {"enabled": True, "noise_multiplier": 0.8},
            "quantization": {"enabled": False},
            "lambda_reg": 1e-4,
        },
        "Ablation_D_DP_High_Noise": {
            "dp": {"enabled": True, "noise_multiplier": 2.5},
            "quantization": {"enabled": True, "bits": 8},
            "lambda_reg": 1e-4,
        },
    }

    shared = {
        "data": {"split_mode": "non_iid", "force_repartition": True},
        "drift": {"enabled": False},
        "experiment": {"compare_models": ["mlp"]},
        "rounds": 6,
    }

    ablation_results = {}
    for name, patch in ablation_matrix.items():
        logger.info(f"Running {name}...")
        overrides = {**shared}
        for k, v in patch.items():
            if isinstance(v, dict) and k in overrides and isinstance(overrides[k], dict):
                overrides[k] = {**overrides[k], **v}
            else:
                overrides[k] = v
        results = run_with_overrides(overrides)
        history = results["mlp"]
        ablation_results[name] = {
            "rounds": history["round"],
            "accuracy": history["accuracy"],
            "final_accuracy": history["accuracy"][-1],
            "traffic_mb": history["total_traffic_mb"][-1],
        }

    plt.figure(figsize=(10, 6))
    for name, data in ablation_results.items():
        plt.plot(
            data["rounds"], data["accuracy"], marker="o",
            label=f"{name} ({data['final_accuracy']*100:.1f}%)",
        )
    plt.title("Ablation: Accuracy vs Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plot_path = "results/ablation_performance_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Ablation chart saved -> {plot_path}")
    for name, data in ablation_results.items():
        logger.info(f"  {name}: acc={data['final_accuracy']*100:.2f}% traffic={data['traffic_mb']:.2f}MB")


if __name__ == "__main__":
    run_ablation_suite()
