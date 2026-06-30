import os
import numpy as np
import matplotlib.pyplot as plt


class FederatedVisualizer:
    def __init__(self, output_dir="results/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ==========================================================
    # Privacy (ε) vs Utility (Accuracy)
    # ==========================================================
    def plot_privacy_utility_curve(self, history, model_name="MLP"):
        if not history:
            return

        rounds = history.get("round", [])
        eps = history.get("epsilon", [])
        acc = history.get("accuracy", [])

        if len(rounds) == 0:
            return

        plt.figure(figsize=(8, 5))

        plt.plot(rounds, eps, label="Privacy Budget (ε)")
        plt.plot(rounds, acc, label="Accuracy")

        plt.xlabel("Communication Round")
        plt.ylabel("Metric Value")
        plt.title(f"{model_name}: Privacy vs Utility")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                f"{model_name.lower()}_privacy_utility.png",
            )
        )
        plt.close()

    # ==========================================================
    # Continual Learning Forgetting Matrix
    # ==========================================================
    def plot_forgetting_matrix(self, performance_matrix):
        if performance_matrix is None or len(performance_matrix) == 0:
            return

        matrix = np.array(performance_matrix)

        plt.figure(figsize=(6, 5))

        plt.imshow(matrix, aspect="auto")
        plt.colorbar(label="Accuracy")

        plt.xlabel("Distribution / Task")
        plt.ylabel("Training Round")
        plt.title("Catastrophic Forgetting Matrix")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                "forgetting_matrix.png",
            )
        )
        plt.close()

    # ==========================================================
    # LR vs MLP Comparison Dashboard
    # ==========================================================
    def plot_system_comparison_dashboard(self, results):
        if "lr" not in results or "mlp" not in results:
            return

        lr = results["lr"]
        mlp = results["mlp"]

        lr_rounds = lr.get("round", [])
        mlp_rounds = mlp.get("round", [])

        plt.figure(figsize=(8, 5))

        plt.plot(
            lr_rounds,
            lr.get("accuracy", []),
            label="Logistic Regression",
        )

        plt.plot(
            mlp_rounds,
            mlp.get("accuracy", []),
            label="MLP",
        )

        plt.xlabel("Communication Round")
        plt.ylabel("Accuracy")
        plt.title("Federated Accuracy Comparison")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                "accuracy_comparison.png",
            )
        )
        plt.close()

        plt.figure(figsize=(8, 5))

        plt.plot(
            lr_rounds,
            lr.get("loss", []),
            label="Logistic Regression",
        )

        plt.plot(
            mlp_rounds,
            mlp.get("loss", []),
            label="MLP",
        )

        plt.xlabel("Communication Round")
        plt.ylabel("Loss")
        plt.title("Federated Loss Comparison")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                "loss_comparison.png",
            )
        )
        plt.close()

        plt.figure(figsize=(8, 5))

        plt.plot(
            lr_rounds,
            lr.get("total_traffic_mb", []),
            label="Logistic Regression",
        )

        plt.plot(
            mlp_rounds,
            mlp.get("total_traffic_mb", []),
            label="MLP",
        )

        plt.xlabel("Communication Round")
        plt.ylabel("Traffic (MB)")
        plt.title("Communication Cost Comparison")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                "bandwidth_comparison.png",
            )
        )
        plt.close()