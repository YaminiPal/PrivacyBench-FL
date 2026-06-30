import numpy as np


class FederatedForgettingAnalyticEngine:
    def __init__(self):
        """
        Tracks catastrophic forgetting in federated continual learning setups.
        performance_matrix[t][k] = performance at round t on task k
        """
        self.performance_matrix = []

    # ========================================================
    # RECORDING
    # ========================================================
    def record_eval_snapshot(self, round_scores: list):
        """
        round_scores: performance on all known distributions/tasks at a given round
        """
        self.performance_matrix.append(np.array(round_scores, dtype=np.float32))

    # ========================================================
    # ABSOLUTE FORGETTING
    # ========================================================
    def compute_absolute_forgetting(self, task_idx=0) -> float:
        """
        Max historical performance drop for a given task.
        """

        if len(self.performance_matrix) < 2:
            return 0.0

        timeline = np.array([row[task_idx] for row in self.performance_matrix])

        peak = np.max(timeline)
        current = timeline[-1]

        return float(max(0.0, peak - current))

    # ========================================================
    # BACKWARD TRANSFER (BWT)
    # ========================================================
    def compute_backward_transfer(self) -> float:
        """
        Standard continual learning BWT definition:

        BWT = average over tasks i<current_round:
              (performance_now_on_task_i - performance_when_task_learned)
        """

        if len(self.performance_matrix) < 2:
            return 0.0

        R = np.array(self.performance_matrix, dtype=np.float32)
        T, K = R.shape

        if T < 2 or K < 2:
            return 0.0

        bwt_sum = 0.0
        count = 0

        for t in range(1, T):
            for k in range(min(t, K)):
                bwt_sum += (R[t][k] - R[k][k])
                count += 1

        return float(bwt_sum / count) if count > 0 else 0.0

    # ========================================================
    # STABILITY REPORT
    # ========================================================
    def generate_stability_report(self) -> dict:
        """
        Produces FL continual learning diagnostics.
        """

        if not self.performance_matrix:
            return {"status": "No evaluation snapshots recorded."}

        abs_forgetting = self.compute_absolute_forgetting(task_idx=0)
        bwt = self.compute_backward_transfer()

        # Stability classification (cleaner thresholds)
        if bwt >= 0:
            stability = "Excellent (Positive Transfer)"
        elif bwt > -0.05:
            stability = "Stable (Minor Forgetting)"
        elif bwt > -0.15:
            stability = "Degrading (Noticeable Forgetting)"
        else:
            stability = "Critical (Catastrophic Forgetting)"

        return {
            "absolute_forgetting": abs_forgetting,
            "backward_transfer": bwt,
            "stability": stability,
            "num_snapshots": len(self.performance_matrix)
        }

    # ========================================================
    # PRINT SUMMARY
    # ========================================================
    def print_text_summary(self):
        report = self.generate_stability_report()

        print("\n🧠 CONTINUAL LEARNING REPORT")
        print("=" * 60)

        if "status" in report:
            print(report["status"])
            return

        print(f"📉 Absolute Forgetting: {report['absolute_forgetting']:.4f}")
        print(f"🔄 Backward Transfer:   {report['backward_transfer']:.4f}")
        print(f"🧩 Stability:           {report['stability']}")
        print(f"📊 Snapshots:           {report['num_snapshots']}")

        print("=" * 60)