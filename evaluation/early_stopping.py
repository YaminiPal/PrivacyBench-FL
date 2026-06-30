import numpy as np
from utils.logger import logger


class FederatedEarlyStopper:
    def __init__(
        self,
        patience=5,
        min_delta=1e-4,
        target_accuracy=0.85,
        max_epsilon=10.0,
        accuracy_window=3
    ):
        """
        Multi-objective early stopping for Federated Learning.
        Optimized to handle noisy DP-SGD parameters seamlessly.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.target_accuracy = target_accuracy
        self.max_epsilon = max_epsilon
        self.accuracy_window = accuracy_window

        self.best_loss = np.inf
        self.patience_counter = 0

        # Stability tracking (filters out per-round DP-SGD noise spikes)
        self.accuracy_history = []
        self.total_epsilon = 0.0
        self.stop_reason = ""

    # ========================================================
    # HARMONIZED STOP CHECK
    # ========================================================
    def should_stop(self, current_loss, current_accuracy, current_epsilon):
        """
        Evaluates system termination bounds. Keyed precisely to match 
        the trainer pipeline invocation footprints.
        """
        # 🔒 Update internal absolute privacy expenditure tracker
        self.total_epsilon = current_epsilon

        # --------------------------------------------------------
        # 1. PRIVACY BUDGET EXHAUSTION
        # --------------------------------------------------------
        if self.total_epsilon >= self.max_epsilon:
            self.stop_reason = (
                f"🔒 Privacy budget exhausted: ε={self.total_epsilon:.3f} "
                f">= max limits ({self.max_epsilon})"
            )
            logger.warning(f"Early Stopping Condition Triggered -> {self.stop_reason}")
            return True

        # --------------------------------------------------------
        # 2. ACCURACY STABILITY WINDOW CHECK
        # --------------------------------------------------------
        self.accuracy_history.append(current_accuracy)
        if len(self.accuracy_history) > self.accuracy_window:
            self.accuracy_history.pop(0)

        if len(self.accuracy_history) == self.accuracy_window:
            avg_acc = np.mean(self.accuracy_history)

            if avg_acc >= self.target_accuracy:
                self.stop_reason = (
                    f"🎯 Stable convergence target hit: "
                    f"Rolling Window Avg={avg_acc*100:.2f}% >= Target={self.target_accuracy*100:.2f}%"
                )
                logger.info(f"Early Stopping Condition Triggered -> {self.stop_reason}")
                return True

        # --------------------------------------------------------
        # 3. LOSS IMPROVEMENT CONVERGENCE CHECK
        # --------------------------------------------------------
        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.stop_reason = (
                    f"🛑 Training optimization stalled: No loss improvement for "
                    f"{self.patience} rounds (Best validation loss recorded: {self.best_loss:.4f})"
                )
                logger.warning(f"Early Stopping Condition Triggered -> {self.stop_reason}")
                return True

        return False

    def get_status_report(self):
        return {
            "stop_reason": self.stop_reason,
            "best_loss": float(self.best_loss),
            "total_epsilon": float(self.total_epsilon),
            "accuracy_history": self.accuracy_history
        }