import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


class FederatedMetricsEngine:
    def __init__(self):
        """
        Comprehensive global and localized tracking engine for Federated Learning telemetry.
        """
        self.history = {
            "round": [],
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "auc_roc": [],
            "client_variance": []  # Tracks statistical performance spread across client nodes
        }

    # ====================================
    # COMPUTE GLOBAL METRICS
    # ====================================
    def compute_global_metrics(self, round_idx: int, all_logits: list, all_targets: list, current_loss: float, client_losses: list = None) -> dict:
        """
        Compiles structural classification metrics over aggregated validation streams.
        
        all_logits: List of raw output tensors from the evaluation pass.
        all_targets: List of actual true label integers.
        current_loss: Average aggregated loss for the round.
        client_losses: Optional list of individual client losses to compute heterogeneity metrics.
        """
        # Convert lists of tensors cleanly into unified numpy arrays
        logits = torch.cat(all_logits, dim=0).cpu().numpy()
        targets = torch.cat(all_targets, dim=0).cpu().numpy()
        
        # Extract class probabilities and structural predictions
        # Softmax converts raw logits into probabilities; class 1 is chosen for binary AUC-ROC
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        pos_probs = probs[:, 1] if probs.shape[1] > 1 else probs.flatten()

        # Calculate robust performance statistics
        acc = accuracy_score(targets, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average="binary", zero_division=0)
        
        try:
            auc = roc_auc_score(targets, pos_probs)
        except ValueError:
            auc = 0.5  # Fallback if a small batch accidentally contains only a single class

        # Calculate variance across client nodes to observe the impact of non-IID data distribution
        variance = np.var(client_losses) if client_losses and len(client_losses) > 0 else 0.0

        # Update core structural metrics dictionary history
        self.history["round"].append(round_idx)
        self.history["loss"].append(current_loss)
        self.history["accuracy"].append(acc)
        self.history["precision"].append(precision)
        self.history["recall"].append(recall)
        self.history["f1_score"].append(f1)
        self.history["auc_roc"].append(auc)
        self.history["client_variance"].append(variance)

        return {
            "round": round_idx,
            "loss": round(current_loss, 4),
            "accuracy": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "auc_roc": round(auc, 4),
            "client_variance": round(variance, 6)
        }

    # ====================================
    # EXPORT METRICS HISTORY
    # ====================================
    def get_history(self) -> dict:
        """Returns the full tracked runtime metrics log."""
        return self.history

    def print_summary_table(self, round_stats: dict):
        """Prints a clean status report to the console for live telemetry tracking."""
        print("\n" + "="*50)
        print(f"📊 SYSTEM TELEMETRY REPORT - ROUND {round_stats['round']}")
        print("="*50)
        print(f" 📉 Global Loss:       {round_stats['loss']:.4f}")
        print(f" 🎯 Global Accuracy:   {round_stats['accuracy']*100:.2f}%")
        print(f" 🔬 Precision / Recall: {round_stats['precision']:.4f} / {round_stats['recall']:.4f}")
        print(f" 🧪 F1-Score / ROC-AUC: {round_stats['f1_score']:.4f} / {round_stats['auc_roc']:.4f}")
        print(f" ⚖️  Client Variance:   {round_stats['client_variance']:.6f}")
        print("="*50 + "\n")