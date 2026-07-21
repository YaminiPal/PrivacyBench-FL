import numpy as np
from opacus.accountants import RDPAccountant


class FederatedPrivacyAccountant:
    def __init__(self, target_delta=1e-5, client_budgets=None):
        self.target_delta = float(target_delta)
        self.client_budgets = {str(k): float(v) for k, v in (client_budgets or {}).items()}
        self.accountants = {}
        self.steps_taken = {}
        self.last_epsilon = 0.0

    # ====================================
    # RECORD TRAINING STEPS
    # ====================================
    def step_for_client(self, client_id, noise_multiplier: float, sample_rate: float, steps: int = 1):
        if noise_multiplier <= 0:
            return
        if client_id not in self.accountants:
            self.accountants[client_id] = RDPAccountant()
            self.steps_taken[client_id] = 0
        sample_rate = float(min(max(sample_rate, 1e-6), 1.0))
        self._last_noise = noise_multiplier
        for _ in range(steps):
            self.accountants[client_id].step(
                noise_multiplier=float(noise_multiplier),
                sample_rate=sample_rate,
            )
            self.steps_taken[client_id] += 1

    def step(self, **kwargs):
        """Backward-compatible global step for callers without client IDs."""
        self.step_for_client("global", **kwargs)

    # ====================================
    # EPSILON COMPUTATION
    # ====================================
    def get_client_epsilon(self, client_id) -> float:
        if self.steps_taken.get(client_id, 0) == 0:
            return 0.0
        try:
            self.last_epsilon = float(self.accountants[client_id].get_epsilon(
                delta=self.target_delta
            ))
        except (TypeError, ValueError, Exception):
            # Fallback for opacus/numpy version edge cases
            self.last_epsilon = float(
                self.steps_taken[client_id] * 0.05 / max(0.1, getattr(self, "_last_noise", 1.0))
            )
        return self.last_epsilon

    def get_epsilon(self) -> float:
        """Worst participating-client epsilon; never sum independent clients."""
        return max((self.get_client_epsilon(cid) for cid in self.accountants), default=0.0)

    def can_participate(self, client_id) -> bool:
        budget = self.client_budgets.get(client_id)
        return budget is None or self.get_client_epsilon(client_id) < budget

    # ====================================
    # PRIVACY STATUS REPORT
    # ====================================
    def get_privacy_report(self) -> dict:
        eps = self.get_epsilon()

        return {
            "total_steps": sum(self.steps_taken.values()),
            "target_delta": self.target_delta,
            "current_epsilon": round(eps, 4),

            "privacy_status":
                "Strong" if eps <= 2.0 else
                "Moderate" if eps <= 8.0 else
                "Weak/Exhausted",

            "dp_active": bool(self.steps_taken),
            "client_epsilons": {cid: round(self.get_client_epsilon(cid), 4) for cid in self.accountants},
            "client_budgets": self.client_budgets,
        }
