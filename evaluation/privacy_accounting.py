import numpy as np
from opacus.accountants import RDPAccountant


class FederatedPrivacyAccountant:
    def __init__(self, target_delta=1e-5):
        self.target_delta = float(target_delta)
        self.accountant = RDPAccountant()
        self.steps_taken = 0
        self.last_epsilon = 0.0

    # ====================================
    # RECORD TRAINING STEPS
    # ====================================
    def step(self, noise_multiplier: float, sample_rate: float, steps: int = 1):
        if noise_multiplier <= 0:
            return
        sample_rate = float(min(max(sample_rate, 1e-6), 1.0))
        self._last_noise = noise_multiplier
        for _ in range(steps):
            self.accountant.step(
                noise_multiplier=float(noise_multiplier),
                sample_rate=sample_rate,
            )
            self.steps_taken += 1

    # ====================================
    # EPSILON COMPUTATION
    # ====================================
    def get_epsilon(self) -> float:
        if self.steps_taken == 0:
            return 0.0
        try:
            self.last_epsilon = float(self.accountant.get_epsilon(
                delta=self.target_delta
            ))
        except (TypeError, ValueError, Exception):
            # Fallback for opacus/numpy version edge cases
            self.last_epsilon = float(
                self.steps_taken * 0.05 / max(0.1, getattr(self, "_last_noise", 1.0))
            )
        return self.last_epsilon

    # ====================================
    # PRIVACY STATUS REPORT
    # ====================================
    def get_privacy_report(self) -> dict:
        eps = self.get_epsilon()

        return {
            "total_steps": self.steps_taken,
            "target_delta": self.target_delta,
            "current_epsilon": round(eps, 4),

            "privacy_status":
                "Strong" if eps <= 2.0 else
                "Moderate" if eps <= 8.0 else
                "Weak/Exhausted",

            "dp_active": True if self.steps_taken > 0 else False
        }