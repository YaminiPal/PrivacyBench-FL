import torch
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier


class DPEngine:
    def __init__(self, noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        
        # Core underlying framework instance
        self.privacy_engine = PrivacyEngine()

    # ============================
    # ATTACH DP TO MODEL + OPTIMIZER
    # ============================
    def make_private(self, model, optimizer, data_loader):
        """
        Converts standard local client loaders and models into DP-SGD configurations.
        Handles per-sample gradient tracking, clipping boundary limits, and noise additions.
        """
        # Validate dataset volume to prevent division-by-zero errors in Opacus
        if len(data_loader.dataset) == 0:
            raise ValueError("Cannot apply DP-SGD to an empty client data loader stream.")

        # If the model is already wrapped from a previous local epoch, unwrap it first
        if hasattr(model, "_module"):
            model = model._module

        model, optimizer, data_loader = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        return model, optimizer, data_loader

    # ============================
    # PRIVACY ACCOUNTING (EPISODIC TRACKING)
    # ============================
    def get_epsilon(self):
        """
        Returns the current privacy budget (epsilon) consumed by this specific node layer.
        """
        if not self.privacy_engine.accountant.history:
            return 0.0  # Training hasn't started yet, so no budget is consumed

        try:
            return self.privacy_engine.get_epsilon(delta=self.delta)
        except Exception:
            # Fallback protection if accountant states aren't ready yet
            return -1.0

    # ============================
    # SAFE WEIGHT EXTRACTION
    # ============================
    @staticmethod
    def get_clean_state_dict(model):
        """
        Extracts clean model weights without Opacus prefix artifacts.
        Ensures compatibility with server aggregation logic.
        """
        if hasattr(model, "_module"):
            return {k: v.clone().detach().cpu() for k, v in model._module.state_dict().items()}
        return {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}

    # ============================
    # TARGET PRIVACY CALCULATOR
    # ============================
    @staticmethod
    def compute_noise_multiplier(target_epsilon, target_delta, sample_rate, epochs):
        """
        Helper function to calculate the noise multiplier required to achieve
        a target privacy budget (epsilon). Useful for populating config.yaml parameters.
        """
        return get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            epochs=epochs
        )