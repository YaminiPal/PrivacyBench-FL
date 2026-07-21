"""Privacy-preserving, local feature-distribution drift detector."""
import torch


class FeatureDriftDetector:
    """Detects covariate drift using only aggregate feature moments per client."""

    def __init__(self, z_threshold=2.5, min_samples=32):
        self.z_threshold = float(z_threshold)
        self.min_samples = int(min_samples)
        self.baselines = {}

    @staticmethod
    def _moments(loader):
        features = [x.detach().float().cpu() for x, _ in loader]
        if not features:
            return None
        X = torch.cat(features, dim=0)
        return X.mean(dim=0), X.std(dim=0, unbiased=False).clamp_min(1e-6), len(X)

    def observe(self, client_id, loader):
        moments = self._moments(loader)
        if moments is None:
            return {"detected": False, "score": 0.0, "reason": "empty loader"}
        mean, std, n = moments
        if client_id not in self.baselines:
            self.baselines[client_id] = (mean, std)
            return {"detected": False, "score": 0.0, "reason": "baseline established"}
        reference_mean, reference_std = self.baselines[client_id]
        score = torch.mean(torch.abs(mean - reference_mean) / reference_std).item()
        detected = n >= self.min_samples and score >= self.z_threshold
        return {"detected": detected, "score": score, "reason": "feature-moment shift"}
