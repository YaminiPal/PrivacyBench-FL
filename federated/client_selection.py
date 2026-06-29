import numpy as np


class ClientSelector:
    def __init__(self, fraction=1.0, strategy="random", seed=42):
        self.fraction = max(0.0, min(1.0, fraction))
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)
        self.client_scores = {}
        self.client_noise_levels = {}

    def update_client_metrics(self, client_id, update_norm, noise_level=0.0):
        """Track per-client update quality and privacy noise for privacy-aware selection."""
        self.client_scores[client_id] = float(update_norm)
        self.client_noise_levels[client_id] = float(noise_level)

    def select(self, client_dict, train_loaders, client_metrics=None):
        all_client_ids = list(client_dict.keys())
        total_clients = len(all_client_ids)

        if self.fraction <= 0:
            return []

        num_to_select = int(np.ceil(self.fraction * total_clients))
        num_to_select = min(num_to_select, total_clients)

        if num_to_select == total_clients:
            return all_client_ids

        if self.strategy == "random":
            return self.rng.choice(
                all_client_ids, size=num_to_select, replace=False
            ).tolist()

        if self.strategy == "utility_proportional":
            volumes = []
            for cid in all_client_ids:
                loader = train_loaders.get(cid)
                volumes.append(len(loader.dataset) if loader else 0)
            probabilities = np.array(volumes, dtype=np.float64) + 1e-8
            probabilities /= probabilities.sum()
            return self.rng.choice(
                all_client_ids, size=num_to_select, replace=False, p=probabilities
            ).tolist()

        if self.strategy == "privacy_aware":
            scores = []
            for cid in all_client_ids:
                loader = train_loaders.get(cid)
                data_score = len(loader.dataset) if loader else 1
                update_norm = self.client_scores.get(cid, 1.0)
                noise = self.client_noise_levels.get(cid, 0.0)
                quality = 1.0 / (1.0 + abs(update_norm - 1.0))
                privacy_penalty = 1.0 / (1.0 + noise)
                scores.append(data_score * quality * privacy_penalty)
            scores = np.array(scores, dtype=np.float64)
            if client_metrics:
                for i, cid in enumerate(all_client_ids):
                    if cid in client_metrics:
                        scores[i] *= client_metrics[cid].get("quality", 1.0)
            scores = scores + 1e-8
            probabilities = scores / scores.sum()
            return self.rng.choice(
                all_client_ids, size=num_to_select, replace=False, p=probabilities
            ).tolist()

        raise ValueError(f"Unknown strategy: {self.strategy}")
