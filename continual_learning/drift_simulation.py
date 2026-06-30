import torch
from torch.utils.data import TensorDataset, DataLoader


class TabularDriftSimulator:
    def __init__(self, mode="covariate_shift", severity=0.3, seed=42):
        """
        Simulates realistic data drift in federated tabular environments.

        Modes:
        - covariate_shift: P(X) changes
        - concept_drift: P(Y|X) changes
        - temporal_decay: distribution shift over time
        """
        self.mode = mode
        self.severity = float(max(0.0, min(1.0, severity)))
        torch.manual_seed(seed)

    # ============================
    # PUBLIC API
    # ============================
    def apply_drift(self, dataloader: DataLoader) -> DataLoader:
        """
        Applies drift without breaking dataset structure.
        """

        if self.severity <= 0.0:
            return dataloader

        X_list, y_list = [], []

        for x, y in dataloader:
            X_list.append(x.detach().cpu())
            y_list.append(y.detach().cpu())

        X = torch.cat(X_list, dim=0)
        y = torch.cat(y_list, dim=0)

        # Apply drift
        if self.mode == "covariate_shift":
            X = self._covariate_shift(X)

        elif self.mode == "concept_drift":
            y = self._concept_drift(y)

        elif self.mode == "temporal_decay":
            X, y = self._temporal_decay(X, y)

        else:
            raise ValueError(f"Invalid drift mode: {self.mode}")

        dataset = TensorDataset(X, y)

        return DataLoader(
            dataset,
            batch_size=dataloader.batch_size,
            shuffle=True,
            drop_last=getattr(dataloader, "drop_last", False)
        )

    # ============================
    # 1. COVARIATE SHIFT P(X)
    # ============================
#  FIXED
    def _covariate_shift(self, X: torch.Tensor) -> torch.Tensor:
        X_drifted = X.clone()
        
        # Isolate continuous variables (assuming they are indices 0 to 5)
        continuous_features = X_drifted[:, :6]
        
        noise = torch.randn_like(continuous_features) * self.severity * 0.1
        scale = 1.0 + self.severity * 0.2
        
        # Mutate continuous features while preserving one-hot categorical matrices
        X_drifted[:, :6] = (continuous_features + noise) * scale
        return X_drifted

    # ============================
    # 2. CONCEPT DRIFT P(Y|X)
    # ============================
    def _concept_drift(self, y: torch.Tensor) -> torch.Tensor:
        y = y.clone()

        flip_mask = torch.rand(y.shape[0]) < self.severity

        # safe binary assumption (Adult dataset)
        y[flip_mask] = 1 - y[flip_mask]

        return y.long()

    # ============================
    # 3. TEMPORAL DECAY
    # ============================
    def _temporal_decay(self, X: torch.Tensor, y: torch.Tensor):
        """
        Simulates aging / distribution contraction.
        """

        if len(X) == 0:
            return X, y

        # age assumed column 0 (safe for your pipeline)
        order = torch.argsort(X[:, 0])

        cutoff = int(len(order) * (self.severity * 0.5))

        cutoff = min(max(cutoff, 0), len(order) - 1)

        selected = order[cutoff:]

        return X[selected], y[selected]