import torch
import torch.nn as nn


class FederatedMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=[64, 32],
        output_dim=2,
        dropout_rate=0.2,
        use_normalization=True,
        num_groups=4,
        personalized_head=False,
    ):
        """
        DP-SGD compatible Federated MLP optimized for:
        - non-IID data
        - differential privacy (Opacus-safe)
        - federated optimization stability
        """

        super().__init__()

        layers = []
        current_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))

            # ============================
            # NORMALIZATION (FL + DP SAFE)
            # ============================
            if use_normalization:
                # GroupNorm requires divisible structure
                if h_dim % num_groups == 0:
                    layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=h_dim))
                else:
                    # fallback for stability in edge cases
                    layers.append(nn.LayerNorm(h_dim))

            layers.append(nn.ReLU())

            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))

            current_dim = h_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.personalized_head = personalized_head

        self._initialize_weights()

    # ============================
    # WEIGHT INITIALIZATION
    # ============================
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ============================
    # FORWARD PASS
    # ============================
    def forward(self, x):
        # Safety check (important in FL with corrupted clients)
        if torch.isnan(x).any():
            raise ValueError("NaN detected in input features")

        x = x.to(torch.float32)
        return self.network(x)

    # ============================
    # L2 REGULARIZATION (FedProx-ready)
    # ============================
    def get_l2_penalty(self, lambda_reg=1e-4):
        """
        Used for:
        - L2 regularization
        - FedProx-style proximal updates (optional extension)
        """
        device = next(self.parameters()).device
        l2 = torch.tensor(0.0, device=device)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                l2 += torch.sum(module.weight ** 2)

        return 0.5 * lambda_reg * l2

    # ============================
    # PROBABILITY OUTPUT (SAFE)
    # ============================
    def predict_proba(self, x):
        device = next(self.parameters()).device
        x = x.to(device).to(torch.float32)

        with torch.inference_mode():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)

        return probs

    def personalized_parameter_keys(self):
        """Keep only the classifier head local for personalized FL."""
        if not self.personalized_head:
            return []
        for name, module in reversed(list(self.network.named_children())):
            if isinstance(module, nn.Linear):
                return [f"network.{name}.weight", f"network.{name}.bias"]
        return []
