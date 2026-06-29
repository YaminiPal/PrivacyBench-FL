import torch
import torch.nn as nn


class FederatedLogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)

        # Xavier initialization (stable convergence in FL)
        nn.init.xavier_uniform_(self.linear.weight)

        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    # ============================
    # FORWARD PASS
    # ============================
    def forward(self, x):
        x = x.to(torch.float32)
        return self.linear(x)

    # ============================
    # L2 REGULARIZATION LOSS PENALTY
    # ============================
    def get_l2_penalty(self, lambda_reg=1e-4):
        """
        Calculates the explicit L2 regularization penalty for the model weights.
        Excludes the bias vector from regularization, matching industry best practices.
        """
        # Sum of squared errors for weights: 0.5 * lambda * ||W||_2^2
        l2_reg = torch.tensor(0.0, device=self.linear.weight.device)
        l2_reg += torch.sum(self.linear.weight ** 2)
        return 0.5 * lambda_reg * l2_reg

    # ============================
    # PROBABILITY OUTPUT (EVAL ONLY)
    # ============================
    def predict_proba(self, x):
        was_training = self.training
        self.eval()

        device = next(self.parameters()).device
        x = x.to(device).to(torch.float32)

        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)

        self.train(was_training)
        return probs