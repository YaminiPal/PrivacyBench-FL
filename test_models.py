import torch
import torch.nn as nn
import numpy as np
from models.logistic_regression import FederatedLogisticRegression
from models.mlp import FederatedMLP


def run_model_sanity_checks():
    print("🧪 Kicking off Model Architecture Verification Suite...\n")

    # The input feature dimension size matching your Adult dataset preprocessor
    input_dim = 108
    batch_size = 16
    output_dim = 2

    # Generate a mock batch mimicking processed client tensors (float32)
    mock_X = torch.randn(batch_size, input_dim)
    # Binary classification labels (0 or 1)
    mock_y = torch.randint(0, output_dim, (batch_size,))

    # Define standard loss function used in FLTrainer
    loss_fn = nn.CrossEntropyLoss()

    # ========================================================
    # 1. VERIFY LOGISTIC REGRESSION
    # ========================================================
    print("🔵 Testing: FederatedLogisticRegression")
    log_reg = FederatedLogisticRegression(input_dim=input_dim, output_dim=output_dim)
    
    # Check forward pass shape
    logits_lr = log_reg(mock_X)
    assert logits_lr.shape == (batch_size, output_dim), f"❌ LR Logit shape mismatch: {logits_lr.shape}"
    print("  ✅ Forward pass shapes match.")

    # Check L2 Regularization output
    l2_lr = log_reg.get_l2_penalty(lambda_reg=1e-4)
    assert isinstance(l2_lr, torch.Tensor) and l2_lr.ndim == 0, "❌ LR L2 penalty must be a scalar tensor"
    assert l2_lr.item() > 0, "❌ LR L2 penalty should be greater than zero"
    print(f"  ✅ L2 Penalty method executes correctly (Value: {l2_lr.item():.6f})")

    # Check backward pass stability
    loss_lr = loss_fn(logits_lr, mock_y) + l2_lr
    loss_lr.backward()
    assert log_reg.linear.weight.grad is not None, "❌ LR Gradients failed to backpropagate"
    print("  ✅ Backward pass gradients compute without errors.")

    # Check predict_proba
    probs_lr = log_reg.predict_proba(mock_X)
    assert torch.allclose(probs_lr.sum(dim=1), torch.ones(batch_size)), "❌ LR Probabilities must sum to 1.0"
    print("  ✅ Predict_proba normalizes values correctly.")


    print("\n--------------------------------------------------\n")


    # ========================================================
    # 2. VERIFY MULTI-LAYER PERCEPTRON (MLP)
    # ========================================================
    print("🟢 Testing: FederatedMLP (with GroupNorm)")
    mlp = FederatedMLP(input_dim=input_dim, hidden_dims=[64, 32], output_dim=output_dim, use_normalization=True)

    # Check forward pass shape
    logits_mlp = mlp(mock_X)
    assert logits_mlp.shape == (batch_size, output_dim), f"❌ MLP Logit shape mismatch: {logits_mlp.shape}"
    print("  ✅ Forward pass shapes match.")

    # Check L2 Regularization output
    l2_mlp = mlp.get_l2_penalty(lambda_reg=1e-4)
    assert isinstance(l2_mlp, torch.Tensor) and l2_mlp.ndim == 0, "❌ MLP L2 penalty must be a scalar tensor"
    assert l2_mlp.item() > 0, "❌ MLP L2 penalty should be greater than zero"
    print(f"  ✅ L2 Penalty method executes correctly (Value: {l2_mlp.item():.6f})")

    # Check backward pass stability
    loss_mlp = loss_fn(logits_mlp, mock_y) + l2_mlp
    loss_mlp.backward()
    
    # Verify gradient flow through the first linear layer in Sequential block
    first_layer = mlp.network[0]
    assert first_layer.weight.grad is not None, "❌ MLP Gradients failed to flow back to input layers"
    print("  ✅ Backward pass gradients compute without errors.")

    # Check predict_proba
    probs_mlp = mlp.predict_proba(mock_X)
    assert torch.allclose(probs_mlp.sum(dim=1), torch.ones(batch_size)), "❌ MLP Probabilities must sum to 1.0"
    print("  ✅ Predict_proba normalizes values correctly.")

    print("\n🎉 All core model structural pipelines passed verification successfully!")


if __name__ == "__main__":
    run_model_sanity_checks()