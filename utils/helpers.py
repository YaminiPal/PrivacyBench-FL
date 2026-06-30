import os
import random
import numpy as np
import torch


# =========================================================
# SEEDING (FULL REPRODUCIBILITY ACROSS FL + PYTORCH + NUMPY)
# =========================================================
def set_seed(seed: int = 42):
    """
    Freezes random generation states across all library layers 
    to guarantee absolute numeric reproducibility across runs.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic mode (IMPORTANT for research reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# CONFIG VALIDATION (STRICT MODE - prevents silent crashes)
# =========================================================
def verify_config_keys(config: dict, required_keys: list) -> bool:
    """
    Validates the structure of configuration parameters at startup,
    preventing key-error failures deep inside multi-hour training routines.
    """
    missing = []
    for key in required_keys:
        if key not in config:
            missing.append(key)

    if missing:
        raise KeyError(
            f"❌ Config validation failed. Missing keys: {missing}"
        )

    return True


# =========================================================
# SAFE DICT ACCESS (prevents runtime KeyErrors in FL loops)
# =========================================================
def safe_get(dct, key, default=None):
    """Safely extracts configurations from dictionaries without throwing errors."""
    if dct is None:
        return default
    return dct.get(key, default)


# =========================================================
# FL-SAFE TENSOR SANITIZER (prevents GPU/CPU mismatch bugs)
# =========================================================
def to_cpu_state_dict(state_dict):
    """
    Clones model weights asynchronously and safely drops graph history 
    to prevent cross-node GPU/CPU mismatch bugs or VRAM leaks.
    """
    return {k: v.detach().clone().to("cpu", non_blocking=True) for k, v in state_dict.items()}


# =========================================================
# GRADIENT HEALTH CHECK (debugging DP-SGD instability)
# =========================================================
def check_gradient_health(model) -> str:
    """
    Audits model parameters for NaN or Infinite numerical issues.
    Essential for tuning DP clipping bounds and noise multipliers.
    """
    for name, p in model.named_parameters():
        if p.grad is not None:
            # torch.isfinite checks for both NaN and +/- Inf simultaneously
            if not torch.isfinite(p.grad).all():
                if torch.isnan(p.grad).any():
                    return f"NaN anomaly detected in layer: {name}"
                return f"Infinity anomaly detected in layer: {name}"
    return "HEALTHY"

import torch

def has_nan_gradients(model) -> bool:
    """
    Returns True if any parameter gradient contains NaN or Inf values.
    """
    for param in model.parameters():
        if param.grad is None:
            continue

        if torch.isnan(param.grad).any():
            return True

        if torch.isinf(param.grad).any():
            return True

    return False