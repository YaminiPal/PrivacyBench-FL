import torch


class AdaptiveClipper:
    def __init__(self, initial_target_clip=1.0, learning_rate=0.2, target_quantile=0.5):
        """
        Dynamically scales down exploding user updates based on running norm statistics.
        
        initial_target_clip: Starting clipping bound threshold (C).
        learning_rate: Tuning step speed for updating the clipping threshold.
        target_quantile: The fraction of unclipped updates we want to maintain (0.5 = median).
        """
        self.clip_bound = float(initial_target_clip)
        self.lr = learning_rate
        self.target_quantile = target_quantile

    # ====================================
    # COMPUTE VECTOR NORM
    # ====================================
    def _compute_total_norm(self, update_dict):
        """
        Calculates the global Frobenius norm across all layer matrices combined.
        """
        total_sq_sum = 0.0
        for val in update_dict.values():
            if isinstance(val, torch.Tensor):
                total_sq_sum += torch.sum(val ** 2).item()
            else:
                total_sq_sum += float(torch.sum(torch.tensor(val) ** 2))
        return torch.sqrt(torch.tensor(total_sq_sum)).item()

    # ====================================
    # APPLY CLIPPING TO AN UPDATE
    # ====================================
    def clip_update(self, update_dict):
        """
        Clips a single update dictionary in-place if its norm exceeds the current bound.
        Returns the clipped update and a boolean flag indicating if it was clipped.
        """
        total_norm = self._compute_total_norm(update_dict)
        
        # Determine the scaling factor: min(1, C / ||delta_W||)
        clip_coef = self.clip_bound / (total_norm + 1e-6)
        
        clipped_update = {}
        was_clipped = False

        if clip_coef < 1.0:
            was_clipped = True
            for key, tensor in update_dict.items():
                clipped_update[key] = tensor * clip_coef
        else:
            clipped_update = {k: v.clone() for k, v in update_dict.items()}

        return clipped_update, was_clipped

    # ====================================
    # DYNAMICALLY ADJUST THE THRESHOLD (C)
    # ====================================
    def update_clipping_bound(self, fraction_clippedThisRound):
        """
        Adjusts the value of C using an exponential gradient descent step 
        based on how many clients were clipped this round.
        """
        # Multiplicative gradient adjustment step equation
        grad = fraction_clippedThisRound - self.target_quantile
        self.clip_bound = self.clip_bound * torch.exp(torch.tensor(-self.lr * grad)).item()
        
        # Enforce stability floor boundaries
        self.clip_bound = max(1e-4, self.clip_bound)
        return self.clip_bound