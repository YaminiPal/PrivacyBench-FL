import torch
from utils.logger import logger
from utils.helpers import to_cpu_state_dict


class UnifiedServer:
    def __init__(self, model: torch.nn.Module):
        """
        The Central Parameter Authority for the Private Federated System.
        
        Manages the canonical global model state, isolates computational histories 
        to prevent memory leaks, and handles model hydration for local updates.
        """
        self.model = model
        
        # ✅ HARDENING STRATEGY: Pin the definitive global model parameters to the CPU.
        # This keeps the server's memory footprint footprint small and prevents VRAM 
        # fragmentation when running multiple edge validation client nodes.
        self.global_state = to_cpu_state_dict(self.model.state_dict())
        
        logger.info(f"Unified Central Server initialized with architecture: {type(model).__name__}")

    # ========================================================
    # PUBLIC API: PARAMETER EXPORT (DOWNLINK HANDSHAKE)
    # ========================================================
    def get_parameters(self) -> dict:
        """
        Returns a clean, completely detached snapshot copy of the global weights.
        Prevents downstream client backpropagation from modifying the server's parameters.
        """
        # Return a deep, independent copy of the tensor dict values
        return {k: v.clone() for k, v in self.global_state.items()}

    # ========================================================
    # PUBLIC API: PARAMETER IMPORT (UPLINK HANDSHAKE)
    # ========================================================
    def set_parameters(self, new_params: dict):
        """
        Updates the global state dictionary with newly aggregated parameter updates,
        and synchronizes the server's evaluation model structure.
        """
        if not new_params:
            logger.error("Refusing server update parameter step: Target parameter payload is empty.")
            return

        # 1. Update the central CPU state registry cache
        for k, v in new_params.items():
            if k in self.global_state:
                # Ensure the incoming data matches the server's shape layout rules
                if self.global_state[k].shape == v.shape:
                    self.global_state[k] = v.detach().clone().to("cpu")
                else:
                    logger.error(f"Shape mismatch detected for layer '{k}'. Server: {self.global_state[k].shape}, Incoming: {v.shape}")
                    raise ValueError(f"Structural shape collision in layer: {k}")
            else:
                logger.warning(f"Incoming parameter key '{k}' does not exist in Server's structural skeleton.")

        # 2. Hydrate the weights back into the structural server model for evaluation passes
        # strict=False allows robustness if specific feature-embedding layers are frozen mid-run
        self.model.load_state_dict(self.global_state, strict=False)
        logger.debug("Server global state map updated and synchronized successfully.")