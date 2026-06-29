import os
import copy
import torch


class ModelUtils:
    # ====================================
    # COUNT TRAINING PARAMETERS
    # ====================================
    @staticmethod
    def get_model_size(model: torch.nn.Module):
        """
        Calculates and tracks parameter allocations inside a network.
        Returns a dictionary containing total, trainable, and footprint sizes.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate approximate memory bandwidth footprint in Megabytes (assuming float32 = 4 bytes)
        memory_size_mb = (total_params * 4) / (1024 * 1024)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "estimated_payload_size_mb": round(memory_size_mb, 4)
        }

    # ====================================
    # DEEP CLONE MODEL PARAMETERS
    # ====================================
    @staticmethod
    def clone_parameters(source_state_dict: dict) -> dict:
        """
        Creates a clean, independent CPU-mapped memory clone of a state dictionary.
        This prevents tracking-tensor cross-contamination between client and server layers.
        """
        cloned_state = {}
        for key, tensor in source_state_dict.items():
            # Detach from computational autograd graphs and clone to host CPU memory
            cloned_state[key] = tensor.detach().clone().cpu()
        return cloned_state

    # ====================================
    # SAVE MODEL CHECKPOINT
    # ====================================
    @staticmethod
    def save_checkpoint(model: torch.nn.Module, export_path: str, current_round: int = 0, metrics: dict = None):
        """
        Saves a structured PyTorch snapshot archive to disk storage.
        """
        directory = os.path.dirname(export_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        payload = {
            "model_state_dict": model.state_dict(),
            "federated_round": current_round,
            "performance_metrics": metrics if metrics else {}
        }
        
        torch.save(payload, export_path)
        print(f"💾 Checkpoint safely exported to target location -> '{export_path}'")

    # ====================================
    # LOAD MODEL CHECKPOINT
    # ====================================
    @staticmethod
    def load_checkpoint(model: torch.nn.Module, import_path: str) -> dict:
        """
        Loads a snapshot dictionary from disk and reconstructs weights into the input model object.
        Returns the parsed metadata container (round count and metrics log).
        """
        if not os.path.exists(import_path):
            raise FileNotFoundError(f"Target checkpoint could not be discovered at location: '{import_path}'")
            
        # Map parameters explicitly to CPU to prevent out-of-memory errors on initialization
        checkpoint = torch.load(import_path, map_location="cpu")
        
        # Strip away potential Opacus '_module.' prefixes if present inside the saved artifact
        sanitized_state = {}
        for k, v in checkpoint["model_state_dict"].items():
            if k.startswith("_module."):
                sanitized_state[k.replace("_module.", "")] = v
            else:
                sanitized_state[k] = v
                
        model.load_state_dict(sanitized_state)
        print(f"✅ Model structural parameters successfully restored from -> '{import_path}'")
        
        return {
            "federated_round": checkpoint.get("federated_round", 0),
            "performance_metrics": checkpoint.get("performance_metrics", {})
        }