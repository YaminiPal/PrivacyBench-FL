import numpy as np
import torch


class QuantizationEngine:
    def __init__(self, bits=8):
        """
        Compresses/Decompresses weight updates to minimize communication overhead.
        
        bits: Target bit-width for quantization (typically 8).
        """
        if bits not in [4, 8, 16]:
            raise ValueError("Quantization currently only supports 4, 8, or 16-bit compression scales.")
        
        self.bits = bits
        # Calculate maximum integer boundary for symmetric distribution
        self.qmin = -(2 ** (bits - 1))
        self.qmax = (2 ** (bits - 1)) - 1

    # ============================
    # COMPRESS (FLOAT32 -> INT8)
    # ============================
    def quantize(self, update_dict):
        """
        Compresses float32 tensors into low-bit integers alongside structural scale vectors.
        """
        quantized_update = {}

        for key, tensor in update_dict.items():
            # If the matrix is a NumPy array from communication layers, convert it to a PyTorch tensor
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor)
            
            tensor = tensor.to(torch.float32)
            max_val = torch.max(torch.abs(tensor)).item()

            # Handle edge case: empty layer or zeroed gradient updates
            if max_val == 0:
                quantized_update[key] = {
                    "quantized": tensor.to(torch.int8).numpy(),
                    "scale": 0.0
                }
                continue

            # Calculate uniform mapping scale factor
            scale = max_val / self.qmax

            # Scale, round, and clamp weights into integer boundaries
            q_tensor = torch.round(tensor / scale)
            q_tensor = torch.clamp(q_tensor, self.qmin, self.qmax)

            quantized_update[key] = {
                "quantized": q_tensor.to(torch.int8).numpy(),
                "scale": scale
            }

        return quantized_update

    # ============================
    # DECOMPRESS (INT8 -> FLOAT32)
    # ============================
    def dequantize(self, quantized_update):
        """
        Reconstructs compressed integer payloads back into float32 space tensors.
        """
        dequantized_update = {}

        for key, payload in quantized_update.items():
            q_array = payload["quantized"]
            scale = payload["scale"]

            # Convert payload container back to float format
            q_tensor = torch.tensor(q_array, dtype=torch.float32)
            
            # Project integers back into the real number scale domain
            dequantized_update[key] = q_tensor * scale

        return dequantized_update