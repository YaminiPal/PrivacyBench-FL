import io
import zlib
import torch


class CommunicationChannel:
    def __init__(self, quantizer=None):
        """
        Manages data serialization, compression, and bandwidth tracking for FL handshakes.
        
        quantizer: Optional quantization engine (e.g., UniformQuantizer) to compress floats to 8-bit.
        """
        self.quantizer = quantizer
        
        # Bandwidth Telemetry Trackers (in Bytes)
        self.total_bytes_sent = 0
        self.total_bytes_received = 0

    # ====================================
    # PACKAGE AND COMPRESS PAYLOAD (TX)
    # ====================================
    def package_payload(self, state_dict: dict) -> bytes:
        """
        Serializes, optionally quantizes, and compresses a model state dictionary into an optimized byte stream.
        """
        buffer = io.BytesIO()
        
        # 1. Check for Quantization Layer
        if self.quantizer is not None and hasattr(self.quantizer, "quantize"):
            packaged_dict = self.quantizer.quantize(state_dict)
        else:
            # Fallback to converting standard tensors to CPU numpy matrices for clean pickling
            packaged_dict = {
                k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v 
                for k, v in state_dict.items()
            }

        # 2. Serialize to binary stream
        torch.save(packaged_dict, buffer)
        raw_bytes = buffer.getvalue()
        
        # 3. Apply lossless zlib compression to strip redundant bit spaces
        compressed_bytes = zlib.compress(raw_bytes, level=6)
        
        # Track metric egress payload size
        self.total_bytes_sent += len(compressed_bytes)
        return compressed_bytes

    # ====================================
    # DECOMPRESS AND UNPACK PAYLOAD (RX)
    # ====================================
    def unpack_payload(self, compressed_bytes: bytes) -> dict:
        """
        Decompresses, deserializes, and optionally dequantizes an incoming byte stream back into a PyTorch state dict.
        """
        self.total_bytes_received += len(compressed_bytes)
        
        # 1. Reverse lossless zlib compression
        raw_bytes = zlib.decompress(compressed_bytes)
        buffer = io.BytesIO(raw_bytes)
        
        # 2. Deserialization
        unpacked_dict = torch.load(buffer, map_location="cpu", weights_only=False)
        
        # 3. Reverse Quantization layer if active
        final_state_dict = {}
        if self.quantizer is not None and hasattr(self.quantizer, "dequantize"):
            final_state_dict = self.quantizer.dequantize(unpacked_dict)
        else:
            final_state_dict = {
                k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v 
                for k, v in unpacked_dict.items()
            }
            
        return final_state_dict

    # ====================================
    # RESET TELEMETRY COUNTERS
    # ====================================
    def reset_telemetry(self):
        """Resets the network traffic tracking metrics."""
        self.total_bytes_sent = 0
        self.total_bytes_received = 0

    def get_bandwidth_report(self) -> dict:
        """Returns data transfer metrics formatted into Kilobytes."""
        return {
            "data_sent_kb": round(self.total_bytes_sent / 1024, 2),
            "data_received_kb": round(self.total_bytes_received / 1024, 2),
            "total_traffic_mb": round((self.total_bytes_sent + self.total_bytes_received) / (1024 * 1024), 4)
        }