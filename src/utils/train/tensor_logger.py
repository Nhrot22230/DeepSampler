from typing import Optional

import torch


class TensorLogger:
    """Utility class for logging tensor shapes during model execution.

    Args:
        debug: Whether to enable debug logging
        prefix: Optional prefix for log messages
    """

    def __init__(self, debug: bool = False, prefix: Optional[str] = None):
        self.debug = debug
        self.prefix = prefix + ": " if prefix else ""

    def log(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Log tensor shape and optionally other statistics.

        Args:
            name: Name identifier for the tensor
            tensor: The tensor to log information about

        Returns:
            The input tensor (for chaining operations)
        """
        if self.debug:
            print(f"{self.prefix}{name} shape: {tensor.shape}")
        return tensor
