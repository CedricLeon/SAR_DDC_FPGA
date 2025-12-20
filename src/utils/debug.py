import numbers
from typing import Any

import numpy as np
import torch


def log_tensor_shape(name: str, tensor: Any) -> None:
    """Utility print for debugging tensor shapes during DPU export."""
    prefix = f"[DEBUG] {name}: "
    if isinstance(tensor, torch.Tensor):
        shape = tuple(tensor.shape)
        print(f"{prefix}torch.Tensor shape={shape} dtype={tensor.dtype} device={tensor.device}")
    elif isinstance(tensor, np.ndarray):
        print(f"{prefix}np.ndarray shape={tensor.shape} dtype={tensor.dtype}")
    elif isinstance(tensor, numbers.Number):
        print(f"{prefix}{type(tensor).__name__} value={tensor}")
    else:
        print(f"{prefix}type={type(tensor)}")
