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


def print_statistics(name: str, tensor: Any) -> None:
    """Print basic statistics of a tensor or numpy array for debugging."""
    if isinstance(tensor, torch.Tensor):
        print(
            f"{name:<40}: min={torch.min(tensor):.6f}, max={torch.max(tensor):.6f}, mean={torch.mean(tensor):.6f}, std={torch.std(tensor):.6f}"
        )
    elif isinstance(tensor, np.ndarray):
        print(
            f"{name:<40}: min={np.min(tensor):.6f}, max={np.max(tensor):.6f}, mean={np.mean(tensor):.6f}, std={np.std(tensor):.6f}"
        )
    else:
        print(f"{name:<40}: Unsupported type {type(tensor)} for statistics printing.")
