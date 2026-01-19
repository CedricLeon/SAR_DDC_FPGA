from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.models.components.res_scale_hyperprior_dpu import (
    ResidualScaleHyperpriorPatched,
)


class ResidualScaleHyperpriorDPUWrapper(nn.Module):
    """Unified Wrapper for Vitis-AI Calibration and Deployment.

    This class handles two distinct modes of operation to ensure node name consistency
    between Calibration (full graph) and Deployment (split graph).

    Mode 1: Calibration/Evaluation (Single Input)
    - Input: x (image)
    - Behavior: Runs full model logic (g_a -> entropy(CPU) -> g_s).
    - Purpose: Generate valid statistics for quantization. Neural layers are exposed,
      Entropy layers are hidden (so VAI_Q doesn't trace them).

    Mode 2: Deployment/Split (Multiple Inputs)
    - Inputs: x, abs_y, z_hat, y_hat
    - Behavior: Runs disjoint subgraphs.
    - Purpose: Export to XIR as 4 separate subgraphs for DPU.
    """

    def __init__(self, original_model: Union[ResidualScaleHyperpriorPatched, nn.Module]):
        super().__init__()
        # 1. Registered Submodules (Visible to VAI_Q)
        # These will be quantized. Node names will be 'ResidualScaleHyperpriorDPUWrapper/Sequential[g_a]/...'
        self.g_a = original_model.g_a
        self.h_a = original_model.h_a
        self.h_s = original_model.h_s
        self.g_s = original_model.g_s

        # 2. Hidden Submodules (Invisible to VAI_Q)
        # We need these for Calibration Mode to calculate correct data flow.
        if hasattr(original_model, "entropy_bottleneck"):
            self._entropy_bottleneck = [original_model.entropy_bottleneck]
            self._gaussian_conditional = [original_model.gaussian_conditional]
        elif hasattr(original_model, "_entropy_bottleneck"):
            # If already wrapped, reuse the existing hidden lists
            self._entropy_bottleneck = original_model._entropy_bottleneck
            self._gaussian_conditional = original_model._gaussian_conditional
        else:
            raise AttributeError(
                "Input model must have 'entropy_bottleneck' or '_entropy_bottleneck'."
            )

    def forward(
        self,
        x_in: Tensor,
        abs_y_in: Tensor = None,
        z_hat_in: Tensor = None,
        y_hat_in: Tensor = None,
    ) -> Union[Dict[str, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Forward pass for the wrapper."""
        # --- Mode 2: Deployment (Split Graph) ---
        # Triggered when all 4 inputs are provided
        if abs_y_in is not None and z_hat_in is not None and y_hat_in is not None:
            # 1. Main Encoder (g_a)
            y_out = self.g_a(x_in)
            # 2. Hyper Encoder (h_a)
            z_out = self.h_a(abs_y_in)
            # 3. Hyper Decoder (h_s)
            scales_out = self.h_s(z_hat_in)
            # 4. Main Decoder (g_s)
            x_out = self.g_s(y_hat_in)

            return y_out, z_out, scales_out, x_out

        # --- Mode 1: Calibration (Full Graph) ---
        # Triggered when only x_in is provided

        # Retrieve hidden modules (Unused in DPU Calibration Trace to avoid Unsupported Ops)
        # entropy_bottleneck = self._entropy_bottleneck[0]
        # gaussian_conditional = self._gaussian_conditional[0]

        # Enforce 2-channel input for logic consistency
        x_real = x_in[:, :1, :, :]
        x_imag = x_in[:, 1:, :, :]

        # Analysis
        y_real = self.g_a(x_real)
        y_imag = self.g_a(x_imag)
        y = torch.cat((y_real, y_imag), dim=1)

        # Hyperprior
        # We use torch.abs locally. Vitis-AI will flag this as CPU layer, which is fine.
        # It creates a graph: DPU(g_a) -> CPU(abs) -> DPU(h_a)
        z = torch.abs(y)

        z = self.h_a(z)

        # BYPASS Complex Entropy Logic for Vitis-AI Trace
        # Real logic: z_hat, z_likelihoods = entropy_bottleneck(z)
        # Trace logic: Direct pass-through.
        # We lose the quantization noise (z_hat ~ z), but this is acceptable for
        # calibrating the activation ranges of the subsequent layers.
        z_hat = z

        scales = self.h_s(z_hat)

        # Entropy coding
        # Real logic: y_hat, y_likelihoods = gaussian_conditional(y, scales)
        # Trace logic: Direct pass-through.
        y_hat = y

        # Split
        y_hat_real = y_hat[:, : y_hat.shape[1] // 2, :, :]
        y_hat_imag = y_hat[:, y_hat.shape[1] // 2 :, :, :]

        # Synthesis
        x_hat_real = self.g_s(y_hat_real)
        x_hat_imag = self.g_s(y_hat_imag)
        x_hat = torch.cat((x_hat_real, x_hat_imag), dim=1)

        # Return Dictionary to match original model output structure for Loss calculation
        # We return dummy likelihoods to prevent downstream code from crashing if it checks keys.
        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "likelihoods": {"y": torch.ones_like(y_hat), "z": torch.ones_like(z_hat)},
        }


# Helper alias to keep model_quant.py happy if it imports this name
ResidualScaleHyperpriorCalibrationWrapper = ResidualScaleHyperpriorDPUWrapper
