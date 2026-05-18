from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.models.components.res_scale_hyperprior_dpu import (
    ResidualScaleHyperpriorPatched,
)
from src.utils.debug import log_tensor_shape


class ResidualScaleHyperpriorDPUWrapper(nn.Module):
    """Unified Wrapper for Vitis-AI Calibration and Deployment.

    This class handles two distinct modes of operation: Calibration (full graph) and Deployment (split graph).
    It is necessary to implement both modes in the same graph because Vitis-AI's quantization tool (VAI_Q) expects consistent
    node names throughout the deployment process.

    Mode 1: Calibration/Evaluation (Single Input)
    - Input: x (image)
    - Behavior: Runs full model logic (g_a -> h_a -> entropy(CPU) -> g_s).
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

        self.DEBUG_MODE = True
        self.i_batch = 0

        # 2. Hidden Submodules (Invisible to VAI_Q)
        # We need these for Calibration Mode to calculate correct data flow.
        if hasattr(original_model, "entropy_bottleneck"):
            self._entropy_bottleneck = [original_model.entropy_bottleneck]
            self._gaussian_conditional = [original_model.gaussian_conditional]
        elif hasattr(
            original_model, "_entropy_bottleneck"
        ):  # If already wrapped, reuse the existing hidden lists
            self._entropy_bottleneck = original_model._entropy_bottleneck
            self._gaussian_conditional = original_model._gaussian_conditional
        else:
            raise AttributeError(
                "Input model must have 'entropy_bottleneck' or '_entropy_bottleneck'."
            )

    def manual_abs_operaion(self, x: Tensor) -> Tensor:
        """Manual Absolute Operation to replace torch.abs for Vitis-AI compatibility."""
        return torch.sqrt(x * x + 1e-6)  # Adding a small epsilon for numerical stability

    def forward_mode1_calibration(self, x_in: Tensor) -> Dict[str, Tensor]:
        """Forward pass for Calibration Mode (Full Graph)."""
        print(f"Calibration Mode Forward Pass, batch {self.i_batch}")
        # Enforce 2-channel input for logic consistency
        x_real = x_in[:, :1, :, :]
        x_imag = x_in[:, 1:, :, :]

        # Analysis
        y_real = self.g_a(x_real)
        y_imag = self.g_a(x_imag)
        y = torch.cat((y_real, y_imag), dim=1)

        # Hyperprior
        z = self.manual_abs_operaion(
            y
        )  # z = torch.abs(y) # `torch.abs()` is flagged as a CPU layer by Vitis-AI
        z = self.h_a(z)
        # z_hat, z_likelihoods = self._entropy_bottleneck[0](z)
        z_likelihoods = torch.ones_like(z)
        z_hat = z  # Trace logic: Direct pass-through.
        scales = self.h_s(z_hat)

        # y_hat, y_likelihoods = self._gaussian_conditional[0](y, scales)
        y_likelihoods = torch.ones_like(y)
        y_hat = y  # Trace logic: Direct pass-through.

        # Split
        y_hat_real = y_hat[:, : y_hat.shape[1] // 2, :, :]
        y_hat_imag = y_hat[:, y_hat.shape[1] // 2 :, :, :]

        # Synthesis
        x_hat_real = self.g_s(y_hat_real)
        x_hat_imag = self.g_s(y_hat_imag)
        x_hat = torch.cat((x_hat_real, x_hat_imag), dim=1)

        if self.DEBUG_MODE:  # and self.i_batch == 10:
            # self.i_batch = 0
            log_tensor_shape("Input x", x_in)
            log_tensor_shape("g_a.y", y)
            log_tensor_shape("h_a.z", z)
            log_tensor_shape("GC.z_likelihoods", z_likelihoods)
            log_tensor_shape("h_s.scales", scales)
            log_tensor_shape("EB.y_likelihoods", y_likelihoods)
            log_tensor_shape("g_s.y_hat_real", y_hat_real)
            log_tensor_shape("Output x_hat", x_hat)

        self.i_batch += 1
        results_dict = {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
        return results_dict

    def forward_mode2_deployment(
        self, x_in: Tensor, abs_y_in: Tensor, z_hat_in: Tensor, y_hat_in: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass for Deployment Mode (Split Graph)."""
        # 1. Main Encoder (g_a)
        y_out = self.g_a(x_in)
        # 2. Hyper Encoder (h_a)
        z_out = self.h_a(abs_y_in)
        # 3. Hyper Decoder (h_s)
        scales_out = self.h_s(z_hat_in)
        # 4. Main Decoder (g_s)
        x_out = self.g_s(y_hat_in)

        return y_out, z_out, scales_out, x_out

    def forward(
        self,
        x_in: Tensor,
        abs_y_in: Optional[Tensor] = None,
        z_hat_in: Optional[Tensor] = None,
        y_hat_in: Optional[Tensor] = None,
    ) -> Union[Dict[str, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Forward pass for the wrapper."""
        # --- Mode 2: Deployment (Split Graph) ---
        # Triggered when all 4 inputs are provided
        if abs_y_in is not None and z_hat_in is not None and y_hat_in is not None:
            return self.forward_mode2_deployment(x_in, abs_y_in, z_hat_in, y_hat_in)

        # --- Mode 1: Calibration (Full Graph) ---
        # Triggered when only x_in is provided
        return self.forward_mode1_calibration(x_in)


# ---------------------------------------------------------------------------
# DPU Wrapper for FactorizedPriorPatched (2-subgraph: g_a + g_s only)
# ---------------------------------------------------------------------------
class FactorizedPriorDPUWrapper(nn.Module):
    """Unified Wrapper for Vitis-AI Calibration and Deployment — FactorizedPrior variant.

    Mirrors ``ResidualScaleHyperpriorDPUWrapper`` but for the simpler 2-subgraph topology:
    there is no hyperprior path (no h_a, h_s) and no GaussianConditional.

    Mode 1: Calibration/Evaluation (Single Input)
    - Input: x [B, 2, H, W] (real + imag)
    - Behavior: g_a → EB passthrough → split → g_s (real + imag) → cat
    - Purpose: Generate valid activation statistics for VAI_Q. EB is bypassed
      (direct pass-through, same pattern as ResidualScaleHyperpriorDPUWrapper)
      so VAI_Q never sees the entropy layers.

    Mode 2: Deployment/Split (Two Inputs)
    - Inputs: x_in [B, 1, H, W], y_hat_in [B, M, H', W']
    - Behavior: g_a(x_in) and g_s(y_hat_in) run as independent subgraphs.
    - Purpose: Export to XIR as 2 separate DPU subgraphs.
    """

    def __init__(self, original_model: nn.Module) -> None:
        super().__init__()
        # 1. Registered Submodules (Visible to VAI_Q)
        # Node names: 'FactorizedPriorDPUWrapper/Sequential[g_a]/...'
        self.g_a = original_model.g_a
        self.g_s = original_model.g_s

        self.DEBUG_MODE = True
        self.i_batch = 0

        # 2. Hidden Submodules (Invisible to VAI_Q)
        # Needed to give the calibration forward access to the EB weights for future
        # entropy parameter export, even though EB itself is not traced by VAI_Q.
        if hasattr(original_model, "entropy_bottleneck"):
            self._entropy_bottleneck = [original_model.entropy_bottleneck]
        elif hasattr(original_model, "_entropy_bottleneck"):
            self._entropy_bottleneck = original_model._entropy_bottleneck
        else:
            raise AttributeError(
                "Input model must have 'entropy_bottleneck' or '_entropy_bottleneck'."
            )

    def forward_mode1_calibration(self, x_in: Tensor) -> Dict[str, object]:
        """Forward pass for Calibration Mode (Full Graph, single input)."""
        print(f"[FactorizedPriorDPUWrapper] Calibration Mode Forward Pass, batch {self.i_batch}")
        x_real = x_in[:, :1, :, :]
        x_imag = x_in[:, 1:, :, :]

        # Analysis
        y_real = self.g_a(x_real)
        y_imag = self.g_a(x_imag)
        y = torch.cat((y_real, y_imag), dim=1)

        # Entropy Bottleneck — direct pass-through so VAI_Q never traces it.
        # The real EB compress/decompress happens on the CPU at inference time.
        y_likelihoods = torch.ones_like(y)
        y_hat = y  # Trace logic: direct pass-through

        # Split real/imag back out
        y_hat_real = y_hat[:, : y_hat.shape[1] // 2, :, :]
        y_hat_imag = y_hat[:, y_hat.shape[1] // 2 :, :, :]

        # Synthesis
        x_hat_real = self.g_s(y_hat_real)
        x_hat_imag = self.g_s(y_hat_imag)
        x_hat = torch.cat((x_hat_real, x_hat_imag), dim=1)

        if self.DEBUG_MODE:
            log_tensor_shape("[FP Wrapper] Input x", x_in)
            log_tensor_shape("[FP Wrapper] g_a.y", y)
            log_tensor_shape("[FP Wrapper] EB.y_likelihoods (passthrough)", y_likelihoods)
            log_tensor_shape("[FP Wrapper] g_s.y_hat_real", y_hat_real)
            log_tensor_shape("[FP Wrapper] Output x_hat", x_hat)

        self.i_batch += 1
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods},
        }

    def forward_mode2_deployment(self, x_in: Tensor, y_hat_in: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass for Deployment Mode (Split Graph, two inputs).

        Args:
            x_in: Single-channel input slice [B, 1, H, W] fed to g_a.
            y_hat_in: Decoded latent [B, N, H', W'] fed to g_s (N = nb_channels_main).
        Returns:
            Tuple of (y_out, x_out): encoder output and decoder output.
        """
        # 1. Main Encoder (g_a)  — subgraph 1
        y_out = self.g_a(x_in)
        # 2. Main Decoder (g_s)  — subgraph 2
        x_out = self.g_s(y_hat_in)
        return y_out, x_out

    def forward(
        self,
        x_in: Tensor,
        y_hat_in: Optional[Tensor] = None,
    ) -> Union[Dict[str, object], Tuple[Tensor, Tensor]]:
        """Dispatch between calibration and deployment modes.

        Args:
            x_in: Image input [B, 2, H, W] (calibration) or [B, 1, H, W] (deployment g_a slice).
            y_hat_in: If provided, triggers deployment mode. Latent for g_s [B, N, H', W'].
        """
        if y_hat_in is not None:
            return self.forward_mode2_deployment(x_in, y_hat_in)
        return self.forward_mode1_calibration(x_in)
