"""
Technical Context: Training vs. Inference
-----------------------------------------
1. Training (PyTorch):
   - Uses "soft" quantization (additive uniform noise) to allow gradient flow.
   - Likelihoods are estimated using continuous probability density functions (PDFs).
   - `GaussianConditional` calculates likelihoods using a continuous Gaussian distribution.

2. Inference (Deployment/DPU):
   - Uses "hard" quantization (rounding).
   - Likelihoods must be discrete probabilities (PMF) for the Entropy Coder (ANS/Range Coder).
   - `EntropyBottleneck` uses a pre-computed Cumulative Distribution Function (CDF) table.
   - `GaussianConditional` often maps the continuous scale parameter to a discrete index (via a "Scale Table")
     to lookup a pre-computed CDF for that specific scale.

The 'Hybrid' Approach:
----------------------
This file implements the **Inference** logic using pure Numpy. This allows the ARM CPU on the Zynq MPSoC
to handle the entropy coding (which requires complex control flow and table lookups) while the DPU
handles the heavy Neural Network convolutions.

Key Differences from PyTorch Checkpoints:
-----------------------------------------
- Checkpoints save *parameters* (weights, biases).
- They do NOT always save the *auxiliary tables* (CDFs, Scale Tables) needed for inference,
  unless `model.update()` was called explicitly before saving.
- This implementation expects these tables to be exported to a `.npz` file via `scripts/fpga/export_entropy_params.py`.
"""

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# NUMPY IMPLEMENTATION OF COMPRESSAI MODULES (DPU / INFERENCE ONLY)
# -----------------------------------------------------------------------------


class EntropyBottleneckDPU:
    """Numpy-only implementation of EntropyBottleneck for inference.

    Detailed Role:
    --------------
    Used for the hyper-latent `z`. Since `z` is modeled non-parametrically, we learn
    the PDF of each channel independently using a small fully connected network (in training).

    For inference, this learned PDF is baked into a static CDF Table.

    Operations:
    1. Quantization:
       Inputs are centered around a 'median' (learned parameter) before rounding.
       y_hat = round(x - median) + median

    2. Likelihood Estimation (or Bitrate estimation):
       Instead of evaluating the neural network, we look up the probability of the
       quantized symbol in the pre-computed CDF table.
       P(symbol) = CDF(symbol + 1) - CDF(symbol)
    """

    def __init__(
        self,
        channels: int,
        quantized_cdf: np.ndarray,
        cdf_length: np.ndarray,
        offset: np.ndarray,
        medians: np.ndarray,
    ):
        self.channels = channels
        self.quantized_cdf = quantized_cdf
        self.cdf_length = cdf_length
        self.offset = offset
        self.medians = medians

        # Validate dimensions
        if self.quantized_cdf.size == 0:
            raise ValueError(
                "EntropyBottleneckDPU: quantized_cdf is empty! The model was likely exported without running model.update(force=True) properly."
            )

        if self.channels != self.medians.size and self.medians.size != 1:
            raise ValueError(
                f"EntropyBottleneckDPU: Dimension mismatch! channels={self.channels} (from CDF) but medians={self.medians.size}. They should match."
            )

        # Ensure medians is 1D for easier reshaping later
        self.medians = self.medians.flatten()

    def _get_layout_and_reshape(self, inputs: np.ndarray) -> Tuple[str, Tuple[int, ...]]:
        """Determine layout (NCHW vs NHWC) and return appropriate broadcast shape."""
        # Heuristic: Check if channels dimension matches
        if inputs.ndim == 4:
            if inputs.shape[1] == self.channels and inputs.shape[3] != self.channels:
                # NCHW
                return "NCHW", (1, self.channels, 1, 1)
            elif inputs.shape[3] == self.channels:
                # NHWC (Common in DPU)
                return "NHWC", (1, 1, 1, self.channels)

        # Fallback or ambiguous (e.g. C=H=W), default to NCHW or try to guess
        if inputs.shape[1] == self.channels:
            return "NCHW", (1, self.channels, 1, 1)
        elif inputs.shape[-1] == self.channels:
            return "NHWC", (1, 1, 1, self.channels)

        raise ValueError(f"Input shape {inputs.shape} does not match channels {self.channels}")

    def _quantize(self, inputs: np.ndarray, mode: str = "dequantize") -> np.ndarray:
        """Quantize the inputs.

        Args:
            inputs: Input tensor (N, C, H, W) or (N, H, W, C)
            mode: "noise" or "dequantize". For inference, usually "dequantize" (rounding).
        """
        layout, broadcast_shape = self._get_layout_and_reshape(inputs)
        medians_b = self.medians.reshape(broadcast_shape)

        if mode == "noise":
            # We shouldn't be adding noise in inference usually, but if requested:
            noise = np.random.uniform(-0.5, 0.5, size=inputs.shape)
            return inputs + noise

        # Dequantize (Round)
        # inputs - medians_b broadcasts correctly based on determined layout
        outputs = np.round(inputs - medians_b) + medians_b
        return outputs

    def _likelihood(self, inputs: np.ndarray) -> np.ndarray:
        """Estimate likelihoods using the loaded quantized CDF."""
        layout, broadcast_shape = self._get_layout_and_reshape(inputs)

        medians_b = self.medians.reshape(broadcast_shape)
        offset_b = self.offset.reshape(broadcast_shape)

        # 1. Get Index
        indices = (inputs - medians_b + offset_b).astype(np.int32)

        # 2. Lookup CDF
        pmf = np.zeros(indices.shape, dtype=np.float32)
        cdf_lengths = self.cdf_length  # (C,)

        # Loop over channels
        for c in range(self.channels):
            length = cdf_lengths[c]

            # Access proper slice
            if layout == "NCHW":
                idx_c = indices[:, c, :, :]
            else:  # NHWC
                idx_c = indices[..., c]

            # Clamp
            idx_c = np.clip(idx_c, 0, length - 2)

            # Lookup
            cdf_row = self.quantized_cdf[c]
            prob = cdf_row[idx_c + 1] - cdf_row[idx_c]
            prob = np.maximum(prob, 1e-10)

            # SUPER WEIRD: Normalize to probabilities (assuming 16-bit precision standard in CompressAI)
            prob = prob / 65536.0

            # Assign back
            if layout == "NCHW":
                pmf[:, c, :, :] = prob
            else:
                pmf[..., c] = prob

        return pmf

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass - Quantize and Calc Likelihoods."""
        outputs = self._quantize(x)
        likelihood = self._likelihood(outputs)
        return outputs, likelihood


class GaussianConditionalDPU:
    """Numpy-only implementation of GaussianConditional for inference.

    Detailed Role:
    --------------
    Used for the latent `y`. The distribution of `y` is modeled as a Gaussian
    whose parameters (scales) are predicted by the hyper-network `h_s` from `z_hat`.

    Training vs Inference (Likelihoods):
    ------------------------------------
    - Training: We use the analytical PDF of the Gaussian distribution.
      L = product( PDF_Gaussian(y, scale) )

    - Inference (Theoretical): We calculate the probability mass under the Gaussian PDF
      integrated over the quantization bin [y-0.5, y+0.5].
      P(y) = CDF(y + 0.5) - CDF(y - 0.5)
      where CDF is based on the error function `erf`.

    - Inference (Practical / CompressAI standard):
      CompressAI usually maps the continuous `scale` to a discrete index (using a logarithmic Scale Table).
      It then effectively looks up a specific pre-computed CDF for that index.

      HOWEVER, in this minimal DPU implementation, we stick to the **Continuous Approximation**
      using `math.erf`. This avoids managing the massive CDF tables (64 scales * 65536 symbols)
      on the embedded board memory if we don't strictly need bit-perfect compatibility with the
      Entropy Coder C++ engine right away.

      We output `likelihood` which is an estimate of the probability.
      Shannon Information (-log2(likelihood)) gives us the theoretical bitrate (BPP).
    """

    def __init__(self, scale_table: Optional[np.ndarray] = None, scale_bound: float = 0.11):
        """
        Args:
            scale_table: (Optional) 1D array of valid scales used in discrete mode.
                         Not strictly used if we compute CDFs on the fly with `erf`.
            scale_bound: Minimum value for scale to avoid division by zero (default 0.11).
        """
        self.scale_table = scale_table
        self.scale_bound = scale_bound

        # Vectorized Error Function for fast numpy execution
        self._erf = np.vectorize(math.erf)

    def _standard_gaussian_cdf(self, x: np.ndarray) -> np.ndarray:
        """CDF of N(0, 1)."""
        # 0.5 * (1 + erf(x / sqrt(2)))
        return 0.5 * (1 + self._erf(x * 0.70710678))

    def _quantize(self, inputs: np.ndarray, means: Optional[np.ndarray] = None) -> np.ndarray:
        """Quantize inputs.

        y_hat = round(y - mean) + mean
        """
        if means is not None:
            outputs = np.round(inputs - means) + means
        else:
            outputs = np.round(inputs)
        return outputs

    def _likelihood(
        self, inputs: np.ndarray, scales: np.ndarray, means: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate likelihood p(y|z)."""
        # p(x) = CDF(x + 0.5) - CDF(x - 0.5)
        # where CDF is N(mean, scale).
        # CDF_N(x) = StdCDF((x - mean) / scale)

        if means is None:
            means = np.zeros_like(inputs)

        # Clamp scales
        scales = np.maximum(scales, self.scale_bound)

        # Derived inputs
        # centered = inputs - means
        # upper = (centered + 0.5) / scales
        # lower = (centered - 0.5) / scales

        # We can do this in steps to avoid massive allocations? No, numpy assumes memory.

        half = 0.5
        values = inputs - means

        upper = (values + half) / scales
        lower = (values - half) / scales

        cdf_upper = self._standard_gaussian_cdf(upper)
        cdf_lower = self._standard_gaussian_cdf(lower)

        likelihood = np.abs(cdf_upper - cdf_lower)

        # Lower bound likelihood
        likelihood = np.maximum(likelihood, 1e-10)

        return likelihood

    def forward(
        self, inputs: np.ndarray, scales: np.ndarray, means: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass - Quantize and Likelihood."""
        outputs = self._quantize(inputs, means)
        likelihood = self._likelihood(outputs, scales, means)
        return outputs, likelihood


def load_entropy_models_dpu(npz_path: Path) -> Tuple[EntropyBottleneckDPU, GaussianConditionalDPU]:
    """Load Entropy Models from an .npz parameter file."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Entropy parameters not found at {npz_path}")

    data = np.load(npz_path)

    # Check for medians
    if "eb_medians" in data:
        eb_medians = data["eb_medians"]
        # Handle 0-d array issues if any
        if eb_medians.shape == ():  # Scalar
            eb_medians = np.array([eb_medians])
    else:
        # Fallback to zeros if old export used
        print(
            "WARNING: 'eb_medians' not found in parameters. Defaulting to 0. (Re-run export for better accuracy)"
        )
        eb_medians = np.zeros(data["eb_quantized_cdf"].shape[0])

    eb = EntropyBottleneckDPU(
        channels=data["eb_quantized_cdf"].shape[0],
        quantized_cdf=data["eb_quantized_cdf"],
        cdf_length=data["eb_cdf_length"],
        offset=data["eb_offset"],
        medians=eb_medians,
    )

    # Load GaussianConditional params
    gc_scale = None
    if "gc_scale_table" in data:
        gc_scale = data["gc_scale_table"]

    gc = GaussianConditionalDPU(gc_scale)

    return eb, gc
