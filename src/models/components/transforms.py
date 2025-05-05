"""
SAR Preprocessing Transforms.

This module contains the specialized transformations for SAR image processing.
"""

import torch


class SARPreprocessTransform:
    """Transform that implements the SAR preprocessing chain."""

    def __init__(self, patch_size=256, epsilon=1e-10, seed=42):
        """Initialize the transform.

        Args:
            patch_size: Size of image patches (default: 256)
            epsilon: Small constant for numerical stability (default: 1e-10)
            seed: Random seed for deterministic behavior (default: 42)
        """
        self.patch_size = patch_size
        self.epsilon = epsilon

        # For deterministic behavior in operations that need randomness
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

        print("Should not use this transform, unupdated after HDF5 preprocessing.")

    def __call__(self, sample):
        """Apply the transform to a sample."""
        # Extract real and imaginary parts
        real_part = sample["real"]
        imag_part = sample["imag"]

        # 1. Symmetrization (spectrum centering)
        real_part, imag_part = self._symmetrize(real_part, imag_part)

        # 2. Point-like scatterer preservation (9dB threshold)
        real_part, imag_part = self._preserve_scatterers(real_part, imag_part)

        # 3. Log transformation and normalization
        real_part_norm = self._log_normalize(real_part)
        imag_part_norm = self._log_normalize(imag_part)

        # Calculate squared versions for the noise2noise training
        real_squared = real_part**2
        imag_squared = imag_part**2

        # Also normalize the squared versions
        real_squared_norm = self._log_normalize(real_squared)
        imag_squared_norm = self._log_normalize(imag_squared)

        # Update sample with processed data
        updated_sample = {
            "real": real_part_norm,
            "imag": imag_part_norm,
            "real_squared": real_squared_norm,
            "imag_squared": imag_squared_norm,
            "intensity": sample.get("intensity", None),  # Pass through if available
            "file_path": sample.get("file_path", None),  # Pass through if available
        }

        return updated_sample

    def _symmetrize(self, real, imag):
        """Apply symmetrization to ensure real and imaginary parts independence."""
        # Import here to avoid circular imports
        from src.utils.sar_utils import symetrisation_patch

        # Reshape for compatibility with the symetrisation_patch function
        if len(real.shape) == 3:  # Single image with channel dimension
            real_reshaped = real.permute(1, 2, 0)  # [H, W, C]
            imag_reshaped = imag.permute(1, 2, 0)  # [H, W, C]

            # Apply symmetrization (zero Doppler centering)
            real_sym, imag_sym = symetrisation_patch(real_reshaped, imag_reshaped)

            # Reshape back to original dimensions
            real_out = real_sym.permute(2, 0, 1)  # [C, H, W]
            imag_out = imag_sym.permute(2, 0, 1)  # [C, H, W]
            return real_out, imag_out
        elif len(real.shape) == 4:  # Batch of images
            # Handle batched inputs
            batch_size = real.shape[0]
            real_list, imag_list = [], []

            for i in range(batch_size):
                r = real[i].permute(1, 2, 0)  # [H, W, C]
                im = imag[i].permute(1, 2, 0)  # [H, W, C]

                r_sym, im_sym = symetrisation_patch(r, im)

                real_list.append(r_sym.permute(2, 0, 1))  # [C, H, W]
                imag_list.append(im_sym.permute(2, 0, 1))  # [C, H, W]

            return torch.stack(real_list), torch.stack(imag_list)
        else:
            raise ValueError(f"Unexpected shape for real/imag parts: {real.shape}")

    def _preserve_scatterers(self, real, imag):
        """Preserve point-like scatterers with power greater than 9dB threshold."""
        # Calculate power
        power = real**2 + imag**2

        # 9dB threshold in linear scale
        threshold = 10 ** (9 / 10)
        mask = power > threshold

        # Set equal values for real and imag parts (half the power each)
        half_sqrt_power = torch.sqrt(power[mask] / 2)

        # Create new tensors to avoid in-place operations
        real_out = real.clone()
        imag_out = imag.clone()

        # Apply the transform
        real_out[mask] = half_sqrt_power
        imag_out[mask] = half_sqrt_power

        return real_out, imag_out

    def _log_normalize(self, x):
        """Apply log transform and normalize to reduce dynamic range."""
        # Log transformation with epsilon for numerical stability
        x_log = torch.log(x + self.epsilon)

        # Normalize to [0, 1] range
        x_min = x_log.min()
        x_max = x_log.max()
        x_norm = (x_log - x_min) / (x_max - x_min + self.epsilon)

        return x_norm
