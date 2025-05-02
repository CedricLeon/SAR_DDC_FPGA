"""
SAR Despeckling and Data Compression (DDC) Lightning Module.

This module handles the training and testing logic for joint despeckling
and compression of SAR images using a Noise2Noise approach.
"""

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics.functional.image import structural_similarity_index_measure

from src.models.components.compression_models.sar_hyperprior import SARHyperprior


class SARDDCModule(LightningModule):
    """Lightning Module for SAR Despeckling and Data Compression.

    This module implements the training and testing logic for joint
    despeckling and compression of SAR images using a Noise2Noise approach.
    """

    def __init__(
        self,
        lambda_=0.01,
        learning_rate=1e-4,
        model_kwargs=None,
    ):
        """Initialize the Lightning Module.

        Args:
            lambda_: Rate-distortion tradeoff parameter (default: 0.01)
            learning_rate: Learning rate for optimizer (default: 1e-4)
            model_kwargs: Additional model parameters (default: None)
        """
        super().__init__()

        # Save hyperparameters to be accessible via self.hparams
        self.save_hyperparameters()

        # Initialize model with default or provided parameters
        model_params = model_kwargs or {}
        self.model = SARHyperprior(**model_params)

    def on_fit_start(self):
        """Called at the beginning of fit."""
        # Here _rng was set using thew seed but I suspect it's useless because done with lightning.seed_everything

    def calculate_bpp(self, likelihoods, input_shape):
        """Calculate bits per pixel.

        Args:
            likelihoods: Dictionary of likelihoods from model output
            input_shape: Shape of the input tensor

        Returns:
            Bits per pixel value as a tensor
        """
        num_pixels = input_shape[0] * input_shape[2] * input_shape[3]
        bpp = 0

        for likelihood in likelihoods.values():
            bpp += torch.sum(torch.log2(likelihood)) / (-num_pixels)

        return bpp

    def calculate_psnr(self, x, x_hat):
        """Calculate Peak Signal-to-Noise Ratio.

        Args:
            x: Original image
            x_hat: Reconstructed image

        Returns:
            PSNR value as a tensor
        """
        mse = torch.mean((x - x_hat) ** 2)
        if mse == 0:
            return torch.tensor(float("inf"))
        max_val = 1.0
        return 10 * torch.log10(max_val**2 / mse)

    def calculate_ssim(self, x, x_hat):
        """Calculate Structural Similarity Index Measure.

        Args:
            x: Original image
            x_hat: Reconstructed image

        Returns:
            SSIM value as a tensor
        """
        return structural_similarity_index_measure(x_hat, x)

    def training_step(self, batch, batch_idx):
        """Training step using Noise2Noise approach."""
        # Get real and imaginary parts (already squared and normalized)
        real_squared, imag_squared = batch["real"], batch["imag"]

        # Deterministic random switching of inputs/targets using seeded generator
        if torch.rand(1).item() > 0.5:
            input_data, target_data = real_squared, imag_squared
        else:
            input_data, target_data = imag_squared, real_squared

        # Forward pass
        output = self.model(input_data)
        x_hat = output["x_hat"]
        likelihoods = output["likelihoods"]

        # Calculate rate (bits per pixel)
        bpp = self.calculate_bpp(likelihoods, input_data.shape)

        # Calculate distortion (MSE between output and target)
        mse = F.mse_loss(x_hat, target_data)

        # Rate-distortion loss
        loss = self.hparams.lambda_ * mse + bpp

        # Log metrics
        self.log("train/loss", loss)
        self.log("train/mse", mse)
        self.log("train/bpp", bpp)
        self.log("train/psnr", self.calculate_psnr(target_data, x_hat))

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with optimized processing of both real and imaginary parts."""
        # Get real and imaginary parts (already squared and normalized)
        real_squared, imag_squared = batch["real"], batch["imag"]

        # Process real part
        real_output = self.model(real_squared)
        real_x_hat = real_output["x_hat"]
        real_likelihoods = real_output["likelihoods"]

        # Process imaginary part
        imag_output = self.model(imag_squared)
        imag_x_hat = imag_output["x_hat"]
        imag_likelihoods = imag_output["likelihoods"]

        # Calculate rate (bits per pixel) for both parts
        real_bpp = self.calculate_bpp(real_likelihoods, real_squared.shape)
        imag_bpp = self.calculate_bpp(imag_likelihoods, imag_squared.shape)
        total_bpp = real_bpp + imag_bpp

        # Calculate distortion (MSE)
        real_mse = F.mse_loss(
            real_x_hat, imag_squared
        )  # Cross-validation (Noise2Noise style)
        imag_mse = F.mse_loss(
            imag_x_hat, real_squared
        )  # Cross-validation (Noise2Noise style)
        avg_mse = (real_mse + imag_mse) / 2

        # Calculate validation loss
        val_loss = self.hparams.lambda_ * avg_mse + total_bpp

        # Log metrics
        self.log("val/loss", val_loss)
        self.log("val/mse", avg_mse)
        self.log("val/bpp", total_bpp)
        self.log("val/psnr", 10 * torch.log10(1.0 / avg_mse))

        return val_loss

    def test_step(self, batch, batch_idx):
        """Test step with optimized processing of both real and imaginary parts."""
        # Get real and imaginary parts (already squared and normalized)
        real_squared, imag_squared = batch["real"], batch["imag"]

        # Process real part directly through analysis transform
        real_output = self.model(real_squared, training=False)
        real_y_hat = real_output["y_hat"]
        real_x_hat = real_output["x_hat"]
        real_likelihoods = real_output["likelihoods"]

        # Process imaginary part directly through analysis transform
        imag_output = self.model(imag_squared, training=False)
        imag_y_hat = imag_output["y_hat"]
        imag_x_hat = imag_output["x_hat"]
        imag_likelihoods = imag_output["likelihoods"]

        # Calculate rate (bits per pixel)
        real_bpp = self.calculate_bpp(real_likelihoods, real_squared.shape)
        imag_bpp = self.calculate_bpp(imag_likelihoods, imag_squared.shape)
        total_bpp = real_bpp + imag_bpp

        # Average to get reflectivity estimate (despeckled result)
        reflectivity = (real_x_hat + imag_x_hat) / 2

        # WHy were these quality metrics computed on the intensity?
        # if intensity is not None:
        #     # Calculate PSNR between predicted reflectivity and original intensity
        #     psnr = self.calculate_psnr(intensity, reflectivity)
        #     self.log("test/psnr", psnr)

        #     # Calculate SSIM between predicted reflectivity and original intensity
        #     ssim = self.calculate_ssim(intensity, reflectivity)
        #     self.log("test/ssim", ssim)

        # Log rate metrics
        self.log("test/bpp", total_bpp)
        self.log("test/real_bpp", real_bpp)
        self.log("test/imag_bpp", imag_bpp)

        return {
            "reflectivity": reflectivity,
            "bpp": total_bpp,
            # "psnr": psnr if intensity is not None else None,
            # "ssim": ssim if intensity is not None else None,
        }

    def configure_optimizers(self):
        """Configure optimizers."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
