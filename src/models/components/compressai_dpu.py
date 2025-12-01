"""DPU-friendly variants of CompressAI operators.

This module re-implements the key operators from `context/compressai_original.py` without using
custom `torch.autograd.Function`. The goal is to keep behavior close to the original while
avoiding constructs that are problematic for Vitis-AI DPU export.

Provided classes:
  - LowerBoundPatched
  - NonNegativeParametrizerPatched
  - GDNPatched
  - EntropyModelPatched
  - EntropyBottleneckPatched
  - src/models/components/res_scale_hyperprior_dpu.py
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
from compressai.entropy_models.entropy_models import _EntropyCoder
from torch import Tensor


# -------------------------------------------------------------------------
# -------------- Original from compressai/ops/bound_ops.py ----------------
# -------------------------------------------------------------------------
class LowerBoundPatched(nn.Module):  # (no custom autograd.Function)
    """Lower bound operator, computes `torch.max(x, bound)`.

    This is a *pure* clamp with standard autograd (no custom backward). For
    inference / DPU deployment this is sufficient, because no gradients are
    used on the target.

    Note: During *training*, CompressAI's original LowerBound uses a custom
    gradient that still passes gradients when moving towards the bound. If you
    need identical training behavior, use the original implementation from
    `context/compressai_original.py` instead of this patched version.
    """

    bound: Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.tensor(float(bound)))

    def forward(self, x: Tensor) -> Tensor:
        return torch.max(x, self.bound)


# -------------------------------------------------------------------------
# -------------- Original from compressai/ops/parametrizer.py -------------
# -------------------------------------------------------------------------
class NonNegativeParametrizerPatched(nn.Module):
    """Non negative reparametrization (DPU-friendly).

    Same logic as CompressAI's NonNegativeParametrizer but using LowerBoundPatched.
    """

    pedestal: Tensor

    def __init__(self, minimum: float = 0.0, reparam_offset: float = 2**-18):
        super().__init__()
        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)

        pedestal = self.reparam_offset**2
        self.register_buffer("pedestal", torch.tensor([pedestal]))
        bound = (self.minimum + self.reparam_offset**2) ** 0.5
        self.lower_bound = LowerBoundPatched(bound)

    def init(self, x: Tensor) -> Tensor:
        """Comes directly from CompressAI."""
        return torch.sqrt(torch.max(x + self.pedestal, self.pedestal))

    def forward(self, x: Tensor) -> Tensor:
        """Comes directly from CompressAI."""
        out = self.lower_bound(x)
        out = out**2 - self.pedestal
        return out


# -------------------------------------------------------------------------
# -------------- Original from compressai/layers/gdn.py -------------------
# -------------------------------------------------------------------------
class GDNPatched(nn.Module):
    r"""Generalized Divisive Normalization layer (DPU-friendly).

    Identical math to CompressAI's GDN, but references `NonNegativeParametrizerPatched`
    and `LowerBoundPatched`.

    y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}
    """

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizerPatched(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizerPatched()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) -> Tensor:
        """Comes directly from CompressAI."""
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm
        return out


# Copilot rewrote half of the original CompressAI code, I don't trust it, so for the moment its just the original one commented out.


def default_entropy_coder():
    """Comes directly from CompressAI."""
    from compressai import get_entropy_coder

    return get_entropy_coder()


def pmf_to_quantized_cdf(pmf: Tensor, precision: int = 16) -> Tensor:
    """Comes directly from CompressAI."""
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf


def _forward(self, *args: Any) -> Any:
    """Comes directly from CompressAI."""
    raise NotImplementedError()


# -------------------------------------------------------------------------
# ------ Original from compressai/entropy_models/entropy_models.py --------
# -------------------------------------------------------------------------
class EntropyModelPatched(nn.Module):
    r"""Entropy model base class.

    Args:
        likelihood_bound (float): minimum likelihood bound
        entropy_coder (str, optional): set the entropy coder to use, use default
            one if None
        entropy_coder_precision (int): set the entropy coder precision
    """

    def __init__(
        self,
        likelihood_bound: float = 1e-9,
        entropy_coder: str | None = None,
        entropy_coder_precision: int = 16,
    ):
        super().__init__()

        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)

        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBoundPatched(likelihood_bound)

        # to be filled on update()
        self.register_buffer("_offset", torch.IntTensor())
        self.register_buffer("_quantized_cdf", torch.IntTensor())
        self.register_buffer("_cdf_length", torch.IntTensor())

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["entropy_coder"] = self.entropy_coder.name
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.entropy_coder = _EntropyCoder(self.__dict__.pop("entropy_coder"))

    @property
    def offset(self):
        """Comes directly from CompressAI."""
        return self._offset

    @property
    def quantized_cdf(self):
        """Comes directly from CompressAI."""
        return self._quantized_cdf

    @property
    def cdf_length(self):
        """Comes directly from CompressAI."""
        return self._cdf_length

    # See: https://github.com/python/mypy/issues/8795
    forward: Callable[..., Any] = _forward

    def quantize(self, inputs: Tensor, mode: str, means: Tensor | None = None) -> Tensor:
        """Comes directly from CompressAI."""
        if mode not in ("noise", "dequantize", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs

        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        outputs = torch.round(outputs)

        if mode == "dequantize":
            if means is not None:
                outputs += means
            return outputs

        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs

    def _quantize(self, inputs: Tensor, mode: str, means: Tensor | None = None) -> Tensor:
        """Comes directly from CompressAI."""
        warnings.warn("_quantize is deprecated. Use quantize instead.", stacklevel=2)
        return self.quantize(inputs, mode, means)

    @staticmethod
    def dequantize(
        inputs: Tensor, means: Tensor | None = None, dtype: torch.dtype = torch.float
    ) -> Tensor:
        """Comes directly from CompressAI."""
        if means is not None:
            outputs = inputs.type_as(means)
            outputs += means
        else:
            outputs = inputs.type(dtype)
        return outputs

    @classmethod
    def _dequantize(cls, inputs: Tensor, means: Tensor | None = None) -> Tensor:
        """Comes directly from CompressAI."""
        warnings.warn("_dequantize. Use dequantize instead.", stacklevel=2)
        return cls.dequantize(inputs, means)

    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        """Comes directly from CompressAI."""
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def _check_cdf_size(self):
        """Comes directly from CompressAI."""
        if self._quantized_cdf.numel() == 0:
            raise ValueError("Uninitialized CDFs. Run update() first")

        if len(self._quantized_cdf.size()) != 2:
            raise ValueError(f"Invalid CDF size {self._quantized_cdf.size()}")

    def _check_offsets_size(self):
        """Comes directly from CompressAI."""
        if self._offset.numel() == 0:
            raise ValueError("Uninitialized offsets. Run update() first")

        if len(self._offset.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._offset.size()}")

    def _check_cdf_length(self):
        """Comes directly from CompressAI."""
        if self._cdf_length.numel() == 0:
            raise ValueError("Uninitialized CDF lengths. Run update() first")

        if len(self._cdf_length.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._cdf_length.size()}")

    def compress(
        self,
        inputs: torch.Tensor,
        indexes: torch.Tensor,
        means: torch.Tensor | None = None,
    ):
        """Compress input tensors to char strings.

        Args:
            inputs (torch.Tensor): input tensors
            indexes (torch.Tensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """
        symbols = self.quantize(inputs, "symbols", means)

        if len(inputs.size()) < 2:
            raise ValueError(
                "Invalid `inputs` size. Expected a tensor with at least 2 dimensions."
            )

        if inputs.size() != indexes.size():
            raise ValueError("`inputs` and `indexes` should have the same size.")

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        strings = []
        for i in range(symbols.size(0)):
            rv = self.entropy_coder.encode_with_indexes(
                symbols[i].reshape(-1).int().tolist(),
                indexes[i].reshape(-1).int().tolist(),
                self._quantized_cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            strings.append(rv)
        return strings

    def decompress(
        self,
        strings: list[bytes],
        indexes: torch.Tensor,
        dtype: torch.dtype = torch.float,
        means: torch.Tensor | None = None,
    ):
        """Decompress char strings to tensors.

        Args:
            strings (list[bytes]): compressed tensors
            indexes (torch.Tensor): tensors CDF indexes
            dtype (torch.dtype): type of dequantized output
            means (torch.Tensor, optional): optional tensor means
        """

        if not isinstance(strings, (tuple, list)):
            raise ValueError("Invalid `strings` parameter type.")

        if not len(strings) == indexes.size(0):
            raise ValueError("Invalid strings or indexes parameters")

        if len(indexes.size()) < 2:
            raise ValueError(
                "Invalid `indexes` size. Expected a tensor with at least 2 dimensions."
            )

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        if means is not None:
            if means.size()[:2] != indexes.size()[:2]:
                raise ValueError("Invalid means or indexes parameters")
            if means.size() != indexes.size():
                for i in range(2, len(indexes.size())):
                    if means.size(i) != 1:
                        raise ValueError("Invalid means parameters")

        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size())

        for i, s in enumerate(strings):
            values = self.entropy_coder.decode_with_indexes(
                s,
                indexes[i].reshape(-1).int().tolist(),
                cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            outputs[i] = torch.tensor(values, device=outputs.device, dtype=outputs.dtype).reshape(
                outputs[i].size()
            )
        outputs = self.dequantize(outputs, means, dtype)
        return outputs


class EntropyBottleneck(EntropyModelPatched):
    r"""Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://github.com/tensorflow/compression/blob/v1.3/docs/entropy_bottleneck.md>`__
    for an introduction.
    """

    _offset: Tensor

    def __init__(
        self,
        channels: int,
        *args: Any,
        tail_mass: float = 1e-9,
        init_scale: float = 10,
        filters: tuple[int, ...] = (3, 3, 3, 3),
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        # Create parameters
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels

        self.matrices = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.factors = nn.ParameterList()

        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self.matrices.append(nn.Parameter(matrix))

            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.biases.append(nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.factors.append(nn.Parameter(factor))

        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)

        target = np.log(2 / self.tail_mass - 1)
        self.register_buffer("target", torch.Tensor([-target, 0, target]))

    def _get_medians(self) -> Tensor:
        """Comes directly from CompressAI."""
        medians = self.quantiles[:, :, 1:2]
        return medians

    def update(self, force: bool = False, update_quantiles: bool = False) -> bool:
        """Comes directly from CompressAI."""
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditional model is update()'d.
        if self._offset.numel() > 0 and not force:
            return False

        if update_quantiles:
            self._update_quantiles()

        medians = self.quantiles[:, 0, 1]

        minima = medians - self.quantiles[:, 0, 0]
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0)

        maxima = self.quantiles[:, 0, 2] - medians
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)

        self._offset = -minima

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1

        max_length = pmf_length.max().item()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)
        samples = samples[None, :] + pmf_start[:, None, None]

        pmf, lower, upper = self._likelihood(samples, stop_gradient=True)
        pmf = pmf[:, 0, :]
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return True

    def loss(self) -> Tensor:
        """Comes directly from CompressAI."""
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = torch.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs: Tensor, stop_gradient: bool) -> Tensor:
        """Comes directly from CompressAI."""
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = self.matrices[i]
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)

            bias = self.biases[i]
            if stop_gradient:
                bias = bias.detach()
            logits = logits + bias

            if i < len(self.filters):
                factor = self.factors[i]
                if stop_gradient:
                    factor = factor.detach()
                logits = logits + torch.tanh(factor) * torch.tanh(logits)
        return logits

    def _likelihood(
        self, inputs: Tensor, stop_gradient: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Comes directly from CompressAI."""
        half = float(0.5)
        lower = self._logits_cumulative(inputs - half, stop_gradient=stop_gradient)
        upper = self._logits_cumulative(inputs + half, stop_gradient=stop_gradient)
        likelihood = torch.sigmoid(upper) - torch.sigmoid(lower)
        return likelihood, lower, upper

    def forward(self, x: Tensor, training: bool | None = None) -> tuple[Tensor, Tensor]:
        """Comes directly from CompressAI."""
        if training is None:
            training = self.training

        if not torch.jit.is_scripting():
            # x from B x C x ... to C x B x ...
            perm = torch.cat(
                (
                    torch.tensor([1, 0], dtype=torch.long, device=x.device),
                    torch.arange(2, x.ndim, dtype=torch.long, device=x.device),
                )
            )
            inv_perm = perm
        else:
            raise NotImplementedError()
            # TorchScript in 2D for static inference
            # Convert to (channels, ... , batch) format
            # perm = (1, 2, 3, 0)
            # inv_perm = (3, 0, 1, 2)

        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)

        # Add noise or quantize

        outputs = self.quantize(values, "noise" if training else "dequantize", self._get_medians())

        if not torch.jit.is_scripting():
            likelihood, _, _ = self._likelihood(outputs)
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            raise NotImplementedError()
            # TorchScript not yet supported
            # likelihood = torch.zeros_like(outputs)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):
        """Comes directly from CompressAI."""
        dims = len(size)
        N = size[0]
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(N, 1, *size[2:])

    @staticmethod
    def _extend_ndims(tensor, n):
        """Comes directly from CompressAI."""
        return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    @torch.no_grad()
    def _update_quantiles(self, search_radius=1e5, rtol=1e-4, atol=1e-3):
        """Fast quantile update via bisection search.

        Often faster and much more precise than minimizing aux loss.
        """
        device = self.quantiles.device
        shape = (self.channels, 1, 1)
        low = torch.full(shape, -search_radius, device=device)
        high = torch.full(shape, search_radius, device=device)

        def f(y, self=self):
            return self._logits_cumulative(y, stop_gradient=True)

        for i in range(len(self.target)):
            q_i = self._search_target(f, self.target[i], low, high, rtol, atol)
            self.quantiles[:, :, i] = q_i[:, :, 0]

    @staticmethod
    def _search_target(f, target, low, high, rtol=1e-4, atol=1e-3, strict=False):
        """Comes directly from CompressAI."""
        assert (low <= high).all()
        if strict:
            assert ((f(low) <= target) & (target <= f(high))).all()
        else:
            low = torch.where(target <= f(high), low, high)
            high = torch.where(f(low) <= target, high, low)
        while not torch.isclose(low, high, rtol=rtol, atol=atol).all():
            mid = (low + high) / 2
            f_mid = f(mid)
            low = torch.where(f_mid <= target, mid, low)
            high = torch.where(f_mid >= target, mid, high)
        return (low + high) / 2

    def compress(self, x):
        """Comes directly from CompressAI."""
        indexes = self._build_indexes(x.size())
        medians = self._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self._extend_ndims(medians, spatial_dims)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        return super().compress(x, indexes, medians)

    def decompress(self, strings, size):
        """Comes directly from CompressAI."""
        output_size = (len(strings), self._quantized_cdf.size(0), *size)
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        medians = self._extend_ndims(self._get_medians().detach(), len(size))
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))
        return super().decompress(strings, indexes, medians.dtype, medians)


class GaussianConditionalPatched(EntropyModelPatched):
    r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the Gaussian conditional layer in
    *tensorflow/compression*. See the `tensorflow documentation
    <https://github.com/tensorflow/compression/blob/v1.3/docs/api_docs/python/tfc/GaussianConditional.md>`__
    for more information.
    """

    def __init__(
        self,
        scale_table: list | tuple | None,
        *args: Any,
        scale_bound: float = 0.11,
        tail_mass: float = 1e-9,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if not isinstance(scale_table, (type(None), list, tuple)):
            raise ValueError(f'Invalid type for scale_table "{type(scale_table)}"')

        if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
            raise ValueError(f'Invalid scale_table length "{len(scale_table)}"')

        if scale_table and (
            scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)
        ):
            raise ValueError(f'Invalid scale_table "({scale_table})"')

        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            scale_bound = self.scale_table[0]
        if scale_bound <= 0:
            raise ValueError("Invalid parameters")
        self.lower_bound_scale = LowerBoundPatched(scale_bound)

        self.register_buffer(
            "scale_table",
            self._prepare_scale_table(scale_table) if scale_table else torch.Tensor(),
        )

        self.register_buffer(
            "scale_bound",
            torch.Tensor([float(scale_bound)]) if scale_bound is not None else None,
        )

    @staticmethod
    def _prepare_scale_table(scale_table):
        """Comes directly from CompressAI."""
        return torch.Tensor(tuple(float(s) for s in scale_table))

    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        """Comes directly from CompressAI."""
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        """Comes directly from CompressAI."""
        return scipy.stats.norm.ppf(quantile)

    def update_scale_table(self, scale_table, force=False):
        """Comes directly from CompressAI."""
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditional model is
        # updated.
        if self._offset.numel() > 0 and not force:
            return False
        device = self.scale_table.device
        self.scale_table = self._prepare_scale_table(scale_table).to(device)
        self.update()
        return True

    def update(self):
        """Comes directly from CompressAI."""
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.abs(torch.arange(max_length, device=device).int() - pmf_center[:, None])
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def _likelihood(self, inputs: Tensor, scales: Tensor, means: Tensor | None = None) -> Tensor:
        """Comes directly from CompressAI."""
        half = float(0.5)

        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = self.lower_bound_scale(scales)

        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower

        return likelihood

    def forward(
        self,
        inputs: Tensor,
        scales: Tensor,
        means: Tensor | None = None,
        training: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Comes directly from CompressAI."""
        if training is None:
            training = self.training
        outputs = self.quantize(inputs, "noise" if training else "dequantize", means)
        likelihood = self._likelihood(outputs, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

    def build_indexes(self, scales: Tensor) -> Tensor:
        """Comes directly from CompressAI."""
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes
