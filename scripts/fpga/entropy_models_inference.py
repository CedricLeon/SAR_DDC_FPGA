from typing import List, Optional, Tuple

import numpy as np

try:
    # Try importing the compiled C++ extension
    import ans  # type: ignore
except ImportError:
    ans = None
    print("Warning: 'ans' module (C++ rANS) not found. Real bitstream generation will fail.")
    exit(1)


class EntropyBottleneck:
    """Real implementation of EntropyBottleneck using C++ rANS coder.

    Expects tables loaded from .npz file exported by `export_entropy_params.py`.
    """

    def __init__(
        self,
        channels: int,
        quantized_cdf: np.ndarray,
        cdf_length: np.ndarray,
        offset: np.ndarray,
        medians: Optional[np.ndarray] = None,
    ):
        self.channels = channels
        # Ensure tables are contiguous int32 for C++
        self.quantized_cdf = np.ascontiguousarray(quantized_cdf, dtype=np.int32)
        self.cdf_length = np.ascontiguousarray(cdf_length, dtype=np.int32)
        self.offset = np.ascontiguousarray(offset, dtype=np.int32)
        self.medians = (
            np.ascontiguousarray(medians.flatten(), dtype=np.float32)
            if medians is not None
            else np.zeros(channels, dtype=np.float32)
        )

        self.encoder = ans.RansEncoder()  # pyright: ignore[reportOptionalMemberAccess]
        self.decoder = ans.RansDecoder()  # pyright: ignore[reportOptionalMemberAccess]

    def compress(self, inputs: np.ndarray) -> List[bytes]:
        """Compresses latent `z`.

        Args:
            inputs: Latent `z` (N, C, H, W) or (N, H, W, C).
        Returns:
            List of byte strings (one per batch item).
        """
        # 1. Handle Layout (Default to NHWC for DPU, but check)
        if inputs.ndim == 4 and inputs.shape[3] == self.channels:
            # NHWC -> NCHW? No, we work with flattened data usually.
            pass
        elif inputs.ndim == 4 and inputs.shape[1] == self.channels:
            # NCHW -> NHWC
            inputs = inputs.transpose(0, 2, 3, 1)

        # inputs is now (N, H, W, C)
        N, H, W, C = inputs.shape
        assert C == self.channels

        # Quantize symbols: round(inputs - medians)
        # We assume discrete symbols for entropy coding
        # Note: In CompressAI, the symbols passed to RansEncoder are integers derived from (x - mean).
        # The offset is handled inside the encoder C++ (value - offset).

        medians_broad = self.medians.reshape(1, 1, 1, C)
        symbols = np.round(inputs - medians_broad).astype(np.int32)

        # We need a list of indexes. For EB, the index is simply the channel index.
        # We need to construct a vector of indexes matching the flattened symbols.
        # Create a channel index map: [0, 1, 2, ... C-1] repeated H*W times.

        # (H, W, C)
        indexes = np.tile(np.arange(C, dtype=np.int32), (H, W, 1))

        strings = []
        for i in range(N):
            # Process one image at a time
            sym = symbols[i].flatten()
            idx = indexes.flatten()

            # Encode
            cdfs_list = self.quantized_cdf.tolist()

            encoded_bytes = self.encoder.encode_with_indexes(
                sym.tolist(),
                idx.tolist(),
                cdfs_list,
                self.cdf_length.tolist(),
                self.offset.tolist(),
            )
            strings.append(encoded_bytes)

        return strings

    def decompress(self, strings: List[bytes], shape: Tuple[int, int]) -> np.ndarray:
        """Decompress latent `z`.

        Args:
            strings: List of byte strings.
            shape: (H, W) spatial dimensions to reconstruct.
        """
        H, W = shape
        C = self.channels

        # Reconstruct indexes
        indexes = np.tile(np.arange(C, dtype=np.int32), (H, W, 1)).flatten().tolist()

        cdfs_list = self.quantized_cdf.tolist()
        cdf_lengths = self.cdf_length.tolist()
        offsets = self.offset.tolist()

        outputs = []
        for s in strings:
            decoded_syms = self.decoder.decode_with_indexes(
                s, indexes, cdfs_list, cdf_lengths, offsets
            )
            data = np.array(decoded_syms, dtype=np.float32).reshape(H, W, C)

            # Add medians back: inputs = symbols + medians
            data = data + self.medians.reshape(1, 1, C)
            outputs.append(data)

        return np.stack(outputs)  # (N, H, W, C)


class GaussianConditional:
    """Real implementation of GaussianConditional using C++ RANS coder.

    Requires:
    1. Scale Table (mapping scale_idx -> sigma)
    2. Quantized CDFs for each scale index.
    """

    def __init__(
        self,
        scale_table: np.ndarray,
        quantized_cdf: np.ndarray,
        cdf_length: np.ndarray,
        offset: np.ndarray,
    ):
        self.scale_table = np.ascontiguousarray(scale_table.flatten(), dtype=np.float32)
        self.quantized_cdf = np.ascontiguousarray(quantized_cdf, dtype=np.int32)
        self.cdf_length = np.ascontiguousarray(cdf_length, dtype=np.int32)
        self.offset = np.ascontiguousarray(offset, dtype=np.int32)

        self.encoder = ans.RansEncoder()  # pyright: ignore[reportOptionalMemberAccess]
        self.decoder = ans.RansDecoder()  # pyright: ignore[reportOptionalMemberAccess]

    def _get_scale_indexes(self, scales: np.ndarray) -> np.ndarray:
        """Map continuous scales to discrete indexes using the scale table.

        This is a nearest-neighbor search or lower-bound search. CompressAI typically uses
        LowerBound ops. Here we use numpy searchsorted.
        """
        # searchsorted finds indices where elements should be inserted to maintain order.
        # scale_table is sorted.
        # We clamp the result to valid range [0, len(scale_table)-1]
        indexes = np.searchsorted(self.scale_table, scales, side="right") - 1
        indexes = np.clip(indexes, 0, len(self.scale_table) - 1)
        return indexes.astype(np.int32)

    def compress(
        self, inputs: np.ndarray, scales: np.ndarray, means: Optional[np.ndarray] = None
    ) -> List[bytes]:
        """Compress latent `y`.

        Args:
            inputs: Latent y (N, H, W, C)
            scales: Sigma (N, H, W, C) estimated by Hyper-decoder.
            means: mean (N, H, W, C) estimated by Hyper-decoder.
        """
        if means is not None:
            inputs = inputs - means

        # Quantize symbols
        symbols = np.round(inputs).astype(np.int32)

        # Get indexes from scales
        indexes = self._get_scale_indexes(scales)

        N = inputs.shape[0]
        strings = []

        # Preparing lists for C++ (optimization: do this once if constant, but here it depends on input)
        cdfs_list = self.quantized_cdf.tolist()
        cdf_lengths = self.cdf_length.tolist()
        offsets = self.offset.tolist()

        for i in range(N):
            sym_flat = symbols[i].flatten().tolist()
            idx_flat = indexes[i].flatten().tolist()

            encoded_bytes = self.encoder.encode_with_indexes(
                sym_flat, idx_flat, cdfs_list, cdf_lengths, offsets
            )
            strings.append(encoded_bytes)

        return strings

    def decompress(
        self, strings: List[bytes], scales: np.ndarray, means: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Decompress latent `y`."""
        N, H, W, C = scales.shape
        indexes = self._get_scale_indexes(scales)

        cdfs_list = self.quantized_cdf.tolist()
        cdf_lengths = self.cdf_length.tolist()
        offsets = self.offset.tolist()

        outputs = []
        for i, s in enumerate(strings):
            idx_flat = indexes[i].flatten().tolist()
            decoded_syms = self.decoder.decode_with_indexes(
                s, idx_flat, cdfs_list, cdf_lengths, offsets
            )
            out_array = np.array(decoded_syms, dtype=np.float32).reshape(H, W, C)
            outputs.append(out_array)

        outputs = np.stack(outputs)

        if means is not None:
            outputs = outputs + means

        return outputs
