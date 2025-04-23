# Math and theoritical explanations
*This file intends to keep track and explain the main data transformation of the repository. I will use it for my own understanding as well as to keep consistent naming conventions.*

## SAR preprocessing

### Normalize SAR Image
This function logarithmically normalizes SAR image data to a standard range.
[normalize_sar()](src/utils/sar_utils.py)

$$\frac{\log(I^2 + 10^{-12}) - 2m}{2(M - m)} \tag{1}$$

Where:

$I$ is the input SAR image
$M$ and $m$ are constants that define the normalization range
The logarithm is applied to the squared magnitude with a small epsilon ($10^{-12}$) to avoid log of zero

### Denormalize SAR Image
This function converts normalized SAR data back to its original dynamic range.
[denormalize_sar()](src/utils/sar_utils.py)

$$\exp((M - m) \cdot \text{clip}(I_{norm}, 0, 1) + m) \tag{2}$$

Where:

$I_{norm}$ is the normalized image
$\text{clip}(I_{norm}, 0, 1)$ ensures values stay in the range $[0,1]$
The exponential function reverses the logarithm applied during normalization