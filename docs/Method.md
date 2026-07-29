# Method & Theory

*Theoretical background, signal model, and key equations for the SAR DDC pipeline. See [Data.md](Data.md) for data formats and normalisation, and [FPGA_inference.md](FPGA_inference.md) for FPGA deployment details.*

@TODO: To be renamed MOdel and focus on the model theory and math. So far it's only about MERLIN it should be completed.

## Method

The pipeline relies on Pytorch Ligthning on [Compressai](https://github.com/InterDigitalInc/CompressAI) [2] to implement Hyper-autoencoders solutions based on Johannes Ballé's work [3-5].
In addition, the despeckling task is inspired from MERLIN's self-supervised training pipeline [6].

### MERLIN Theory

**The big picture (mostly written by ChatGPT)**:
Dalsasso et al. introduce MERLIN, a fully self-supervised strategy for training deep despeckling networks directly on single-look complex (SLC) SAR images. By exploiting Goodman’s speckle model—which shows that the real and imaginary components of an SLC pixel are two independent, Gaussian-distributed realizations with variance proportional to the local reflectivity $r$—they train a U-Net to predict pixel-wise variance maps (i.e., the effective “blurred” reflectivity $r$) from one component (say, the real part) and evaluate the loss on the other component (the imaginary part).

Taking a SLC SAR image with Real part $\tilde{a}$ and Imaginary part $\tilde{b}$ (We use tilde notation to indicate the transformation from the SAR transfer function **H**, see MERLIN Eq.(4) and (5)). Its intensity $I$ is
$$
I = \tilde{a}^2 + \tilde{b}^2
\tag{1}
$$
MERLIN aims at depesckling, i.e., reconstructing the underlying reflectivity image $\tilde r$, with a self-supervised model $f_\theta()$ as $\frac{f_\theta(\tilde{a}^2) + f_\theta(\tilde{b}^2)}{2}$.
To achieve this, the model is trained with the loss function below, where $k$ represent the iteration over pixels:
$$
\mathcal{L}(\tilde{r}, \tilde{b}) = \sum_k \frac{log(\tilde{r}_k)}{2} + \frac{\tilde{b}_k^2}{\tilde{r}_k}
\tag{2a}
$$
In practice, it is useful to work in log-scale to reduce the dynamic range. We use the check notation to represent log-scaled variables: $\check r = log~\tilde r$, $\check a = log|\tilde a|$ and $\check b = log|\tilde b|$.
> Note: The absolute value operator |.| is only used for mathematical correctness as $\tilde a$ and $\tilde b$ can have negative values. In practice it is not needed as these values are squared before being fed to the network. As a last detail, it is also necessary to add a small $\epsilon$ to $\tilde a$ or $\tilde b$, to avoid $log(0)$. In python, this is done with `1e-6` or, preferably, `np.spacing(1)`.

Working in log-scale implies to modify the loss:
$$
\mathcal{L}(\check r, \check b) = \sum_k \frac{\check r_k}{2} + exp(2 \check b_k - \check r_k)
\tag{2b}
$$

**A few mode details**:  ==@TO UPDATE==

- **Further normalization**. In addition of the log-scale, it is beneficial to "normalize the images using a fixed affine transform".

> What this mean is using the minmax formula, but not with the minimum and maximum values of the image (because of the strong outliers, the whole distribution would end up being very narrow and the network would struggle differentiating values). Instead percentile values are used, typically 5 and 95%.

- **Misconception about the data range of the output**. As (most of) the data lies between $[0;1]$, one could expect the reconstruction of the network be in the same interval. However, the network learns to map from noisy realizations of $\tilde a \sim \mathcal{N}(0,r/2)$ to the total reflectivity $r$, not to $\frac{r}{2}$. Same for $\tilde b$.

> This means that to compare 2 images of the same scale one must visualize the Intensity $I = \tilde{a}^2 + \tilde{b}^2$ and a single prediction , e.g., $f_\theta(\tilde{a}^2)$. During inference both network estimations are averaged to decrease the variance of the reconstructed reflectivity, but if a simple proxy is needed, one could use only one of the reconstructions.

#### References

- [1] Joel Amao-Oliva, Nils Foix-Colonier, Francescopaolo Sica. (2024). Joint compression and despeckling by SAR representation learning. ISPRS Journal of Photogrammetry and Remote Sensing.
- [2] J. Bégaint, F. Racapé, S. Feltman, and A. Pushparaja, “CompressAI: a PyTorch library and evaluation platform for end-to-end compression research,” Nov. 05, 2020, arXiv: arXiv:2011.03029. doi: 10.48550/arXiv.2011.03029.
- [3] D. Minnen, J. Ballé, and G. D. Toderici, “Joint Autoregressive and Hierarchical Priors for Learned Image Compression,” in Advances in Neural Information Processing Systems, Curran Associates, Inc., 2018. Accessed: Jun. 14, 2023.
- [4] J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational image compression with a scale hyperprior,” presented at the International Conference on Learning Representations, Feb. 2018. Accessed: Apr. 10, 2024.
- [5] J. Ballé, V. Laparra, and E. P. Simoncelli, “End-to-end Optimized Image Compression,” Mar. 03, 2017, arXiv: arXiv:1611.01704. doi: 10.48550/arXiv.1611.01704.
- [6] Dalsasso, E., Denis, L., & Tupin, F. (2022). As if by magic: Self-supervised training of deep despeckling networks with MERLIN. IEEE Transactions on Geoscience and Remote Sensing, 60, 1–13. <https://doi.org/10.1109/TGRS.2021.3128621>
- [7] AMD Vitis™ AI Software. (2019). AMD. <https://www.amd.com/en/products/software/vitis-ai.html>
