import torch
import torch.nn as nn
import torch.nn.functional as F


class MerlinUNet(nn.Module):
    """
    Re-implementation of the U-Net network used by the MERLIN paper:
        Dalsasso, E., Denis, L., & Tupin, F. (2022). As if by magic: Self-supervised training of deep despeckling networks with MERLIN. IEEE Transactions on Geoscience and Remote Sensing, 60, 1–13. https://doi.org/10.1109/TGRS.2021.3128621
    In which they refer the reader the the architecture described in:
        Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., Karras, T., Aittala, M., & Aila, T. (2018). Noise2Noise: Learning Image Restoration without Clean Data (No. arXiv:1803.04189). arXiv. https://doi.org/10.48550/arXiv.1803.04189

    Implementation details:
    - Network weights initialized following He et al. (2015)
    - No batch norm, dropout or other regularization tehcniques
    - Optimizer ADAM (β1 = 0.9, β2 = 0.99, epsilon = 10−8)
    """

    def __init__(self, nb_in, nb_out):
        """
        Args:
            nb_in (int): Number of input channels
            nb_out (int): Number of output channels
        """
        super().__init__()

        # Encoder layers
        self.enc_conv0 = nn.Conv2d(nb_in, 48, kernel_size=3, padding=1)
        self.enc_conv1 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv2 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv3 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv4 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv5 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv6 = nn.Conv2d(48, 48, kernel_size=3, padding=1)

        # Decoder layers
        self.dec_conv5a = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.dec_conv5b = nn.Conv2d(96, 96, kernel_size=3, padding=1)

        self.dec_conv4a = nn.Conv2d(144, 96, kernel_size=3, padding=1)
        self.dec_conv4b = nn.Conv2d(96, 96, kernel_size=3, padding=1)

        self.dec_conv3a = nn.Conv2d(144, 96, kernel_size=3, padding=1)
        self.dec_conv3b = nn.Conv2d(96, 96, kernel_size=3, padding=1)

        self.dec_conv2a = nn.Conv2d(144, 96, kernel_size=3, padding=1)
        self.dec_conv2b = nn.Conv2d(96, 96, kernel_size=3, padding=1)

        self.dec_conv1a = nn.Conv2d(96 + nb_in, 64, kernel_size=3, padding=1)
        self.dec_conv1b = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec_conv1c = nn.Conv2d(32, nb_out, kernel_size=3, padding=1)

        # Leaky ReLU activation
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)  # alpha in paper

        # Initialize weights using He initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization as in the original TensorFlow
        implementation."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for all conv layers except the final one
                if m == self.dec_conv1c:
                    # Final layer uses gain=1.0
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                    nn.init.constant_(m.bias, 0)
                else:
                    # Other layers use default gain=sqrt(2)
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.1
                    )
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the U-Net architecture."""
        # Store input for final concatenation
        input_tensor = x

        # Encoder path
        x = self.leaky_relu(self.enc_conv0(x))
        x = self.leaky_relu(self.enc_conv1(x))
        pool1_out = self.pool1(x)

        x = self.leaky_relu(self.enc_conv2(pool1_out))
        pool2_out = self.pool2(x)

        x = self.leaky_relu(self.enc_conv3(pool2_out))
        pool3_out = self.pool3(x)

        x = self.leaky_relu(self.enc_conv4(pool3_out))
        pool4_out = self.pool4(x)

        x = self.leaky_relu(self.enc_conv5(pool4_out))
        x = self.pool5(x)

        x = self.leaky_relu(self.enc_conv6(x))

        # Decoder path
        # Upsample5 and concatenate with pool4
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, pool4_out], dim=1)  # 48 + 48 = 96 channels
        x = self.leaky_relu(self.dec_conv5a(x))
        x = self.leaky_relu(self.dec_conv5b(x))

        # Upsample4 and concatenate with pool3
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, pool3_out], dim=1)  # 96 + 48 = 144 channels
        x = self.leaky_relu(self.dec_conv4a(x))
        x = self.leaky_relu(self.dec_conv4b(x))

        # Upsample3 and concatenate with pool2
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, pool2_out], dim=1)  # 96 + 48 = 144 channels
        x = self.leaky_relu(self.dec_conv3a(x))
        x = self.leaky_relu(self.dec_conv3b(x))

        # Upsample2 and concatenate with pool1
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, pool1_out], dim=1)  # 96 + 48 = 144 channels
        x = self.leaky_relu(self.dec_conv2a(x))
        x = self.leaky_relu(self.dec_conv2b(x))

        # Upsample1 and concatenate with input
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, input_tensor], dim=1)  # 96 + n channels
        x = self.leaky_relu(self.dec_conv1a(x))
        x = self.leaky_relu(self.dec_conv1b(x))

        # Final output layer with linear activation (no activation function)
        x = self.dec_conv1c(x)

        return x
