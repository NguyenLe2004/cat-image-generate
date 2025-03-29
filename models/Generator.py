import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SelfAttention import SelfAttention

class Generator(nn.Module):
    """Generator network with skip connections and self-attention."""

    def __init__(self, latent_dim: int) -> None:
        """
        Initialize the Generator.
        
        Args:
            latent_dim: Dimension of the latent space input
        """
        super(Generator, self).__init__()
        
        # Initial upsampling block
        self.initial = self._build_upsample_block(latent_dim, 512, 4, 1, 0)
        
        # Main upsampling path
        self.upsample_blocks = nn.ModuleList([
            self._build_upsample_block(512, 256),
            self._build_upsample_block(256, 128),
            self._build_upsample_block(128, 64),
            self._build_upsample_block(64, 32)
        ])
        
        # Self-attention layer
        self.attention = SelfAttention(64)
        
        # Skip connection path
        self.skip_blocks = nn.ModuleList([
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.Conv2d(64, 32, kernel_size=1)
        ])
        
        # Final output layers
        self.final = nn.Sequential(
            nn.Conv2d(32, 3 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Tanh()
        )
        
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    @staticmethod
    def _build_upsample_block(
        in_dims: int,
        out_dims: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1
    ) -> nn.Sequential:
        """
        Create an upsampling block with transposed convolution and instance norm.
        
        Args:
            in_dims: Input channels
            out_dims: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            
        Returns:
            Sequential upsampling block
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_dims, out_dims, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generator.
        
        Args:
            x: Input latent tensor (batch_size, latent_dim, 1, 1)
            
        Returns:
            Generated image tensor
        """
        x = self.initial(x)
        
        for i, (block, skip_block) in enumerate(zip(self.upsample_blocks, self.skip_blocks)):
            # Apply self-attention at specific layer
            if i == 3:
                x = self.attention(x)
                
            # Skip connection path
            skip = F.interpolate(
                x,
                scale_factor=2,
                mode="bilinear",
                align_corners=True
            )
            skip = skip_block(skip)
            
            # Main path
            x = self.activation(block(x) + skip)
            
        return self.final(x)


class ResidualBlock(nn.Module):
    """Residual block with multiple convolutions and skip connection."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_block: int = 3
    ) -> None:
        """
        Initialize the residual block.
        
        Args:
            in_dim: Input channels
            out_dim: Output channels
            n_block: Number of convolution layers
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Main convolution path
        self.convs = nn.ModuleList([
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)
            for _ in range(n_block - 1)
        ])
        
        self.norms = nn.ModuleList([
            nn.InstanceNorm2d(in_dim)
            for _ in range(n_block - 1)
        ])

        # Final output convolution
        self.out_conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Skip connection convolution
        self.skip_conv = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after residual operations
        """
        skip = x
        
        # Main convolution path
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = norm(x)
            x = self.activation(x)
            
        x = self.out_conv(x)
        
        # Skip connection
        if self.in_dim != self.out_dim:
            skip = self.skip_conv(skip)
            
        return self.activation(x + skip)
