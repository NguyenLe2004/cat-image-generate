import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    """Discriminator network with spectral normalization and skip connections."""

    def __init__(self) -> None:
        """Initialize the Discriminator."""
        super(Discriminator, self).__init__()
        
        # Main convolution path
        self.conv_layers = nn.ModuleList([
            spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),  
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),  
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), 
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)), 
            spectral_norm(nn.Conv2d(512, 1024, 4, 2, 1)),
            spectral_norm(nn.Conv2d(1024, 1, 4, 1, 0)) 
        ])
        
        # Skip connection path
        self.skip_layers = nn.ModuleList([
            spectral_norm(nn.Conv2d(64, 128, 1, 1, 0)),
            spectral_norm(nn.Conv2d(128, 256, 1, 1, 0)),
            spectral_norm(nn.Conv2d(256, 512, 1, 1, 0)),
            spectral_norm(nn.Conv2d(512, 1024, 1, 1, 0))
        ])
        
        # Pooling layer for skip connections
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator.
        
        Args:
            x: Input image tensor
            
        Returns:
            Discriminator output logits
        """
        # Initial layer
        x = self.activation(self.conv_layers[0](x))
        features = [x]
        
        # Process through remaining layers with skip connections
        for i in range(1, 5):
            x = self.activation(self.conv_layers[i](features[-1]))
            
            # Add skip connection
            if i - 1 < len(self.skip_layers):
                skip = self.skip_layers[i - 1](self.pool(features[-1]))
                x = x + skip
                
            features.append(x)
        
        # Final layer
        x = self.sigmoid(self.conv_layers[-1](features[-1]))
        return self.flatten(x)