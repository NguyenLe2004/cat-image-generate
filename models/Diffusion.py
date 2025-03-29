import torch
import torch.nn as nn
from typing import Tuple

class DiffusionProcess(nn.Module):
    """Implements the diffusion process for noise scheduling and image corruption."""

    def __init__(
        self,
        T: int,
        beta_start: float,
        beta_end: float,
        device: torch.device,
        sigma: float = 0.05
    ) -> None:
        """
        Initialize the diffusion process.
        
        Args:
            T: Number of diffusion steps
            beta_start: Starting value for beta schedule
            beta_end: Ending value for beta schedule
            device: Target device for computation
            sigma: Noise scaling factor
        """
        super(DiffusionProcess, self).__init__()
        
        self.T = T
        self.sigma = sigma
        
        # Register buffers for diffusion parameters
        self.register_buffer(
            'betas',
            torch.linspace(beta_start, beta_end, T)
        )
        self.register_buffer(
            'alphas',
            1 - self.betas
        )
        self.register_buffer(
            'alpha_bars',
            torch.cumprod(self.alphas, dim=0)
        )
        
        # Move all buffers to the specified device
        self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply diffusion process to input images.
        
        Args:
            x: Input images (batch_size, channels, height, width)
            t: Diffusion timesteps (batch_size,)
            
        Returns:
            Tuple of (noisy_images, added_noise)
        """
        # Get alpha_bar for current timesteps
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        
        # Generate scaled noise
        noise = torch.randn_like(x) * self.sigma
        
        # Compute noisy images
        noisy_x = (
            torch.sqrt(alpha_bar_t) * x + 
            torch.sqrt(1 - alpha_bar_t) * noise
        )
        
        return noisy_x, noise
