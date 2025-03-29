import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainingArgs:
    """Configuration class for training parameters."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    fe_step: int = 10
    eval_step: int = 3
    epsilon: float = 0.1
    latent_dim: int = 100
    num_epochs: int = 200
    T_max: int = 500
    T_min: int = 50
    betas: Tuple[float, float] = (0.5, 0.999)
    device: str = "cuda"
    checkpoint_dir: str = "./checkpoints"

def update_T(T_current, r_d, d_target=0.6, C=1, T_min = 5, T_max = 500):
    if r_d > d_target:
        T_new = T_current + C  
    else:
        T_new = T_current - C 
    return max(T_min, min(T_new, T_max))


def weights_init(m):
    classname = m.__class__.__name__
    
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02) 
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    
    # Khởi tạo cho BatchNorm2d
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)  
        nn.init.constant_(m.bias.data, 0) 