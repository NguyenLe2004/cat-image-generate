import torch
import wandb
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from utils.helper import weights_init, update_T, TrainingArgs
from utils.metrics import compute_gan_metrics
from models.Generator import Generator
from models.Discriminator import Discriminator
from models.Diffusion import DiffusionProcess


class Trainer:
    """Main training class for GAN with diffusion process and feature extraction."""
    
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        feature_extract: nn.Module,
        training_args: TrainingArgs
    ):
        """
        Initialize the trainer with models and training configuration.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            feature_extract: Feature extraction model
            diffusion: Diffusion process module
            training_args: Training configuration parameters
        """
        self.generator = generator
        self.discriminator = discriminator
        self.feature_extract = feature_extract
        self.diffusion = DiffusionProcess(training_args.T_max, beta_start=0.0001, beta_end=0.02, device=training_args.device, sigma=0.05)
        self.training_args = training_args
        
        # Training state
        self.T_max = training_args.T_max
        self.T_min = training_args.T_min
        self.T = training_args.T_max
        self.min_fid = 1000
        
        # Initialize Weights & Biases
        self.wandb = wandb
        self.wandb.init("Image Cat GANs", config=training_args.__dict__)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        feature_extract_loader: DataLoader
    ) -> None:
        """
        Main training loop.
        
        Args:
            train_loader: DataLoader for training images
            val_loader: DataLoader for validation images
            feature_extract_loader: DataLoader for feature extraction
        """
        self._setup_training()
        
        for epoch in range(1, self.training_args.num_epochs + 1):
            self._train_epoch(epoch, train_loader, feature_extract_loader)
            self.eval_epoch(epoch, val_loader)

    def _setup_training(self) -> None:
        """Initialize optimizers, weights and criterions"""
        self.optimizer_D = Adam(
            self.discriminator.parameters(),
            lr=self.training_args.learning_rate,
            betas=self.training_args.betas
        )
        self.optimizer_G = Adam(
            self.generator.parameters(),
            lr=self.training_args.learning_rate,
            betas=self.training_args.betas
        )
        self.optimizer_FE = Adam(
            self.feature_extract.parameters(),
            lr=self.training_args.learning_rate
        )

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        
        self.gan_criterion = nn.BCELoss()
        self.fe_criterion = nn.TripletMarginLoss(swap=True, margin=2.0)

    def _train_epoch(
        self,
        epoch: int,
        train_loader: DataLoader,
        feature_extract_loader: DataLoader
    ) -> None:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        # Select appropriate loader based on feature extraction step
        if epoch % self.training_args.fe_step == 0:
            loader = zip(train_loader, feature_extract_loader)
            total_batches = len(feature_extract_loader)
        else:
            loader = train_loader
            total_batches = len(train_loader)
            
        progress_bar = tqdm(
            enumerate(loader),
            total=total_batches,
            desc=f"Epoch {epoch}/{self.training_args.num_epochs}"
        )
        
        for step, batch in progress_bar:
            d_loss, g_loss = self._train_step(epoch, step, batch)
            self._update_progress(progress_bar, epoch, d_loss, g_loss)

    def _train_step(
        self,
        epoch: int,
        step: int,
        batch: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Process one training batch.
        
        Returns:
            Tuple of discriminator loss and generator loss
        """
        device = self.training_args.device
        
        # Prepare real images
        if epoch % self.training_args.fe_step == 0:
            real_images = batch[0].to(device)
            anchor_images = batch[1].to(device)
        else:
            real_images = batch.to(device)
            
        batch_size = real_images.size(0)
        
        # Train discriminator
        d_loss, real_output = self._train_discriminator(real_images, batch_size)
        
        # Train feature extractor if needed
        if epoch % self.training_args.fe_step == 0:
            self._train_feature_extractor(real_images, anchor_images, batch_size)
            
        # Train generator
        g_loss = self._train_generator(real_images, batch_size, epoch)
        
        # Update diffusion parameters
        if step % 4 == 0:
            self._update_diffusion_params(real_output)
            
        return d_loss.item(), g_loss.item()

    def _train_discriminator(
        self,
        real_images: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train discriminator on real and fake images."""
        self.optimizer_D.zero_grad()
        
        # Prepare real and fake data
        t = torch.randint(0, self.T, (batch_size,), device=self.training_args.device)
        noisy_real_images, _ = self.diffusion(real_images, t)
        
        real_labels = torch.ones(batch_size, 1, device=self.training_args.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.training_args.device)
        
        # Real images pass
        real_output = self.discriminator(noisy_real_images.detach())
        d_loss_real = self.gan_criterion(real_output, real_labels)
        
        # Fake images pass
        z = torch.randn(batch_size, self.training_args.latent_dim, 1, 1, 
                       device=self.training_args.device)
        fake_images = self.generator(z)
        noisy_fake_images, _ = self.diffusion(fake_images, t)
        fake_output = self.discriminator(noisy_fake_images.detach())
        d_loss_fake = self.gan_criterion(fake_output, fake_labels)
        
        # Combined loss and backprop
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.optimizer_D.step()
        
        return d_loss, real_output

    def _train_feature_extractor(
        self,
        real_images: torch.Tensor,
        anchor_images: torch.Tensor,
        batch_size: int
    ) -> None:
        """Train feature extractor using triplet loss."""
        self.feature_extract.train()
        self.optimizer_FE.zero_grad()
        
        # Generate fake images
        z = torch.randn(batch_size, self.training_args.latent_dim, 1, 1,
                       device=self.training_args.device)
        fake_images = self.generator(z)
        
        # Compute triplet loss
        positive_extract = self.feature_extract(real_images.detach())
        anchor_extract = self.feature_extract(anchor_images.detach())
        negative_extract = self.feature_extract(fake_images.detach())
        
        fe_loss = self.fe_criterion(anchor_extract, positive_extract, negative_extract)
        fe_loss.backward()
        
        # Clip gradients and update
        nn.utils.clip_grad_norm_(self.feature_extract.parameters(), max_norm=1.0)
        self.optimizer_FE.step()

    def _train_generator(
        self,
        real_images: torch.Tensor,
        batch_size: int,
        epoch: int
    ) -> torch.Tensor:
        """Train generator to fool discriminator and match features."""
        self.optimizer_G.zero_grad()
        
        # Generate fake images
        z = torch.randn(batch_size, self.training_args.latent_dim, 1, 1,
                       device=self.training_args.device)
        fake_images = self.generator(z)
        t = torch.randint(0, self.T, (batch_size,), device=self.training_args.device)
        noisy_fake_images, _ = self.diffusion(fake_images, t)
        
        # Basic GAN loss
        fake_output = self.discriminator(noisy_fake_images)
        g_loss = self.gan_criterion(fake_output, torch.ones_like(fake_output))
        
        # Add feature matching loss if needed
        if epoch >= self.training_args.fe_step:
            with torch.no_grad():
                self.feature_extract.eval()
                real_emb = self.feature_extract(real_images).detach()
                fake_emb = self.feature_extract(fake_images).detach()
            
            g_loss = g_loss + self.training_args.epsilon * nn.functional.mse_loss(real_emb, fake_emb)
        
        g_loss.backward()
        self.optimizer_G.step()
        
        return g_loss

    def _update_diffusion_params(self , real_output) -> None:
        """Update diffusion parameters based on discriminator performance."""
        r_d = torch.sign(real_output - 0.5).mean().item()
        self.T = update_T(
            self.T,
            r_d,
            T_min=self.T_min,
            T_max=self.T_max
        )

    def _update_progress(
        self,
        progress_bar: tqdm,
        epoch: int,
        d_loss: float,
        g_loss: float
    ) -> None:
        """Update progress bar and log metrics."""
        description = (
            f"Epoch {epoch}/{self.training_args.num_epochs} | "
            f"T: {self.T} | "
            f"D Loss: {d_loss:.4f} | "
            f"G Loss: {g_loss:.4f}"
        )
        progress_bar.set_description(description)
        
        self.wandb.log({
            "D_loss": d_loss,
            "G_loss": g_loss,
            "epoch": epoch,
            "T": self.T
        })

    def eval_epoch(self, epoch: int, dataloader: DataLoader) -> None:
        """Run evaluation for one epoch."""
        self.generator.eval()
        
        # Calculate metrics periodically
        if epoch % self.training_args.eval_step == 0:
            metrics = self._calculate_metrics(dataloader)
            self.wandb.log(metrics)
            self._update_training_strategy(metrics["FID"])
        
        # Generate sample images
        self._log_generated_images(epoch)

    def _calculate_metrics(self, dataloader: DataLoader) -> Dict[str, float]:
        """Calculate GAN evaluation metrics."""
        return compute_gan_metrics(
            generator=self.generator,
            dataloader=dataloader,
            latent_dim=self.training_args.latent_dim,
            device=self.training_args.device
        )

    def _update_training_strategy(self, fid: float) -> None:
        """Adjust training parameters based on FID score."""
        self.min_fid = min(self.min_fid, fid)
        
        if self.min_fid <= 30:
            self.training_args.T_max = 30
            self.training_args.T_min = 1
            self.training_args.fe_step = 2
        elif self.min_fid <= 40:
            self.training_args.T_max = 50
            self.training_args.T_min = 5
            self.training_args.fe_step = 5

    def _log_generated_images(self, epoch: int) -> None:
        """Generate and log sample images."""
        with torch.no_grad():
            z = torch.randn(16, self.training_args.latent_dim, 1, 1,
                          device=self.training_args.device)
            fake_images = self.generator(z).cpu()
            fake_images = fake_images * 0.5 + 0.5
            
        fake_grid = vutils.make_grid(
            fake_images,
            nrow=4,
            padding=2,
            normalize=False
        )
        
        self.wandb.log({
            "Generated Images": [
                wandb.Image(fake_grid, caption=f"Epoch {epoch}")
            ]
        })