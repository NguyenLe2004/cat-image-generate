#!/usr/bin/env python3
"""
Main training script for GAN with diffusion process.
"""

import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.transforms import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    RandomRotation,
    ColorJitter,
    RandomResizedCrop,
    ToTensor,
    Normalize,
)

from data.dataloader import get_dataloader
from models.Generator import Generator
from models.Discriminator import Discriminator
from trainer.trainer import Trainer
from utils.helper import TrainingArgs


def main(args: argparse.Namespace) -> None:
    """
    Main training function for the GAN model.

    Args:
        args: Command-line arguments containing training configuration.

    Returns:
        None
    """
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ngpu = torch.cuda.device_count()

    # Data loading and transformations
    train_loader, valid_loader = get_dataloader(
        data_path=args.src_data_path,
        image_size = args.image_size,
        test_set=True,
        test_size=0.2
    )

    transform = Compose([
        Resize((128, 128)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=10),
        ColorJitter(brightness=0.2, contrast=0.2),
        RandomResizedCrop(128, scale=(0.8, 1.0)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    feature_extract_loader = get_dataloader(
        data_path=args.src_data_path,
        image_size=args.image_size,
        test_set=False,
        transform=transform
    )

    # Model initialization
    generator = Generator(latent_dim=args.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    feature_extract = initialize_feature_extractor().to(device)
    if (device == 'cuda') and (ngpu > 1):
        generator = nn.DataParallel(generator, list(range(ngpu)))
        discriminator = nn.DataParallel(discriminator, list(range(ngpu)))
        feature_extract = nn.DataParallel(feature_extract, list(range(ngpu)))

    # Training configuration
    train_args = TrainingArgs(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        T_max=500,
        T_min=50,
        device=device,
        num_epochs=args.epochs
    )

    # Training
    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        feature_extract=feature_extract,
        training_args=train_args
    )

    trainer.train(
        train_loader=train_loader,
        val_loader=valid_loader,
        feature_extract_loader=feature_extract_loader
    )


def initialize_feature_extractor() -> torch.nn.Module:
    """
    Initialize and configure the feature extractor model.

    Returns:
        Configured feature extractor model
    """
    model = vgg16(pretrained=True)
    
    # Freeze feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Modify classifier head
    model.classifier[-1] = torch.nn.Linear(4096, 256)
    
    return model


def get_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Train GAN with diffusion process"
    )
    
    parser.add_argument(
        "--src_data_path", "-sp",
        type=str,
        default="./cats",
        help="Path to source dataset directory"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=64,
        help="Batch size for training"
    )
    parser.add_argument(
        "--latent_dim", "-ld",
        type=int,
        default=100,
        help="Dimension of latent space"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=200,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", "-lr",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--image_size" , "-is" ,
        type=int,
        default=128,
        help="Size of the image"
    )
    parser.add_argument(
        "--step_size", "-ss",
        type=int,
        default=60,
        help="Step size for learning rate scheduling"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)