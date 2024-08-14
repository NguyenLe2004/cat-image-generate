from utils.preprocessing import get_dataloader, DisplayExampleEachDataloader
from utils.model import Generator, Discriminator, model_preprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.transforms import Compose, Normalize
def main(args) -> None:
    """
    The main function that trains the model.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        None
    """
    train(
        batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        step_size = args.step_size,
        data_path = args.src_data_path
    )

def get_args():
    """
    Parses the command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--src_data_path","-sp", type=str, default="./cats", help="Path to source data")
    parser.add_argument("--batch_size","-b", type=int, default=64)
    parser.add_argument("--epochs","-e",type=int, default=200)
    parser.add_argument("--learning_rate","-lr",type=float, default=1e-3)
    parser.add_argument("--step_size","-ss",type=int, default=60)
    args = parser.parse_args()
    return args

def train( batch_size : float, learning_rate : float, step_size : int, data_path : str) -> None:
    """
    Trains the model.

    Args:
        batch_size (float): The batch size.
        learning_rate (float): The learning rate.
        step_size (int): The step size for the learning rate scheduler.
        data_path (str): The path to the source data.

    Returns:
        None
    """
    real_criterion = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter("Tensorboard")
    fixed_noise = torch.randn(batch_size, 100, 1, 1, device=device)
    generator, discriminator = model_preprocessing(Generator(), Discriminator(), device)
    optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_scheduler = StepLR(optimizerD, step_size=step_size, gamma=0.2, verbose=True)
    g_scheduler = StepLR(optimizerG, step_size=step_size, gamma=0.2,verbose=True)
    print("Starting Training Loop...")
    epoch_interval = 20
    idx = -1
    for epoch in range(200):
        if epoch % epoch_interval == 0 :
            idx +=1
        train_dataloader = get_dataloader(data_path= data_path, image_size=16 * (2**idx), max_image_size= 64)
        train_progess = tqdm(train_dataloader)
        for data in train_progess:
            cur_batch_size = data.shape[0]
            discriminator.train()
            generator.train()

            discriminator.zero_grad()
            real_images = data.to(device)
            output = discriminator(real_images,idx).view(-1)
            real_label = torch.ones_like(output)
            lossD_real = real_criterion(output, real_label)
            lossD_real.backward()
            
            noise = torch.randn(cur_batch_size, 100, 1, 1, device=device)
            fake_images = generator(noise,idx)
            output = discriminator(fake_images.detach(),idx).view(-1)
            fake_label = torch.zeros_like(output)
            lossD_fake = real_criterion(output, fake_label)
            lossD_fake.backward()
            optimizerD.step()

            generator.zero_grad()
            output = discriminator(fake_images,idx).view(-1)
            real_label = torch.ones_like(output)
            lossG = real_criterion(output, real_label)  
            lossG.backward()
            optimizerG.step()

            train_progess.set_description("epochs: {}: discriminator loss: {:.4f}, {:.4f}; generator loss: {:.4f}".format(epoch,lossD_real,lossD_fake, lossG))
        g_scheduler.step()
        d_scheduler.step()

        writer.add_scalar("Discriminator loss",(lossD_real + lossD_fake)/2, epoch)
        writer.add_scalar("Generator loss",lossG, epoch)

        torch.save(generator.state_dict(),"last_model.pth")
        generator.eval()
        with torch.no_grad():
            fake_images = generator(fixed_noise,idx).detach().cpu()
            fake_images = invert_normalize(fake_images)
            fake_images_grid = make_grid(fake_images, padding=2, normalize=True)
            writer.add_image("cat generate", fake_images_grid)

def invert_normalize(image : torch.Tensor) :
    """
    Inverts the normalization applied to the image.

    Args:
        image (torch.Tensor): The input image tensor.

    Returns:
        torch.Tensor: The denormalized image tensor.
    """
    transform = Compose([
        Normalize(mean = [ 0., 0., 0. ],
                        std = [ 2, 2, 2 ]),
        Normalize(mean = [ -0.5, -0.5, -0.5 ],
                            std = [ 1., 1., 1. ])
        ])
    return transform(image)
    
if __name__ == "__main__" :
    args = get_args()
    main(args)