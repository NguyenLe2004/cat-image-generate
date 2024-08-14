import torch
import argparse
from torch.nn import DataParallel
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from utils.model import Generator
from torchvision import transforms

def generate(checkpoint_path: str, output_path: str) -> None:
    """
    Generate an image using a pre-trained generator model.

    Args:
        checkpoint_path (str): Path to the saved model checkpoint.
        output_path (str): Path to save the generated image.

    Returns:
        None
    """
    # Load the pre-trained generator model
    model = DataParallel(Generator()).eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # Generate a random noise vector and use the generator to produce an image
    noise = torch.randn(1, 100, 1, 1)
    with torch.no_grad():
        output = invert_normalize(model(noise))

    # Save and display the generated image
    save_image(output, output_path)
    show_image(output)

def get_args():
    """
    Parses the command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate")
    parser.add_argument("--checkpoint_path","-c", type=str, default="./last_model.pth", help="Path to trained model")
    parser.add_argument("--output_path","-o", type=str, default="./generated_image.png")
    args = parser.parse_args()
    return args

def show_image(output) -> None:
    """
    Display the generated image.

    Args:
        output (torch.Tensor): The generated image tensor.

    Returns:
        None
    """
    output = output.squeeze(0)
    output = output.permute(1, 2, 0)
    plt.imshow(output)
    plt.show()

def invert_normalize(image: torch.Tensor) -> torch.Tensor:
    """
    Invert the normalization applied to the input image.

    Args:
        image (torch.Tensor): The input image tensor.

    Returns:
        torch.Tensor: The normalized image tensor.
    """
    transform = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[2, 2, 2]),
        transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1, 1, 1])
    ])
    return transform(image)

if __name__ == "__main__":
    args = get_args()
    generate(
        checkpoint_path = args.checkpoint_path,
        output_path = args.output_path
    )