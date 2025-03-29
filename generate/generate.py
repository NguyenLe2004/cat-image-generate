import torch
import argparse
from torch.nn import DataParallel
import matplotlib.pyplot as plt
from torchvision.utils import save_image
# from utils.model import Generator
from torchvision import transforms

def generate(checkpoint_path: str, output_path: str , device : str) -> None:
    """
    Generate an image using a pre-trained generator model.

    Args:
        checkpoint_path (str): Path to the saved model checkpoint.
        output_path (str): Path to save the generated image.

    Returns:
        None
    """
    # Load the pre-trained generator model
    model = torch.jit.load(checkpoint_path, map_location = device)

    # Generate a random noise vector and use the generator to produce an image
    noise = torch.randn(1, 100, 1, 1)
    with torch.no_grad():
        output = model(noise) * 0.5 + 0.5

    # Save and display the generated image
    save_image(output, output_path)

def get_args():
    """
    Parses the command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate")
    parser.add_argument("--checkpoint_path","-c", type=str, default="./cat_face_generate_model.pt", help="Path to trained model")
    parser.add_argument("--output_path","-o", type=str, default="./generated_image.png")
    parser.add_argument("--device","-d", type=str, default="cpu")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args()
    generate(
        checkpoint_path = args.checkpoint_path,
        output_path = args.output_path,
        device = args.device
    )