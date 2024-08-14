from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, Resize

class ImageDataset(Dataset):
    """
    PyTorch Dataset class for loading and transforming image data.
    """
    def __init__(self, image_path, transform=None):
        self.image_paths = glob(os.path.join(image_path,"*.jpg"))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image
    
def get_dataloader(data_path : str, image_size : int, max_image_size : int = 64, batch_size : int = 64) :
    """
    Create a PyTorch DataLoader for the image dataset.
    
    Args:
        data_path (str): Path to the directory containing image files.
        image_size (int): Target size for the images.
        max_image_size (int, optional): Maximum image size. Defaults to 64.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 64.
    
    Returns:
        DataLoader: PyTorch DataLoader for the image dataset.
    """
    image_size = min(image_size, max_image_size)
    transform = Compose([
        ToTensor(),
        Resize((image_size,image_size)),
        RandomHorizontalFlip(0.5),
        Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
    ])
    dataset = ImageDataset(image_path = data_path, transform= transform)
    return DataLoader(dataset, batch_size = batch_size, shuffle = True)

def DisplayExampleEachDataloader(dataloader : DataLoader) -> None :
    """
    Display example images from each batch in the DataLoader.
    
    Args:
        dataloader (DataLoader): PyTorch DataLoader containing the image dataset.
    """
    _ , ax = plt.subplots(1, 3, figsize=(10, 10))
    batch = next(iter(dataloader))
    first_image, seconds_image, thirds_image = batch[0], batch[1], batch[2]

    first_image = np.transpose(first_image, (1, 2, 0))
    seconds_image = np.transpose(seconds_image, (1, 2, 0))
    thirds_image = np.transpose(thirds_image, (1, 2, 0))

    ax[0].imshow(first_image)
    ax[1].imshow(seconds_image)
    ax[2].imshow(thirds_image)
    plt.suptitle("Example Images from Each DataLoader")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataloader = get_dataloader(data_path= "../cats",image_size= 64, max_image_size= 64)
    DisplayExampleEachDataloader(dataloader) 