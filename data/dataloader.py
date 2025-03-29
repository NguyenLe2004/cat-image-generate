import os
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image

def get_dataloader(data_path : str, image_size : int, test_set : bool = False , test_size : float = 0.2, transform : transforms = None,  batch_size : int = 64) :
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
    paths = glob(os.path.join(data_path, "*.jpg"))
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    if test_set :
        train_paths, val_paths = train_test_split(paths, test_size = test_size)
        train_dataset = ImageDataset(train_paths, transform)
        val_dataset = ImageDataset(val_paths, transform)
    
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size)
        )
    dataset = ImageDataset(paths, transform=transform)
    DataLoader(dataset, batch_size=batch_size, shuffle=True)