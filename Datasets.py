
import os
from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

class CatsAndDogsDataset(Dataset):
    """
    A custom dataset class for loading and preprocessing images of cats and dogs.
    """
    label_dict = {"cat": 0, "dog": 1}
    
    def __init__(self, location: str, train: bool = True):
        """
        Initializes the CatsAndDogsDataset.

        Args:
            location (str): The root directory of the dataset.
            train (bool, optional): Whether to load the training set or validation set. Defaults to True.
        """
        self.train = train
        self.set_location(location, train)   
        self.file_list = glob(os.path.join(self.location , "*.jpg"))
        self.labels = [0 if "cat" in file else 1 for file in self.file_list]
        self.transforms = self.get_transforms(train)
        
    def set_location(self, location: str, train: bool):
        """
        Sets the location of the dataset based on the train flag.

        Args:
            location (str): The root directory of the dataset.
            train (bool): Whether to load the training set or validation set.
        """
        if train:
            self.location = os.path.join(location, "train")
        else:
            self.location = os.path.join(location, "val")
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.file_list)
      
    def get_transforms(self, train: bool):
        """
        Returns the data transformations to be applied on the images.

        Args:
            train (bool): Whether to return transformations for training or validation.

        Returns:
            torchvision.transforms.Compose: The composed transformations.
        """
        if train:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        """
        Returns the image and its corresponding label at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            torch.Tensor: The preprocessed image.
            list: The one-hot encoded label.
        """
        img = Image.open(self.file_list[idx])
        img = self.transforms(img)
        return img, self.labels[idx]


# plt.imshow(img.permute(1, 2, 0) * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406]))
# # %%
