
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from Datasets import CatsAndDogsDataset
from torchvision.models import resnet18

# https://www.kaggle.com/code/tirendazacademy/cats-dogs-classification-with-pytorch
class CatsNDogsModelKaggle(nn.Module):
        """
        This is a PyTorch model for classifying images of cats and dogs.
        It consists of multiple convolutional layers followed by a fully connected layer.

        Args:
                None

        Returns:
                torch.Tensor: The output tensor of the model.
        """

        def __init__(self):
                super().__init__()
                self.conv_layer_1 = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    nn.MaxPool2d(2))
                self.conv_layer_2 = nn.Sequential(
                    nn.Conv2d(64, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(512),
                    nn.MaxPool2d(2))
                self.conv_layer_3 = nn.Sequential(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(512),
                    nn.MaxPool2d(2)) 
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(in_features=512*3*3, out_features=2))

        def forward(self, x: torch.Tensor):
                x = self.conv_layer_1(x)
                x = self.conv_layer_2(x)
                x = self.conv_layer_3(x)
                x = self.conv_layer_3(x)
                x = self.conv_layer_3(x)
                x = self.conv_layer_3(x)
                x = self.classifier(x)
                return x


class CatsNDogsModelTL(nn.Module):
    def __init__(self):
        """
        Initializes a CatsNDogsModelTL object.
        """
        super(CatsNDogsModelFC, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
        # Modify the last fully connected layer for binary classification
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        """
        Performs forward pass through the CatsNDogsModelTL network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.resnet18(x)


class CatsNDogsModelFC(nn.Module):
    def __init__(self):
        """
        Cats and Dogs Model using Fully Connected Layers.

        This model consists of convolutional layers followed by fully connected layers.
        The convolutional layers extract features from the input images, and the fully connected layers
        perform classification based on the extracted features.

        Attributes:
            conv01 (nn.Conv2d): First convolutional layer.
            bnorm01 (nn.BatchNorm2d): Batch normalization layer after the first convolutional layer.
            conv02 (nn.Conv2d): Second convolutional layer.
            bnorm02 (nn.BatchNorm2d): Batch normalization layer after the second convolutional layer.
            conv03 (nn.Conv2d): Third convolutional layer.
            bnorm03 (nn.BatchNorm2d): Batch normalization layer after the third convolutional layer.
            conv04 (nn.Conv2d): Fourth convolutional layer.
            bnorm04 (nn.BatchNorm2d): Batch normalization layer after the fourth convolutional layer.
            fc01 (nn.Linear): First fully connected layer.
            fc02 (nn.Linear): Second fully connected layer.
        """
        super(CatsNDogsModelFC, self).__init__()
        self.conv01 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=2)
        self.bnorm01 = nn.BatchNorm2d(16)
        self.conv02 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2)
        self.bnorm02 = nn.BatchNorm2d(32)
        self.conv03 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2)
        self.bnorm03 = nn.BatchNorm2d(64)
        self.conv04 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2)
        self.bnorm04 = nn.BatchNorm2d(128)

        self.fc01 = nn.Linear(in_features=21632, out_features=128)
        self.fc02 = nn.Linear(in_features=128, out_features=2)
        
    def forward(self, x):
        """
        Forward pass of the Cats and Dogs Model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = F.relu(self.conv01(x))
        x = self.bnorm01(x)
        x = F.relu(self.conv02(x))
        x = self.bnorm02(x)
        x = F.relu(self.conv03(x))
        x = self.bnorm03(x) 
        x = F.relu(self.conv04(x))
        x = self.bnorm04(x)

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc01(x))
        x = self.fc02(x)
        return x
# %%

class CatsNDogsModelConvOnly(nn.Module):
    """
    Convolutional Neural Network model for classifying images of cats and dogs.

    This model consists of multiple convolutional layers followed by batch normalization,
    global average pooling, and fully connected layers.

    Args:
        None

    Returns:
        torch.Tensor: The output tensor of the model.

    """

    def __init__(self):
        super(CatsNDogsModelConvOnly, self).__init__()
        self.conv01 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=2)
        self.bnorm01 = nn.BatchNorm2d(16)
        self.conv02 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.bnorm02 = nn.BatchNorm2d(32)
        self.conv03 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2)
        self.bnorm03 = nn.BatchNorm2d(64)
        self.conv04 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.bnorm04 = nn.BatchNorm2d(128)
        self.conv05 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2)
        self.bnorm05 = nn.BatchNorm2d(256)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc01 = nn.Linear(in_features=256, out_features=128)
        self.dropout = nn.Dropout(0.2)
        self.fc02 = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        """
        Forward pass of the CatsNDogsModelConvOnly.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output tensor of the model.

        """
        x = F.relu(self.conv01(x))
        x = self.bnorm01(x)
        x = F.relu(self.conv02(x))
        x = self.bnorm02(x)
        x = F.relu(self.conv03(x))
        x = self.bnorm03(x)
        x = F.relu(self.conv04(x))
        x = self.bnorm04(x)
        x = F.relu(self.conv05(x))
        x = self.bnorm05(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc01(x))
        x = self.dropout(x)
        x = self.fc02(x)

        return x

# # %%
# loc = "F:\\Datasets\\Pussies and Puppies"
# ds = CatsAndDogsDataset(loc, train=False)
# img, label = ds.__getitem__(3)
# img = img.unsqueeze(0)
# # %%
# model = CatsNDogsModelConvOnly()
# # %%
# out = model(img)
# print(out.shape)
# # %%

