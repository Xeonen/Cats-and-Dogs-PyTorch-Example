# %%
from CatsNDogsModel import CatsNDogsModelConvOnly, CatsNDogsModelFC, CatsNDogsModelTL, CatsNDogsModelKaggle
from Datasets import CatsAndDogsDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from typing import Tuple, List, Dict, Union, Any
# %%
class Trainer:
    metrics_dict = {"train": {"loss": [], "acc": []}, "val": {"loss": [], "acc": []}}
    def __init__(self, model: nn.Module, root: str, batch_size: int, epochs: int, device: str = "cuda:0") -> None:
        self.root = root
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

        self.writer = SummaryWriter()
        self.criterion = nn.CrossEntropyLoss()
        self.set_optimizer()
        self.set_datasets()
        self.model.to(self.device)


    def set_optimizer(self, lr: float = 1e-3) -> None:
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def set_datasets(self) -> None:
        self.train_dataset = CatsAndDogsDataset(self.root, train=True)
        self.val_dataset = CatsAndDogsDataset(self.root, train=False)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def train_step(self, train: bool, epoch: int=0) -> Tuple:
        if train:
            step_type = "Train"
            self.model.train()
            loader = self.train_loader
            
        else:
            step_type = "Validation"
            self.model.eval()
            loader = self.val_loader

        pbar = tqdm(loader)
        step_loss = []
        step_acc = []

        for img, label in pbar:

            img = img.to(self.device)
            label = label.to(self.device)

            if train:
                self.optimizer.zero_grad()
                
            out = self.model(img)
            loss = self.criterion(out, label)

            if train:
                loss.backward()
                self.optimizer.step()

            pred = torch.argmax(out, dim=1)
            correct = torch.sum(pred == label).item()
            acc = correct / len(label)
            step_acc.append(acc)
            step_loss.append(loss.item())
            pbar.set_description(f"Epoch: {epoch}/{self.epochs} Step Type: {step_type} Loss: {torch.mean(torch.tensor(step_loss)):.4f} Acc: {torch.mean(torch.tensor(step_acc)):.4f}")
            self.writer.add_scalar(f"{step_type} Loss", loss.item(), epoch * len(loader) + pbar.n)
        return torch.mean(torch.tensor(step_loss)), torch.mean(torch.tensor(step_acc))

    def train(self, model_name: str) -> None:
        best_loss = torch.inf
        for epoch in range(self.epochs):
            loss, acc = self.train_step(True, epoch)
            self.metrics_dict["train"]["loss"].append(loss)
            self.metrics_dict["train"]["acc"].append(acc)
            
            loss, acc = self.train_step(False, epoch)
            self.metrics_dict["val"]["loss"].append(loss)
            self.metrics_dict["val"]["acc"].append(acc)

            if loss < best_loss:
                best_loss = loss
                torch.save(self.model.state_dict(), f"{model_name}_{epoch}_{best_loss:.4f}.pth")
                print(f"Model saved at epoch {epoch} with loss {best_loss:.4f}")
        self.writer.close()

# %%
if __name__ == "__main__":
    # model = CatsNDogsModelKaggle()
    # model = CatsNDogsModelFC()
    model = CatsNDogsModelConvOnly()
    root = "F:\\Datasets\\Pussies and Puppies"
    trainer = Trainer(model, root, 32, 1)
    trainer.train("conv_only")
    # for i in range(1):
    #     trainer.train_step(True)
    #     trainer.train_step(False)
# %%
# model = CatsNDogsModelFC()
# # model = CatsNDogsModelConvOnly()
# root = "F:\\Datasets\\Pussies and Puppies"
# trainer = Trainer(model, root, 32, 10)
# for i in range(3):
#     trainer.train_step(True)
#     trainer.train_step(False)
# %%

