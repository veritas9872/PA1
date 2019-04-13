import torch
import torchvision
from torchvision import transforms
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader

data_root = '/home/veritas/PycharmProjects/PA1/data'

batch_size = 32
num_workers = 4

# Do more fancy transforms later.
train_dataset = torchvision.datasets.CIFAR100(data_root, train=True, transform=transforms.ToTensor(), download=True)
val_dataset = torchvision.datasets.CIFAR100(data_root, train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

