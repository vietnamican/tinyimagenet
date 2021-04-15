import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder, DatasetFolder, default_loader
from datasets import TinyImagenetDataset, transformer
import random

transform = transforms.Compose([
    transforms.ToTensor(),
])
torch.random.manual_seed(42)
if __name__ == '__main__':
    dataset = TinyImagenetDataset('sets/random/0/wnids10.txt', 'tiny-imagenet-200/train', transform=transformer['train'])
    print(dataset.__len__())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

    for image, label in dataloader:
        # print(image.shape)
        print(label)
        # break
