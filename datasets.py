import torch
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder, make_dataset, IMG_EXTENSIONS


transformer = {
    'train': transforms.Compose([
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(15, (0.1, 0.1), (0.9, 1.1), 10, ),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
}


class TinyImagenetDataset(ImageFolder):
    def __init__(
        self,
        class_to_idx_file,
        root: str,
        transform=None,
        target_transform=None
    ):
        super().__init__(root, transform, target_transform, is_valid_file=None)
        self.make_class_to_idx(class_to_idx_file)
        self.samples = self.make_dataset(root, self.class_to_idx)
        self.imgs = self.samples
        # self.imgs = self.make_dataset(root, self.class_to_idx)

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx,
        extensions=IMG_EXTENSIONS,
        is_valid_file=None,
    ):
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def make_class_to_idx(self, class_to_idx_file):
        with open(class_to_idx_file, 'r') as f:
            content = f.read().split('\n')[:-1]
            class_to_idx = {}
            idx = 0
            for c in content:
                class_to_idx[c] = idx
                idx += 1
            self.class_to_idx = class_to_idx
