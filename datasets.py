from glob import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob(os.path.join(root, mode, 'A', '*')))
        self.files_B = sorted(glob(os.path.join(root, mode, 'B', '*')))

    def __getitem__(self, index):
        img_A = self.files_A[index % len(self.files_A)]
        item_A = self.transform(Image.open(img_A))

        if self.unaligned:
            img_B = self.files_B[random.randint(0, len(self.files_B) - 1)]
        else:
            img_B = self.files_B[index % len(self.files_B)]

        item_B = self.transform(Image.open(img_B))

        return {'A': item_A, 'B': item_B, 'img_A': img_A, 'img_B': img_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
