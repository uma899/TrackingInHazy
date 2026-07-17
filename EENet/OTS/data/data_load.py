import os
import torch
import numpy as np
from PIL import Image as Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_dataloader(path, batch_size=64, num_workers=0):
    image_dir = os.path.join(path, 'train')

    dataloader = DataLoader(
        DeblurDataset(image_dir, ps=256),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test'), is_valid=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader

import random

import glob

class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False, is_valid=False, ps=None):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'hazy/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.is_valid = is_valid
        self.ps = ps
    
    def __len__(self):
        return len(self.image_list)

    def _get_gt_path(self, filename):
        # Extracts the base name without extension (e.g., "001.jpg" -> "001")
        base_name = os.path.splitext(filename)[0]
        gt_dir = os.path.join(self.image_dir, 'gt')
        
        # Searches for 001.png, 001.jpg, or 001.jpeg in the gt directory
        search_pattern = os.path.join(gt_dir, f"{base_name}.*")
        matches = glob.glob(search_pattern)
        
        if not matches:
            raise FileNotFoundError(f"Could not find a matching ground truth image for {filename} in {gt_dir}")
            
        return matches[0]

    def __getitem__(self, idx):
        filename = self.image_list[idx]
        image = Image.open(os.path.join(self.image_dir, 'hazy', filename)).convert('RGB')
        
        # Automatically finds the correct gt file path even if the extension differs
        gt_path = self._get_gt_path(filename)
        label = Image.open(gt_path).convert('RGB')
        
        ps = self.ps

        if self.ps is not None:
            image = F.to_tensor(image)
            label = F.to_tensor(label)

            hh, ww = label.shape[1], label.shape[2]

            rr = random.randint(0, hh-ps)
            cc = random.randint(0, ww-ps)
            
            image = image[:, rr:rr+ps, cc:cc+ps]
            label = label[:, rr:rr+ps, cc:cc+ps]

            if random.random() < 0.5:
                image = image.flip(2)
                label = label.flip(2)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)

        if self.is_test:
            return image, label, filename
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1].lower() not in ['png', 'jpg', 'jpeg']:
                raise ValueError(f"Unsupported file format found: {x}")
