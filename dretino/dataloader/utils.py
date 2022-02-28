import os

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm import tqdm

train_transforms = A.Compose(
    [
        A.Resize(width=250, height=250),
        A.RandomCrop(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Blur(p=0.3),
        A.CLAHE(p=0.3),
        A.ColorJitter(p=0.3),
        A.Affine(shear=30, rotate=0, p=0.2),
        A.Normalize(
            mean=[0.3740, 0.4216, 0.5143],
            std=[0.8616, 1.1727, 1.4168],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(height=224, width=224),
        A.Normalize(
            mean=[0.3740, 0.4216, 0.5143],
            std=[0.8616, 1.1727, 1.4168],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.Resize(height=224, width=224),
        A.Normalize(
            mean=[0.3740, 0.4216, 0.5143],
            std=[0.8616, 1.1727, 1.4168],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)


class CustomDataset(Dataset):
    def __init__(self, dfx, image_dir, transform=None):
        self.dfx = dfx
        self.image_ids = self.dfx['Image name'].values
        self.targets = self.dfx['Retinopathy grade'].values
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        index = torch.tensor(self.targets[idx])
        target = F.one_hot(index, num_classes=5)

        img = np.array(Image.open(os.path.join(self.image_dir, img_name + '.jpg')).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target


def get_sampler(train_data):
    targets = []
    for _, target in tqdm(train_data):
        targets.append(torch.argmax(target, dim=-1))

    targets = torch.stack(targets)
    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)]
    )
    weight = 1. / class_sample_count.float()
    sample_weight = torch.tensor([weight[t] for t in targets])
    sampler = WeightedRandomSampler(sample_weight, len(sample_weight))

    return sampler
