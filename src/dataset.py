import pathlib
import torch
import numpy as np
from typing import Tuple, Generator
from PIL import Image
from torch.utils.data import Dataset
from .config import Train_Config, COLOR_MAP, Model_Config
from .utils import *
import albumentations as A
from albumentations.pytorch import ToTensorV2

TRAIN_CONFIG = Train_Config()
MODEL_CONFIG = Model_Config()

# 定义共享的几何变换
shared_transforms = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.RandomResizedCrop(height=MODEL_CONFIG.IMAGE_SIZE[0], width=MODEL_CONFIG.IMAGE_SIZE[1], scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1),
])

# 定义仅应用于图像的增强
image_specific_transforms = A.Compose([
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.8),
        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.8),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.8),
    ], p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# 数据集类
class RoomDataset(Dataset):
    def __init__(self, one_hot: bool = False):
        super().__init__()
        self.image_dir = self._getlist(TRAIN_CONFIG.IMAGE_DIR)
        self.label_dir = self._getlist(TRAIN_CONFIG.LABEL_DIR)
        self.data = {i:{'image':image, 'label':label} for i, (image, label) in enumerate(zip(self.image_dir, self.label_dir))}
        self.num_classes = len(COLOR_MAP)
        self.one_hot = one_hot

    def _getlist(self, data_dir: pathlib.Path):
        jpg_list = data_dir.glob("*.jpg")
        png_list = data_dir.glob("*.png")
        return list(jpg_list) + list(png_list)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self.data[idx]['image']).convert('RGB'))
        label = np.array(Image.open(self.data[idx]['label']))
        label = process_image_vectorized(label, COLOR_MAP)

        # 应用共享的几何变换
        transformed = shared_transforms(image=image, mask=label)
        image, label = transformed['image'], transformed['mask']

        # 应用图像特定的增强
        image = image_specific_transforms(image=image)['image']

        # transform to tensor
        label = torch.from_numpy(label).long().squeeze()
        # 将标签转换为one-hot编码
        if self.one_hot:
            label_one_hot = torch.nn.functional.one_hot(label, num_classes=self.num_classes)
            label_one_hot = label_one_hot.permute(2, 0, 1).float()
            return image, label_one_hot
        return image, label
    
    def _visualize(self, idx):
        image = np.array(Image.open(self.data[idx]['image']).convert('RGB'))
        label = np.array(Image.open(self.data[idx]['label']))
        label = process_image_vectorized(label, COLOR_MAP)

        # 应用共享的几何变换
        transformed = shared_transforms(image=image, mask=label)
        image, label = transformed['image'], transformed['mask']

        # 应用图像特定的增强
        image = image_specific_transforms(image=image)['image']

        return image, label