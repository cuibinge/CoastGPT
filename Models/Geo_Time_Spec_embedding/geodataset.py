import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from datetime import datetime
from torch.utils.data import DataLoader


class GeoDataset(Dataset):
    """地理空间CLIP训练数据集，加载图片和对应的坐标信息"""
    
    def __init__(self, root_dir, image_transform=None):
        """
        初始化数据集
        Args:
            root_dir (string): 数据集根目录
            image_transform (callable, optional): 应用于图片的变换
        """
        self.root_dir = root_dir
        self.image_transform = image_transform

        # 如果没有指定变换，使用默认变换
        if self.image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        # 收集所有图片路径和对应的坐标信息
        self.data_samples = self._collect_samples()

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.data_samples)

    def __getitem__(self, idx):
        """返回（处理后图像，仅含月/日/小时的时间戳张量）"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data_samples[idx]
        # 加载并应用图像变换
        image = Image.open(sample['image_path']).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)
        # 时间戳转换为张量（仅反映月/日/小时）
        timestamp = torch.tensor(sample['timestamp'], dtype=torch.float32)
        
        return image, timestamp

