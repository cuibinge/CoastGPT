
import os
import json
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from datetime import datetime

class GeoDataset(Dataset):
    """地理空间CLIP训练数据集，支持图像数据增强以扩充数据集"""
    
    def __init__(self, root_dir, image_transform=None, recursive_search=True, augmentation_factor=1):
        """
        初始化数据集
        Args:
            root_dir (string): 数据集根目录
            image_transform (callable, optional): 应用于图片的变换
            recursive_search (bool): 是否递归查找子文件夹中的图片
            augmentation_factor (int): 数据增强倍数（每幅图像将生成的增强样本数）
        """
        self.root_dir = root_dir
        self.recursive_search = recursive_search
        self.augmentation_factor = max(1, augmentation_factor)  # 确保至少为1
        
        # 初始化数据增强变换
        self.image_transform = self._get_augmentation_transforms(image_transform)
        
        # 收集所有原始样本
        self.orgsamples = self._collect_samples()
        # 扩充样本列表（每个原始样本重复augmentation_factor次）
        self.data_samples = self._augment_samples()
        
        print(f"原始样本数: {len(self.orgsamples)}, 增强后总样本数: {len(self.data_samples)}")

    

    def __len__(self):
        """返回增强后的总样本数"""
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        """获取指定索引的样本（应用数据增强）"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data_samples[idx]
        
        # 加载图片并应用增强变换
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            if self.image_transform:
                image = self.image_transform(image)
        except Exception as e:
            print(f"加载图片 {sample['image_path']} 时出错: {str(e)}")
            # 返回一个黑色图片作为备用
            image = Image.new('RGB', (256, 256), color='black')
            if self.image_transform:
                image = self.image_transform(image)
        
        # 获取对应的坐标
        coordinates = sample['coordinates']
        
        return image, coordinates