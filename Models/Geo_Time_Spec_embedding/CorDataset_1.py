
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
    
    def _get_augmentation_transforms(self, custom_transform):
        """获取数据增强变换（组合基础变换和增强变换）"""
        if custom_transform is not None:
            return custom_transform
        
        # 基础变换 + 数据增强
        return transforms.Compose([
            # 随机裁剪
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
            # 随机水平翻转
            transforms.RandomHorizontalFlip(p=0.5),
            # 随机垂直翻转
            transforms.RandomVerticalFlip(p=0.3),
            # 随机旋转
            transforms.RandomRotation(degrees=15),
            # 颜色抖动
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=0.8),
            # 随机灰度化
            transforms.RandomGrayscale(p=0.2),
            # 高斯模糊
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
            # 转换为张量并归一化
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def _collect_samples(self):
        """收集所有原始样本"""
        samples = []
        
        # 处理GeoJSON文件
        geojson_files = glob.glob(os.path.join(self.root_dir, "*.geojson"))
        print(f"找到 {len(geojson_files)} 个GeoJSON文件")
        for geojson_path in geojson_files:
            base_name = os.path.splitext(os.path.basename(geojson_path))[0]
            image_dir = os.path.join(self.root_dir, f"{base_name}_Image")
            
            if not os.path.exists(image_dir):
                print(f"警告: 图片文件夹 {image_dir} 不存在，已跳过")
                continue
            
            try:
                with open(geojson_path, 'r', encoding='utf-8') as f:
                    geojson_data = json.load(f)
                
                for item in geojson_data.get('data', []):
                    image_name = item.get('name')
                    if not image_name:
                        continue
                    
                    image_path = self._find_image_path(image_dir, str(image_name))
                    if not image_path:
                        print(f"警告: 图片 {image_name} 在 {image_dir} 中未找到，已跳过")
                        continue
                    
                    features = item.get('features', [])
                    if features and len(features) > 0:
                        coordinates = features[0].get('geometry', {}).get('coordinates', [])
                        coordinates = self._convert_coordinates(coordinates)
                        
                        samples.append({
                            'image_path': image_path,
                            'coordinates': coordinates,
                            'source': 'geojson'
                        })
            
            except Exception as e:
                print(f"处理 {geojson_path} 时出错: {str(e)}")
                continue
        
        # 处理CSV文件
        csv_files = glob.glob(os.path.join(self.root_dir, "*.csv"))
        print(f"找到 {len(csv_files)} 个CSV文件")
        for csv_path in csv_files:
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            image_dir = os.path.join(self.root_dir, f"{base_name}_Image")
            
            if not os.path.exists(image_dir):
                print(f"警告: 图片文件夹 {image_dir} 不存在，已跳过CSV文件 {csv_path}")
                continue
            
            try:
                df = pd.read_csv(csv_path)
                required_cols = ['LAT', 'LON', 'IMG_ID']
                if not all(col in df.columns for col in required_cols):
                    #print(f"警告: CSV文件 {csv_path} 缺少必要的列 {required_cols}，已跳过")
                    continue
                
                for idx_row, row in df.iterrows():
                    img_id = row['IMG_ID']
                    if pd.isna(img_id):
                        #CSV第{idx_row+1}行IMG_ID为空，已跳过")
                        continue
                    
                    # 新功能：IMG_ID作为相对路径处理
                    img_id = str(img_id).strip()
                    
                    # 构建完整的图片路径：image_dir + IMG_ID
                    image_path = os.path.join(image_dir, img_id)
                    
                    # 检查图片路径是否存在
                    if not os.path.exists(image_path):
                        #print(f"警告: 图片路径 {image_path} 不存在，尝试查找替代文件...")
                        
                        # 如果直接路径不存在，尝试查找可能的变体（大小写、扩展名等）
                        found_path = self._find_image_by_path_variants(image_dir, img_id)
                        if found_path:
                            image_path = found_path
                            print(f"找到替代文件: {image_path}")
                        else:
                            #print(f"警告: 图片 {img_id} 在 {image_dir} 中未找到，已跳过")
                            continue
                    
                    lat = row['LAT']
                    lon = row['LON']
                    if pd.isna(lat) or pd.isna(lon):
                        print(f"警告: CSV第{idx_row+1}行（IMG_ID={img_id}）的经纬度为空，已跳过")
                        continue
                    
                    try:
                        lat = float(lat)
                        lon = float(lon)
                    except ValueError:
                        #print(f"警告: CSV第{idx_row+1}行（IMG_ID={img_id}）的经纬度不是有效数字，已跳过")
                        continue
                    
                    coordinates = [[lon, lat]]
                    coordinates = self._convert_coordinates(coordinates)
                    
                    samples.append({
                        'image_path': image_path,
                        'coordinates': coordinates,
                        'source': 'csv'
                    })
            
            except Exception as e:
                print(f"处理CSV文件 {csv_path} 时出错: {str(e)}")
                continue
        
        return samples
    
    def _find_image_by_path_variants(self, image_dir, img_path):
        """
        当直接路径不存在时，尝试查找可能的变体
        支持：大小写不敏感、不同扩展名、在子目录中查找等
        """
        # 分离目录和文件名
        dir_part, file_part = os.path.split(img_path)
        file_name, file_ext = os.path.splitext(file_part)
        
        # 如果原路径有扩展名，只尝试该扩展名的大小写变体
        # 如果没有扩展名，尝试常见图片扩展名
        if file_ext:
            extensions = [file_ext]
        else:
            extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif']
        
        # 构建搜索模式
        search_patterns = []
        for ext in extensions:
            # 原文件名的大小写变体
            search_patterns.extend([
                os.path.join(dir_part, file_name + ext),
                os.path.join(dir_part, file_name.upper() + ext),
                os.path.join(dir_part, file_name.lower() + ext),
            ])
        
        # 递归搜索
        if self.recursive_search:
            for root, _, files in os.walk(os.path.join(image_dir, dir_part) if dir_part else image_dir):
                for file in files:
                    current_path = os.path.join(root, file)
                    relative_path = os.path.relpath(current_path, image_dir)
                    
                    # 检查是否匹配任何搜索模式（大小写不敏感）
                    for pattern in search_patterns:
                        if relative_path.lower() == pattern.lower():
                            return current_path
        else:
            # 非递归搜索
            for pattern in search_patterns:
                full_pattern = os.path.join(image_dir, pattern)
                # 使用glob进行通配符匹配（支持大小写不敏感需要额外处理）
                potential_matches = glob.glob(full_pattern)
                if potential_matches:
                    return potential_matches[0]
                
                # 尝试大小写不敏感的匹配（在Unix系统上可能需要）
                try:
                    # 列出目录内容进行手动匹配
                    search_dir = os.path.join(image_dir, os.path.dirname(pattern))
                    if os.path.exists(search_dir):
                        for file in os.listdir(search_dir):
                            if file.lower() == os.path.basename(pattern).lower():
                                return os.path.join(search_dir, file)
                except:
                    pass
        
        return None
    
    def _augment_samples(self):
        """扩充样本列表：每个原始样本重复augmentation_factor次"""
        augmented_samples = []
        for sample in self.orgsamples:
            # 重复添加同一个样本（坐标相同，图像会在__getitem__中动态增强）
            for _ in range(self.augmentation_factor):
                augmented_samples.append(sample.copy())
        return augmented_samples
    
    def _find_image_path(self, image_dir, image_name):
        """查找图片路径（用于GeoJSON处理，支持递归、大小写不敏感、多种扩展名）"""
        exts = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif']
        possible_names = [image_name + ext for ext in exts] + \
                         [image_name.upper() + ext for ext in exts] + \
                         [image_name.lower() + ext for ext in exts]
        
        if self.recursive_search:
            for root, _, files in os.walk(image_dir):
                for file in files:
                    if file in possible_names:
                        return os.path.join(root, file)
        else:
            for name in possible_names:
                potential_path = os.path.join(image_dir, name)
                if os.path.exists(potential_path):
                    return potential_path
        
        return None
    
    def _convert_coordinates(self, coordinates):
        """将坐标从字符串列表转换为浮点数张量"""
        try:
            def convert_recursive(coord):
                if isinstance(coord, list):
                    return [convert_recursive(c) for c in coord]
                elif isinstance(coord, str):
                    return float(coord)
                return coord
            
            converted = convert_recursive(coordinates)
            return torch.tensor(converted, dtype=torch.float32)
        except Exception as e:
            print(f"转换坐标时出错: {str(e)}")
            return torch.tensor([], dtype=torch.float32)
    
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