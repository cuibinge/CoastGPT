import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob

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
    
    def _collect_samples(self):
        """收集所有样本：图片路径和对应的坐标信息"""
        samples = []
        
        # 找到所有geojson文件
        geojson_files = glob.glob(os.path.join(self.root_dir, "*.geojson"))
        
        for geojson_path in geojson_files:
            # 提取对应的图片文件夹名称
            base_name = os.path.splitext(os.path.basename(geojson_path))[0]
            image_dir = os.path.join(self.root_dir, f"{base_name}_Image")
            
            # 检查图片文件夹是否存在
            if not os.path.exists(image_dir):
                print(f"警告: 图片文件夹 {image_dir} 不存在，已跳过")
                continue
            
            # 解析geojson文件
            try:
                with open(geojson_path, 'r', encoding='utf-8') as f:
                    geojson_data = json.load(f)
                
                # 提取每个图片的信息
                for item in geojson_data.get('data', []):
                    image_name = item.get('name')
                    if not image_name:
                        continue
                    
                    # 查找对应的图片文件（处理可能的扩展名）
                    image_path = None
                    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                        potential_path = os.path.join(image_dir, f"{image_name}{ext}")
                        if os.path.exists(potential_path):
                            image_path = potential_path
                            break
                    
                    if not image_path:
                        print(f"警告: 图片 {image_name} 在 {image_dir} 中未找到，已跳过")
                        continue
                    
                    # 提取坐标信息
                    features = item.get('features', [])
                    if features and len(features) > 0:
                        coordinates = features[0].get('geometry', {}).get('coordinates', [])
                        # 将坐标从字符串转换为浮点数
                        coordinates = self._convert_coordinates(coordinates)
                        
                        # 添加到样本列表
                        samples.append({
                            'image_path': image_path,
                            'coordinates': coordinates
                        })
            
            except Exception as e:
                print(f"处理 {geojson_path} 时出错: {str(e)}")
                continue
        
        return samples
    
    def _convert_coordinates(self, coordinates):
        """将坐标从字符串列表转换为浮点数张量"""
        try:
            # 递归转换所有字符串坐标为浮点数
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
        """返回数据集中样本的数量"""
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        """
        获取指定索引的样本
        Returns:
            tuple: (image, coordinates)，分别为处理后的图片张量和坐标张量
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data_samples[idx]
        
        # 加载图片
        image = Image.open(sample['image_path']).convert('RGB')
        
        # 应用变换
        if self.image_transform:
            image = self.image_transform(image)
        
        # 获取坐标
        coordinates = sample['coordinates']
        
        return image, coordinates

# 使用示例
# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
    
#     # 数据集根目录
#     dataset_dir = "/home/ma-user/work/data/caption/images_rgb"  # 替换为实际的数据集路径
    
#     # 创建数据集实例
#     dataset = GeoDataset(root_dir=dataset_dir)
    
#     # 创建数据加载器
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    
#     # 测试数据加载
#     for batch_idx, (images, coordinates) in enumerate(dataloader):
#         print(f"批次 {batch_idx}:")
#         print(f"  图片形状: {images.shape}")
#         print(f"  坐标形状: {coordinates.shape}")
#         print(f"  坐标: {coordinates[0][0]}")

        
#         # 只测试一个批次
#         if batch_idx == 0:
#             break