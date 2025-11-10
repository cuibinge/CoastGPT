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
    
    def _collect_samples(self):
        """收集所有样本：图片路径和对应的坐标信息"""
        samples = []
        
        # 找到所有geojson文件
        geojson_files = glob.glob(os.path.join(self.root_dir, "*.geojson"))
        print("geojson_files: ",geojson_files)
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





class TimeDataset(Dataset):
    """适配指定GeoJSON格式，仅提取temporal_info中datetime_local的【月份、日、小时】作为时间源"""
    
    def __init__(self, root_dir, image_transform=None):
        self.root_dir = root_dir
        self.image_transform = image_transform
        
        # 保持与原GeoDataset一致的默认图像变换（适配CLIP输入）
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
        
        # 按GeoJSON结构收集（图片路径 + 仅包含月/日/小时的时间戳）
        self.data_samples = self._collect_samples()

    def _collect_samples(self):
        """核心：按GeoJSON层级提取datetime_local，仅保留月/日/小时"""
        samples = []
        geojson_files = glob.glob(os.path.join(self.root_dir, "*.geojson"))
        print(f"找到 {len(geojson_files)} 个GeoJSON文件")

        for geojson_path in geojson_files:
            # 匹配图片文件夹（规则不变：GeoJSON文件名+_Image）
            base_name = os.path.splitext(os.path.basename(geojson_path))[0]
            image_dir = os.path.join(self.root_dir, f"{base_name}_Image")
            
            if not os.path.exists(image_dir):
                print(f"警告: 图片文件夹 {image_dir} 不存在，已跳过")
                continue

            try:
                with open(geojson_path, 'r', encoding='utf-8') as f:
                    geojson_data = json.load(f)

                # 遍历每个图像条目（GeoJSON的data列表）
                for item in geojson_data.get('data', []):
                    image_name = item.get('name')
                    if not image_name:
                        continue

                    # 查找图像文件（支持多种常见格式）
                    image_path = None
                    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                        potential_path = os.path.join(image_dir, f"{image_name}{ext}")
                        if os.path.exists(potential_path):
                            image_path = potential_path
                            break
                    
                    if not image_path:
                        print(f"警告: 图像 {image_name} 在 {image_dir} 中未找到，已跳过")
                        continue

                    # 提取datetime_local（路径：item → features[0] → properties → temporal_info → datetime_local）
                    features = item.get('features', [])
                    if not features:
                        print(f"警告: 图像 {image_name} 无features字段，已跳过")
                        continue
                    
                    properties = features[0].get('properties', {})
                    temporal_info = properties.get('temporal_info', {})
                    datetime_local_str = temporal_info.get('datetime_local')
                    
                    if not datetime_local_str:
                        print(f"警告: 图像 {image_name} 无temporal_info.datetime_local字段，已跳过")
                        continue
                    cleaned_timestamp = self._clean_datetime_local(datetime_local_str)
                    if not cleaned_timestamp:
                        print(f"警告: 图像 {image_name} 的datetime_local {datetime_local_str} 格式无效，已跳过")
                        continue

                    # 新增有效样本（时间戳仅包含月/日/小时信息）
                    samples.append({
                        'image_path': image_path,
                        'timestamp': cleaned_timestamp  # 仅反映月/日/小时的时间戳
                    })

            except Exception as e:
                print(f"处理GeoJSON {geojson_path} 时出错: {str(e)}，已跳过")
                continue

        print(f"最终加载 {len(samples)} 个有效（图像+月/日/小时）样本")
        return samples

    def _clean_datetime_local(self, datetime_local_str):
        """
        仅提取datetime_local中的【月份、日、小时】，忽略年份、分钟、秒
        输入示例："2018-07-20 11:26:57" → 提取 07（月）、20（日）、11（小时）
        输出：基于固定年份（2000年）的月/日/小时时间戳（确保仅月/日/小时影响特征）
        """
        try:
            # 步骤1：去除所有非数字符号，保留纯数字串（如"2018-07-20 11:26:57" → "20180720112657"）
            cleaned = ''.join([c for c in datetime_local_str if c.isdigit()])
            
            # 步骤2：验证长度（需至少包含"年月日时"共10位，如2018072011）
            if len(cleaned) < 10:
                print(f"datetime_local {datetime_local_str} 清理后长度不足10位（需至少包含年月日时）")
                return None
            
            # 步骤3：仅提取月份（4-6位）、日（6-8位）、小时（8-10位）
            # 示例：cleaned为"20180720112657" → 月=07，日=20，小时=11
            month = int(cleaned[4:6])   # 第4-5位（索引4-6）
            day = int(cleaned[6:8])     # 第6-7位（索引6-8）
            hour = int(cleaned[8:10])   # 第8-9位（索引8-10）
            
            # 步骤4：验证月/日/小时的有效性
            if not (1 <= month <= 12):
                print(f"无效月份 {month}（需1-12）")
                return None
            if not (1 <= day <= 31):
                print(f"无效日期 {day}（需1-31）")
                return None
            if not (0 <= hour <= 23):
                print(f"无效小时 {hour}（需0-23）")
                return None
            
            # 步骤5：固定年份为2000年（忽略原年份），生成仅反映月/日/小时的时间戳
            # 原因：确保不同年份的相同月/日/小时生成相同时间特征（如2018-07-20 11点 和 2020-07-20 11点 视为同一时间特征）
            dt = datetime(
                year=2000,  # 固定年份，消除年份影响
                month=month,
                day=day,
                hour=hour,
                minute=0,   # 忽略分钟
                second=0    # 忽略秒
            )
            
            # 步骤6：转换为秒级时间戳（仅体现月/日/小时差异）
            return dt.timestamp()
        
        except Exception as e:
            print(f"清理datetime_local {datetime_local_str} 失败: {str(e)}")
            return None

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

# -------------------------- 测试代码：验证datetime_local提取正确性 --------------------------
if __name__ == "__main__":

    # 替换为你的数据集根目录（需包含GeoJSON和对应_Image文件夹）
    dataset_dir = "/home/ma-user/work/data/caption/images_rgb"
    # 初始化数据集
    dataset = TimeDataset(root_dir=dataset_dir)
    # 初始化数据加载器
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    # 测试1个批次，验证数据格式和时间准确性
    for batch_idx, (images, timestamps) in enumerate(dataloader):
        print(f"\n=== 批次 {batch_idx + 1} 验证结果 ===")
        print(f"1. 图像张量形状: {images.shape} → 符合 (batch_size, 3, 256, 256) 要求")
        print(f"2. 时间戳张量形状: {timestamps.shape} → 符合 (batch_size,) 要求")
        # 验证时间戳转回本地时间是否与原datetime_local一致
        for i in range(len(timestamps)):
            ts = timestamps[i].item()
            dt = datetime.fromtimestamp(ts)  # 按本地时间解析（匹配Asia/Shanghai）
            print(f"3. 样本{i+1}：")
            print(f"   - 秒级时间戳: {ts:.1f}")
            print(f"   - 转回本地时间: {dt.strftime('%Y-%m-%d %H:%M:%S')} → 与原datetime_local格式一致")
        
        # 仅测试1个批次
        if batch_idx == 0:
            break