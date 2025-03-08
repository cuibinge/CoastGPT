import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from .transform import *


# 定义基础数据集类，继承自 torch.utils.data.Dataset
class BaseDataset(Dataset):
    def __init__(
        self,
        size: Union[Tuple, List],  # 图像的目标尺寸
        data_root: Optional[Union[str, Path]],  # 数据根目录
        label_root: Optional[Union[str, Path]] = None,  # 标签根目录
        mode: str = "pretrain",  # 数据集模式，如预训练、验证、训练、测试
        txt_file: bool = False,  # 是否使用文本文件指定图像列表
        txt_file_dir: Optional[Union[str, Path]] = None,  # 文本文件所在目录
        custom_transform: Callable = None,  # 自定义的图像变换函数
    ):
        super(BaseDataset, self).__init__()
        # 确保模式为合法值
        assert mode.lower() in ["pretrain", "val", "train", "test"]
        # 如果使用文本文件，确保文本文件目录已指定
        if txt_file:
            assert txt_file_dir is not None
        # 如果是训练或验证模式，确保标签根目录已指定
        if mode.lower() in ["train", "val"]:
            assert label_root is not None

        self.data_root = Path(data_root)  # 将数据根目录转换为 Path 对象
        self.mode = mode.lower()  # 将模式转换为小写
        self.size = size  # 存储图像目标尺寸
        if label_root is not None and self.mode in ["train", "val"]:
            self.label_root = Path(label_root)  # 将标签根目录转换为 Path 对象
        if txt_file:
            self.txt_file_dir = Path(txt_file_dir)  # 将文本文件目录转换为 Path 对象
            txt_file_name = mode + ".txt"  # 生成文本文件名称
            self.txt_file_dir = self.txt_file_dir / txt_file_name  # 构建完整的文本文件路径

        self.img_list = []  # 存储图像文件名列表
        if txt_file:
            f = open(self.txt_file_dir, "r")  # 打开文本文件
            for line in f.readlines():
                self.img_list.append(line.strip("\n") + ".png")  # 读取文本文件中的图像文件名
            f.close()
        else:
            self.img_list = os.listdir(self.data_root)  # 直接获取数据根目录下的所有文件名

        if custom_transform is not None:
            self.transform = custom_transform  # 使用自定义的图像变换函数
        elif self.mode in ["pretrain"]:
            self.transform = get_pretrain_transform_BYOL(size)  # 预训练模式使用特定的变换
        elif self.mode in ["train"]:
            self.transform = get_train_transform(size)  # 训练模式使用特定的变换
        else:
            self.transform = get_test_transform(size)  # 其他模式使用测试变换

    def __len__(self) -> int:
        return len(self.img_list)  # 返回图像列表的长度

    def __getitem__(self, idx: int) -> Dict:
        img_name = self.img_list[idx]  # 获取指定索引的图像文件名
        img = Image.open(self.data_root / img_name)  # 打开图像文件
        img = np.asarray(img)  # 将图像转换为 NumPy 数组
        if self.mode in ["train", "val"]:
            label = Image.open(self.label_root / img_name)  # 打开对应的标签文件
            label = np.asarray(label)  # 将标签转换为 NumPy 数组

        if self.mode == "pretrain":
            # 预训练模式下，对图像进行两次不同的变换
            view1 = self.transform[0](image=img)["image"].type(torch.float32)
            view2 = self.transform[1](image=img)["image"].type(torch.float32)
            img_dict = dict(view1=view1, view2=view2)  # 构建包含两个视图的字典
        elif self.mode in ["train", "val"]:
            # 训练或验证模式下，对图像和标签同时进行变换
            ts = self.transform(image=img, mask=label)
            img = ts["image"]
            label = ts["mask"]
            img_dict = dict(img=img, label=label)  # 构建包含图像和标签的字典
        else:
            # 其他模式下，只对图像进行变换
            ts = self.transform(image=img)
            img = ts["image"]
            img_dict = dict(img=img)  # 构建包含图像的字典

        return img_dict


# 定义基础掩码数据集类，继承自 BaseDataset
class BaseMaskDataset(BaseDataset):
    def __init__(
        self,
        grid_size: int = 7,  # 网格大小
        input_size: Union[List, Tuple] = [224, 224],  # 输入图像尺寸
        crop_size: Union[List, Tuple] = [224, 224],  # 裁剪后的图像尺寸
        crop_num: int = 2,  # 裁剪的数量
        **kwargs,
    ) -> None:
        super(BaseMaskDataset, self).__init__(**kwargs)
        # 确保模式为预训练模式
        assert self.mode == "pretrain", "BaseMaskDataset Only Support to mask dataset"
        self.input_size = input_size  # 存储输入图像尺寸
        self.crop_size = crop_size  # 存储裁剪后的图像尺寸
        self.crop_num = crop_num  # 存储裁剪数量
        self.grid_size = grid_size  # 存储网格大小
        self.transform = get_pretrain_transform(self.crop_size, type="image")  # 获取图像变换函数
        self.interpolate = get_pretrain_transform(self.crop_size, type="mask")  # 获取掩码变换函数

    def __getitem__(self, idx: int) -> Dict:
        img_name = self.img_list[idx]  # 获取指定索引的图像文件名
        img = Image.open(self.data_root / img_name).resize(self.input_size)  # 打开图像并调整大小
        img = np.asarray(img)  # 将图像转换为 NumPy 数组
        images = []  # 存储裁剪后的图像列表
        masks = []  # 存储对应的掩码列表
        for _ in range(self.crop_num):
            crop = img.copy()  # 复制图像
            # 生成网格掩码
            mask = np.arange(self.grid_size * self.grid_size, dtype=np.uint8).reshape(
                self.grid_size, self.grid_size
            )
            mask = self.interpolate(image=mask)["image"]  # 对掩码进行变换
            transformed = self.transform(image=crop, mask=mask)  # 对图像和掩码同时进行变换
            crop, mask = (
                transformed["image"].type(torch.float32),
                transformed["mask"].float(),
            )  # 获取变换后的图像和掩码
            # 对掩码进行插值操作
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(1), size=56, mode="nearest"
            ).squeeze()
            images.append(crop), masks.append(mask)  # 将图像和掩码添加到列表中

        return dict(views=images, masks=masks)  # 构建包含图像和掩码的字典


# 定义 Potsdam 数据集类，继承自 BaseDataset
class Potsdam(BaseDataset):
    # 定义类别名称列表
    CLASSES = [
        "Impervious surfaces",
        "Building",
        "Low vegetation",
        "Tree",
        "Car",
        "background",
    ]
    # 定义调色板，用于可视化标签
    PALETTE = [
        [255, 255, 255],
        [0, 0, 255],
        [0, 255, 255],
        [0, 255, 0],
        [255, 255, 0],
        [255, 0, 0],
    ]

    def __init__(
        self,
        size: Union[Tuple, List],  # 图像的目标尺寸
        data_dir: str,  # 数据目录
        mode: str,  # 数据集模式
        custom_transform: Callable = None,  # 自定义的图像变换函数
    ):
        data_dir = Path(data_dir)  # 将数据目录转换为 Path 对象
        super(Potsdam, self).__init__(
            size=size,
            data_root=data_dir / "Image",  # 数据根目录为 Image 子目录
            label_root=data_dir / "Label",  # 标签根目录为 Label 子目录
            mode=mode,
            txt_file=True,  # 使用文本文件指定图像列表
            txt_file_dir=data_dir,  # 文本文件所在目录
            custom_transform=custom_transform,
        )


# 定义 Potsdam 掩码数据集类，继承自 BaseMaskDataset
class PotsdamMask(BaseMaskDataset):
    def __init__(
        self,
        size: Union[Tuple, List],  # 图像的目标尺寸
        data_dir: str,  # 数据目录
        mode: str,  # 数据集模式
        custom_transform: Callable = None,  # 自定义的图像变换函数
    ):
        data_dir = Path(data_dir)  # 将数据目录转换为 Path 对象
        super(PotsdamMask, self).__init__(
            size=size,
            data_root=data_dir / "Image",  # 数据根目录为 Image 子目录
            label_root=data_dir / "Label",  # 标签根目录为 Label 子目录
            mode=mode,
            txt_file=True,  # 使用文本文件指定图像列表
            txt_file_dir=data_dir,  # 文本文件所在目录
            custom_transform=custom_transform,
        )


# 定义 LoveDA 数据集函数
def LoveDA(
    size: Union[Tuple, List],  # 图像的目标尺寸
    root_dir: str,  # 数据集根目录
    mode: str,  # 数据集模式
    custom_transform: Callable = None,  # 自定义的图像变换函数
) -> Optional[ConcatDataset]:
    root_dir = Path(root_dir)  # 将数据集根目录转换为 Path 对象
    dataset = None  # 初始化数据集
    if mode == "pretrain":
        split = ["Train", "Val", "Test"]  # 预训练模式使用所有分割数据
    elif mode == "train":
        split = ["train"]  # 训练模式使用训练分割数据
    elif mode == "val":
        split = ["val"]  # 验证模式使用验证分割数据
    else:
        split = ["test"]  # 其他模式使用测试分割数据

    for name in ["Urban", "Rural"]:  # 遍历城市和乡村数据
        for s in split:  # 遍历分割数据
            dir = root_dir / s / name  # 构建子目录路径
            sub_dataset = BaseMaskDataset(
                size=size,
                data_root=dir / "images",  # 数据根目录为 images 子目录
                label_root=dir / "annfiles",  # 标签根目录为 annfiles 子目录
                mode=mode,
                custom_transform=custom_transform,
            )  # 创建子数据集
            if dataset is None:
                dataset = sub_dataset  # 如果数据集为空，直接赋值
            else:
                dataset += sub_dataset  # 否则将子数据集合并到主数据集中

    return dataset