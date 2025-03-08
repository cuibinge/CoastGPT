from typing import List
import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2




# 该函数用于获取预训练阶段的图像或掩码的变换操作组合
def get_pretrain_transform(size, type) -> A.Compose:
    # 确保输入的类型为 'image' 或 'mask'
    assert type in ["image", "mask"]
    if type == "image":
        # 定义图像的变换操作组合
        transform = A.Compose(
            [
                # 随机裁剪并调整大小，裁剪的比例范围是 0.08 到 1.0
                A.RandomResizedCrop(size[0], size[1], scale=(0.08, 1.0)),
                # 以 0.5 的概率进行水平翻转
                A.HorizontalFlip(p=0.5),
                # 以 0.3 的概率进行 90 度随机旋转
                A.RandomRotate90(p=0.3),
                # 以 0.8 的概率对图像的颜色进行随机调整
                A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                # 以 0.5 的概率对图像进行高斯模糊，模糊核大小在 3 到 7 之间，标准差在 0.1 到 2.0 之间
                A.GaussianBlur(blur_limit=(3, 7), sigma_limit=[0.1, 2.0], p=0.5),
                # 以 0.6 的概率为图像添加高斯噪声
                A.GaussNoise(p=0.6),
                # 以 0.2 的概率对图像进行反色处理
                A.Solarize(p=0.2),
                # 以 0.2 的概率将图像转换为灰度图
                A.ToGray(p=0.2),
                # 对图像进行归一化处理，使用给定的均值和标准差
                A.Normalize(
                    mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407]
                ),
                # 将处理后的图像转换为 PyTorch 张量
                ToTensorV2(),
            ]
        )
    else:
        # 对于掩码，只进行调整大小操作，使用最近邻插值方法
        transform = A.Resize(*size, interpolation=cv2.INTER_NEAREST_EXACT)

    return transform

# 该函数用于获取 BYOL（Bootstrap Your Own Latent）预训练方法的两种图像变换操作组合
def get_pretrain_transform_BYOL(size) -> List[A.Compose]:
    # 定义第一种变换操作组合
    transform1 = A.Compose(
        [
            # 随机裁剪并调整大小，裁剪的比例范围是 0.2 到 1.0
            A.RandomResizedCrop(size[0], size[1], scale=(0.2, 1.0)),
            # 以 0.5 的概率进行水平翻转
            A.HorizontalFlip(p=0.5),
            # 以 0.8 的概率对图像的颜色进行随机调整
            A.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
            # 以 1.0 的概率对图像进行高斯模糊，模糊核大小在 3 到 7 之间，标准差在 0.1 到 2.0 之间
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=[0.1, 2.0], p=1.0),
            # 以 0.6 的概率为图像添加高斯噪声
            A.GaussNoise(p=0.6),
            # 不进行反色处理
            A.Solarize(p=0.0),
            # 以 0.2 的概率将图像转换为灰度图
            A.ToGray(p=0.2),
            # 对图像进行归一化处理，使用给定的均值和标准差
            A.Normalize(mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407]),
            # 将处理后的图像转换为 PyTorch 张量
            ToTensorV2(),
        ]
    )

    # 定义第二种变换操作组合
    transform2 = A.Compose(
        [
            # 随机裁剪并调整大小，裁剪的比例范围是 0.2 到 1.0
            A.RandomResizedCrop(size[0], size[1], scale=(0.2, 1.0)),
            # 以 0.5 的概率进行水平翻转
            A.HorizontalFlip(p=0.5),
            # 以 0.8 的概率对图像的颜色进行随机调整
            A.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
            # 以 0.1 的概率对图像进行高斯模糊，模糊核大小在 3 到 7 之间，标准差在 0.1 到 2.0 之间
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=[0.1, 2.0], p=0.1),
            # 以 0.6 的概率为图像添加高斯噪声
            A.GaussNoise(p=0.6),
            # 以 0.2 的概率对图像进行反色处理
            A.Solarize(p=0.2),
            # 以 0.2 的概率将图像转换为灰度图
            A.ToGray(p=0.2),
            # 对图像进行归一化处理，使用给定的均值和标准差
            A.Normalize(mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407]),
            # 将处理后的图像转换为 PyTorch 张量
            ToTensorV2(),
        ]
    )

    # 将两种变换操作组合存储在列表中
    transforms = [transform1, transform2]
    return transforms

# 该函数用于获取训练阶段的图像变换操作组合
def get_train_transform(size) -> A.Compose:
    # 定义训练阶段的图像变换操作组合
    transform = A.Compose(
        [
            # 随机裁剪并调整大小，裁剪的比例范围是 0.2 到 1.0
            A.RandomResizedCrop(size[0], size[1], scale=(0.2, 1.0)),
            # 以 0.5 的概率进行水平翻转
            A.HorizontalFlip(p=0.5),
            # 以 0.5 的概率进行 90 度随机旋转
            A.RandomRotate90(p=0.5),
            # 以 0.5 的概率为图像添加高斯噪声
            A.GaussNoise(p=0.5),
            # 以 0.3 的概率对图像进行高斯模糊，模糊核大小为 3，标准差在 1.5 到 1.5 之间
            A.GaussianBlur((3, 3), (1.5, 1.5), p=0.3),
            # 对图像进行归一化处理，使用给定的均值和标准差
            A.Normalize(mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407]),
            # 将处理后的图像转换为 PyTorch 张量
            ToTensorV2(),
        ]
    )

    return transform

# 该函数用于获取测试阶段的图像变换操作组合
def get_test_transform(size) -> A.Compose:
    # 定义测试阶段的图像变换操作组合
    transform = A.Compose(
        [
            # 调整图像大小
            A.Resize(size[0], size[1]),
            # 对图像进行归一化处理，使用给定的均值和标准差
            A.Normalize(mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407]),
            # 将处理后的图像转换为 PyTorch 张量
            ToTensorV2(),
        ]
    )

    return transform