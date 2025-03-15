import ml_collections
import PIL
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from transformers import CLIPImageProcessor

# 定义四通道数据的均值和标准差
IMAGENET_FOUR_CHANNEL_DEFAULT_MEAN = IMAGENET_DEFAULT_MEAN + (0.5,)
IMAGENET_FOUR_CHANNEL_DEFAULT_STD = IMAGENET_DEFAULT_STD + (0.5,)

def build_cls_transform(config, is_train=True, num_channels=3):
    """
    构建用于分类任务的图像变换。

    参数:
    config (ml_collections.ConfigDict): 配置字典，包含图像变换的相关参数。
    is_train (bool): 是否为训练阶段，默认为True。
    num_channels (int): 图像的通道数，默认为3。

    返回:
    torchvision.transforms.Compose: 图像变换组合。
    """
    # 根据通道数选择均值和标准差
    if num_channels == 3:
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    elif num_channels == 4:
        mean = IMAGENET_FOUR_CHANNEL_DEFAULT_MEAN
        std = IMAGENET_FOUR_CHANNEL_DEFAULT_STD
    else:
        raise ValueError(f"Unsupported number of channels: {num_channels}")

    if is_train:
        transform = create_transform(
            input_size=config.transform.input_size,
            is_training=True,
            color_jitter=config.color_jitter,
            auto_augment=config.aa,
            interpolation="bicubic",
            re_prob=config.reprob,  # re means random erasing
            re_mode=config.remode,
            re_count=config.recount,
            mean=mean,
            std=std,
        )
        return transform

    t = []
    crop_pct = 224 / 256
    size = int(config.transform.input_size[0] / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(config.transform.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def build_vlp_transform(config: ml_collections.ConfigDict, is_train: bool = True, num_channels=3):
    """
    构建用于视觉语言预训练（VLP）任务的图像变换。

    参数:
    config (ml_collections.ConfigDict): 配置字典，包含图像变换的相关参数。
    is_train (bool): 是否为训练阶段，默认为True。
    num_channels (int): 图像的通道数，默认为3。

    返回:
    torchvision.transforms.Compose 或 CLIPImageProcessor: 图像变换组合或CLIP图像处理器。
    """
    if config.rgb_vision.arch.startswith("vit"):
        return CLIPImageProcessor.from_pretrained(config.rgb_vision.vit_name)

    # 根据通道数选择均值和标准差
    if num_channels == 3:
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    elif num_channels == 4:
        mean = IMAGENET_FOUR_CHANNEL_DEFAULT_MEAN
        std = IMAGENET_FOUR_CHANNEL_DEFAULT_STD
    else:
        raise ValueError(f"Unsupported number of channels: {num_channels}")

    if is_train:
        transform = create_transform(
            is_training=True,
            input_size=config.transform.input_size,
            auto_augment=config.transform.rand_aug,
            interpolation="bicubic",
            mean=mean,
            std=std,
        )
        return transform

    t = []
    crop_pct = 224 / 256
    size = int(config.transform.input_size[0] / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(config.transform.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)