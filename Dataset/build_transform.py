import ml_collections
import PIL
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from transformers import CLIPImageProcessor
# Keep NPU runtime optional for CUDA and CPU environments.
try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

# Normalization constants for four-channel imagery.
IMAGENET_FOUR_CHANNEL_DEFAULT_MEAN = IMAGENET_DEFAULT_MEAN + (0.5,)
IMAGENET_FOUR_CHANNEL_DEFAULT_STD = IMAGENET_DEFAULT_STD + (0.5,)

def build_cls_transform(config, is_train=True, num_channels=3):
    """


    """
    if num_channels == 3:
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    elif num_channels == 4:
        mean = IMAGENET_FOUR_CHANNEL_DEFAULT_MEAN
        std = IMAGENET_FOUR_CHANNEL_DEFAULT_STD
    else:
        raise ValueError(f"Unsupported number of channels: {num_channels}")

    if is_train:
        input_size = getattr(config.transform, 'default_input_size', None) \
                  or getattr(config.transform, 'input_size', [224, 224])
        transform = create_transform(
            input_size=input_size,
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
    input_size = getattr(config.transform, 'default_input_size', None) \
              or getattr(config.transform, 'input_size', [224, 224])

    crop_pct = getattr(config.transform, 'crop_pct', None)
    if crop_pct is not None:
        resize_size = int(input_size[0] / crop_pct)
    else:
        resize_size = input_size[0]

    t.append(transforms.Resize(resize_size, interpolation=PIL.Image.BICUBIC))

    if crop_pct is not None:
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def build_vlp_transform(config: ml_collections.ConfigDict, is_train: bool = True, num_channels=3):
    """


    """
    if config.rgb_vision.arch.startswith("vit"):
        return CLIPImageProcessor.from_pretrained(config.rgb_vision.vit_name)

    if num_channels == 3:
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    elif num_channels == 4:
        mean = IMAGENET_FOUR_CHANNEL_DEFAULT_MEAN
        std = IMAGENET_FOUR_CHANNEL_DEFAULT_STD
    else:
        raise ValueError(f"Unsupported number of channels: {num_channels}")

    if is_train:
        input_size = getattr(config.transform, 'default_input_size', None) \
                  or getattr(config.transform, 'input_size', [224, 224])
        transform = create_transform(
            is_training=True,
            input_size=input_size,
            auto_augment=config.transform.rand_aug,
            interpolation="bicubic",
            mean=mean,
            std=std,
        )
        return transform

    t = []
    input_size = getattr(config.transform, 'default_input_size', None) \
              or getattr(config.transform, 'input_size', [224, 224])

    crop_pct = getattr(config.transform, 'crop_pct', None)
    if crop_pct is not None:
        resize_size = int(input_size[0] / crop_pct)
    else:
        resize_size = input_size[0]

    t.append(transforms.Resize(resize_size, interpolation=PIL.Image.BICUBIC))

    if crop_pct is not None:
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
