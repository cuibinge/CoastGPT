import logging
import math

import braceexpand
import ml_collections
import torch
import webdataset as wds
from torch.utils.data import DataLoader, DistributedSampler

from ..CustomTrainer.utils import InfiniteSampler
from .build_transform import build_cls_transform, build_vlp_transform
from .cap_dataset import (
    CaptionDatasetVQA,
    DataCollatorForSupervisedDataset,
    InstructDataset,
    InstructDatasetWithTaskId,
    RS5MDataset,
)
from .ImageFolderInstance import ImageFolderInstance
from .meterml import METERMLDataset
from .UCM import UCM

logger = logging.getLogger("train")


def build_loader_hepler(
    config: ml_collections.ConfigDict,
    dataset: torch.utils.data.Dataset,
    collate_fn=None,
    is_train: bool = True,
):
    """
    辅助函数，用于构建数据加载器。

    参数:
    config (ml_collections.ConfigDict): 配置字典，包含数据加载的相关参数。
    dataset (torch.utils.data.Dataset): 数据集对象。
    collate_fn (callable, optional): 用于合并样本列表以形成小批量的函数。
    is_train (bool, optional): 是否为训练阶段，默认为True。

    返回:
    torch.utils.data.DataLoader: 数据加载器对象。
    """
    if config.is_distribute:
        sampler = DistributedSampler(dataset, shuffle=True)
    elif config.inf_sampler and is_train:
        sampler = InfiniteSampler(dataset, shuffle=True)
    else:
        sampler = None

    if is_train and sampler is None:
        drop_last = True
        shuffer = True
    else:
        drop_last = False
        shuffer = False

    dataloader = DataLoader(
        dataset,
        config.batch_size,
        sampler=sampler,
        num_workers=config.workers,
        pin_memory=True,
        drop_last=drop_last,
        shuffle=shuffer,
        collate_fn=collate_fn,
    )

    return dataloader


def build_vlp_loader(
    config: ml_collections.ConfigDict, is_train: bool = True, num_channels=3, **kwargs
):
    """
    构建用于视觉语言预训练（VLP）的数据集加载器。

    参数:
    config (ml_collections.ConfigDict): 配置字典，包含数据加载的相关参数。
    is_train (bool, optional): 是否为训练阶段，默认为True。
    num_channels (int, optional): 图像的通道数，默认为3。
    **kwargs: 其他关键字参数。

    返回:
    torch.utils.data.DataLoader 或 wds.WebLoader: 数据加载器对象。
    """
    # 构建VLP数据变换，传递通道数信息
    transform = build_vlp_transform(config, is_train=is_train, num_channels=num_channels)
    logger.info(f"Evaluate data transform:\n{transform}")

    if is_train and config.stage == 1:
        if "RS5M" in config.data_path:
            dataset = RS5MDataset(root=config.data_path, transform=transform, **kwargs)
        else:
            dataset = CaptionDatasetVQA(
                root=config.data_path, transform=transform, **kwargs
            )
    elif is_train and config.stage >= 2:
        if not config.weight_sample:
            dataset = InstructDataset(
                root=config.data_path,
                transform=transform,
                crop_size=config.transform.input_size[0],
                **kwargs,
            )
        else:
            dataset = InstructDatasetWithTaskId(
                root=config.data_path,
                transform=transform,
                crop_size=config.transform.input_size[0],
                **kwargs,
            )
            from torch.utils.data import WeightedRandomSampler
            from .utils import DistributedSamplerWrapper

            weight_sampler = WeightedRandomSampler(
                dataset.sample_weight, num_samples=len(dataset), replacement=False
            )
            distribute_weight_sampler = DistributedSamplerWrapper(weight_sampler)
            loader = DataLoader(
                dataset,
                sampler=distribute_weight_sampler,
                batch_size=config.batch_size,
                num_workers=config.workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=DataCollatorForSupervisedDataset(
                    tokenizer=kwargs["tokenizer"]
                ),
            )
            return loader

    if "RS5M" in config.data_path:
        if config.is_distribute:
            dataset, url = dataset
            dataset.extend(
                [
                    wds.batched(
                        config.batch_size,
                        partial=False,
                        collation_fn=DataCollatorForSupervisedDataset(
                            tokenizer=kwargs["tokenizer"]
                        ),
                    )
                ]
            )
            dataset = wds.DataPipeline(*dataset)
            num_shards = len(expand_urls(url)[0])
            assert (
                num_shards >= config.workers * config.world_size
            ), "number of shards must be >= total workers"

            round_fn = math.ceil
            global_batch_size = config.batch_size * config.world_size
            num_samples = 5070186
            num_batches = round_fn(num_samples / global_batch_size)

            num_workers = max(1, config.workers)
            num_worker_batches = round_fn(
                num_batches / num_workers
            )  # per dataloader worker
            num_batches = num_worker_batches * num_workers
            num_samples = num_batches * global_batch_size
            dataset = dataset.with_epoch(
                num_worker_batches
            )  # each worker is iterating over this

            loader = wds.WebLoader(
                dataset,
                num_workers=config.workers,
                batch_size=None,
                shuffle=False,
                persistent_workers=config.workers > 0,
            )
            loader.num_batches = num_batches
            loader.num_samples = num_samples
            loader.length = round_fn(num_samples / global_batch_size)
            return loader
    else:
        logger.info(f"Build dataset: Train images = {len(dataset)}")
        dataloader = build_loader_hepler(
            config,
            dataset,
            is_train=is_train,
            collate_fn=DataCollatorForSupervisedDataset(tokenizer=kwargs["tokenizer"]),
        )
        logger.info(f"Build dataloader: Epoch length = {len(dataloader)}")
        return dataloader


def build_zero_shot_loader(
    config: ml_collections.ConfigDict, mode: str = "zero_shot_cls", num_channels=3
):
    """
    构建零样本学习的数据加载器。

    参数:
    config (ml_collections.ConfigDict): 配置字典，包含数据加载的相关参数。
    mode (str, optional): 数据加载器的模式，默认为"zero_shot_cls"。
    num_channels (int, optional): 图像的通道数，默认为3。

    返回:
    torch.utils.data.DataLoader: 数据加载器对象。
    """
    assert mode in ["zero_shot_cls", "zero_shot_retrieval"], (
        "Please choose mode for dataloder from [zero_shot_cls, " "zero_shot_retrieval]"
    )
    if mode == "zero_shot_cls":
        # 构建分类数据变换，传递通道数信息
        transform = build_cls_transform(config, is_train=False, num_channels=num_channels)
        if config.eval.dataset == "UCM":
            dataset = UCM(
                config.data_path, split="all", transform=transform, return_idx=False
            )
        elif config.eval.dataset == "METERML":
            dataset = METERMLDataset(
                root=config.data_path,
                split="test",
                mode="naip_rgb",
                transform=transform,
            )
        else:
            dataset = ImageFolderInstance(
                dataset_name=config.eval.dataset,
                return_index=False,
                root=config.data_path,
                transform=transform,
            )
    else:
        raise NotImplementedError("Zero-shot retrieval not implemented")
        # transform = build_vlp_transform(config, is_train=False)

    dataloader = build_loader_hepler(config, dataset, is_train=False)
    logger.info(f"Build dataloader: Epoch length = {len(dataloader)}")
    return dataloader


def build_loader(
    config: ml_collections.ConfigDict,
    mode: str = "pretrain",
    is_train: bool = True,
    num_channels=3,
    **kwargs,
):
    """
    构建数据加载器的主函数。

    参数:
    config (ml_collections.ConfigDict): 配置字典，包含数据加载的相关参数。
    mode (str, optional): 数据加载器的模式，默认为"pretrain"。
    is_train (bool, optional): 是否为训练阶段，默认为True。
    num_channels (int, optional): 图像的通道数，默认为3。
    **kwargs: 其他关键字参数。

    返回:
    torch.utils.data.DataLoader 或 wds.WebLoader: 数据加载器对象。
    """
    assert mode in [
        "pretrain",
    ], "Please choose mode for dataloder from [pretrain]"
    if mode == "pretrain":
        return build_vlp_loader(config, is_train=is_train, num_channels=num_channels, **kwargs)


def expand_urls(urls, weights=None):
    """
    扩展URL列表。

    参数:
    urls (str 或 list): URL列表或单个URL字符串。
    weights (str 或 list, optional): 每个URL的权重列表或单个权重字符串。

    返回:
    tuple: 扩展后的URL列表和对应的权重列表。
    """
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split("::")
        assert len(weights) == len(
            urllist
        ), f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights