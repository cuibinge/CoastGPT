import json
import logging
import math
from pathlib import Path

import braceexpand
import ml_collections
import torch
import webdataset as wds
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from Trainer.utils import InfiniteSampler
from .build_transform import build_cls_transform, build_vlp_transform
from .cap_dataset import (
    CaptionDatasetVQA,
    Cut300SegDataset,
    DataCollatorForSupervisedDataset,
    InstructDatasetWithTaskId,
    PromptConditionedSegDataset,
    RS5MDataset,
)
from .ImageFolderInstance import ImageFolderInstance
from .meterml import METERMLDataset
from .UCM import UCM

logger = logging.getLogger("train")


CUT300_IMAGE_DIR_CANDIDATES = ("images", "image", "img", "images_", "image_", "img_")
CUT300_GT_DIR_CANDIDATES = ("gt", "groundtruth", "mask", "masks", "gt_", "groundtruth_", "mask_", "masks_")


def _resolve_cut300_split_dir(root: Path, split: str, kind: str):
    return _resolve_cut300_split_dir_by_files(root, split, kind, file_names=None)


def _resolve_cut300_split_dir_by_files(root: Path, split: str, kind: str, file_names=None):
    if kind == "image":
        candidates = CUT300_IMAGE_DIR_CANDIDATES
    elif kind == "gt":
        candidates = CUT300_GT_DIR_CANDIDATES
    else:
        raise ValueError(f"Unsupported cut300 dir kind: {kind}")

    split_root = root / split
    existing = []
    for name in candidates:
        candidate = split_root / name
        if candidate.is_dir():
            existing.append(candidate)
    if not existing:
        return None
    if not file_names:
        return existing[0]

    best_dir = None
    best_score = -1
    best_file_count = -1
    for candidate in existing:
        score = _count_cut300_matches(candidate, file_names)
        file_count = sum(1 for p in candidate.iterdir() if p.is_file())
        if score > best_score or (score == best_score and file_count > best_file_count):
            best_dir = candidate
            best_score = score
            best_file_count = file_count
    return best_dir


def _count_cut300_matches(directory: Path, file_names) -> int:
    if directory is None or file_names is None:
        return 0
    return sum((directory / file_name).is_file() for file_name in file_names)


def _has_cut300_split_layout(split_dir: Path) -> bool:
    if (not split_dir.is_dir()) or (not (split_dir / "annotation.json").is_file()):
        return False
    has_image_dir = any((split_dir / name).is_dir() for name in CUT300_IMAGE_DIR_CANDIDATES)
    has_gt_dir = any((split_dir / name).is_dir() for name in CUT300_GT_DIR_CANDIDATES)
    return has_image_dir and has_gt_dir


def _has_cut300_layout(root: Path) -> bool:
    return (
        root.is_dir()
        and (root / "train" / "annotation.json").is_file()
        and _resolve_cut300_split_dir(root, "train", "image") is not None
        and _resolve_cut300_split_dir(root, "train", "gt") is not None
    )


def _infer_cut300_root_from_data_path(path_value):
    raw_path = Path(str(path_value)).expanduser()
    if _has_cut300_layout(raw_path):
        return raw_path
    if _has_cut300_split_layout(raw_path):
        return raw_path.parent
    train_candidate = raw_path / "train"
    if _has_cut300_split_layout(train_candidate):
        return raw_path
    return None


def _resolve_cut300_root(path_value) -> Path:
    raw_root = Path(str(path_value)).expanduser()
    inferred_root = _infer_cut300_root_from_data_path(raw_root)
    if inferred_root is not None:
        return inferred_root
    candidates = []
    seen = set()

    def _add_candidate(path_obj: Path):
        try:
            key = str(path_obj.resolve(strict=False))
        except Exception:
            key = str(path_obj)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path_obj)

    _add_candidate(raw_root)
    _add_candidate(raw_root.parent)
    _add_candidate(raw_root.parent / "cut_300")

    for probe_root in (raw_root, raw_root.parent, raw_root.parent.parent, raw_root.parent.parent.parent):
        if not probe_root.exists() or not probe_root.is_dir():
            continue
        _add_candidate(probe_root / "cut_300")
        for child_name in ("cut_300", "cut300"):
            _add_candidate(probe_root / child_name)
        for child in probe_root.iterdir():
            if child.is_dir() and child.name.lower() in {"cut_300", "cut300"}:
                _add_candidate(child)
                _add_candidate(child / "cut_300")

    for candidate in candidates:
        if _has_cut300_layout(candidate):
            return candidate

    attempted = [str(candidate / "train" / "annotation.json") for candidate in candidates]
    raise FileNotFoundError(
        "Unable to resolve a valid cut300 root from "
        f"'{path_value}'. Expected a directory containing train/annotation.json, "
        "and a train image dir plus ground-truth dir. Tried: "
        f"{attempted}"
    )


def _config_to_plain(value):
    if hasattr(value, "to_dict"):
        return _config_to_plain(value.to_dict())
    if isinstance(value, dict):
        return {k: _config_to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_config_to_plain(v) for v in value]
    return value


def _expand_path(base_root: Path, path_value):
    if path_value is None:
        return None
    path_obj = Path(str(path_value)).expanduser()
    if path_obj.is_absolute():
        return path_obj
    return base_root / path_obj


def _load_annotation_file_names(
    ann_json: Path,
    annotation_image_key: str = "images",
    annotation_file_key: str = "file_name",
):
    with open(ann_json, "r", encoding="utf-8") as f:
        ann = json.load(f)
    ann_items = ann.get(annotation_image_key, [])
    if isinstance(ann_items, dict):
        ann_items = list(ann_items.values())
    elif not isinstance(ann_items, list):
        ann_items = []
    file_names = [
        str(item.get(annotation_file_key, "")).strip()
        for item in ann_items
        if isinstance(item, dict) and str(item.get(annotation_file_key, "")).strip()
    ]
    return file_names, len(ann_items)


def _build_seg_dataset_entries(config: ml_collections.ConfigDict):
    entries = []
    raw_entries = getattr(config, "seg_datasets", None)
    if raw_entries:
        for raw_entry in raw_entries:
            entry = _config_to_plain(raw_entry)
            if not entry:
                continue
            if not bool(entry.get("enabled", True)):
                continue
            if (not entry.get("root", entry.get("path", None))) and (not entry.get("ann_json", None)):
                logger.warning(
                    "Skip an empty seg_datasets entry because both root/path and ann_json are missing: %s",
                    entry,
                )
                continue
            entries.append(entry)

    has_named_cut300 = any(str(entry.get("name", "")).lower() == "cut300" for entry in entries)
    if getattr(config, "cut300_path", None) and not has_named_cut300:
        entries.append(
            {
                "name": "cut300",
                "root": getattr(config, "cut300_path"),
                "split": "train",
                "task_text": "element extraction",
                "element_text": "aquaculture region",
                "task_id": 4,
                "category_id": 1,
                "physical_prompt": "[Dataset: cut300] [Band: RGB] [Target: aquaculture]",
                "_legacy_cut300": True,
            }
        )

    has_named_coastline = any(str(entry.get("name", "")).lower() == "coastline" for entry in entries)
    coastline_source = getattr(config, "coastline_seg_path", None)
    if coastline_source and not has_named_coastline:
        entries.append(
            {
                "name": "coastline",
                "root": coastline_source,
                "split": "train",
                "task_text": "element extraction",
                "element_text": "coastline",
                "task_id": 4,
                "category_id": 5,
                "mask_dilate_kernel": 3,
                "physical_prompt": "[Dataset: coastline] [Band: RGB] [Target: coastline]",
                "question_templates": [
                    "[seg]Please segment the coastline in this remote sensing image. <image>",
                    "[seg]Extract the shoreline mask from this image. <image>",
                    "[seg]Identify the coastline pixels in the scene. <image>",
                ],
            }
        )

    deduped = []
    seen = set()
    for entry in entries:
        signature = (
            str(entry.get("name", "")),
            str(entry.get("root", entry.get("path", ""))),
            str(entry.get("ann_json", "")),
            str(entry.get("image_dir", entry.get("img_dir", ""))),
            str(entry.get("gt_dir", "")),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(entry)
    return deduped


def _build_seg_dataset_from_entry(
    entry: dict,
    transform,
    config: ml_collections.ConfigDict,
    **kwargs,
):
    dataset_name = str(entry.get("name", entry.get("dataset_name", "seg_dataset"))).strip() or "seg_dataset"
    dataset_name_lc = dataset_name.lower()
    split = str(entry.get("split", "train"))
    annotation_image_key = str(entry.get("annotation_image_key", "images"))
    annotation_file_key = str(entry.get("annotation_file_key", "file_name"))

    root_value = entry.get("root", entry.get("path", None))
    base_root = Path(".")
    if root_value:
        if bool(entry.get("_legacy_cut300", False)) or dataset_name_lc == "cut300":
            base_root = _resolve_cut300_root(root_value)
        else:
            base_root = Path(str(root_value)).expanduser()

    ann_json = _expand_path(base_root, entry.get("ann_json"))
    if ann_json is None:
        ann_json = base_root / split / "annotation.json"
    if not ann_json.is_file():
        raise FileNotFoundError(
            f"Segmentation dataset '{dataset_name}' annotation json was not found: {ann_json}"
        )

    file_names, ann_count = _load_annotation_file_names(
        ann_json,
        annotation_image_key=annotation_image_key,
        annotation_file_key=annotation_file_key,
    )

    image_dir_value = entry.get("image_dir", entry.get("img_dir", None))
    gt_dir_value = entry.get("gt_dir", entry.get("mask_dir", entry.get("label_dir", None)))
    if image_dir_value is not None:
        img_dir = _expand_path(base_root, image_dir_value)
    else:
        img_dir = _resolve_cut300_split_dir_by_files(base_root, split, "image", file_names=file_names)
    if gt_dir_value is not None:
        gt_dir = _expand_path(base_root, gt_dir_value)
    else:
        gt_dir = _resolve_cut300_split_dir_by_files(base_root, split, "gt", file_names=file_names)

    if img_dir is None or gt_dir is None:
        raise FileNotFoundError(
            f"Segmentation dataset '{dataset_name}' is missing a usable image/gt directory. "
            f"root={base_root}, split={split}, image_dir={img_dir}, gt_dir={gt_dir}"
        )

    dataset_cls = Cut300SegDataset if bool(entry.get("_legacy_cut300", False)) else PromptConditionedSegDataset
    dataset = dataset_cls(
        img_dir=img_dir,
        gt_dir=gt_dir,
        ann_json=ann_json,
        transform=transform,
        crop_size=config.transform.input_size[0],
        question_templates=entry.get("question_templates", None),
        answer_text=entry.get("answer_text", "Segmentation completed."),
        task_text=entry.get("task_text", "element extraction"),
        element_text=entry.get("element_text", dataset_name),
        task_id=entry.get("task_id", None),
        category_id=entry.get("category_id", None),
        dataset_name=entry.get("dataset_name", dataset_name),
        physical_prompt=entry.get("physical_prompt", None),
        sample_weight=float(entry.get("sample_weight", 1.0)),
        mask_threshold=int(entry.get("mask_threshold", 127)),
        mask_positive_values=entry.get("mask_positive_values", None),
        mask_dilate_kernel=int(entry.get("mask_dilate_kernel", 1)),
        gt_suffix=entry.get("gt_suffix", ""),
        gt_extension=entry.get("gt_extension", None),
        annotation_image_key=annotation_image_key,
        annotation_file_key=annotation_file_key,
        **kwargs,
    )
    info = {
        "name": dataset_name,
        "root": base_root,
        "ann_json": ann_json,
        "img_dir": img_dir,
        "gt_dir": gt_dir,
        "img_matches": _count_cut300_matches(img_dir, file_names),
        "gt_matches": _count_cut300_matches(gt_dir, file_names),
        "ann_count": ann_count,
        "element_text": entry.get("element_text", dataset_name),
    }
    return dataset, info


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
    if hasattr(config, "coord_bins") and "coord_bins" not in kwargs:
        kwargs["coord_bins"] = config.coord_bins

    if is_train and config.stage == 1:
        if "RS5M" in config.data_path:
            dataset = RS5MDataset(root=config.data_path, transform=transform, **kwargs)
        else:
            dataset = CaptionDatasetVQA(
                root=config.data_path, transform=transform, **kwargs
            )
    elif is_train and config.stage >= 2:
        stage_value = int(getattr(config, "stage", 2))
        geojson_priority = bool(getattr(config, "geojson_priority", stage_value >= 3))
        base_dataset = InstructDatasetWithTaskId(
            root=config.data_path,
            transform=transform,
            crop_size=config.transform.input_size[0],
            stage=stage_value,
            geojson_priority=geojson_priority,
            **kwargs,
        )
        dataset = base_dataset
        seg_entries = _build_seg_dataset_entries(config)
        has_cut300_entry = any(
            bool(entry.get("_legacy_cut300", False)) or str(entry.get("name", "")).lower() == "cut300"
            for entry in seg_entries
        )
        if not has_cut300_entry:
            auto_cut300_root = _infer_cut300_root_from_data_path(getattr(config, "data_path", None))
            if auto_cut300_root is not None:
                seg_entries.append(
                    {
                        "name": "cut300",
                        "root": auto_cut300_root,
                        "split": "train",
                        "task_text": "element extraction",
                        "element_text": "aquaculture region",
                        "task_id": 4,
                        "category_id": 1,
                        "physical_prompt": "[Dataset: cut300] [Band: RGB] [Target: aquaculture]",
                        "_legacy_cut300": True,
                    }
                )
                logger.warning(
                    "cut300_path is not set; inferred cut300 root '%s' from data_path='%s'",
                    auto_cut300_root,
                    config.data_path,
                )

        seg_datasets = []
        for entry in seg_entries:
            seg_ds, seg_info = _build_seg_dataset_from_entry(
                entry,
                transform=transform,
                config=config,
                **kwargs,
            )
            if len(seg_ds) == 0:
                logger.warning(
                    "Configured segmentation dataset is empty after matching files: "
                    "name=%s, element=%s, ann_json=%s, images=%s, gt=%s, img_matches=%s/%s, gt_matches=%s/%s",
                    seg_info["name"],
                    seg_info["element_text"],
                    seg_info["ann_json"],
                    seg_info["img_dir"],
                    seg_info["gt_dir"],
                    seg_info["img_matches"],
                    seg_info["ann_count"],
                    seg_info["gt_matches"],
                    seg_info["ann_count"],
                )
                continue
            seg_datasets.append(seg_ds)
            logger.info(
                "Appended segmentation dataset: name=%s, element=%s, samples=%s, images=%s, gt=%s, img_matches=%s/%s, gt_matches=%s/%s",
                seg_info["name"],
                seg_info["element_text"],
                len(seg_ds),
                seg_info["img_dir"],
                seg_info["gt_dir"],
                seg_info["img_matches"],
                seg_info["ann_count"],
                seg_info["gt_matches"],
                seg_info["ann_count"],
            )

        dataset_parts = []
        if len(base_dataset) > 0:
            dataset_parts.append(base_dataset)
        if seg_datasets:
            dataset_parts.extend(seg_datasets)

        if not dataset_parts:
            dataset = base_dataset
        elif len(dataset_parts) == 1:
            dataset = dataset_parts[0]
        else:
            dataset = ConcatDataset(dataset_parts)
            merged_sample_weight = []
            for ds in dataset_parts:
                ds_weights = getattr(ds, "sample_weight", None)
                if ds_weights is None:
                    ds_weights = [1.0] * len(ds)
                merged_sample_weight.extend(list(ds_weights))
            dataset.sample_weight = merged_sample_weight

        if len(base_dataset) == 0 and len(seg_datasets) > 0:
            logger.warning(
                "Base instruction dataset is empty for data_path='%s'; training will use segmentation dataset(s) only.",
                config.data_path,
            )
        if len(dataset) == 0:
            raise ValueError(
                "Resolved stage-3 training dataset is empty. "
                f"data_path={config.data_path}, cut300_path={getattr(config, 'cut300_path', None)}, "
                f"coastline_seg_path={getattr(config, 'coastline_seg_path', None)}. "
                "Please verify the instruction json root and configured segmentation image/gt directories."
            )
        if config.weight_sample:
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
                    tokenizer=kwargs["tokenizer"],
                    physical_prompt_max_len=int(getattr(config, "physical_prompt_max_len", 64)),
                    task_text_max_len=int(getattr(config, "task_text_max_len", 16)),
                    element_text_max_len=int(getattr(config, "element_text_max_len", 16)),
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
                            tokenizer=kwargs["tokenizer"],
                            physical_prompt_max_len=int(getattr(config, "physical_prompt_max_len", 64)),
                            task_text_max_len=int(getattr(config, "task_text_max_len", 16)),
                            element_text_max_len=int(getattr(config, "element_text_max_len", 16)),
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
            collate_fn=DataCollatorForSupervisedDataset(
                tokenizer=kwargs["tokenizer"],
                physical_prompt_max_len=int(getattr(config, "physical_prompt_max_len", 64)),
                task_text_max_len=int(getattr(config, "task_text_max_len", 16)),
                element_text_max_len=int(getattr(config, "element_text_max_len", 16)),
            ),
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
