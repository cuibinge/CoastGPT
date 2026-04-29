"""Shared task/category ids for instruction and segmentation datasets."""

# Task ids used by Dataset.cap_dataset.InstructDatasetWithTaskId.
# Keep the first five ids aligned with the hard-coded tag mapping:
# [cls] -> 0, [vqa]/[qa] -> 1, [vg]/[loc] -> 2, [cap] -> 3, [det]/[seg] -> 4.
TASK2ID = {
    "场景分类": 0,
    "视觉问答": 1,
    "视觉定位": 2,
    "描述": 3,
    "要素提取": 4,
    "GeoJSON定位": 5,
}


# Element/category ids used across instruction tuning and segmentation data.
# These ids align with the aliases in Dataset.cap_dataset and the legacy
# defaults in Dataset.build_loader (e.g. aquaculture=1, coastline=5, land cover=10).
ELEMENT2ID = {
    "无": 0,
    "网箱养殖区": 1,
    "筏式养殖区": 2,
    "赤潮": 3,
    "浒苔": 4,
    "海岸线": 5,
    "风力发电机": 6,
    "海上钻井平台": 7,
    "滩涂": 8,
    "红树林湿地": 9,
    "土地覆盖": 10,
}
