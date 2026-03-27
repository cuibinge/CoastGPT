"""
Global mappings for tasks and elements.
Keep IDs in sync with MoEProjection(num_tasks, num_elements).
"""

# Task taxonomy
TASK2ID = {
    "场景分类": 0,
    "视觉问答": 1,
    "视觉定位": 2,
    "描述": 3,
    "要素提取": 4,
}

# Element taxonomy (expand as needed)
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
}
