import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.ViT_MTP import VisionTransformer
import torch.nn.functional as F


def test_vit_mtp():
    # 测试配置
    batch_size = 2
    img_size = 224
    num_classes = 10
    num_future = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 生成测试数据（新增主分类标签）
    x = torch.randn(batch_size, 3, img_size, img_size).to(device)
    y_cls = torch.randint(0, num_classes, (batch_size,)).to(device)        # 主分类标签
    y_patch = torch.randint(0, num_classes, (batch_size, 14, 14)).to(device) # 多任务标签

    # 初始化模型
    model = VisionTransformer(
        num_classes=num_classes,
        num_future_predict=num_future,
        predict_head_layers=2
    ).to(device)

    # 测试2：训练模式前向传播（传入元组标签）
    print("测试2：训练模式前向传播")
    model.train()
    logits, loss = model(x, (y_cls, y_patch))  # 传入元组形式的标签

    # 验证输出维度
    assert logits.shape == (batch_size, num_classes), "主分类头输出形状错误"
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0, "损失值应该是标量"
    print("-> 训练模式前向传播验证通过\n")


if __name__ == "__main__":
    test_vit_mtp()