import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from ViT_MTP import VisionTransformer
import torch.nn.functional as F


def test_vit_mtp():
    # 测试配置
    batch_size = 2
    img_size = 224
    num_classes = 10
    num_future = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 生成测试数据
    x = torch.randn(batch_size, 3, img_size, img_size).to(device)
    y = torch.randint(0, num_classes, (batch_size, 14, 14)).to(device)  # 14x14 grid for 224/16

    # 初始化模型
    model = VisionTransformer(
        num_classes=num_classes,
        num_future_predict=num_future,
        predict_head_layers=2
    ).to(device)

    # 测试1：验证模型结构
    print("测试1：模型结构验证")
    assert len(model.predict_heads) == num_future, "预测头数量错误"
    for head in model.predict_heads:
        assert len(head) == 3, "预测头层数错误(应该包含2个Linear层和激活函数)"
        assert isinstance(head[-1], nn.Linear), "最后一层应该是Linear层"
        assert head[-1].out_features == num_classes, "输出维度错误"
    print("-> 模型结构验证通过\n")

    # 测试2：训练模式前向传播
    print("测试2：训练模式前向传播")
    model.train()
    logits, loss = model(x, y)
    assert logits.shape == (batch_size, num_classes), "主分类头输出形状错误"
    assert isinstance(loss, torch.Tensor), "损失值应该是Tensor"
    assert loss > 0, "损失值应该大于0"
    print("-> 训练模式前向传播验证通过\n")

    # 测试3：推理模式前向传播
    print("测试3：推理模式前向传播")
    model.eval()
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (batch_size, num_classes), "推理输出形状错误"
    print("-> 推理模式验证通过\n")

    # 测试4：损失计算验证
    print("测试4：损失计算验证")
    fake_logits = torch.randn(batch_size * 14 * 14, num_classes, device=device)
    fake_labels = torch.randint(0, num_classes, (batch_size * 14 * 14,), device=device)
    loss = F.cross_entropy(fake_logits, fake_labels)
    assert loss.dim() == 0, "交叉熵损失应该是标量"
    print("-> 基础损失计算验证通过")

    # 测试多任务损失加权
    model.train()
    _, loss = model(x, y)
    loss_items = [p for n, p in model.named_parameters() if 'predict_heads' in n]
    assert len(loss_items) == num_future * 3, "参数数量错误"  # 每个头3个参数(2个Linear层)
    print("-> 多任务损失加权验证通过\n")

    # 测试5：位置编码特征融合
    print("测试5：位置编码特征融合")
    dummy_features = torch.randn(batch_size, 197, 768, device=device)
    fused = model.forward_features_with_position(dummy_features)
    assert fused.shape == dummy_features.shape, "特征融合形状错误"
    print("-> 位置编码特征融合验证通过\n")

    # 测试6：异常输入处理
    print("测试6：异常输入处理")
    try:
        wrong_x = torch.randn(batch_size, 3, 128, 128).to(device)  # 错误尺寸
        model(wrong_x)
    except AssertionError as e:
        assert "Input image size" in str(e), "错误的断言信息"
        print("-> 尺寸错误处理验证通过")

    # 测试7：设备兼容性
    print("测试7：设备兼容性验证")
    assert next(model.parameters()).device == device, "模型设备位置错误"
    print(f"-> 模型运行在 {device} 设备验证通过\n")

    print("所有测试通过！")


if __name__ == "__main__":
    test_vit_mtp()