

import logging

import ml_collections
import torch.nn as nn
from timm.optim.optim_factory import create_optimizer_v2

# 创建一个名为 'train' 的日志记录器
logger = logging.getLogger("train")

# 检查名称中是否包含指定关键字的函数
def check_keywords_in_name(name, keywords=()):
    isin = False
    # 遍历关键字列表
    for keyword in keywords:
        # 原代码有误，这里应该是 keyword.split('.')[-1]，修正后检查名称中是否包含关键字的最后一部分
        if keyword.split('.')[-1] in name:
            isin = True
    return isin

# 获取预训练时的参数分组，区分需要权重衰减和不需要权重衰减的参数
def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):
    # 存储需要权重衰减的参数
    has_decay = []
    # 存储不需要权重衰减的参数
    no_decay = []
    # 存储需要权重衰减的参数名称
    has_decay_name = []
    # 存储不需要权重衰减的参数名称
    no_decay_name = []

    # 遍历模型的所有命名参数
    for name, param in model.named_parameters():
        # 如果参数不需要梯度更新，则跳过
        if not param.requires_grad:
            continue
        # 判断参数是否不需要权重衰减
        if (
            # 若参数维度为 1（通常是偏置或归一化层的参数）
            len(param.shape) == 1
            # 或者参数名称以 '.bias' 结尾
            or name.endswith(".bias")
            # 或者参数名称的最后一部分在跳过列表中
            or (name.split(".")[-1] in skip_list)
            # 或者参数名称包含需要跳过的关键字
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    # 返回参数分组列表，包含需要权重衰减和不需要权重衰减的参数组
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]

# 设置模型参数的权重衰减，区分需要权重衰减和不需要权重衰减的参数
def set_weight_decay(model, skip_list=(), skip_keywords=()):
    # 存储需要权重衰减的参数
    has_decay = []
    # 存储不需要权重衰减的参数
    no_decay = []

    # 遍历模型的所有命名参数
    for name, param in model.named_parameters():
        # 如果参数不需要梯度更新，则跳过
        if not param.requires_grad:
            continue  # frozen weights
        # 判断参数是否不需要权重衰减
        if (
            # 若参数维度为 1（通常是偏置或归一化层的参数）
            len(param.shape) == 1
            # 或者参数名称以 '.bias' 结尾
            or name.endswith(".bias")
            # 或者参数名称的最后一部分在跳过列表中
            or (name.split(".")[-1] in skip_list)
            # 或者参数名称包含需要跳过的关键字
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
            # 记录不需要权重衰减的参数名称
            logger.info(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    # 返回参数分组列表，包含需要权重衰减和不需要权重衰减的参数组
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]

# 根据是否为预训练阶段获取模型的参数分组
def get_param_group(model: nn.Module, is_pretrain: bool = True):
    # 存储不需要权重衰减的参数名称集合
    skip = {}
    # 存储不需要权重衰减的关键字集合
    skip_keywords = {}
    # 如果模型有 'no_weight_decay' 方法，获取不需要权重衰减的参数名称
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()
    # 如果模型有 'no_weight_decay_keywords' 方法，获取不需要权重衰减的关键字
    if hasattr(model, "no_weight_decay_keywords"):
        skip_keywords = model.no_weight_decay_keywords()

    # 根据是否为预训练阶段调用不同的参数分组函数
    if is_pretrain:
        parameters = get_pretrain_param_groups(model, skip, skip_keywords)
    else:
        parameters = set_weight_decay(model, skip, skip_keywords)
    return parameters

# 构建优化器
def build_optimizer(
    model: nn.Module, config: ml_collections.ConfigDict, is_pretrain: bool = True
):
    # 获取模型的参数分组
    parameters = get_param_group(model, is_pretrain)
    # 使用 timm 库的 create_optimizer_v2 函数创建优化器
    return create_optimizer_v2(
        parameters,
        opt=config.optimizer,  # 优化器类型
        lr=config.lr,  # 学习率
        weight_decay=config.wd,  # 权重衰减系数
        filter_bias_and_bn=False,  # 是否过滤偏置和归一化层的参数
    )