import argparse
import logging
import os
from argparse import Namespace
from copy import deepcopy
from typing import List, Optional

import yaml

# 初始化日志记录器，用于记录程序运行过程中的信息
logger = logging.getLogger("train")


class ConfigArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        # 创建一个不显示帮助信息的子解析器，用于解析配置文件路径
        self.config_parser = argparse.ArgumentParser(add_help=False)
        # 为子解析器添加一个参数，用于指定 YAML 配置文件的路径
        self.config_parser.add_argument(
            "-c",
            "--config",
            default="Config/inference.yaml",
            metavar="FILE",
            help="where to load YAML configuration",
        )
        # 用于存储所有添加的参数的目标名称
        self.option_names = []
        # 调用父类的构造函数，继承子解析器的参数，并设置格式化器
        super().__init__(
            *args,
            # 从 config_parser 继承选项
            parents=[self.config_parser],
            # 不修改描述的格式
            formatter_class=argparse.RawDescriptionHelpFormatter,
            **kwargs,
        )

    def add_argument(self, *args, **kwargs):
        # 调用父类的 add_argument 方法添加参数
        arg = super().add_argument(*args, **kwargs)
        # 将参数的目标名称添加到 option_names 列表中
        self.option_names.append(arg.dest)
        return arg

    def parse_args(self, wandb=False, args=None):
        # 首先使用子解析器解析出用户指定的配置文件的路径，保存在 res.config 中
        res, remaining_argv = self.config_parser.parse_known_args(args)

        if res.config is not None:
            # 如果指定了配置文件，则打开该文件并使用 yaml 解析
            with open(res.config, "r") as f:
                config_vars = yaml.safe_load(f)
            # 解析剩余的命令行参数
            namespace = vars(super().parse_args(remaining_argv))
            if wandb:
                # 如果使用 wandb，将命令行参数更新到配置文件的参数中
                config_vars.update(namespace)
                return config_vars
            else:
                # 否则，将配置文件的参数更新到命令行参数中
                namespace.update(config_vars)
                return namespace
        else:
            # 如果没有指定配置文件，直接解析命令行参数
            return vars(super().parse_args(remaining_argv))


def save_args(
    args: Namespace, filepath: str, excluded_fields: Optional[List[str]] = None
) -> None:
    """
    保存解析后的参数到一个 YAML 文件中，同时可以排除一些不需要保存的字段。

    Args:
        args (Namespace): 解析后的参数对象。
        filepath (str): 保存文件的路径，必须以 ".yaml" 结尾。
        excluded_fields (list[str]): 一些不需要保存的字段的名称列表。
            默认值为 ["config"]。
    """
    # 确保传入的 args 是 Namespace 类型
    assert isinstance(args, Namespace)
    # 确保文件路径以 ".yaml" 结尾
    assert filepath.endswith(".yaml")
    # 创建保存文件的目录（如果不存在）
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    # 深拷贝 args 的字典，避免修改原始数据
    save_dict = deepcopy(args.__dict__)
    # 遍历需要排除的字段，从保存字典中移除这些字段
    for field in excluded_fields or ["config"]:
        save_dict.pop(field)
    # 打开文件并将保存字典以 YAML 格式写入文件
    with open(filepath, "w") as f:
        yaml.dump(save_dict, f)
    # 记录日志，提示参数已保存到指定文件
    logger.info(f"Args is saved to {filepath}")

