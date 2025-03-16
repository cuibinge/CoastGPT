#源于LHRS
import logging
from typing import List

from .hook import (
    CosineAnnealingLrUpdaterHook,
    FixedLrUpdaterHook,
    HookBase,
    IterCheckpointerHook,
    LoggerHook,
)
from .trainer import Trainer
from .utils import collect_env, is_main_process

# 创建一个名为 'train' 的日志记录器
logger = logging.getLogger("train")


# 定义基于迭代（iteration）的训练器类，继承自Trainer类
class IterBasedTrainer(Trainer):
    def __init__(self, max_iters: int, **kwargs):
        """
        Args:
            max_iters (int): 总的训练迭代次数。
        """
        # 调用父类的构造函数
        super().__init__(**kwargs)
        # 存储总的训练迭代次数
        self.target_iters = max_iters
        # 训练开始的迭代次数，初始化为0
        self.begin = 0

        # 如果是主进程或者使用了DeepSpeed
        if is_main_process() or self.deepspeed:
            # 注册默认的钩子函数
            self.register_hook(self._build_default_hook())
            # 记录日志，显示已注册的默认钩子函数名称
            logger.info(
                f"Registered default hooks for main process: {self.registered_hook_names}"
            )

        # 记录环境信息日志
        logger.info("Environment info:\n" + collect_env())

    # 当前的训练状态（迭代次数）的属性
    @property
    def cur_stat(self) -> int:
        return self.inner_iter

    # 总的迭代次数的属性
    @property
    def max_iters(self) -> int:
        return self.target_iters

    # 当前的迭代次数的属性
    @property
    def cur_iter(self) -> int:
        return self.inner_iter

    # 开始的迭代次数的属性
    @property
    def start_iter(self) -> int:
        return self.begin

    # 构建默认钩子函数列表的方法
    def _build_default_hook(self) -> List[HookBase]:
        return [
            # 构建检查点保存的钩子函数
            self.build_ckpt_hook(),
            # 日志记录的钩子函数，设置日志记录周期、TensorBoard日志目录和是否使用WandB
            LoggerHook(
                self._log_period, tb_log_dir=self.tb_log_dir, use_wandb=self.wandb
            ),
        ]

    # 加载当前训练状态的方法
    def load_cur_stat(self, value):
        # 设置当前的迭代次数
        self.inner_iter = value
        # 设置开始的迭代次数
        self.begin = value

    # 获取特定钩子函数列表的方法
    def get_specific_hooks(self) -> List[HookBase]:
        # 如果学习率调度器的名称是 'cosine'
        if self.lr_scheduler.name == "cosine":
            # 创建余弦退火学习率更新钩子函数
            lr_scheduler = CosineAnnealingLrUpdaterHook(
                by_epoch=False,
                warmup=self.lr_scheduler.warmup_method,
                warmup_ratio=self.lr_scheduler.warmup_factor,
                warmup_by_epoch=False,
                min_lr=self.lr_scheduler.min_lr,
                warmup_iters=self.lr_scheduler.warmup_epochs,
            )
        # 如果学习率调度器的名称是 'const'
        elif self.lr_scheduler.name == "const":
            # 创建固定学习率更新钩子函数
            lr_scheduler = FixedLrUpdaterHook()
        else:
            # 如果是不支持的学习率调度器，抛出未实现错误
            raise NotImplementedError(
                f"Unsupported lr scheduler: {self.lr_scheduler.name}"
            )

        return [lr_scheduler]

    # 子类训练的方法
    def sub_classes_train(self):
        # 记录开始训练的日志，显示开始迭代次数和结束迭代次数
        logger.info(
            f"Start training from iteration {self.inner_iter} to {self.target_iters}"
        )
        # 将模型设置为训练模式
        self.model.train()
        # 遍历从开始迭代次数到总迭代次数
        for self.inner_iter in range(self.start_iter, self.target_iters):
            # 调用 'before_iter' 钩子函数
            self._call_hooks("before_iter")
            # 进行一次迭代的训练
            self.train_on_iter()
            # 调用 'after_iter' 钩子函数
            self._call_hooks("after_iter")

