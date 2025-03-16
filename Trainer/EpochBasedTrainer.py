
#源于mmengine.runner.EpochBasedRunner
import logging
from typing import List

from .hook import (
    CosineAnnealingLrUpdaterHook,
    DistributedHook,
    EpochCheckpointerHook,
    FixedLrUpdaterHook,
    HookBase,
    IterCheckpointerHook,
    LoggerHook,
)
from .trainer import Trainer
from .utils import collect_env, is_main_process

# 创建一个名为 'train' 的日志记录器
logger = logging.getLogger("train")


# 定义基于轮次（epoch）的训练器类，继承自 Trainer 类
class EpochBasedTrainer(Trainer):
    def __init__(self, max_epochs: int, **kwargs):
        """
        Args:
            max_epochs (int): 总的训练轮次。
        """
        # 调用父类的构造函数
        super().__init__(**kwargs)
        # 存储总的训练轮次
        self.max_epochs = max_epochs

        # 当前训练轮次，初始化为 0
        self.epoch = 0
        # 开始训练的轮次，初始化为 0
        self.start_epoch = 0
        # 当前轮次内的迭代次数，初始化为 0
        self.inner_iter = 0

        # 如果是主进程或者使用了 DeepSpeed
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
        return self.cur_iter

    # 总的迭代次数的属性
    @property
    def max_iters(self) -> int:
        return self.max_epochs * self.epoch_len

    # 当前的迭代次数的属性
    @property
    def cur_iter(self) -> int:
        return self.epoch * self.epoch_len + self.inner_iter

    # 开始的迭代次数的属性
    @property
    def start_iter(self) -> int:
        return self.start_epoch * self.epoch_len

    # 构建默认钩子函数列表的方法
    def _build_default_hook(self) -> List[HookBase]:
        return [
            # 构建检查点保存的钩子函数
            self.build_ckpt_hook(),
            # 日志记录的钩子函数，设置日志记录周期、TensorBoard 日志目录和是否使用 WandB
            LoggerHook(
                self._log_period, tb_log_dir=self.tb_log_dir, use_wandb=self.wandb
            ),
        ]

    # 加载当前训练状态的方法
    def load_cur_stat(self, value):
        # 计算当前轮次
        epoch = value // self.epoch_len
        # 计算当前轮次内的迭代次数
        inner_iter = value % self.epoch_len
        self.epoch = epoch
        self.start_epoch = epoch
        self.inner_iter = inner_iter

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

        return [lr_scheduler, DistributedHook()]

    # 训练一个轮次的方法
    def _train_one_epoch(self) -> None:
        # 将模型设置为训练模式
        self.model.train()
        # 遍历当前轮次内的迭代次数
        for self.inner_iter in range(self.inner_iter, self.epoch_len):
            # 调用 'before_iter' 钩子函数
            self._call_hooks("before_iter")
            # 以下代码被注释掉，功能是打印需要梯度更新的参数名称
            # for name, param in self.model_or_module.named_parameters():
            #     if param.requires_grad:
            #         print(name)

            # 进行一次迭代的训练
            self.train_on_iter()
            # 调用 'after_iter' 钩子函数
            self._call_hooks("after_iter")
        # 重新初始化数据迭代器
        self._data_iter = iter(self.data_loader)

    # 子类训练的方法
    def sub_classes_train(self):
        # 记录开始训练的日志，显示开始轮次和结束轮次
        logger.info(
            f"Start training from epoch {self.start_epoch} to {self.max_epochs}."
        )
        # 遍历从开始轮次到结束轮次
        for self.epoch in range(self.start_epoch, self.max_epochs):
            # 调用 'before_epoch' 钩子函数
            self._call_hooks("before_epoch")
            # 训练一个轮次
            self._train_one_epoch()
            # 调用 'after_epoch' 钩子函数
            self._call_hooks("after_epoch")