
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

logger = logging.getLogger("train")


class EpochBasedTrainer(Trainer):
    def __init__(self, max_epochs: int, max_iters_override: int = None, **kwargs):
        """
        Args:
            max_epochs (int): genericgenericgeneric?
            max_iters_override (int, optional): genericヨgenericgenericgenericヨgenericｆgeneric€?
        """
        super().__init__(**kwargs)
        self.max_epochs = max_epochs
        self.max_iters_override = (
            int(max_iters_override)
            if max_iters_override is not None and int(max_iters_override) > 0
            else None
        )

        self.epoch = 0
        self.start_epoch = 0
        self.inner_iter = 0

        if is_main_process() or self.deepspeed:
            self.register_hook(self._build_default_hook())
            logger.info(
                f"Registered default hooks for main process: {self.registered_hook_names}"
            )

        logger.info("Environment info:\n" + collect_env())

    @property
    def cur_stat(self) -> int:
        return self.cur_iter

    @property
    def max_iters(self) -> int:
        base_iters = self.max_epochs * self.epoch_len
        if self.max_iters_override is None:
            return base_iters
        return min(base_iters, self.max_iters_override)

    @property
    def cur_iter(self) -> int:
        return self.epoch * self.epoch_len + self.inner_iter

    @property
    def start_iter(self) -> int:
        return self.start_epoch * self.epoch_len

    def _build_default_hook(self) -> List[HookBase]:
        return [
            self.build_ckpt_hook(),
            LoggerHook(
                self._log_period, tb_log_dir=self.tb_log_dir, use_wandb=self.wandb
            ),
        ]

    def load_cur_stat(self, value):
        epoch = value // self.epoch_len
        inner_iter = value % self.epoch_len
        self.epoch = epoch
        self.start_epoch = epoch
        self.inner_iter = inner_iter

    def get_specific_hooks(self) -> List[HookBase]:
        if self.lr_scheduler.name == "cosine":
            lr_scheduler = CosineAnnealingLrUpdaterHook(
                by_epoch=False,
                warmup=self.lr_scheduler.warmup_method,
                warmup_ratio=self.lr_scheduler.warmup_factor,
                warmup_by_epoch=False,
                min_lr=self.lr_scheduler.min_lr,
                warmup_iters=self.lr_scheduler.warmup_epochs,
            )
        elif self.lr_scheduler.name == "const":
            lr_scheduler = FixedLrUpdaterHook()
        else:
            raise NotImplementedError(
                f"Unsupported lr scheduler: {self.lr_scheduler.name}"
            )

        return [lr_scheduler, DistributedHook()]

    def _train_one_epoch(self) -> None:
        self.model.train()
        for self.inner_iter in range(self.inner_iter, self.epoch_len):
            if self.max_iters_override is not None and self.cur_iter >= self.max_iters_override:
                break
        # for self.inner_iter in range(0, 10):
            self._call_hooks("before_iter")
            # for name, param in self.model_or_module.named_parameters():
            #     if param.requires_grad:
            #         print(name)

            self.train_on_iter()
            self._call_hooks("after_iter")
        self._data_iter = iter(self.data_loader)
        self.inner_iter = 0
        sampler = getattr(self.data_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            try:
                sampler.set_epoch(self.epoch + 1)
            except Exception:
                pass

    def sub_classes_train(self):
        logger.info(
            f"Start training from epoch {self.start_epoch} to {self.max_epochs}."
        )
        if self.max_iters_override is not None:
            logger.info(f"Early-stop by max_iters_override={self.max_iters_override}.")
        for self.epoch in range(self.start_epoch, self.max_epochs):
            if self.max_iters_override is not None and self.cur_iter >= self.max_iters_override:
                break
            self._call_hooks("before_epoch")
            self._train_one_epoch()
            self._call_hooks("after_epoch")
