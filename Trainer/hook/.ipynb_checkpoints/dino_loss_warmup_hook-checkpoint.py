from .hookbase import HookBase
import torch_npu


class DINOLossWarmUp(HookBase):
    def __init__(self):
        super(DINOLossWarmUp, self).__init__()

    def before_epoch(self) -> None:
        self.trainer.model_or_module.class_loss.epoch = self.trainer.epoch
