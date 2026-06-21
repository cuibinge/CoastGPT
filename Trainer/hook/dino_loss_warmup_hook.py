from .hookbase import HookBase
try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None


class DINOLossWarmUp(HookBase):
    def __init__(self):
        super(DINOLossWarmUp, self).__init__()

    def before_epoch(self) -> None:
        self.trainer.model_or_module.class_loss.epoch = self.trainer.epoch
