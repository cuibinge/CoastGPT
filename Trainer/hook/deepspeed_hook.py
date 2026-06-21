from .hookbase import HookBase
from .runtime import synchronize_device
try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None


class DeepSpeedHook(HookBase):
    def after_iter(self) -> None:
        self.trainer.model.backward(self.trainer.loss_dict["total_loss"])
        self.trainer._call_hooks("after_backward")
        self.trainer.model.step()
        self.trainer._call_hooks("after_step")

        synchronize_device(self.trainer.device)

        if (
            self.trainer._clip_grad_norm is not None
            and self.trainer._clip_grad_norm > 0.0
        ):
            self.trainer.log(
                self.trainer.cur_iter,
                smooth=False,
                grad_norm=self.trainer.optimizer._global_grad_norm,
            )
