from .hookbase import HookBase
import torch_npu


class DeepSpeedHook(HookBase):
    def after_iter(self) -> None:
        self.trainer.model.backward(self.trainer.loss_dict["total_loss"])
        self.trainer._call_hooks("after_backward")
        self.trainer.model.step()
        self.trainer._call_hooks("after_step")

        try:
            import torch_npu
            torch_npu.npu.synchronize()
        except Exception:
            pass

        if (
            self.trainer._clip_grad_norm is not None
            and self.trainer._clip_grad_norm > 0.0
        ):
            self.trainer.log(
                self.trainer.cur_iter,
                smooth=False,
                grad_norm=self.trainer.optimizer._global_grad_norm,
            )
