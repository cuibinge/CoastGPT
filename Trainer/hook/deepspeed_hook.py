import logging
from .hookbase import HookBase
import time
import torch_npu

logger = logging.getLogger("train")


class DeepSpeedHook(HookBase):
    def after_iter(self) -> None:
        backward_start = time.perf_counter()
        self.trainer.model.backward(self.trainer.loss_dict["total_loss"])
        backward_time = time.perf_counter() - backward_start

        after_backward_start = time.perf_counter()
        self.trainer._call_hooks("after_backward")
        after_backward_time = time.perf_counter() - after_backward_start

        step_start = time.perf_counter()
        self.trainer.model.step()
        step_time = time.perf_counter() - step_start

        after_step_start = time.perf_counter()
        self.trainer._call_hooks("after_step")
        after_step_time = time.perf_counter() - after_step_start

        sync_time = 0.0
        try:
            import torch_npu
            sync_start = time.perf_counter()
            torch_npu.npu.synchronize()
            sync_time = time.perf_counter() - sync_start
        except Exception:
            pass

        if (
            getattr(self.trainer, "_debug_phase_timing", False)
            and self.trainer.cur_iter < getattr(self.trainer, "_debug_phase_iters", 0)
        ):
            self.trainer._last_phase_timing.update(
                {
                    "backward_time": backward_time,
                    "after_backward_time": after_backward_time,
                    "step_time": step_time,
                    "after_step_time": after_step_time,
                    "sync_time": sync_time,
                }
            )
            logger.info(
                "[phase][iter %s] backward=%.4fs after_backward=%.4fs step=%.4fs after_step=%.4fs sync=%.4fs",
                self.trainer.cur_iter,
                backward_time,
                after_backward_time,
                step_time,
                after_step_time,
                sync_time,
            )

        if (
            self.trainer._clip_grad_norm is not None
            and self.trainer._clip_grad_norm > 0.0
        ):
            self.trainer.log(
                self.trainer.cur_iter,
                smooth=False,
                grad_norm=self.trainer.optimizer._global_grad_norm,
            )
