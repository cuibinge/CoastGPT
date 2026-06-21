import torch


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return
    if device.type == "npu":
        try:
            import torch_npu

            torch_npu.npu.synchronize()
        except Exception:
            pass

