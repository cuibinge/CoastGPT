import os
from typing import Optional

import torch


ACCELERATOR_ALIASES = {
    "cuda": "gpu",
    "gpu": "gpu",
    "nvidia": "gpu",
    "npu": "npu",
    "ascend": "npu",
    "cpu": "cpu",
    "mps": "mps",
    "auto": "auto",
}


def normalize_accelerator(accelerator: Optional[str] = "auto") -> str:
    key = str(accelerator or "auto").strip().lower()
    return ACCELERATOR_ALIASES.get(key, key)


def get_local_rank(default: int = 0) -> int:
    for key in ("LOCAL_RANK", "SLURM_LOCALID"):
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
    return int(default)


def is_npu_available() -> bool:
    if not hasattr(torch, "npu"):
        try:
            import torch_npu  # noqa: F401
        except Exception:
            return False
    try:
        return bool(torch.npu.is_available())
    except Exception:
        return False


def require_npu() -> None:
    if not is_npu_available():
        raise RuntimeError(
            "NPU runtime is unavailable. Install/enable torch_npu or use --accelerator gpu/cpu."
        )


def resolve_accelerator(accelerator: Optional[str] = "auto") -> str:
    acc = normalize_accelerator(accelerator)
    if acc != "auto":
        return acc
    if torch.cuda.is_available():
        return "gpu"
    if is_npu_available():
        return "npu"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(
    accelerator: Optional[str] = "auto",
    local_rank: Optional[int] = None,
    prefer_indexed: bool = True,
) -> torch.device:
    acc = resolve_accelerator(accelerator)
    rank = get_local_rank() if local_rank is None else int(local_rank)

    if acc == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable, but accelerator='gpu' was requested.")
        idx = rank if torch.cuda.device_count() > 1 else 0
        torch.cuda.set_device(idx)
        return torch.device(f"cuda:{idx}" if prefer_indexed else "cuda")

    if acc == "npu":
        require_npu()
        idx = rank
        torch.npu.set_device(idx)
        return torch.device(f"npu:{idx}" if prefer_indexed else "npu")

    return torch.device(acc)


def resolve_distributed_backend(
    accelerator: Optional[str] = "auto",
    backend: Optional[str] = None,
) -> str:
    if backend:
        return str(backend).lower()
    acc = resolve_accelerator(accelerator)
    if acc == "gpu":
        return "nccl"
    if acc == "npu":
        return "hccl"
    return "gloo"


def get_device_count(accelerator: Optional[str] = "auto") -> int:
    acc = resolve_accelerator(accelerator)
    if acc == "gpu":
        return max(1, int(torch.cuda.device_count()))
    if acc == "npu" and is_npu_available():
        try:
            return max(1, int(torch.npu.device_count()))
        except Exception:
            return 1
    return 1


def supports_bitsandbytes(accelerator: Optional[str] = "auto") -> bool:
    return resolve_accelerator(accelerator) == "gpu" and torch.cuda.is_available()


def cuda_visible_device() -> Optional[str]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_VISABLE_DEVICES")
    if value and "," not in value:
        return value.strip()
    return None

