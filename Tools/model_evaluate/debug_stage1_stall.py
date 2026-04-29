#!/usr/bin/env python3
import argparse
import os
import signal
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List

import ml_collections
import torch
import yaml
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Dataset.build_loader import build_loader
from Models.coastgpt import CoastGPT


class BatchTimeoutError(RuntimeError):
    pass


@contextmanager
def alarm_timeout(seconds: int):
    if seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise BatchTimeoutError(f"next(data_iter) timeout after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def to_device(x: Any, device: torch.device):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(to_device(v, device) for v in x)
    if hasattr(x, "to"):
        return x.to(device=device)
    return x


def summarize_tensor_dict(d: Dict[str, Any]) -> str:
    parts = []
    for k, v in d.items():
        if torch.is_tensor(v):
            parts.append(f"{k}:{tuple(v.shape)}:{str(v.dtype).replace('torch.', '')}")
        elif v is None:
            parts.append(f"{k}:None")
        elif isinstance(v, (list, tuple)):
            parts.append(f"{k}:{type(v).__name__}[{len(v)}]")
        elif isinstance(v, dict):
            parts.append(f"{k}:dict[{len(v)}]")
        else:
            parts.append(f"{k}:{type(v).__name__}")
    return " | ".join(parts)


def load_config(path: str, data_path: str, batch_size: int, workers: int):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    config = ml_collections.ConfigDict(cfg)
    config.data_path = data_path
    config.batch_size = int(batch_size)
    config.workers = int(workers)
    config.is_distribute = False
    config.inf_sampler = False
    return config


def build_train_loader(config: ml_collections.ConfigDict, local_files_only: bool):
    tokenizer = LlamaTokenizerFast.from_pretrained(
        config.text.path,
        local_files_only=local_files_only,
    )
    tokenizer.pad_token_id = tokenizer.unk_token_id
    loader = build_loader(
        config,
        mode="pretrain",
        is_train=True,
        tokenizer=tokenizer,
        prompt_type=config.prompt_template,
    )
    return tokenizer, loader


def run_probe(
    config_path: str,
    data_path: str,
    workers: int,
    batch_size: int,
    iters: int,
    next_timeout: int,
    move_device: str,
    local_files_only: bool,
):
    print(f"\n=== Probe workers={workers}, batch_size={batch_size} ===", flush=True)
    config = load_config(config_path, data_path, batch_size, workers)
    _, loader = build_train_loader(config, local_files_only=local_files_only)

    print(f"loader_len={len(loader)}", flush=True)
    data_iter = iter(loader)

    fetch_times: List[float] = []
    move_times: List[float] = []

    if move_device != "none":
        device = torch.device(move_device)
    else:
        device = None

    for i in range(iters):
        t0 = time.perf_counter()
        try:
            with alarm_timeout(next_timeout):
                batch = next(data_iter)
        except StopIteration:
            print(f"[iter {i}] dataloader ended", flush=True)
            break
        except BatchTimeoutError as e:
            print(f"[iter {i}] TIMEOUT: {e}", flush=True)
            return False
        fetch_dt = time.perf_counter() - t0
        fetch_times.append(fetch_dt)

        move_dt = 0.0
        if device is not None:
            m0 = time.perf_counter()
            _ = to_device(batch, device)
            if device.type == "npu":
                import torch_npu  # noqa: F401

                torch.npu.synchronize()
            move_dt = time.perf_counter() - m0
            move_times.append(move_dt)

        if i == 0:
            print(f"[iter {i}] batch: {summarize_tensor_dict(batch)}", flush=True)
        print(f"[iter {i}] fetch={fetch_dt:.3f}s move={move_dt:.3f}s", flush=True)

    if fetch_times:
        p50 = statistics.median(fetch_times)
        p95 = sorted(fetch_times)[max(0, int(len(fetch_times) * 0.95) - 1)]
        print(
            f"fetch_time p50={p50:.3f}s p95={p95:.3f}s max={max(fetch_times):.3f}s",
            flush=True,
        )
    if move_times:
        p50 = statistics.median(move_times)
        p95 = sorted(move_times)[max(0, int(len(move_times) * 0.95) - 1)]
        print(
            f"move_time  p50={p50:.3f}s p95={p95:.3f}s max={max(move_times):.3f}s",
            flush=True,
        )
    return True


def build_model_for_probe(config: ml_collections.ConfigDict):
    compute_dtype = torch.float16 if bool(config.fp16) else (
        torch.bfloat16 if bool(config.bf16) else torch.float32
    )
    model = CoastGPT(config)
    model.prepare_for_training(
        freeze_vision=not bool(config.tune_rgb_bk),
        freeze_text=not bool(config.lora.enable),
        tune_multimodal=bool(getattr(config, "tune_multimodal", False)),
        model_path=getattr(config, "model_path", None),
        tune_im_start=bool(getattr(config, "tune_im_start", False)),
        compute_dtype=compute_dtype,
    )
    return model, compute_dtype


def run_model_probe(
    config_path: str,
    data_path: str,
    workers: int,
    batch_size: int,
    iters: int,
    next_timeout: int,
    local_files_only: bool,
    run_backward: bool,
):
    print(
        f"\n=== Model probe workers={workers}, batch_size={batch_size}, backward={run_backward} ===",
        flush=True,
    )
    config = load_config(config_path, data_path, batch_size, workers)
    config.accelerator = "npu"
    config.is_distribute = False
    config.rank = 0
    config.local_rank = 0
    config.world_size = 1

    _, loader = build_train_loader(config, local_files_only=local_files_only)
    data_iter = iter(loader)

    model, compute_dtype = build_model_for_probe(config)
    device = torch.device("npu")
    model.to(device)
    model.train()

    fetch_times: List[float] = []
    fwd_times: List[float] = []
    bwd_times: List[float] = []

    for i in range(iters):
        t0 = time.perf_counter()
        try:
            with alarm_timeout(next_timeout):
                batch = next(data_iter)
        except StopIteration:
            print(f"[iter {i}] dataloader ended", flush=True)
            break
        except BatchTimeoutError as e:
            print(f"[iter {i}] FETCH TIMEOUT: {e}", flush=True)
            return False
        fetch_dt = time.perf_counter() - t0
        fetch_times.append(fetch_dt)

        batch = to_device(batch, device)
        torch.npu.synchronize()

        f0 = time.perf_counter()
        with torch.autocast(device_type="npu", enabled=True, dtype=compute_dtype):
            out = model(batch)
        torch.npu.synchronize()
        fwd_dt = time.perf_counter() - f0
        fwd_times.append(fwd_dt)

        if isinstance(out, dict):
            loss = out.get("total_loss", None)
            if loss is None:
                loss = out.get("text_loss", None)
            if loss is None:
                raise RuntimeError(f"Model output dict does not contain total_loss/text_loss: {list(out.keys())}")
        else:
            loss = out

        bwd_dt = 0.0
        if run_backward:
            model.zero_grad(set_to_none=True)
            b0 = time.perf_counter()
            loss.backward()
            torch.npu.synchronize()
            bwd_dt = time.perf_counter() - b0
            bwd_times.append(bwd_dt)

        print(
            f"[iter {i}] fetch={fetch_dt:.3f}s fwd={fwd_dt:.3f}s bwd={bwd_dt:.3f}s loss={float(loss.detach().cpu()):.4f}",
            flush=True,
        )

    if fetch_times:
        print(
            f"fetch_time p50={statistics.median(fetch_times):.3f}s max={max(fetch_times):.3f}s",
            flush=True,
        )
    if fwd_times:
        print(
            f"fwd_time   p50={statistics.median(fwd_times):.3f}s max={max(fwd_times):.3f}s",
            flush=True,
        )
    if bwd_times:
        print(
            f"bwd_time   p50={statistics.median(bwd_times):.3f}s max={max(bwd_times):.3f}s",
            flush=True,
        )
    return True


def parse_worker_list(s: str) -> List[int]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if x:
            vals.append(int(x))
    return vals


def main():
    parser = argparse.ArgumentParser(
        description="Debug stage1 training stalls by probing dataloader batch fetch latency."
    )
    parser.add_argument("--config", type=str, default="./Configs/step1_dual.yaml")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--scan-workers", type=str, default="")
    parser.add_argument("--next-timeout", type=int, default=60)
    parser.add_argument(
        "--move-device",
        type=str,
        default="none",
        choices=["none", "cpu", "npu"],
        help="Optionally test host->device transfer after fetching each batch.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force tokenizer/model file resolution from local cache/path only.",
    )
    parser.add_argument(
        "--run-model-forward",
        action="store_true",
        help="Build CoastGPT and run forward timing on NPU for fetched batches.",
    )
    parser.add_argument(
        "--run-model-backward",
        action="store_true",
        help="When used with --run-model-forward, also run loss.backward() timing.",
    )
    args = parser.parse_args()

    # Critical for tokenizer + dataloader workers stability diagnostics.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONHASHSEED", "0")

    if args.scan_workers:
        worker_list = parse_worker_list(args.scan_workers)
    else:
        worker_list = [args.workers]

    print(
        "debug args:",
        {
            "config": args.config,
            "data_path": args.data_path,
            "batch_size": args.batch_size,
            "iters": args.iters,
            "workers": worker_list,
            "next_timeout": args.next_timeout,
            "move_device": args.move_device,
            "local_files_only": args.local_files_only,
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        },
        flush=True,
    )

    all_ok = True
    for w in worker_list:
        if args.run_model_forward:
            ok = run_model_probe(
                config_path=args.config,
                data_path=args.data_path,
                workers=w,
                batch_size=args.batch_size,
                iters=args.iters,
                next_timeout=args.next_timeout,
                local_files_only=args.local_files_only,
                run_backward=args.run_model_backward,
            )
        else:
            ok = run_probe(
                config_path=args.config,
                data_path=args.data_path,
                workers=w,
                batch_size=args.batch_size,
                iters=args.iters,
                next_timeout=args.next_timeout,
                move_device=args.move_device,
                local_files_only=args.local_files_only,
            )
        all_ok = all_ok and ok

    if not all_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
