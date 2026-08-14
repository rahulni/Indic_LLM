"""Device auto-detection so every trainer runs on whatever hardware is
available without code changes.

Verified on this machine: CUDA (NVIDIA GeForce RTX 3070 Laptop GPU, 8GB,
torch 2.11+cu128). Falls back to MPS or CPU elsewhere rather than failing.
"""
from __future__ import annotations

import torch


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def device_report() -> dict:
    device = get_device()
    report = {"device": device.type}
    if device.type == "cuda":
        report["cuda_device_name"] = torch.cuda.get_device_name(0)
        report["torch_version"] = torch.__version__
    return report
