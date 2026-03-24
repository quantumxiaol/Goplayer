from __future__ import annotations

from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - depends on optional rl deps
    torch = None


def require_torch():
    if torch is None:
        raise ImportError("PyTorch is required. Install with: uv sync --extra rl")


def get_default_device():
    require_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(device_name: str | None = None):
    """
    Resolve device from user config.
    Supports: auto, cpu, cuda, mps.
    """
    require_torch()
    name = (device_name or "auto").strip().lower()
    if name == "auto":
        return get_default_device()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Requested device 'cuda' but CUDA is not available.")
    if name == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Requested device 'mps' but MPS is not available.")
    raise ValueError(f"Unsupported device '{device_name}'. Use one of: auto, cpu, cuda, mps.")


def ensure_checkpoint_dir(path: str | Path):
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_checkpoint(path: str | Path, model, optimizer=None, metadata=None):
    require_torch()
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metadata": metadata or {},
    }
    torch.save(payload, str(path))


def load_checkpoint(path: str | Path, map_location="cpu"):
    require_torch()
    return torch.load(str(path), map_location=map_location)
