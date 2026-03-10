"""
Utility functions for the Efficient TTT project.

Provides config loading, JSON I/O, model checkpointing, and logging setup.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
import torch


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Dictionary of configuration values.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Any, path: str) -> None:
    """Save data to a JSON file with indent=2.

    Args:
        data: Dictionary or list to serialize.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    """Load data from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Deserialized data (dict or list).
    """
    with open(path, "r") as f:
        return json.load(f)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model checkpoint (trainable parameters only, not frozen encoders).

    Args:
        model: FullVQAModel instance.
        optimizer: Optimizer state to save (optional).
        epoch: Current epoch number.
        path: Output checkpoint path.
        extra: Additional metadata to include.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    checkpoint = {
        "fusion": model.fusion.state_dict(),
        "gate": model.gate.state_dict(),
        "prediction_head": model.prediction_head.state_dict(),
        "epoch": epoch,
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if extra is not None:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    load_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Load checkpoint into model. Frozen encoders are not saved/loaded.

    Args:
        model: FullVQAModel instance.
        path: Path to the checkpoint file.
        load_optimizer: Whether to load optimizer state.
        optimizer: Optimizer to load state into (required if load_optimizer=True).

    Returns:
        Checkpoint dict (for accessing epoch, extra metadata).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.fusion.load_state_dict(checkpoint["fusion"])
    model.gate.load_state_dict(checkpoint["gate"])
    model.prediction_head.load_state_dict(checkpoint["prediction_head"])

    if load_optimizer and optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Set up Python logging to file + console.

    Args:
        log_dir: Directory for log files.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("efficient_ttt")
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
    file_handler.setLevel(level)
    file_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).

    Returns:
        torch.device for the best available accelerator.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch module.
        trainable_only: If True, only count parameters with requires_grad=True.

    Returns:
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
