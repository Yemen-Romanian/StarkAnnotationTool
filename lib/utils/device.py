"""Single place that decides which torch device inference runs on.

The tracker code used to call ``.cuda()`` unconditionally, which made it
impossible to run (or benchmark) the models on a machine without a GPU.
Everything now goes through :func:`get_device`, which resolves once to:

    1. whatever :func:`set_device` was told (e.g. the ``--device`` CLI flag),
    2. else the ``STARK_DEVICE`` environment variable ("cpu", "cuda", "cuda:1"),
    3. else "cuda" when it is available, "cpu" otherwise.

Set the device *before* building a network: the corner heads bake their
coordinate grids onto the device at construction time.
"""

import os

import torch


_device = None


def set_device(device):
    """Force the device used from now on. Accepts a string or torch.device."""
    global _device
    _device = torch.device(device) if device is not None else None
    return _device


def get_device():
    """Resolve (and cache) the device inference should run on."""
    global _device
    if _device is None:
        requested = os.environ.get("STARK_DEVICE", "").strip()
        if requested:
            _device = torch.device(requested)
        elif torch.cuda.is_available():
            _device = torch.device("cuda")
        else:
            _device = torch.device("cpu")
        if _device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Device '%s' was requested but CUDA is not available. "
                "Use --device cpu (or STARK_DEVICE=cpu) to run on the CPU." % _device)
    return _device
