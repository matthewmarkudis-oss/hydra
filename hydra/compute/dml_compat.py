"""DirectML compatibility patches for stable-baselines3.

SB3 calls `tensor.item()` in ~15 places across PPO/SAC/A2C training loops
for scalar loss extraction. DirectML tensors cannot call `.item()` without
first transferring to CPU. This module monkey-patches `torch.Tensor.item`
and `torch.Tensor.numpy` to transparently handle the CPU transfer.

The patches are idempotent (safe to call multiple times) and add zero
overhead for tensors already on CPU.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("hydra.compute.dml_compat")

_patched = False


def patch_tensor_for_directml() -> bool:
    """Monkey-patch torch.Tensor.item and .numpy for DirectML compatibility.

    Returns True if patches were applied (or already applied), False if
    torch-directml is not installed.
    """
    global _patched

    if _patched:
        return True

    try:
        import torch_directml  # noqa: F401
    except ImportError:
        logger.debug("torch-directml not installed, skipping tensor patches")
        return False

    import torch

    _original_item = torch.Tensor.item
    _original_numpy = torch.Tensor.numpy

    def _safe_item(self):
        if not self.is_cpu:
            return _original_item(self.cpu())
        return _original_item(self)

    def _safe_numpy(self, *args, **kwargs):
        # Always detach: torch serialization calls .cpu().numpy() on tensors
        # that still require grad, which fails even on CPU tensors.
        t = self if self.is_cpu else self.cpu()
        if t.requires_grad:
            t = t.detach()
        return _original_numpy(t, *args, **kwargs)

    torch.Tensor.item = _safe_item
    torch.Tensor.numpy = _safe_numpy

    _patched = True
    logger.info("Applied DirectML tensor patches (item, numpy)")
    return True
