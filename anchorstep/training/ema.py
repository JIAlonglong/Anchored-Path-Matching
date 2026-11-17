"""Exponential moving average updates for routed student adapters."""

from __future__ import annotations

from torch import nn


def update_ema(ema_model: nn.Module, online_model: nn.Module, decay: float) -> None:
    """Update matching EMA parameters and copy buffers from the online model."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    online_parameters = dict(online_model.named_parameters())
    ema_parameters = dict(ema_model.named_parameters())
    if online_parameters.keys() != ema_parameters.keys():
        raise ValueError("online and EMA parameter topologies must match")
    for name, ema_parameter in ema_parameters.items():
        online_parameter = online_parameters[name]
        ema_parameter.data.mul_(decay).add_(online_parameter.data, alpha=1.0 - decay)
    online_buffers = dict(online_model.named_buffers())
    ema_buffers = dict(ema_model.named_buffers())
    if online_buffers.keys() != ema_buffers.keys():
        raise ValueError("online and EMA buffer topologies must match")
    for name, ema_buffer in ema_buffers.items():
        ema_buffer.copy_(online_buffers[name])
