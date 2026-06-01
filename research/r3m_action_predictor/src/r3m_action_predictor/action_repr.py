"""Continuous (sin/cos) action representation.

The raw VLABench action is 7-D Euler: ``[x, y, z, roll, pitch, yaw, gripper]``.
Two of the Euler columns live right on the +/-pi wraparound boundary, which makes
them discontinuous: two physically identical orientations (e.g. -pi and +pi) sit
maximally far apart under an MSE/Gaussian loss. To remove this, the *model* works
in a continuous representation where each Euler angle is replaced by its
``(sin, cos)`` pair::

    raw  (7-D): [x, y, z, roll,       pitch,      yaw,        gripper]
    repr(10-D): [x, y, z, sin r,cos r, sin p,cos p, sin y,cos y, gripper]

Only the action path uses this representation. ``raw_*`` tensors and everything
downstream of the model (control, risk labels) stay in the 7-D Euler space, so the
public predictor contract is unchanged. ``repr_to_euler`` is the single inverse and
``atan2`` makes it robust to off-unit-circle predictions.

All helpers accept either NumPy arrays or Torch tensors (dispatched on the input
type) and operate on the last axis, so they work for any leading batch/time shape.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import ACTION_DIM

# Raw Euler action dim (7) and the continuous model-space dim (10).
RAW_ACTION_DIM = ACTION_DIM
REPR_ACTION_DIM = ACTION_DIM + 3  # +1 extra column per Euler angle (sin & cos)

# Column layout of the 10-D representation: position (0:3), the six sin/cos angle
# columns (3:9), and gripper (9). SINCOS_REPR_SLICE selects the sin/cos columns,
# which must be left in their natural unit-circle scale (never per-dim standardized).
SINCOS_REPR_SLICE = slice(3, REPR_ACTION_DIM - 1)

_POS = slice(0, 3)
_ROT = slice(3, 6)
_GRIP = slice(6, 7)


def euler_to_repr(actions):
    """``(..., 7)`` Euler actions -> ``(..., 10)`` continuous representation."""
    if torch.is_tensor(actions):
        pos, ang, grip = actions[..., _POS], actions[..., _ROT], actions[..., _GRIP]
        sincos = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1).reshape(*ang.shape[:-1], 6)
        return torch.cat([pos, sincos, grip], dim=-1)
    actions = np.asarray(actions, dtype=np.float32)
    pos, ang, grip = actions[..., _POS], actions[..., _ROT], actions[..., _GRIP]
    sincos = np.stack([np.sin(ang), np.cos(ang)], axis=-1).reshape(*ang.shape[:-1], 6)
    return np.concatenate([pos, sincos, grip], axis=-1).astype(np.float32)


def repr_to_euler(rep):
    """``(..., 10)`` continuous representation -> ``(..., 7)`` Euler actions."""
    if torch.is_tensor(rep):
        pos, sincos, grip = rep[..., 0:3], rep[..., 3:9], rep[..., 9:10]
        sincos = sincos.reshape(*sincos.shape[:-1], 3, 2)
        ang = torch.atan2(sincos[..., 0], sincos[..., 1])
        return torch.cat([pos, ang, grip], dim=-1)
    rep = np.asarray(rep, dtype=np.float32)
    pos, sincos, grip = rep[..., 0:3], rep[..., 3:9], rep[..., 9:10]
    sincos = sincos.reshape(*sincos.shape[:-1], 3, 2)
    ang = np.arctan2(sincos[..., 0], sincos[..., 1])
    return np.concatenate([pos, ang, grip], axis=-1).astype(np.float32)


def repr_std_to_euler_std(std):
    """Map a per-dim ``(..., 10)`` repr std to an interpretable ``(..., 7)`` Euler std.

    Position and gripper stds pass through unchanged. For each angle, the magnitude
    of ``d(atan2)/d(sin,cos)`` is ``1/r`` with ``r ~= 1`` on the unit circle, so the
    angular std is approximated (in radians) by ``sqrt(sigma_sin^2 + sigma_cos^2)``.
    This keeps the public ``action_std`` output 7-D and monotone in uncertainty.
    """
    if torch.is_tensor(std):
        pos, sincos, grip = std[..., 0:3], std[..., 3:9], std[..., 9:10]
        sincos = sincos.reshape(*sincos.shape[:-1], 3, 2)
        ang = torch.sqrt(sincos[..., 0] ** 2 + sincos[..., 1] ** 2)
        return torch.cat([pos, ang, grip], dim=-1)
    std = np.asarray(std, dtype=np.float32)
    pos, sincos, grip = std[..., 0:3], std[..., 3:9], std[..., 9:10]
    sincos = sincos.reshape(*sincos.shape[:-1], 3, 2)
    ang = np.sqrt(sincos[..., 0] ** 2 + sincos[..., 1] ** 2)
    return np.concatenate([pos, ang, grip], axis=-1).astype(np.float32)
