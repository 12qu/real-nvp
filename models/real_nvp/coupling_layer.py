import torch
import torch.nn as nn

from enum import IntEnum
from models.resnet import ResidualBlock
from util import checkerboard_mask

from .realnvp import AffineCouplingBijection, ConvAffineCoupler


class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1


class CouplingLayer(nn.Module):
    """Coupling layer in RealNVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the `s` and `t` network.
        num_blocks (int): Number of residual blocks in the `s` and `t` network.
        mask_type (MaskType): One of `MaskType.CHECKERBOARD` or `MaskType.CHANNEL_WISE`.
        reverse_mask (bool): Whether to reverse the mask. Useful for alternating masks.
    """
    def __init__(self, in_channels, mid_channels, num_blocks, mask_type, reverse_mask):
        super(CouplingLayer, self).__init__()

        # Save mask info
        self.mask_type = mask_type
        self.reverse_mask = reverse_mask

        # Build scale and translate network
        if self.mask_type == MaskType.CHANNEL_WISE:
            in_channels //= 2

        # Learnable scale for s
        self.rescale = Rescale(in_channels)

        assert self.mask_type == MaskType.CHECKERBOARD
        mask = checkerboard_mask(28, 28, self.reverse_mask).squeeze(0)
        self.bijection = AffineCouplingBijection(mask, coupler=ConvAffineCoupler(in_channels, mid_channels), num_u_channels=0)

    def forward(self, x, sldj=None, reverse=True):
        if reverse:
            result = self.bijection.z_to_x(x)
            assert sldj is None
            return result["x"], sldj
        else:
            result = self.bijection.x_to_z(x)
            sldj += result["log-jac"].view(-1)
            return result["z"], sldj


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.

    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x
