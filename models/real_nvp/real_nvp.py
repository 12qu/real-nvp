import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from oos.bijections import LogitTransformBijection, CompositeBijection, AffineCouplingBijection, ConvAffineCoupler

from models.real_nvp.coupling_layer import CouplingLayer, MaskType
from util import checkerboard_mask


class RealNVP(nn.Module):
    """RealNVP Model

    Based on the paper:
    "Density estimation using Real NVP"
    by Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio
    (https://arxiv.org/abs/1605.08803).

    Args:
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
        `Coupling` layers.
    """
    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8):
        super(RealNVP, self).__init__()
        lam = 1e-6
        self.logit = LogitTransformBijection(input_shape=(1,28,28), lam=lam)

        # Register data_constraint to pre-process images, not learnable
        self.register_buffer('data_constraint', torch.tensor([1 - 2*lam], dtype=torch.float32))

        # self.flows = _RealNVP(0, num_scales, in_channels, mid_channels, num_blocks)

        mask = checkerboard_mask(28, 28, reverse=False).squeeze(0)
        flows = [self.logit]
        for _ in range(4):
            flow = AffineCouplingBijection(
                mask=mask.to(torch.uint8),
                coupler=ConvAffineCoupler(in_channels, mid_channels), num_u_channels=0
            )
            flows.append(flow)
            mask = 1 - mask
        self.flows = CompositeBijection(flows, "x-to-z")

    def forward(self, x, reverse=False):
        if reverse:
            result = self.flows.z_to_x(x)
            return result["x"], result["log-jac"].view(x.shape[0])
        else:
            result = self.flows.x_to_z(x)
            return result["z"], result["log-jac"].view(x.shape[0])

        sldj = None
        if not reverse:
            # Expect inputs in [0, 256]
            if x.min() < 0 or x.max() > 256:
                raise ValueError('Expected x in [0, 1], got x with min/max {}/{}'
                                 .format(x.min(), x.max()))

            # De-quantize and convert to logits
            x, sldj = self._pre_process(x)

        x, sldj = self.flows(x, sldj, reverse)

        return x, sldj

    def _pre_process(self, x):
        """Dequantize the input image `x` and convert to logits.

        Args:
            x (torch.Tensor): Input image.

        Returns:
            y (torch.Tensor): Dequantized logits of `x`.

        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        """
        result = self.logit.x_to_z(x)
        return result["z"], result["log-jac"].view(x.shape[0])

    # def _pre_process(self, x):
    #     y = (2 * x/256 - 1) * self.data_constraint
    #     y = (y + 1) / 2
    #     p = y
    #     y = y.log() - (1. - y).log()

    #     # Save log-determinant of Jacobian of initial transform
    #     ldj = -torch.log(1 - p) - torch.log(p) + np.log(self.data_constraint.item())
    #     sldj = ldj.view(ldj.size(0), -1).sum(-1)

    #     return y, sldj

class _RealNVP(nn.Module):
    """Recursive builder for a `RealNVP` model.

    Each `_RealNVPBuilder` corresponds to a single scale in `RealNVP`,
    and the constructor is recursively called to build a full `RealNVP` model.

    Args:
        scale_idx (int): Index of current scale.
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
    """
    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks):
        super(_RealNVP, self).__init__()

        self.is_last_block = scale_idx == num_scales - 1

        self.in_couplings = nn.ModuleList([
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True),
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False)
        ])

        assert self.is_last_block
        self.in_couplings.append(
            CouplingLayer(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True))

    def forward(self, x, sldj, reverse=False):
        if reverse:
            for coupling in reversed(self.in_couplings):
                x, sldj = coupling(x, sldj, reverse)
        else:
            for coupling in self.in_couplings:
                x, sldj = coupling(x, sldj, reverse)

        return x, sldj
