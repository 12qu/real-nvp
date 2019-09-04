import numpy as np

import torch
import torch.nn as nn

from oos.densities import DiagonalGaussianDensity


class RealNVPLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self, k=256):
        super(RealNVPLoss, self).__init__()
        self.k = k

    def forward(self, z, sldj):
        # prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        # prior_ll = prior_ll.view(z.size(0), -1).sum(-1) \
        #     - np.log(self.k) * np.prod(z.size()[1:])
        # ll = prior_ll + sldj
        # nll = -ll.mean()

        # import ipdb; ipdb.set_trace()

        density = DiagonalGaussianDensity(
            mean=torch.zeros_like(z),
            stddev=torch.ones_like(z),
        )

        lls = density.elbo(z)["elbo"].view(z.shape[0]) + sldj

        return -lls.mean()
