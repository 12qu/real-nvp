import torch
import torch.nn.functional as F


def checkerboard_mask(height, width, reverse=False, dtype=torch.float32,
                      device=None, requires_grad=False):
    """Get a checkerboard mask, such that no two entries adjacent entries
    have the same value. In non-reversed mask, top-left entry is 0.

    Args:
        height (int): Number of rows in the mask.
        width (int): Number of columns in the mask.
        reverse (bool): If True, reverse the mask (i.e., make top-left entry 1).
            Useful for alternating masks in RealNVP.
        dtype (torch.dtype): Data type of the tensor.
        device (torch.device): Device on which to construct the tensor.
        requires_grad (bool): Whether the tensor requires gradient.


    Returns:
        mask (torch.tensor): Checkerboard mask of shape (1, 1, height, width).
    """
    checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)]
    mask = torch.tensor(checkerboard, dtype=dtype, device=device, requires_grad=requires_grad)

    if reverse:
        mask = 1 - mask

    # Reshape to (1, 1, height, width) for broadcasting with tensors of shape (B, C, H, W)
    mask = mask.view(1, 1, height, width)

    return mask
