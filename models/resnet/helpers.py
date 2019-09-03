import os

import numpy as np

import torch
import torch.nn as nn
import torchvision.utils


# TODO: Batch norm correct?
# TODO: Correct to use 1x1 convolution as projection? Do we need it?
class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            init_to_identity=False,
            weight_norm=False,
            batch_norm=True
    ):
        super().__init__()
        self.conv1 = self._get_conv3x3(in_channels, out_channels, weight_norm)
        self.bn1 = self._get_batch_norm(out_channels, enable=batch_norm)
        self.conv2 = self._get_conv3x3(out_channels, out_channels, weight_norm)
        self.bn2 = self._get_batch_norm(out_channels, enable=batch_norm)
        self.relu = nn.ReLU()

        if in_channels != out_channels:
            self.proj = self._get_conv1x1(in_channels, out_channels)
        else:
            self.proj = nn.Identity()

        if init_to_identity:
            for p in self.conv2.parameters():
                p.data.zero_()

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.proj(inputs)

        return out

    def _get_batch_norm(self, channels, enable):
        if enable:
            return nn.BatchNorm2d(channels)
        else:
            return nn.Identity

    def _get_conv1x1(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _get_conv3x3(self, in_channels, out_channels, weight_norm):
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        if weight_norm:
            conv = nn.utils.weight_norm(conv)

        return conv


def get_mlp(input_dim, hidden_layer_dims, output_dim, activation, log_softmax_outputs=False):
    layers = []
    prev_dim = input_dim
    for dim in hidden_layer_dims:
        layers.append(nn.Linear(prev_dim, dim))
        layers.append(activation())
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, output_dim))

    if log_softmax_outputs:
        layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


class ConstantMap(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer("value", value)

    def forward(self, inputs):
        return self.value.expand(inputs.shape[0], *self.value.shape[1:])


# TODO: Remove duplication with masked_map.py
class MultiHeadMLP(nn.Module):
    def __init__(
            self,
            input_shape,
            hidden_layer_dims,
            output_shape,
            output_names,
            activation
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.output_names = output_names
        input_dim = int(np.prod(input_shape))
        output_dim = len(output_names) * int(np.prod(output_shape))
        self._mlp = get_mlp(input_dim, hidden_layer_dims, output_dim, activation)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        assert inputs.shape[1:] == self.input_shape
        flat_inputs = inputs.flatten(start_dim=1)
        flat_outputs = self._mlp(flat_inputs).chunk(len(self.output_names), dim=1)
        outputs = (o.view(batch_size, *self.output_shape) for o in flat_outputs)
        return dict(zip(self.output_names, outputs))


def params_norm(params):
    norm = torch.tensor(0.)
    for param in params:
        norm += torch.norm(param)**2
    return torch.sqrt(norm).item()


def params_grad_norm(params):
    norm = torch.tensor(0.)
    for param in params:
        if param.grad is not None:
            norm += param.grad.norm()**2
    return torch.sqrt(norm).item()


def one_hot(y, num_classes, dtype=torch.float):
    y_one_hot = torch.zeros(len(y), num_classes, device=y.device, dtype=dtype)
    y_one_hot.scatter_(dim=1, index=y.view(-1, 1), value=1)
    return y_one_hot


def isreal(x):
    return torch.all(torch.isfinite(x))


def isreal_params(params):
    for param in params:
        if not isreal(param):
            return False
    return True


def isreal_params_grad(params):
    for param in params:
        if not isreal(param.grad):
            return False
    return True


def num_params(module):
    return sum(p.view(-1).shape[0] for p in module.parameters())


def pass_through(f, loader, return_x=False, return_y=False):
    f_x = []

    if return_x:
        x = []

    if return_y:
        y = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            f_x.append(f(x_batch).cpu())

            if return_x:
                x.append(x_batch.cpu())

            if return_y:
                y.append(y_batch.cpu())

    f_x = torch.cat(f_x, dim=0)

    if return_x:
        x = torch.cat(x, dim=0)

    if return_y:
        y = torch.cat(y, dim=0)

    if return_x and return_y:
        return f_x, x, y
    elif return_x:
        return f_x, x
    elif return_y:
        return f_x, y
    else:
        return f_x


def data_dim(loader):
    x_batch, _ = next(iter(loader))
    _, dim = x_batch.shape
    return dim
