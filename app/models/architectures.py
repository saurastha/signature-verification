from collections import OrderedDict

import numpy as np
from torch import nn
from abc import ABC, abstractmethod
import torch.nn.functional as F


# ------------> Signature Verifier Model Architecture <------------
class SigNet(nn.Module):
    def __init__(self):
        super(SigNet, self).__init__()

        self.feature_space_size = 2048

        self.conv_layers = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", conv_bn_relu(1, 96, 11, stride=4)),
                    ("maxpool1", nn.MaxPool2d(3, 2)),
                    ("conv2", conv_bn_relu(96, 256, 5, pad=2)),
                    ("maxpool2", nn.MaxPool2d(3, 2)),
                    ("conv3", conv_bn_relu(256, 384, 3, pad=1)),
                    ("conv4", conv_bn_relu(384, 384, 3, pad=1)),
                    ("conv5", conv_bn_relu(384, 256, 3, pad=1)),
                    ("maxpool3", nn.MaxPool2d(3, 2)),
                ]
            )
        )

        self.fc_layers = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", linear_bn_relu(256 * 3 * 5, 2048)),
                    ("fc2", linear_bn_relu(self.feature_space_size, self.feature_space_size)),
                ]
            )
        )

    def forward(self, inputs):
        x = self.conv_layers(inputs)
        x = x.view(x.shape[0], 256 * 3 * 5)
        x = self.fc_layers(x)
        return x


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, pad=0):
    return nn.Sequential(
        OrderedDict(
            [
                ("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)),
                ("bn", nn.BatchNorm2d(out_channels)),
                ("relu", nn.ReLU()),
            ]
        )
    )


def linear_bn_relu(in_features, out_features):
    return nn.Sequential(
        OrderedDict(
            [
                ("fc", nn.Linear(in_features, out_features, bias=False)),  # Bias is added after BN
                ("bn", nn.BatchNorm1d(out_features)),
                ("relu", nn.ReLU()),
            ]
        )
    )


# ------------> Signature Cleaner Model Architecture <------------


class NeuralNetwork(nn.Module, ABC):
    """An abstract class representing a neural network.

    All other neural network should subclass it. All subclasses should override
    ``forward``, that makes a prediction for its input argument, and
    ``criterion``, that evaluates the fit between a prediction and a desired
    output. This class inherits from ``nn.Module`` and overloads the method
    ``named_parameters`` such that only parameters that require gradient
    computation are returned. Unlike ``nn.Module``, it also provides a property
    ``device`` that returns the current device in which the network is stored
    (assuming all network parameters are stored on the same device).
    """

    def __init__(self):
        super(NeuralNetwork, self).__init__()

    @property
    def device(self):
        return next(self.parameters()).device

    def named_parameters(self, recurse=True):
        nps = nn.Module.named_parameters(self)
        for name, param in nps:
            if not param.requires_grad:
                continue
            yield name, param

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def criterion(self, y, d):
        pass


class NNRegressor(NeuralNetwork, ABC):

    def __init__(self):
        super(NNRegressor, self).__init__()
        self.mse = nn.MSELoss()

    def criterion(self, y, d):
        return self.mse(y, d)


class DnCNN(NNRegressor):

    def __init__(self, D, C=64):
        super(DnCNN, self).__init__()
        self.D = D

        # convolution layers
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(3, C, 3, padding=1))
        self.conv.extend([nn.Conv2d(C, C, 3, padding=1) for _ in range(D)])
        self.conv.append(nn.Conv2d(C, 3, 3, padding=1))
        # apply He's initialization
        for i in range(len(self.conv[:-1])):
            nn.init.kaiming_normal_(self.conv[i].weight.data, nonlinearity='relu')

        # batch normalization
        self.bn = nn.ModuleList()
        self.bn.extend([nn.BatchNorm2d(C, C) for _ in range(D)])
        # initialize the weights of the Batch normalization layers
        for i in range(D):
            nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(C))

    def forward(self, x):
        D = self.D
        h = F.relu(self.conv[0](x))
        for i in range(D):
            h = F.relu(self.bn[i](self.conv[i + 1](h)))
        y = self.conv[D + 1](h) + x
        return y
