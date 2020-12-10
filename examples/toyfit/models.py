# flake8: noqa
import numpy as np

import torch
import torch.nn as nn

from resnet import resnet18, resnext101_32x8d

# TODO: add conv models


class SimpleGenerator(nn.Module):
    """Simple fully-connected generator"""
    def __init__(
        self#,
        #noise_dim=10,
        #hidden_dim=256,
        #image_resolution=(28, 28),
        #channels=1
    ):
        """

        :param noise_dim:
        :param hidden_dim:
        :param image_resolution:
        :param channels:
        """
        super().__init__()
        #self.noise_dim = noise_dim
        #self.image_resolution = image_resolution
        #self.channels = channels
        self.resnet18 = True

        self.net = resnet18(pretrained=False, progress=True).cuda()

        """nn.Sequential(
            nn.Linear(noise_dim, hidden_dim), nn.LeakyReLU(0.05),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.05),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.05),
            nn.Linear(hidden_dim, np.prod(image_resolution)), nn.Tanh()
        )"""

    def forward(self, x):
        x = self.net(x)
        return x


## Multilayer LSTM based classifier taking in 200 dimensional fixed time series inputs
class LSTMClassifier(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes, batch_size):
        super(LSTMClassifier, self).__init__()
        self.arch = 'lstm'
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_dir = 2 if bidirectional else 1
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional
            )

        self.hidden2label = nn.Sequential(
            nn.Linear(hidden_dim*self.num_dir, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid()
        )

        # self.hidden = self.init_hidden()

    def init_hidden(self):
        if cuda:
            h0 = Variable(torch.zeros(self.num_layers*self.num_dir, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(self.num_layers*self.num_dir, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers*self.num_dir, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(self.num_layers*self.num_dir, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, x): # x is (batch_size, 1, 200), permute to (200, batch_size, 1)
        x = x.permute(2, 0, 1)
        # See: https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426/2
        lstm_out, (h, c) = self.lstm(x, self.init_hidden())
        y  = self.hidden2label(lstm_out[-1])
        return y


class SimpleDiscriminator(nn.Module):
    def __init__(self, image_resolution=(28, 28), channels=1, hidden_dim=100):
        super().__init__()
        self.image_resolution = image_resolution
        self.channels = channels

        self.net = LSTMClassifier(
            in_dim=2,
            hidden_dim=120,
            num_layers=3,
            dropout=0.8,
            bidirectional=True,
            num_classes=1,#bce loss for discriminator
            batch_size=256
        )

    def forward(self, x):
        x = self.net(x)
        return x





__all__ = [
    "SimpleGenerator", "SimpleDiscriminator"
]
