import logging

import torch
from funlib.learn.torch.models import UNet
from funlib.learn.torch.models.unet import ConvPass

logger = logging.getLogger('__name__')


class Unet(torch.nn.Module):
    """ Wrapper around funlib.learn.torch.unet
    """

    def __init__(
            self,
            num_fmaps,
            num_fmaps_out,
            dims=3,
            **kwargs,
    ):
        super().__init__()

        unet = UNet(
            num_fmaps=num_fmaps,
            **kwargs
        )
        self.add_module('unet', unet)

        final_conv_pass = ConvPass(
            in_channels=num_fmaps,
            out_channels=num_fmaps_out,
            kernel_sizes=[(3,) * dims],
            activation='Identity',
            padding='valid',
        )
        self.add_module('final_conv_pass', final_conv_pass)

    def forward(self, x):
        """forward.

        Args:
            x:
        """
        y = self.unet(x)
        out = self.final_conv_pass(y)
        return out
