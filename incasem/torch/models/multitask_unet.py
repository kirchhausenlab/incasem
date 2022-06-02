import logging

import torch
from funlib.learn.torch.models import UNet
from funlib.learn.torch.models.unet import ConvPass

logger = logging.getLogger('__name__')


class MultitaskUnet(torch.nn.Module):
    """ Wrapper around funlib.learn.torch.unet
    """

    def __init__(
            self,
            num_fmaps_out_task=2,
            num_fmaps_out_auxiliary=16,
            activation_out_task='Identity',
            activation_out_auxiliary='Sigmoid',
            dims=3,
            in_channels=1,
            num_fmaps=32,
            fmap_inc_factor=2,
            downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
            kernel_size_down=[[(3, 3, 3), (3, 3, 3)]] * 4,
            kernel_size_up=[[(3, 3, 3), (3, 3, 3)]] * 3,
            activation='ReLU',
            voxel_size=(1, 1, 1),
            constant_upsample=True,
            padding='valid',
    ):
        super().__init__()

        self.num_fmaps_out_task = num_fmaps_out_task

        unet = UNet(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            activation=activation,
            voxel_size=voxel_size,
            constant_upsample=constant_upsample,
            padding=padding,
        )
        self.add_module('unet', unet)

        conv_pass_task = ConvPass(
            in_channels=num_fmaps,
            out_channels=num_fmaps_out_task,
            kernel_sizes=[(3,) * dims],
            activation=activation_out_task,
            padding='valid',
        )
        self.add_module('conv_pass_task', conv_pass_task)

        conv_pass_aux = ConvPass(
            in_channels=num_fmaps,
            out_channels=num_fmaps_out_auxiliary,
            kernel_sizes=[(3,) * dims],
            activation=activation_out_auxiliary,
            padding='valid',
        )
        self.add_module('conv_pass_aux', conv_pass_aux)

    def forward(self, x):
        """forward.

        Args:
            x:
        """
        y = self.unet(x)
        head_task = self.conv_pass_task(y)
        head_aux = self.conv_pass_aux(y)

        return head_task, head_aux
