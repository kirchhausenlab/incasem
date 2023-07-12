import logging

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OneConv3d(torch.nn.Module):
    """OneConv3d.
    """

    def __init__(self, out_channels=2):
        super().__init__()

        self.layer = torch.nn.Conv3d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        logging.debug(f'{x.shape=}')
        out = self.layer(x)
        logging.debug(f'{out.shape=}')
        return out
