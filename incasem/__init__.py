from __future__ import absolute_import

# Fixes multiprocessing + gunpowder error for MacOS
import platform
import multiprocessing

if platform.system() == "Darwin":
    multiprocessing.set_start_method("fork", force=True)


from . import gunpowder
from . import utils
from . import torch
from . import pipeline
from . import metrics
from . import automate
from . import logger
