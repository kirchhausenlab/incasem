from __future__ import absolute_import

import multiprocessing

# Fixes multiprocessing + gunpowder error for MacOS
import platform

if platform.system() == "Darwin":
    multiprocessing.set_start_method("fork", force=True)


from . import automate, gunpowder, incasem, logger, metrics, pipeline, torch, utils
