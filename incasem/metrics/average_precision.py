import logging
import time
import warnings

import numpy as np
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)


def average_precision(target, prediction_probas):
    """Average precision scores for each non-background class

    Args:
        target: n-d numpy array of integers
        prediction_probas: (n+1)-d numpy array (channel, ...)
            of class probabilities

    Returns:
        list: average precision for each class
    """

    start = time.time()

    assert prediction_probas.max() <= 1.0
    assert prediction_probas.min() >= 0.0

    # one-hot encode targets
    target = target.flatten()
    target = np.eye(prediction_probas.shape[0])[target]

    prediction = prediction_probas.reshape(
        (prediction_probas.shape[0], -1)).transpose()

    with warnings.catch_warnings():
        # Catch UndefinedMetricWarning, which informs about AP being
        # set to 0.0 due to division by zero in precision or recall
        scores = average_precision_score(target, prediction, average=None)

    duration = time.time() - start
    logger.info(f"Computed average precision in {duration:.3f} s.")

    return scores
