import logging
import time
import warnings

import numpy as np
from sklearn import metrics

logger = logging.getLogger(__name__)


def precision_recall_curve(target, prediction_probas):
    """Precision-recall curve for each non-background class

    Args:
        target: n-d numpy array of integers
        prediction_probas: (n+1)-d numpy array (channel, ...)
            of class probabilities

    Returns:
        dict of dicts: class_id, precision, recall
    """

    start = time.time()

    assert prediction_probas.max() <= 1.0
    assert prediction_probas.min() >= 0.0

    target = target.flatten()

    curves = {}
    for i in range(1, prediction_probas.shape[0]):
        prediction = prediction_probas[i].flatten()

        with warnings.catch_warnings():
            # Catch UndefinedMetricWarning, due to division by 0
            precision, recall, thresholds = metrics.precision_recall_curve(
                target, prediction, pos_label=i)

            thresholds = np.insert(thresholds, 0, 0.0)
            assert len(recall) == len(thresholds)

            curves[i] = {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds
            }

    duration = time.time() - start
    logger.info(
        f"Computed precision recall curves in {duration:.3f} s.")

    return curves
