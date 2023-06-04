import logging
import time
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def precision_recall(target, prediction_probas, mask=None, threshold=None):
    """Argmax-based precision-recall for each class

    Args:
        target:

            n-d numpy array of integers

        prediction_probas:

            (n+1)-d numpy array (channel, ...) of class probabilities

        threshold (float):

            Probability threshold in [0,1]. Perform `argmax` if not passed.

    Returns:

        list of tuples: precision, recall for each class
    """
    start = time.time()

    assert prediction_probas.max() <= 1.0
    assert prediction_probas.min() >= 0.0

    if mask is None:
        mask = np.ones_like(target, dtype=bool)
        logger.debug("Mask not provided, using default array of ones.")
    else:
        mask = mask.astype(bool)

    assert target.shape == mask.shape

    # boolean selection flattens output the array
    target = target[mask]
    num_classes = prediction_probas.shape[0]
    prediction_probas = prediction_probas[
        np.broadcast_to(mask, prediction_probas.shape)
    ]
    # reshape predictions to original number of channels
    prediction_probas = prediction_probas.reshape(num_classes, -1)

    assert target.shape[0] == prediction_probas.shape[1], \
        (f"Target shape {target.shape} and prediction shape "
         f"{prediction_probas.shape} do not match.")

    if threshold is None:
        prediction = np.argmax(prediction_probas, axis=0)

    scores = []
    for i in range(num_classes):
        target_bool = target == i

        if threshold is None:
            # use argmax
            prediction_bool = prediction == i
        else:
            prediction_bool = prediction_probas[i] >= threshold

        _, fp, fn, tp = np.bincount(
            target_bool * 2 + prediction_bool,
            minlength=4
        )
        if tp == 0 and fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        if tp == 0 and fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)

        scores.append((precision, recall))

    duration = time.time() - start
    logger.info(
        f"Computed precision and recall in {duration:.3f} s.")

    return scores
