import logging
import time
import numpy as np
from sklearn import metrics

logger = logging.getLogger(__name__)


def confusion_matrix(target, prediction_probas, mask=None):
    """Normalized confusion matrix for multiclass prediction

    Args:
        target: n-d numpy array of integers
        prediction_probas: (n+1)-d (channel, ...) numpy array
            of class scores. Channel 0 is assumed to be background
        mask: n-d numpy binary array

    Returns:
        np.array: 2d confusion matrix
    """

    # Channel 0 is assumed to be background
    start = time.time()

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

    prediction = np.argmax(prediction_probas, axis=0)

    if num_classes == 2:
        scores = np.bincount(
            target.astype(bool) * 2 + prediction.astype(bool),
            minlength=4
        ).reshape(2, 2)
        scores = scores / scores.sum()
    else:
        logger.warning(
            f"Confusion matrix for multiclass scenario is incredibly slow.")
        scores = metrics.confusion_matrix(target, prediction, normalize='all')

    duration = time.time() - start
    logger.info(f"Computed confusion matrix in {duration:.3f} s.")
    return scores
