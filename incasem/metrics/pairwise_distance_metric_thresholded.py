import logging
import time
import numpy as np
from scipy.spatial import distance

logger = logging.getLogger(__name__)


def pairwise_distance_metric_thresholded(
        target,
        prediction_probas,
        metric,
        threshold,
        foreground_class,
        mask=None,
):
    """Binary metric for `foreground_class` based on distances
    in `scipy.special.distance`.

    Args:

        target:

            n-d numpy array of integers.

        prediction_probas:

            (n+1)-d (channel, ...) numpy array of class scores.
            Channel 0 is assumed to be background.

        threshold (float):

            Probability threshold in [0,1].

        foreground_class (int):

            Foreground for the binary metric.

        mask:

           n-d numpy array of binary integers to exclude certain positions.

    Returns:

        float: metric
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

    target = target == foreground_class
    prediction_probas = prediction_probas[foreground_class]

    assert target.shape[0] == prediction_probas.shape[0], \
        (f"Target shape {target.shape} and prediction shape "
         f"{prediction_probas.shape} do not match.")

    prediction = prediction_probas >= threshold

    score = 1 - distance.pdist(np.array([target, prediction]), metric=metric)
    score = float(score)

    duration = time.time() - start
    logger.debug(f"Computed {metric} similarity in {duration:.3f} s.")

    return score
