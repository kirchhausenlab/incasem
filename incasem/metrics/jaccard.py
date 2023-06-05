import logging
import time
import numpy as np
from scipy.spatial import distance

logger = logging.getLogger(__name__)


def jaccard(target, prediction_probas, mask=None):
    """Jaccard scores (intersection over union IoU) for all channels in a
    single ROI.

    Args:

        target:

            N-d numpy array of integers.

        prediction_probas:

            N-d or (N+1)-d (channel, d1, d2, ..., dn) numpy array of class
            scores.

            If it is an N-d array or the `channel` dimension is 1, it is
            assumed that `prediciton_probas` is the foreground channel and
            background probabilities `1 - prediction_probas` will be added as
            channel 0.

        mask:

            N-d numpy binary array.

    Returns:

        list: jaccard scores for each class.
    """

    start = time.time()

    assert prediction_probas.max() <= 1.0
    assert prediction_probas.min() >= 0.0

    logger.debug(f"{target.shape=}")
    logger.debug(f"{prediction_probas.shape=}")

    # Add background channel if not given
    if target.shape == prediction_probas.shape:
        prediction_probas = np.array(
            [1.0 - prediction_probas, prediction_probas])

    if (1,) + target.shape == prediction_probas.shape:
        prediction_probas = prediction_probas[0]
        prediction_probas = np.array(
            [1.0 - prediction_probas, prediction_probas])

    num_classes = prediction_probas.shape[0]

    if mask is None:
        mask = np.ones_like(target, dtype=bool)
        logger.debug("Mask not provided, using default array of ones.")
    else:
        mask = mask.astype(bool)

    assert target.shape == mask.shape

    # boolean selection flattens output the array
    target = target[mask]
    prediction_probas = prediction_probas[
        np.broadcast_to(mask, prediction_probas.shape)
    ]
    # reshape predictions to original number of channels
    prediction_probas = prediction_probas.reshape(num_classes, -1)

    assert target.shape[0] == prediction_probas.shape[1], \
        (f"Target shape {target.shape} and prediction shape "
         f"{prediction_probas.shape} do not match.")

    prediction = np.argmax(prediction_probas, axis=0)

    scores = []
    for i in range(num_classes):
        target_bool = target == i
        prediction_bool = prediction == i
        s = 1.0 - \
            distance.pdist(np.array([target_bool, prediction_bool]), 'jaccard')
        scores.append(float(s))

    duration = time.time() - start
    logger.info(f"Computed Jaccard scores in {duration:.3f} s.")
    return scores
