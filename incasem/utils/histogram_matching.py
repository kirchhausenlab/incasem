# adapted from
# https://github.com/scikit-image/scikit-image/blob/master/skimage/exposure/histogram_matching.py

import numpy as np


def match_histograms_masked(source, template, source_mask, template_mask):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices = np.unique(
        source.ravel(), return_inverse=True)

    source_masked = source[source_mask.astype(np.bool)]
    src_values_masked, src_counts = \
        np.unique(source_masked.ravel(), return_counts=True)

    assert np.all(src_values == src_values_masked)

    template_masked = template[template_mask.astype(np.bool)]
    tmpl_values, tmpl_counts = np.unique(
        template_masked.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source_masked.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template_masked.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)

    # TODO only replace values inside mask
    return interp_a_values[src_unique_indices].reshape(source.shape)
