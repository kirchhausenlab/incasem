# #!/usr/bin/env python
# """
# Compare predictions with Neuroglancer using Dask for parallel data loading.
#
# This script loads prediction probability maps (and optional labels) from a Zarr container,
# adds them as layers (with a chosen shader) to a neuroglancer viewer, and sets up an
# interactive grid layout. Prediction arrays for different iterations are loaded in parallel
# using Dask delayed to speed up the process.
#
# Dependencies:
#   - python 3.8+
#   - neuroglancer
#   - funlib.persistence
#   - funlib.geometry
#   - funlib.show.neuroglancer (for add_layer)
#   - dask[distributed]
#   - configargparse
#   - numpy
#   - webbrowser
#   - logging
# """
#
# import os
# import logging
# import webbrowser
# from time import time as now
#
# import neuroglancer
# from funlib.persistence import open_ds  # Keep funlib persistence.
# from funlib.geometry import Roi, Coordinate
# import configargparse as argparse
# import numpy as np
#
# from funlib.show.neuroglancer import add_layer
#
# import dask
# from dask.distributed import Client, as_completed
# from tqdm import tqdm
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
#
#
# # A helper to remove trailing slash.
# def str_rstrip_slash(x):
#     return x.rstrip('/')
#
#
# # A helper to reverse the axes layout.
# def reverse_axes_layout(x):
#     if x == 'xy':
#         return 'yz'
#     elif x == 'yz':
#         return 'xy'
#     else:
#         return x
#
#
# def compare_predictions(
#         input_file,
#         raw_dataset,
#         labels_datasets,
#         predictions_file,
#         predictions_series,
#         iterations,
#         layout='xy',
#         shader='heatmap',
#         predictions_path_prefix='volumes/predictions',
#         predictions_path_suffix='prob_maps/class_1',
#         serve=False):
#     """
#     Set up a neuroglancer viewer to compare predictions.
#
#     This function loads prediction arrays from the specified Zarr container for all given iterations
#     (loading in parallel using Dask delayed), then adds them (with the chosen shader) along with
#     the raw data and optional labels as layers. The viewer layout is configured as a grid.
#     """
#     if serve:
#         neuroglancer.set_server_bind_address('0.0.0.0')
#     else:
#         neuroglancer.set_server_bind_address()
#
#     viewer = neuroglancer.Viewer()
#
#     # Ensure iterations is a list per series.
#     if len(iterations) == 1:
#         iterations = [iterations[0]] * len(predictions_series)
#     num_iterations = [len(x) for x in iterations]
#     if num_iterations.count(num_iterations[0]) != len(num_iterations):
#         logger.warning(
#             f"Different number of iterations for different models. This may affect grid layout.")
#         logger.warning(f"{iterations=}")
#
#     arrays = []
#     datasets = []
#
#     client = Client()  # Start a Dask client for parallel loading.
#     delayed_tasks = []
#
#     for series, iterations_series in zip(predictions_series, iterations):
#         arrays_series = []
#         datasets_series = []
#         for i in iterations_series:
#             ds_path = os.path.join(
#                 predictions_path_prefix,
#                 series,
#                 f"iteration_{i:06d}",
#                 predictions_path_suffix
#             )
#             datasets_series.append(ds_path)
#             logger.debug(f"Scheduling open_ds for {predictions_file}, {ds_path}")
#             # Use dask.delayed to parallelize open_ds calls.
#             task = dask.delayed(open_ds)(predictions_file, ds_path)
#             arrays_series.append(task)
#         arrays.append(arrays_series)
#         datasets.append(datasets_series)
#     # Compute all prediction arrays concurrently.
#     all_futures = client.compute([task for series in arrays for task in series])
#     results = client.gather(all_futures)
#     # Reconstruct the arrays list with computed results.
#     computed_arrays = []
#     idx = 0
#     for series in arrays:
#         n = len(series)
#         computed_arrays.append(results[idx:idx + n])
#         idx += n
#     arrays = computed_arrays
#
#     # Add prediction layers.
#     with viewer.txn() as s:
#         for series_arrays, series_datasets in zip(arrays, datasets):
#             for array, ds_name in zip(series_arrays, series_datasets):
#                 add_layer(
#                     context=s,
#                     array=array,
#                     name=ds_name,
#                     shader=shader
#                 )
#
#     # Add raw and labels layers.
#     input_datasets = [raw_dataset]
#     input_arrays = [open_ds(input_file, raw_dataset)]
#     if labels_datasets is not None:
#         input_datasets.extend(labels_datasets)
#         for ds in labels_datasets:
#             a = open_ds(input_file, ds)
#             input_arrays.append(a)
#     with viewer.txn() as s:
#         for array, ds_name in zip(input_arrays, input_datasets):
#             add_layer(
#                 context=s,
#                 array=array,
#                 name=ds_name
#             )
#
#     # Setup grid layout.
#     with viewer.txn() as s:
#         s.layout = neuroglancer.column_layout([
#             neuroglancer.row_layout([
#                 neuroglancer.LayerGroupViewer(
#                     layout=layout, layers=input_datasets)
#                 for i in iterations[0]
#             ]),
#             *[
#                 neuroglancer.row_layout([
#                     neuroglancer.LayerGroupViewer(
#                         layout=layout,
#                         layers=[input_datasets[0], iteration])
#                     for iteration in series
#                 ]) for series in datasets
#             ],
#         ])
#
#     url = str(viewer)
#     logger.info(f"\n{url}\n")
#     webbrowser.open_new_tab(url)
#
#     logger.info("Press ENTER to quit")
#     input()
#     client.close()
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description="Compare predictions in Neuroglancer.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
#     parser.add_argument('--input_file', type=str_rstrip_slash, help="Path to the container to show.")
#     parser.add_argument('--raw_dataset', type=str_rstrip_slash, help='Raw EM dataset.')
#     parser.add_argument('--labels_datasets', type=str_rstrip_slash, nargs='*', default=None,
#                         help='One or multiple labels datasets.')
#     parser.add_argument('--predictions_file', type=str_rstrip_slash,
#                         help='Path to the .zarr container with all predictions.')
#     parser.add_argument('--predictions_series', type=str_rstrip_slash, action='append',
#                         help='Path to the zarr group for a prediction.')
#     parser.add_argument('--iterations', type=int, nargs='+', action='append',
#                         help='All the iteration numbers to display.')
#     parser.add_argument('--layout', type=reverse_axes_layout, choices=['xy', 'xz', 'yz'],
#                         default='xy', help='The orthogonal view to pick.')
#     parser.add_argument('--shader', choices=['default', 'rgb', 'mask', 'heatmap', 'probmap'],
#                         default='heatmap', help='The shader to be used for all predictions.')
#     parser.add_argument('--predictions_path_prefix', type=str_rstrip_slash, default='volumes/predictions',
#                         help='Prefix path to the predictions to display, equal for all predictions.')
#     parser.add_argument('--predictions_path_suffix', type=str_rstrip_slash, default='prob_maps/class_1',
#                         help='Suffix path to the predictions to display, equal for all predictions.')
#     parser.add_argument('--serve', action='store_true', help='Serve neuroglancer on public IP')
#     args = parser.parse_args()
#
#     compare_predictions(
#         input_file=args.input_file,
#         raw_dataset=args.raw_dataset,
#         labels_datasets=args.labels_datasets,
#         predictions_file=args.predictions_file,
#         predictions_series=args.predictions_series,
#         iterations=args.iterations,
#         layout=args.layout,
#         shader=args.shader,
#         predictions_path_prefix=args.predictions_path_prefix,
#         predictions_path_suffix=args.predictions_path_suffix,
#         serve=args.serve,
#     )
