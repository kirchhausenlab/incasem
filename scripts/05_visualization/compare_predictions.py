import argparse
import glob
import os
import webbrowser
import logging

import neuroglancer
from funlib.persistence import Array, open_ds, prepare_ds
from funlib.show.neuroglancer import add_layer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


choices = ['xy', 'xz', 'yz'],


def compare_predictions(
        input_file,
        raw_dataset,
        labels_datasets,
        predictions_file,
        predictions_series,
        iterations,
        layout='xy',
        shader='heatmap',
        predictions_path_prefix='volumes/predictions',
        predictions_path_suffix='prob_maps/class_1',
        serve=False):

    if serve:
        neuroglancer.set_server_bind_address('0.0.0.0')
    else:
        neuroglancer.set_server_bind_address()

    viewer = neuroglancer.Viewer()

    if len(iterations) == 1:
        iterations = [iterations[0]] * len(predictions_series)
    num_iterations = [len(x) for x in iterations]
    if num_iterations.count(num_iterations[0]) != len(num_iterations):
        logger.warning(
            f"Using different number of iterations for different models. This will lead to an unpleasant grid layout.")
        logger.warning(f"{iterations=}")

    arrays = []
    datasets = []

    for s, iterations_s in zip(predictions_series, iterations):
        arrays_series = []
        datasets_series = []
        for i in iterations_s:
            ds = os.path.join(
                predictions_path_prefix,
                s,
                f"iteration_{i:06d}",
                predictions_path_suffix
            )
            datasets_series.append(ds)

            logger.debug(f"Adding {predictions_file}, {ds}")
            a = open_ds(predictions_file, ds)

            arrays_series.append(a)

        arrays.append(arrays_series)
        datasets.append(datasets_series)

        with viewer.txn() as s:
            for array, dataset in zip(arrays_series, datasets_series):
                add_layer(
                    context=s,
                    array=array,
                    name=dataset,
                    shader=shader
                )

    input_datasets = [raw_dataset]
    input_arrays = [open_ds(input_file, raw_dataset)]

    if labels_datasets is not None:
        input_datasets.extend(labels_datasets)
        for ds in labels_datasets:
            a = open_ds(input_file, ds)
            input_arrays.append(a)

    with viewer.txn() as s:
        for array, dataset in zip(input_arrays, input_datasets):
            add_layer(
                context=s,
                array=array,
                name=dataset
            )

    with viewer.txn() as s:
        # Grid Layout
        s.layout = neuroglancer.column_layout([
            neuroglancer.row_layout([
                neuroglancer.LayerGroupViewer(
                    layout=layout, layers=input_datasets)
                for i in iterations[0]
            ]),
            *[
                neuroglancer.row_layout([
                    neuroglancer.LayerGroupViewer(
                        layout=layout,
                        layers=[input_datasets[0], iteration])
                    for iteration in series
                ]) for series in datasets
            ],
        ])

    url = str(viewer)
    logger.info(f"\n\t\t{url}\n")
    webbrowser.open_new_tab(url)

    logger.info("Press ENTER to quit")
    input()


def str_rstrip_slash(x):
    return x.rstrip('/')


def reverse_axes_layout(x):
    if x == 'xy':
        return 'yz'
    elif x == 'yz':
        return 'xy'
    else:
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # base files: raw and groundtruth, if it exists.
    parser.add_argument(
        '--input_file',
        type=str_rstrip_slash,
        help="The path to the container to show."
    )
    parser.add_argument(
        '--raw_dataset',
        type=str_rstrip_slash,
        help='Raw EM dataset.'
    )
    parser.add_argument(
        '--labels_datasets',
        type=str_rstrip_slash,
        nargs='*',
        default=None,
        help='One or multiple labels datasets.'
    )

    # the predictions to be displayed in grid view
    parser.add_argument(
        '--predictions_file',
        type=str_rstrip_slash,
        help='Path to the .zarr container with all predictions.'
    )
    parser.add_argument(
        '--predictions_series',
        type=str_rstrip_slash,
        action='append',
        help='Path to the zarr group for a prediction.'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        nargs='+',
        action='append',
        help='All the iteration numbers to display.'
    )
    parser.add_argument(
        '--layout',
        type=reverse_axes_layout,
        choices=['xy', 'xz', 'yz'],
        default='xy',
        help='The orthogonal view to pick.'
    )
    parser.add_argument(
        '--shader',
        choices=['default', 'rgb', 'mask', 'heatmap', 'probmap'],
        default='heatmap',
        help='The shaders to be used for all predictions.'
    )
    parser.add_argument(
        '--predictions_path_prefix',
        type=str_rstrip_slash,
        default='volumes/predictions',
        help=(
            'Prefix path to the predictions to display, '
            'equal for all predictions.'
        )
    )
    parser.add_argument(
        '--predictions_path_suffix',
        type=str_rstrip_slash,
        default='prob_maps/class_1',
        help=(
            'Suffix path to the predictions to display, '
            'equal for all predictions.'
        )
    )

    # neuroglancer
    parser.add_argument(
        '--serve',
        action='store_true',
        help='Serve neuroglancer on public IP'
    )

    args = parser.parse_args()

    compare_predictions(
        input_file=args.input_file,
        raw_dataset=args.raw_dataset,
        labels_datasets=args.labels_datasets,
        predictions_file=args.predictions_file,
        predictions_series=args.predictions_series,
        iterations=args.iterations,
        layout=args.layout,
        shader=args.shader,
        predictions_path_prefix=args.predictions_path_prefix,
        predictions_path_suffix=args.predictions_path_suffix,
        serve=args.serve,
    )