import os
import argparse
import json
import webbrowser

import numpy as np
import neuroglancer

from funlib.persistence import Array, open_ds, prepare_ds
from funlib.geometry import Roi, Coordinate
from funlib.show.neuroglancer import add_layer


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    '--dataset_config',
    '-d',
    nargs='+',
    required=True,
    help="The dataset config to visualize.")
parser.add_argument(
    '--data_prefix',
    '-p',
    default='/scratch/fiborganelles/data/',
    help="Location of used zarr containers.")
parser.add_argument(
    '--context',
    type=int,
    default=47,
    help="Context needed for the predictions, no score/loss calculated here."
)
parser.add_argument(
    '--only_cell',
    '-o',
    default=None,
    help="Visualize only ROIs from one cell."
)
parser.add_argument(
    '--no-browser',
    '-n',
    type=bool,
    nargs='?',
    default=False,
    const=True,
    help="If set, do not open a browser, just print a URL")
parser.add_argument(
    '--serve',
    action='store_true',
    help="Serve neuroglancer on public IP."
)

args = parser.parse_args()

if args.serve:
    neuroglancer.set_server_bind_address('0.0.0.0')
else:
    neuroglancer.set_server_bind_address()

viewer = neuroglancer.Viewer()

sources = {}
for config in args.dataset_config:
    with open(config, 'r') as f:
        new_sources = json.load(f)
        for k, v in new_sources.items():
            if k in sources:
                raise ValueError((
                    f"ROI `{k}` already loaded from another file, "
                    "please use unique ROI nicknames."
                ))
            sources[k] = v

# Sort ROIs by cell in a dict of dicts
cells = {}
for nickname, attributes in sources.items():
    cell = attributes['file'].rstrip('/')
    if cell not in cells:
        cells[cell] = {nickname: attributes}
    else:
        cells[cell][nickname] = attributes

# Put the raw and the labels
if args.only_cell:
    cell = cells[args.only_cell]
    roi = list(cell.values())[0]
else:
    if len(cells) > 1:
        raise ValueError(
            (
                "I cannot visualize ROIs in different cells at once. "
                "Please add the `--only_cell` cmd option to"
                "choose one of the following cells:\n"
            ) + '\n'.join(list(cells.keys()))
        )

    cell = list(cells.values())[0]
    roi = list(cell.values())[0]

file_path = os.path.expanduser(
    os.path.join(args.data_prefix, roi['file'].rstrip('/'))
)
raw = open_ds(file_path, roi['raw'])
labels = open_ds(file_path, list(roi['labels'].keys())[0])
offset = Coordinate(roi['offset'])
shape = Coordinate(roi['shape'])
voxel_size = Coordinate(roi['voxel_size'])


def add_bounding_box(s, nickname, attributes, context):
    print(f"Adding bounding box for {nickname}.")
    offset = Coordinate(attributes['offset'])
    print(f"{offset=}")
    shape = Coordinate(attributes['shape'])
    print(f"{shape=}")
    context = Coordinate((context,) * 3)
    print(f"{context=}")

    roi = neuroglancer.AxisAlignedBoundingBoxAnnotation(
        point_a=np.array(offset, dtype=np.int64),
        point_b=np.array(offset + shape, dtype=np.int64),
        id=1,
        props=['salmon'],
    )
    predictions_roi = neuroglancer.AxisAlignedBoundingBoxAnnotation(
        point_a=np.array(offset + context, dtype=np.int64),
        point_b=np.array(offset + shape - context, dtype=np.int64),
        id=2,
        props=['lightgreen'],
    )

    s.layers[nickname] = neuroglancer.LocalAnnotationLayer(
        dimensions=s.dimensions,
        annotations=[roi, predictions_roi],
        annotation_properties=[
            neuroglancer.AnnotationPropertySpec(
                id='color',
                type='rgb',
                default='red',
            ),
        ],
        shader='''
        void main() {
        setBoundingBoxBorderColor(prop_color());
        }
        ''',
        # annotation_color='green',
    )


with viewer.txn() as s:
    add_layer(context=s, array=raw, name='raw')
    add_layer(context=s, array=labels, name='labels')

    s.dimensions = neuroglancer.CoordinateSpace(
        names=["z", "y", "x"], units="nm", scales=list(voxel_size))
    s.position = list(offset + (shape / 2))

    # Draw the bounding boxes
    for name, attributes in cell.items():
        new_file_path = os.path.expanduser(
            os.path.join(args.data_prefix, attributes['file'].rstrip('/'))
        )
        add_bounding_box(s, name, attributes, args.context)


url = str(viewer)
print(url)
if os.environ.get("DISPLAY") and not args.no_browser:
    webbrowser.open_new_tab(url)

print("Press ENTER to quit")
input()