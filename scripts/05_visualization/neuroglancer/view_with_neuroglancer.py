#!/usr/bin/env python

from daisy import Roi, Coordinate, open_ds

from utils import add_layer


import argparse
import glob
import neuroglancer
import os
import webbrowser
import numpy as np
import zarr


def str_rstrip_slash(x):
    return x.rstrip("/")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--file",
    "-f",
    type=str_rstrip_slash,
    action="append",
    help="The path to the container to show",
)
parser.add_argument(
    "--datasets",
    "-d",
    type=str,
    nargs="+",
    action="append",
    help="The datasets in the container to show",
)
parser.add_argument(
    "--graphs",
    "-g",
    type=str,
    nargs="+",
    action="append",
    help="The graphs in the container to show",
)
parser.add_argument(
    "--no-browser",
    "-n",
    type=bool,
    nargs="?",
    default=False,
    const=True,
    help="If set, do not open a browser, just print a URL",
)
parser.add_argument(
    "--serve", action="store_true", help="Serve neuroglancer on public IP."
)
parser.add_argument(
    "--shaders",
    "-s",
    type=str,
    choices=["default", "rgb", "mask", "heatmap", "probmap"],
    default=None,
    nargs="+",
    action="append",
    help=(
        "The shaders to be used for each dataset. "
        "Defaults to `mask` for single channel, "
        "`rgb` for three channel datasets."
    ),
)
parser.add_argument(
    "--add_prefix", action="store_true", help="Add a prefix to each dataset name"
)

args = parser.parse_args()

if args.serve:
    neuroglancer.set_server_bind_address("0.0.0.0")
else:
    neuroglancer.set_server_bind_address()

viewer = neuroglancer.Viewer()


def to_slice(slice_str):
    values = [int(x) for x in slice_str.split(":")]
    if len(values) == 1:
        return values[0]

    return slice(*values)


def parse_ds_name(ds):
    tokens = ds.split("[")

    if len(tokens) == 1:
        return ds, None

    ds, slices = tokens
    slices = list(map(to_slice, slices.rstrip("]").split(",")))

    return ds, slices


class Project:
    def __init__(self, array, dim, value):
        self.array = array
        self.dim = dim
        self.value = value
        self.shape = array.shape[: self.dim] + array.shape[self.dim + 1 :]
        self.dtype = array.dtype

    def __getitem__(self, key):
        slices = key[: self.dim] + (self.value,) + key[self.dim :]
        ret = self.array[slices]
        return ret


def slice_dataset(a, slices):
    dims = a.roi.dims()

    for d, s in list(enumerate(slices))[::-1]:
        if isinstance(s, slice):
            raise NotImplementedError("Slicing not yet implemented!")
        else:
            index = (s - a.roi.get_begin()[d]) // a.voxel_size[d]
            a.data = Project(a.data, d, index)
            a.roi = Roi(
                a.roi.get_begin()[:d] + a.roi.get_begin()[d + 1 :],
                a.roi.get_shape()[:d] + a.roi.get_shape()[d + 1 :],
            )
            a.voxel_size = a.voxel_size[:d] + a.voxel_size[d + 1 :]

    return a


def open_dataset(f, ds):
    original_ds = ds
    ds, slices = parse_ds_name(ds)
    slices_str = original_ds[len(ds) :]

    try:
        dataset_as = []
        if all(key.startswith("s") for key in zarr.open(f)[ds].keys()):
            raise AttributeError("This group is a multiscale array!")
        for key in zarr.open(f)[ds].keys():
            dataset_as.extend(open_dataset(f, f"{ds}/{key}{slices_str}"))
        return dataset_as
    except AttributeError as e:
        # dataset is an array, not a group
        pass

    print("ds    :", ds)
    print("slices:", slices)
    try:
        zarr.open(f)[ds].keys()
        is_multiscale = True
    except BaseException:
        is_multiscale = False

    if not is_multiscale:
        a = open_ds(f, ds)

        if slices is not None:
            a = slice_dataset(a, slices)

        if a.roi.dims() == 2:
            print("ROI is 2D, recruiting next channel to z dimension")
            a.roi = Roi((0,) + a.roi.get_begin(), (a.shape[-3],) + a.roi.get_shape())
            a.voxel_size = Coordinate((1,) + a.voxel_size)

        if a.roi.dims() == 4:
            print("ROI is 4D, stripping first dimension and treat as channels")
            a.roi = Roi(a.roi.get_begin()[1:], a.roi.get_shape()[1:])
            a.voxel_size = Coordinate(a.voxel_size[1:])

        if a.data.dtype == np.int64 or a.data.dtype == np.int16:
            print("Converting dtype in memory...")
            a.data = a.data[:].astype(np.uint64)

        return [(a, ds)]
    else:
        return [([open_ds(f, f"{ds}/{key}") for key in zarr.open(f)[ds].keys()], ds)]


if args.shaders is None:
    args.shaders = [None] * len(args.file)
else:
    assert len(args.shaders) == len(args.file)

for f, datasets, shaders in zip(args.file, args.datasets, args.shaders):
    name_prefix = "/".join(f.strip("/").split("/")[-2:])
    arrays = []
    for ds in datasets:
        try:
            print("Adding %s, %s" % (f, ds))
            dataset_as = open_dataset(f, ds)

        except Exception as e:
            print(type(e), e)
            print("Didn't work, checking if this is multi-res...")

            scales = glob.glob(os.path.join(f, ds, "s*"))
            if len(scales) == 0:
                print(f"Couldn't read {ds}, skipping...")
                raise e
            print("Found scales %s" % ([os.path.relpath(s, f) for s in scales],))
            a = [open_dataset(f, os.path.relpath(scale_ds, f)) for scale_ds in scales]
        for a in dataset_as:
            arrays.append(a)

    if shaders is None:
        shaders = [None] * len(datasets)
    else:
        assert len(shaders) == len(datasets)
        shaders = [None if s == "default" else s for s in shaders]

    with viewer.txn() as s:
        for (array, dataset), shad in zip(arrays, shaders):
            if args.add_prefix:
                dataset = os.path.join(name_prefix, dataset)
            add_layer(context=s, array=array, name=dataset, shader=shad)

if args.graphs:
    for f, graphs in zip(args.file, args.graphs):
        for graph in graphs:
            graph_annotations = []
            try:
                ids = open_ds(f, graph + "-ids").data
                loc = open_ds(f, graph + "-locations").data
            except BaseException:
                loc = open_ds(f, graph).data
                ids = None
            dims = loc.shape[-1]
            loc = loc[:].reshape((-1, dims))
            if ids is None:
                ids = range(len(loc))
            for i, l in zip(ids, loc):
                if dims == 2:
                    l = np.concatenate([[0], l])
                graph_annotations.append(
                    neuroglancer.EllipsoidAnnotation(
                        center=l[::-1], radii=(5, 5, 5), id=i
                    )
                )
            graph_layer = neuroglancer.AnnotationLayer(
                annotations=graph_annotations, voxel_size=(1, 1, 1)
            )

            with viewer.txn() as s:
                s.layers.append(name="graph", layer=graph_layer)

url = str(viewer)
print(url)
if os.environ.get("DISPLAY") and not args.no_browser:
    webbrowser.open_new_tab(url)

print("Press ENTER to quit")
input()
