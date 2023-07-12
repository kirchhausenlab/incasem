import operator
import logging
import neuroglancer
import numpy as np
import zarr
import sys
import glob
import os
import daisy

logger = logging.getLogger(__name__)


class ScalePyramid(neuroglancer.LocalVolume):
    """A neuroglancer layer that provides volume data on different scales.
    Mimics a LocalVolume.

    Args:

            volume_layers (``list`` of ``LocalVolume``):

                One ``LocalVolume`` per provided resolution.
    """

    def __init__(self, volume_layers):
        volume_layers = volume_layers

        super(neuroglancer.LocalVolume, self).__init__()

        logger.debug("Creating scale pyramid...")

        self.min_voxel_size = min(
            [tuple(layer.dimensions.scales) for layer in volume_layers]
        )
        self.max_voxel_size = max(
            [tuple(layer.dimensions.scales) for layer in volume_layers]
        )

        self.dims = len(volume_layers[0].dimensions.scales)
        self.volume_layers = {
            tuple(
                int(x)
                for x in map(
                    operator.truediv, layer.dimensions.scales, self.min_voxel_size
                )
            ): layer
            for layer in volume_layers
        }

        logger.debug("min_voxel_size: %s", self.min_voxel_size)
        logger.debug("scale keys: %s", self.volume_layers.keys())
        logger.debug(self.info())

    @property
    def volume_type(self):
        return self.volume_layers[(1,) * self.dims].volume_type

    @property
    def token(self):
        return self.volume_layers[(1,) * self.dims].token

    def info(self):

        reference_layer = self.volume_layers[(1,) * self.dims]
        # return reference_layer.info()

        reference_info = reference_layer.info()

        info = {
            "dataType": reference_info["dataType"],
            "encoding": reference_info["encoding"],
            "generation": reference_info["generation"],
            "coordinateSpace": reference_info["coordinateSpace"],
            "shape": reference_info["shape"],
            "volumeType": reference_info["volumeType"],
            "voxelOffset": reference_info["voxelOffset"],
            "chunkLayout": reference_info["chunkLayout"],
            "downsamplingLayout": reference_info["downsamplingLayout"],
            "maxDownsampling": int(
                np.prod(np.array(self.max_voxel_size) // np.array(self.min_voxel_size))
            ),
            "maxDownsampledSize": reference_info["maxDownsampledSize"],
            "maxDownsamplingScales": reference_info["maxDownsamplingScales"],
        }

        return info

    def get_encoded_subvolume(self, data_format, start, end, scale_key=None):
        if scale_key is None:
            scale_key = ",".join(("1",) * self.dims)

        scale = tuple(int(s) for s in scale_key.split(","))

        return self.volume_layers[scale].get_encoded_subvolume(
            data_format, start, end, scale_key=",".join(("1",) * self.dims)
        )

    def get_object_mesh(self, object_id):
        return self.volume_layers[(1,) * self.dims].get_object_mesh(object_id)

    def invalidate(self):
        return self.volume_layers[(1,) * self.dims].invalidate()


def add_layer(
    context,
    array,
    name,
    opacity=None,
    shader=None,
    visible=True,
    reversed_axes=False,
    scale_rgb=False,
    c=[0, 1, 2],
    h=[0.0, 0.0, 1.0],
    layer_type='im'
):
    """Add a layer to a neuroglancer context.

    Args:

        context:

            The neuroglancer context to add a layer to, as obtained by
            ``viewer.txn()``.

        array:

            A ``daisy``-like array, containing attributes ``roi``,
            ``voxel_size``, and ``data``. If a list of arrays is given, a
            ``ScalePyramid`` layer is generated.

        name:

            The name of the layer.

        opacity:

            A float to define the layer opacity between 0 and 1

        shader:

            A string to be used as the shader. If set to ``'rgb'``, an RGB
            shader will be used.

        visible:

            A bool which defines layer visibility

        c (channel):

            A list of ints to define which channels to use for an rgb shader

        h (hue):

            A list of floats to define rgb color for an rgba shader
    """

    is_multiscale = isinstance(array, list)

    if not is_multiscale:

        a = array if not is_multiscale else array[0]

        spatial_dim_names = ["t", "z", "y", "x"]
        channel_dim_names = ["b^", "c^"]

        dims = len(a.data.shape)
        spatial_dims = a.roi.dims()
        channel_dims = dims - spatial_dims

        attrs = {
            "names": (channel_dim_names[-channel_dims:] if channel_dims > 0 else [])
            + spatial_dim_names[-spatial_dims:],
            "units": [""] * channel_dims + ["nm"] * spatial_dims,
            "scales": [1] * channel_dims + list(a.voxel_size),
        }
        if reversed_axes:
            attrs = {k: v[::-1] for k, v in attrs.items()}
        dimensions = neuroglancer.CoordinateSpace(**attrs)

        voxel_offset = [0] * channel_dims + \
            list(a.roi.get_offset() / a.voxel_size)

    else:
        dimensions = []
        voxel_offset = None
        for i, a in enumerate(array):

            spatial_dim_names = ["t", "z", "y", "x"]
            channel_dim_names = ["b^", "c^"]

            dims = len(a.data.shape)
            spatial_dims = a.roi.dims()
            channel_dims = dims - spatial_dims

            attrs = {
                "names": (channel_dim_names[-channel_dims:] if channel_dims > 0 else [])
                + spatial_dim_names[-spatial_dims:]
                if spatial_dims > 0
                else [],
                "units": [""] * channel_dims + ["nm"] * spatial_dims,
                "scales": [1] * channel_dims + list(a.voxel_size),
            }
            if reversed_axes:
                attrs = {k: v[::-1] for k, v in attrs.items()}
            dimensions.append(neuroglancer.CoordinateSpace(**attrs))

            if voxel_offset is None:
                voxel_offset = [0] * channel_dims + list(
                    a.roi.get_offset() / a.voxel_size
                )

    if reversed_axes:
        voxel_offset = voxel_offset[::-1]

    if shader is None:
        a = array if not is_multiscale else array[0]
        dims = a.roi.dims()
        if dims < len(a.data.shape):
            channels = a.data.shape[0]
            if channels > 1:
                shader = "rgb"

    if shader == "rgb":
        if scale_rgb:
            shader = """
void main() {
    emitRGB(
        255.0*vec3(
            toNormalized(getDataValue(%i)),
            toNormalized(getDataValue(%i)),
            toNormalized(getDataValue(%i)))
        );
}""" % (
                c[0],
                c[1],
                c[2],
            )

        else:
            shader = """
void main() {
    emitRGB(
        vec3(
            toNormalized(getDataValue(%i)),
            toNormalized(getDataValue(%i)),
            toNormalized(getDataValue(%i)))
        );
}""" % (
                c[0],
                c[1],
                c[2],
            )

    elif shader == "rgba":
        shader = """
void main() {
    emitRGBA(
        vec4(
        %f, %f, %f,
        toNormalized(getDataValue()))
        );
}""" % (
            h[0],
            h[1],
            h[2],
        )

    elif shader == "mask":
        shader = """
void main() {
  emitGrayscale(255.0*toNormalized(getDataValue()));
}"""

    elif shader == 'heatmap':
        shader = """
#uicontrol float thres slider(min=0.0, max=1.0, default=0.0)
void main () {
    float v = toNormalized(getDataValue());
    vec4 rgba = vec4(0,0,0,0);
    if (v > thres) {
        rgba = vec4(colormapJet(v), 1.0);
    } else {
        rgba = vec4(colormapJet(v), 0.25);
    }
    emitRGBA(rgba);
}"""
    elif shader == 'probmap':
        shader = """
#uicontrol float thres slider(min=0.0, max=1.0, default=0.5)
void main () {
    float v = toNormalized(getDataValue());
    vec4 rgba = vec4(0,0,0,0);
    if (v > thres) {
        rgba = vec4(1.0, 0.0, 0.0, 1.0);
    }
    emitRGBA(rgba);
}"""

    kwargs = {}

    if shader is not None:
        kwargs["shader"] = shader
    if opacity is not None:
        kwargs["opacity"] = opacity

    if is_multiscale:

        if layer_type == 'im':
            tt = 'image'
        else:
            tt = 'segmentation'

        layer = ScalePyramid(
            [
                neuroglancer.LocalVolume(
                    data=a.data, voxel_offset=voxel_offset, dimensions=array_dims, volume_type=tt
                )
                for a, array_dims in zip(array, dimensions)
            ]
        )

    else:
        if layer_type == 'im':
            tt = 'image'
        else:
            tt = 'segmentation'

        layer = neuroglancer.LocalVolume(
            data=array.data,
            voxel_offset=voxel_offset,
            dimensions=dimensions,
            volume_type=tt
        )


    context.layers.append(name=name, layer=layer, visible=visible, **kwargs)

def str_rstrip_slash(x):
    return x.rstrip('/')



def to_slice(slice_str):

    values = [int(x) for x in slice_str.split(':')]
    if len(values) == 1:
        return values[0]

    return slice(*values)


def parse_ds_name(ds):

    tokens = ds.split('[')

    if len(tokens) == 1:
        return ds, None

    ds, slices = tokens
    slices = list(map(to_slice, slices.rstrip(']').split(',')))

    return ds, slices


class Project:

    def __init__(self, array, dim, value):
        self.array = array
        self.dim = dim
        self.value = value
        self.shape = array.shape[:self.dim] + array.shape[self.dim + 1:]
        self.dtype = array.dtype

    def __getitem__(self, key):
        slices = key[:self.dim] + (self.value,) + key[self.dim:]
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
            a.roi = daisy.Roi(
                a.roi.get_begin()[:d] + a.roi.get_begin()[d + 1:],
                a.roi.get_shape()[:d] + a.roi.get_shape()[d + 1:])
            a.voxel_size = a.voxel_size[:d] + a.voxel_size[d + 1:]

    return a

def open_dataset(f, ds):
    original_ds = ds
    ds, slices = parse_ds_name(ds)
    slices_str = original_ds[len(ds):]

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
        a = daisy.open_ds(f, ds)

        if slices is not None:
            a = slice_dataset(a, slices)

        if a.roi.dims() == 2:
            print("ROI is 2D, recruiting next channel to z dimension")
            a.roi = daisy.Roi((0,) + a.roi.get_begin(),
                              (a.shape[-3],) + a.roi.get_shape())
            a.voxel_size = daisy.Coordinate((1,) + a.voxel_size)

        if a.roi.dims() == 4:
            print("ROI is 4D, stripping first dimension and treat as channels")
            a.roi = daisy.Roi(a.roi.get_begin()[1:], a.roi.get_shape()[1:])
            a.voxel_size = daisy.Coordinate(a.voxel_size[1:])

        if a.data.dtype == np.int64 or a.data.dtype == np.int16:
            print("Converting dtype in memory...")
            a.data = a.data[:].astype(np.uint64)

        return [(a, ds)]
    else:
        return [([daisy.open_ds(f, f"{ds}/{key}")
                  for key in zarr.open(f)[ds].keys()], ds)]


def add_data_to_viewer(viewer, file, dataset_list):

    shader_list = [None] * len(file)

    for f, datasets, shaders in zip(file, dataset_list, shader_list):

        name_prefix = '/'.join(f.strip('/').split('/')[-2:])
        arrays = []
        for ds in datasets:
            try:

                print("Adding %s, %s" % (f, ds))
                dataset_as = open_dataset(f, ds)

            except Exception as e:

                print(type(e), e)
                print("Didn't work, checking if this is multi-res...")

                scales = glob.glob(os.path.join(f, ds, 's*'))
                if len(scales) == 0:
                    print(f"Couldn't read {ds}, skipping...")
                    raise e
                print("Found scales %s" % ([
                    os.path.relpath(s, f)
                    for s in scales
                ],))
                a = [
                    open_dataset(f, os.path.relpath(scale_ds, f))
                    for scale_ds in scales
                ]
            for a in dataset_as:
                arrays.append(a)

        if shaders is None:
            shaders = [None] * len(datasets)
        else:
            shaders = ['rgb']*len(datasets)
            assert len(shaders) == len(datasets)
            shaders = [None if s == 'default' else s for s in shaders]

        with viewer.txn() as s:
            for (array, dataset), shad in zip(arrays, shaders):

                if "labels" in dataset or "predictions" in dataset:
                    lt = "seg"
                else:
                    lt = "im"

                if True:
                    dataset = os.path.join(name_prefix, dataset)
                add_layer(
                    context=s,
                    array=array,
                    name=dataset,
                    layer_type=lt
                )

    return viewer


