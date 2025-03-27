from __future__ import absolute_import, division

import copy
import numbers
import numpy as np

import zarr
import h5py
import json
import logging


logger = logging.getLogger(__name__)


class Freezable(object):
    __isfrozen = False

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("%r is frozen, you can't add attributes to it" % self)
        object.__setattr__(self, key, value)

    def freeze(self):
        self.__isfrozen = True

    def thaw(self):
        self.__isfrozen = False


class Coordinate(tuple):
    """A ``tuple`` of integers.

    Allows the following element-wise operators: addition, subtraction,
    multiplication, division, absolute value, and negation. This allows to
    perform simple arithmetics with coordinates, e.g.::

        shape = Coordinate((2, 3, 4))
        voxel_size = Coordinate((10, 5, 1))
        size = shape*voxel_size # == Coordinate((20, 15, 4))
    """

    def __new__(cls, array_like):
        return super(Coordinate, cls).__new__(
            cls, [int(x) if x is not None else None for x in array_like]
        )

    def dims(self):
        return len(self)

    def is_multiple_of(self, coordinate):
        """Test if this coordinate is a multiple of the given coordinate."""

        return all([a % b == 0 for a, b in zip(self, coordinate)])

    def __neg__(self):
        return Coordinate(-a if a is not None else None for a in self)

    def __abs__(self):
        return Coordinate(abs(a) if a is not None else None for a in self)

    def __add__(self, other):
        if not isinstance(other, tuple):
            raise TypeError("can only add Coordinate or tuples to Coordinate")
        assert self.dims() == len(other), "can only add Coordinate of equal dimensions"

        return Coordinate(
            a + b if a is not None and b is not None else None
            for a, b in zip(self, other)
        )

    def __sub__(self, other):
        if not isinstance(other, tuple):
            raise TypeError("can only subtract Coordinate or tuples to Coordinate")
        assert self.dims() == len(other), (
            "can only subtract Coordinate of equal dimensions"
        )

        return Coordinate(
            a - b if a is not None and b is not None else None
            for a, b in zip(self, other)
        )

    def __mul__(self, other):
        if isinstance(other, tuple):
            assert self.dims() == len(other), (
                "can only multiply Coordinate of equal dimensions"
            )

            return Coordinate(
                a * b if a is not None and b is not None else None
                for a, b in zip(self, other)
            )

        elif isinstance(other, numbers.Number):
            return Coordinate(a * other if a is not None else None for a in self)

        else:
            raise TypeError(
                "multiplication of Coordinate with type %s not supported" % type(other)
            )

    def __div__(self, other):  # pragma: py3 no cover
        if isinstance(other, tuple):
            assert self.dims() == len(other), (
                "can only divide Coordinate of equal dimensions"
            )

            return Coordinate(
                a / b if a is not None and b is not None else None
                for a, b in zip(self, other)
            )

        elif isinstance(other, numbers.Number):
            return Coordinate(a / other if a is not None else None for a in self)

        else:
            raise TypeError(
                "division of Coordinate with type %s not supported" % type(other)
            )

    def __truediv__(self, other):
        if isinstance(other, tuple):
            assert self.dims() == len(other), (
                "can only divide Coordinate of equal dimensions"
            )

            return Coordinate(
                a / b if a is not None and b is not None else None
                for a, b in zip(self, other)
            )

        elif isinstance(other, numbers.Number):
            return Coordinate(a / other if a is not None else None for a in self)

        else:
            raise TypeError(
                "division of Coordinate with type %s not supported" % type(other)
            )

    def __floordiv__(self, other):
        if isinstance(other, tuple):
            assert self.dims() == len(other), (
                "can only divide Coordinate of equal dimensions"
            )

            return Coordinate(
                a // b if a is not None and b is not None else None
                for a, b in zip(self, other)
            )

        elif isinstance(other, numbers.Number):
            return Coordinate(a // other if a is not None else None for a in self)

        else:
            raise TypeError(
                "division of Coordinate with type %s not supported" % type(other)
            )


class Roi(Freezable):
    """A rectangular region of interest, defined by an offset and a shape.

    Similar to :class:`Coordinate`, supports simple arithmetics, e.g.::

        roi = Roi((1, 1, 1), (10, 10, 10))
        voxel_size = Coordinate((10, 5, 1))
        scale_shift = roi*voxel_size + 1 # == Roi((11, 6, 2), (101, 51, 11))

    Args:

        offset (array-like of ``int``):

            The starting point (inclusive) of the ROI. Entries can be ``None``
            to indicate unboundedness.

        shape (array-like):

            The shape of the ROI. Entries can be ``None`` to indicate
            unboundedness.
    """

    def __init__(self, offset, shape):
        self.__offset = Coordinate(offset)
        self.__shape = Coordinate(shape)
        self.freeze()

        self.__consolidate_offset()

    def set_offset(self, offset):
        self.__offset = Coordinate(offset)
        self.__consolidate_offset()

    def set_shape(self, shape):
        """Set the shape of this ROI.

        Args:

            shape (array-like or ``None``):

                The new shape. Entries can be ``None`` to indicate
                unboundedness. If ``None`` is passed instead of a tuple, all
                dimensions are set to ``None``, if the number of dimensions can
                be inferred from an existing offset or previous shape.
        """

        if shape is None:
            dims = self.__shape.dims()
            self.__shape = Coordinate((None,) * dims)

        else:
            self.__shape = Coordinate(shape)

        self.__consolidate_offset()

    def __consolidate_offset(self):
        """Ensure that offsets for unbound dimensions are None."""

        assert self.__offset.dims() == self.__shape.dims(), (
            "offset dimension %d != shape dimension %d"
            % (self.__offset.dims(), self.__shape.dims())
        )

        self.__offset = Coordinate(
            (o if s is not None else None for o, s in zip(self.__offset, self.__shape))
        )

    def get_offset(self):
        return self.__offset

    def get_begin(self):
        """Smallest coordinate inside ROI."""
        return self.__offset

    def get_end(self):
        """Smallest coordinate which is component-wise larger than any inside
        ROI."""
        return self.__offset + self.__shape

    def get_shape(self):
        return self.__shape

    def get_center(self):
        return self.__offset + self.__shape / 2

    def to_slices(self):
        """Get a ``tuple`` of ``slice`` that represent this ROI and can be used
        to index arrays."""
        return tuple(
            slice(
                int(self.__offset[d]) if self.__shape[d] is not None else None,
                int(self.__offset[d] + self.__shape[d])
                if self.__shape[d] is not None
                else None,
            )
            for d in range(self.dims())
        )

    def dims(self):
        """The the number of dimensions of this ROI."""
        return self.__shape.dims()

    def size(self):
        """Get the volume of this ROI. Returns ``None`` if the ROI is
        unbounded."""

        if self.unbounded():
            return None

        size = 1
        for d in self.__shape:
            size *= d
        return size

    def empty(self):
        """Test if this ROI is empty."""

        return self.size() == 0

    def unbounded(self):
        """Test if this ROI is unbounded."""

        return None in self.__shape

    def contains(self, other):
        """Test if this ROI contains ``other``, which can be another
        :class:`Roi` or a :class:`Coordinate`."""

        if isinstance(other, Roi):
            if other.empty():
                return self.contains(other.get_begin())

            else:
                return self.contains(other.get_begin()) and self.contains(
                    other.get_end() - (1,) * other.dims()
                )

        return all([
            (b is None or p is not None and p >= b)
            and (e is None or p is not None and p < e)
            for p, b, e in zip(other, self.get_begin(), self.get_end())
        ])

    def intersects(self, other):
        """Test if this ROI intersects with another :class:`Roi`."""

        assert self.dims() == other.dims()

        if self.empty() or other.empty():
            return False

        # separated if at least one dimension is separated
        separated = any([
            # a dimension is separated if:
            # none of the shapes is unbounded
            (None not in [b1, b2, e1, e2])
            and (
                # either b1 starts after e2
                (b1 >= e2)
                or
                # or b2 starts after e1
                (b2 >= e1)
            )
            for b1, b2, e1, e2 in zip(
                self.get_begin(), other.get_begin(), self.get_end(), other.get_end()
            )
        ])

        return not separated

    def intersect(self, other):
        """Get the intersection of this ROI with another :class:`Roi`."""

        if not self.intersects(other):
            return Roi((0,) * self.dims(), (0,) * self.dims())  # empty ROI

        begin = Coordinate(
            (
                self.__left_max(b1, b2)
                for b1, b2 in zip(self.get_begin(), other.get_begin())
            )
        )
        end = Coordinate(
            (
                self.__right_min(e1, e2)
                for e1, e2 in zip(self.get_end(), other.get_end())
            )
        )

        return Roi(begin, end - begin)

    def union(self, other):
        """Get the union of this ROI with another :class:`Roi`."""

        begin = Coordinate(
            (
                self.__left_min(b1, b2)
                for b1, b2 in zip(self.get_begin(), other.get_begin())
            )
        )
        end = Coordinate(
            (
                self.__right_max(e1, e2)
                for e1, e2 in zip(self.get_end(), other.get_end())
            )
        )

        return Roi(begin, end - begin)

    def shift(self, by):
        """Shift this ROI."""

        return Roi(self.__offset + by, self.__shape)

    def snap_to_grid(self, voxel_size, mode="grow"):
        """Align a ROI with a given voxel size.

        Args:

            voxel_size (:class:`Coordinate`):

                The voxel size of the grid to snap to.

            mode (string, optional):

                How to align the ROI if it is not a multiple of the voxel size.
                Available modes are 'grow', 'shrink', and 'closest'. Defaults
                to 'grow'.
        """

        assert len(voxel_size) == self.dims(), (
            "dimension of voxel size does not match ROI"
        )

        begin_in_voxel_fractions = np.asarray(
            self.get_begin(), dtype=np.float32
        ) / np.asarray(voxel_size)
        end_in_voxel_fractions = np.asarray(
            self.get_end(), dtype=np.float32
        ) / np.asarray(voxel_size)

        if mode == "closest":
            begin_in_voxel = np.round(begin_in_voxel_fractions)
            end_in_voxel = np.round(end_in_voxel_fractions)
        elif mode == "grow":
            begin_in_voxel = np.floor(begin_in_voxel_fractions)
            end_in_voxel = np.ceil(end_in_voxel_fractions)
        elif mode == "shrink":
            begin_in_voxel = np.ceil(begin_in_voxel_fractions)
            end_in_voxel = np.floor(end_in_voxel_fractions)
        else:
            raise RuntimeError("Unknown mode %s for snap_to_grid" % mode)

        offset = tuple(
            b * v if not np.isnan(b) else None
            for b, v in zip(begin_in_voxel, voxel_size)
        )
        shape = tuple(
            (e - b) * v if not np.isnan(e) and not np.isnan(b) else None
            for b, e, v in zip(begin_in_voxel, end_in_voxel, voxel_size)
        )

        return Roi(offset, shape)

    def grow(self, amount_neg, amount_pos):
        """Grow a ROI by the given amounts in each direction:

        Args:

            amount_neg (:class:`Coordinate` or ``None``):

                Amount (per dimension) to grow into the negative direction.

            amount_pos (:class:`Coordinate` or ``None``):

                Amount (per dimension) to grow into the positive direction.
        """

        if amount_neg is None:
            amount_neg = Coordinate((0,) * self.dims())
        if amount_pos is None:
            amount_pos = Coordinate((0,) * self.dims())

        assert len(amount_neg) == self.dims()
        assert len(amount_pos) == self.dims()

        offset = self.__offset - amount_neg
        shape = self.__shape + amount_neg + amount_pos

        return Roi(offset, shape)

    def copy(self):
        """Create a copy of this ROI."""
        return copy.deepcopy(self)

    def __left_min(self, x, y):
        # None is considered -inf

        if x is None or y is None:
            return None
        return min(x, y)

    def __left_max(self, x, y):
        # None is considered -inf

        if x is None:
            return y
        if y is None:
            return x
        return max(x, y)

    def __right_min(self, x, y):
        # None is considered +inf

        if x is None:
            return y
        if y is None:
            return x
        return min(x, y)

    def __right_max(self, x, y):
        # None is considered +inf

        if x is None or y is None:
            return None
        return max(x, y)

    def __add__(self, other):
        assert isinstance(other, tuple), "can only add Coordinate or tuple to Roi"
        return self.shift(other)

    def __sub__(self, other):
        assert isinstance(other, tuple), (
            "can only subtract Coordinate or tuple from Roi"
        )
        return self.shift(-Coordinate(other))

    def __mul__(self, other):
        assert isinstance(other, tuple) or isinstance(other, numbers.Number), (
            "can only multiply with a number or tuple of numbers"
        )
        return Roi(self.__offset * other, self.__shape * other)

    def __div__(self, other):  # pragma: py3 no cover
        assert isinstance(other, tuple) or isinstance(other, numbers.Number), (
            "can only divide by a number or tuple of numbers"
        )
        return Roi(self.__offset / other, self.__shape / other)

    def __truediv__(self, other):
        assert isinstance(other, tuple) or isinstance(other, numbers.Number), (
            "can only divide by a number or tuple of numbers"
        )
        return Roi(self.__offset / other, self.__shape / other)

    def __floordiv__(self, other):
        assert isinstance(other, tuple) or isinstance(other, numbers.Number), (
            "can only divide by a number or tuple of numbers"
        )
        return Roi(self.__offset // other, self.__shape // other)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented  # pragma: no cover

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented  # pragma: no cover

    def __repr__(self):
        if self.empty():
            return "[empty ROI]"  # pragma: no cover
        slices = ", ".join([
            (str(b) if b is not None else "") + ":" + (str(e) if e is not None else "")
            for b, e in zip(self.get_begin(), self.get_end())
        ])
        dims = ", ".join(str(a) if a is not None else "inf" for a in self.__shape)
        return "[" + slices + "] (" + dims + ")"


class Array(Freezable):
    """A ROI and voxel size annotated ndarray-like. Acts as a view into actual
    data.

    Args:

        data (``ndarray``-like):

            The data to hold. Can be a numpy, HDF5, zarr, etc. array like.
            Needs to have ``shape`` and slicing support for reading/writing. It
            is assumed that slicing returns an ``ndarray``.

        roi (`class:Roi`):

            The region of interest (ROI) represented by this array.

        voxel_size (`class:Coordinate`):

            The size of a voxel.

        data_offset (`class:Coordinate`, optional):

            The start of ``data``, in world units. Defaults to
            ``roi.get_begin()``, if not given.
    """

    def __init__(self, data, roi, voxel_size, data_offset=None):
        self.data = data
        self.roi = roi
        self.voxel_size = Coordinate(voxel_size)
        self.n_channel_dims = len(data.shape) - roi.dims()

        assert self.voxel_size.dims() == self.roi.dims(), (
            "dimension of voxel_size (%d) does not match dimension of roi (%d)"
            % (self.voxel_size.dims(), self.roi.dims())
        )

        if data_offset is None:
            data_offset = roi.get_begin()
        else:
            data_offset = Coordinate(data_offset)

        self.data_roi = Roi(
            data_offset, self.voxel_size * self.data.shape[self.n_channel_dims :]
        )

        assert self.roi.get_begin().is_multiple_of(voxel_size), (
            "roi offset %s is not a multiple of voxel size %s"
            % (self.roi.get_begin(), voxel_size)
        )

        assert self.roi.get_shape().is_multiple_of(voxel_size), (
            "roi shape %s is not a multiple of voxel size %s"
            % (self.roi.get_shape(), voxel_size)
        )

        assert data_offset.is_multiple_of(voxel_size), (
            "data offset %s is not a multiple of voxel size %s"
            % (data_offset, voxel_size)
        )

        assert self.data_roi.contains(roi), (
            "data ROI %s does not contain given ROI %s" % (self.data_roi, roi)
        )

        self.freeze()

    @property
    def shape(self):
        """Get the shape in voxels of this array, possibly including channel
        dimensions. This is equivalent to::

            array.to_ndarray().shape()

        but does not actually create the ``ndarray``.
        """

        view_shape = (self.roi / self.voxel_size).get_shape()
        return self.data.shape[: self.n_channel_dims] + view_shape

    @property
    def dtype(self):
        """Get the dtype of this array."""
        return self.data.dtype

    def __getitem__(self, key):
        """Get a sub-array or a single value.

        Args:

            key (`class:Roi` or `class:Coordinate`):

                The ROI specifying the sub-array or a coordinate for a single
                value.

        Returns:

            If ``key`` is a `class:Roi`, returns a `class:Array` that
            represents this ROI. This is a light-weight operation that does not
            access the actual data held by this array. If ``key`` is a
            `class:Coordinate`, the array value (possible multi-channel)
            closest to the coordinate is returned.
        """

        if isinstance(key, Roi):
            roi = key

            assert self.roi.contains(roi), (
                "Requested roi is not contained in this array."
            )

            return Array(self.data, roi, self.voxel_size, self.data_roi.get_begin())

        elif isinstance(key, Coordinate):
            coordinate = key

            assert self.roi.contains(coordinate), (
                "Requested coordinate is not contained in this array."
            )

            return self.data[self.__index(coordinate)]

    def __setitem__(self, roi, value):
        """Set the data of this array within the given ROI.

        Args:

            roi (`class:Roi`):

                The ROI to write to.

            value (`class:Array`, or broadcastable to ``ndarray``):

                The value to write. If an `class:Array`, the ROIs do not have
                to match, however, the shape of ``value`` has to be
                broadcastable to the voxel shape of ``roi``.
        """

        assert isinstance(roi, Roi), "Roi expected, but got %s" % (type(roi))

        assert roi.get_begin().is_multiple_of(self.voxel_size), (
            "roi offset %s is not a multiple of voxel size %s"
            % (roi.get_begin(), self.voxel_size)
        )

        assert roi.get_shape().is_multiple_of(self.voxel_size), (
            "roi shape %s is not a multiple of voxel size %s"
            % (roi.get_shape(), self.voxel_size)
        )

        target = self.data
        target_slices = self.__slices(roi)

        if not hasattr(value, "__getitem__"):
            target[target_slices] = value
            return

        if isinstance(value, Array):
            array = value
            source = array.data
            source_slices = array.__slices(array.roi)

        else:
            source = value
            source_slices = slice(None)

        target[target_slices] = source[source_slices]

    def materialize(self):
        """Copy the data represented by this array to memory. This is
        equivalent to::

            array = Array(array.to_ndarray(), array.roi, array.voxel_size)

        but modifies this array directly.
        """

        self.data = self.to_ndarray()
        self.data_roi = self.roi.copy()

    def to_ndarray(self, roi=None, fill_value=None):
        """Copy the data represented by this array into an ``ndarray``.

        Args:

            roi (`class:Roi`, optional):

                If given, copy only the data represented by this ROI. This is
                equivalent to::

                    array[roi].to_ndarray()

            fill_value (scalar, optional):

                If given, allow ``roi`` to be outside of this array's ROI.
                Outside values will be filled with ``fill_value``.
        """

        if roi is None:
            return self.data[self.__slices(self.roi)]

        if fill_value is None:
            return self[roi].to_ndarray()

        shape = (roi / self.voxel_size).get_shape()
        data = np.zeros(
            self.data.shape[: self.n_channel_dims] + shape, dtype=self.data.dtype
        )
        if fill_value != 0:
            data[:] = fill_value

        array = Array(data, roi, self.voxel_size)

        shared_roi = self.roi.intersect(roi)

        if not shared_roi.empty():
            array[shared_roi] = self[shared_roi]

        return data

    def intersect(self, roi):
        """Get a sub-array obtained by intersecting this array with the given
        ROI. This is equivalent to::

            array[array.roi.intersect(roi)]

        Args:

            roi (`class:Roi`):

                The ROI to intersect with.
        """

        intersection = self.roi.intersect(roi)
        return self[intersection]

    def __slices(self, roi):
        """Get the voxel slices for the given roi."""

        voxel_roi = (roi - self.data_roi.get_begin()) / self.voxel_size
        return (slice(None),) * self.n_channel_dims + voxel_roi.to_slices()

    def __index(self, coordinate):
        """Get the voxel slices for the given coordinate."""

        index = (coordinate - self.data_roi.get_begin()) / self.voxel_size
        if self.n_channel_dims > 0:
            index = (Ellipsis,) + index
        return index


def _read_voxel_size_offset(ds, order="C"):
    voxel_size = None
    offset = None
    dims = None

    if "resolution" in ds.attrs:
        voxel_size = tuple(ds.attrs["resolution"])
        dims = len(voxel_size)

    if "offset" in ds.attrs:
        offset = tuple(ds.attrs["offset"])

        if dims is not None:
            assert dims == len(offset), (
                "resolution and offset attributes differ in length"
            )
        else:
            dims = len(offset)

    if dims is None:
        dims = len(ds.shape)

    if voxel_size is None:
        voxel_size = (1,) * dims

    if offset is None:
        offset = (0,) * dims

    if order == "F":
        offset = offset[::-1]
        voxel_size = voxel_size[::-1]

    return Coordinate(voxel_size), Coordinate(offset)


def open_ds(filename, ds_name, mode="r"):
    if filename.endswith(".zarr"):
        logger.debug("opening zarr dataset %s in %s", ds_name, filename)
        ds = zarr.open(filename, mode=mode)[ds_name]

        voxel_size, offset = _read_voxel_size_offset(ds, ds.order)
        roi = Roi(offset, voxel_size * ds.shape[-len(voxel_size) :])

        logger.debug("opened zarr dataset %s in %s", ds_name, filename)
        return Array(ds, roi, voxel_size)

    elif filename.endswith(".n5"):
        logger.debug("opening N5 dataset %s in %s", ds_name, filename)
        ds = zarr.open(filename, mode=mode)[ds_name]

        voxel_size, offset = _read_voxel_size_offset(ds, "F")
        roi = Roi(offset, voxel_size * ds.shape[-len(voxel_size) :])

        logger.debug("opened N5 dataset %s in %s", ds_name, filename)
        return Array(ds, roi, voxel_size)

    elif filename.endswith(".h5") or filename.endswith(".hdf"):
        logger.debug("opening H5 dataset %s in %s", ds_name, filename)
        ds = h5py.File(filename, mode=mode)[ds_name]

        voxel_size, offset = _read_voxel_size_offset(ds, "C")
        roi = Roi(offset, voxel_size * ds.shape[-len(voxel_size) :])

        logger.debug("opened H5 dataset %s in %s", ds_name, filename)
        return Array(ds, roi, voxel_size)

    elif filename.endswith(".json"):
        logger.debug("found JSON container spec")
        with open(filename, "r") as f:
            spec = json.load(f)

        array = open_ds(spec["container"], ds_name, mode)
        return Array(
            array.data,
            Roi(spec["offset"], spec["size"]),
            array.voxel_size,
            array.roi.get_begin(),
        )

    else:
        logger.error("don't know data format of %s in %s", ds_name, filename)
        raise RuntimeError("Unknown file format for %s" % filename)
