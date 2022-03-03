import numpy as np
import gunpowder as gp


class ExtractSegmentation(gp.BatchFilter):
    def __init__(
            self, array: gp.ArrayKey,
            output_array: gp.ArrayKey,
            mask: gp.ArrayKey = None,
            dtype=np.uint32):
        """Extract segmentation from multi-channel predictions with argmax

        Args:
            array: ArrayKey of input predictions.
            output_array (gp.ArrayKey): ArrayKey for extracted segmentation.
            dtype: Dtype of the extracted segmentation.
        """

        self.array = array
        self.output_array = output_array
        self.mask = mask
        self.dtype = dtype

    def setup(self):
        self.enable_autoskip()

        spec = self.spec[self.array].copy()
        spec.dtype = self.dtype

        if self.output_array:
            self.provides(self.output_array, spec)
        else:
            self.updates(self.array, spec)

    def prepare(self, request):
        deps = gp.BatchRequest()

        if self.output_array:
            deps[self.array] = request[self.output_array].copy()
            if self.mask:
                deps[self.mask] = request[self.output_array].copy()
        else:
            deps[self.array] = request[self.array].copy()
            if self.mask:
                deps[self.mask] = request[self.array].copy()

        return deps

    def process(self, batch, request):
        output = gp.Batch()

        if self.output_array:
            out_array = self.output_array
        else:
            out_array = self.array

        spec = batch[self.array].spec.copy()
        spec.dtype = self.dtype
        segmentation = np.argmax(
            batch[self.array].data, axis=0).astype(self.dtype)

        if self.mask:
            mask = batch[self.mask].data
            segmentation = (segmentation * mask).astype(self.dtype)

        output[out_array] = gp.Array(
            data=segmentation, spec=spec
        )

        return output
