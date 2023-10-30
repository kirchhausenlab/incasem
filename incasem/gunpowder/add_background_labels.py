from typing import Optional
import numpy as np
import gunpowder as gp


class AddBackgroundLabels(gp.BatchFilter):
    def __init__(
            self,
            raw_array: gp.ArrayKey,
            output_array: gp.ArrayKey,
            value: Optional[int] = 0,
            dtype: Optional[str] = 'uint8'):

        self.raw_array = raw_array
        self.output_array = output_array
        self.value = value
        self.dtype = dtype

    def setup(self):
        self.enable_autoskip()

        spec = self.spec[self.raw_array].copy()
        spec.dtype = np.dtype(self.dtype)
        spec.interpolatable = False
        self.provides(self.output_array, spec)

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.raw_array] = request[self.output_array]
        return deps

    def process(self, batch, request):
        output = gp.Batch()

        spec = self.spec[self.output_array].copy()
        spec.roi = batch[self.raw_array].spec.roi

        labels = np.full_like(
            batch[self.raw_array].data,
            fill_value=self.value,
            dtype=self.dtype
        )

        output[self.output_array] = gp.Array(data=labels, spec=spec)
        return output
