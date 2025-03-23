from setuptools import setup

setup(
    name='incasem',
    version='0.1',
    description='',
    url='https://github.com/kirchhausenlab/incasem',
    author='Benjamin Gallusser',
    author_email='gallusser@tklab.hms.harvard.edu',
    license='MIT',
    py_modules=[],
    install_requires=[
        'numpy',
        'dask',
        'dask[distributed]',
        'zarr',
        'scikit-learn',
        'pyyaml',
        'quilt3',
        'mlpack',
        'tensorboardX',
        'tensorboard',
        'configargparse',
        'protobuf',
        'funlib.learn.torch @ git+https://github.com/kirchhausenlab/funlib.learn.torch@5590fb51aef8381eeae99bbe75800ecb186684a1',
        'gunpowder @ git+https://github.com/bentaculum/gunpowder@total_roi_with_nonspatial_array',
    ],
    python_requires='>=3.9',
)
