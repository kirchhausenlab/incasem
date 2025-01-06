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
        'numpy<2',
        'zarr',
        'scikit-learn',
        'pyyaml',
        'quilt3',
        'tensorboardX',
        'tensorboard',
        'configargparse',
        'protobuf',
        'daisy',
        'sacred @ git+https://github.com/kirchhausenlab/sacred@master',
        'funlib.learn.torch @ git+https://github.com/kirchhausenlab/funlib.learn.torch_TKLAB.git@pinned-version',
        'funlib.persistence @ git+https://github.com/kirchhausenlab/funlib.persistence_TKLAB@pinned-version',
        'gunpowder @ git+https://github.com/bentaculum/gunpowder@total_roi_with_nonspatial_array',
    ],
    python_requires='>=3.9',
)
