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
        'funlib.learn.torch @ git+https://github.com/funkelab/funlib.learn.torch@master',
        'funlib.persistence @ git+https://github.com/funkelab/funlib.persistence@master',
        'funlib.show.neuroglancer @ git+https://github.com/funkelab/funlib.show.neuroglancer.git@master',
        'gunpowder @ git+https://github.com/bentaculum/gunpowder@total_roi_with_nonspatial_array',
    ],
    python_requires='>=3.8, <3.10',
)
