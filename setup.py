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
        'protobuf<=3.20',
        'daisy==0.2.1',
        'funlib.learn.torch @ git+https://github.com/funkelab/funlib.learn.torch@master',
        'gunpowder @ git+https://github.com/bentaculum/gunpowder@total_roi_with_nonspatial_array',
    ],
    python_requires='>=3.8,',
)
