from setuptools import setup, find_packages

setup(
    name="incasem_vanilla",
    version="0.1",
    packages=find_packages(where="incasem"),
    package_dir={"": "incasem"},
    python_requires=">=3.9",  # compatibility with pinned libs
    install_requires=[
        "numpy<2",
        "zarr",
        "wheel",  # adding as a safety
        "imagecodecs",  # prevent image compute errors
        "scikit-learn",
        "pyyaml",
        "quilt3",
        "configargparse",
        "torch",
        "torchvision",
        "protobuf",
        "tqdm",
        "wandb",  # logging
        "hydra-core",  # config management
        "watchdog",
        "omegaconf",
        "loguru",
        "scikit-image",
        "dask",
        "matplotlib",
        "pillow",
        "numba",
        "tifffile",  # tiff file handling
        "streamlit",
        "ipython",
    ],
    extras_require={
        "dev": [  # dev dependencies for code quality
            "pytest",
            "black",
            "ruff",
        ],
    },
    dependency_links=[
        "https://download.pytorch.org/whl/cu118",  # cuda 11.8
        "https://pypi.nvidia.com",
    ],
)
