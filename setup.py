from pathlib import Path
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

THIRD_PARTY_DIR: Path = Path(__file__).parent.joinpath("src", "third_party")

setup(
    name="incasem_vanilla",
    version="0.1",
    packages=find_packages(where="incasem"),
    package_dir={"": "incasem"},
    python_requires=">=3.9",  # compatibility with pinned libs
    install_requires=[
        "numpy",
        "zarr",
        "wheel",  # adding as a safety
        "imagecodecs",  # prevent image compute errors
        "scikit-learn",
        "pyyaml",
        "quilt3",
        "configargparse",
        "torch",
        "ruff",
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
        "daisy",
        "matplotlib",
        "pillow",
        "numba",
        "tifffile",  # tiff file handling
        "streamlit",
        "ipython",
        "sacred @ git+https://github.com/kirchhausenlab/sacred@master",
        "funlib.learn.torch @ git+https://github.com/kirchhausenlab/funlib.learn.torch_TKLAB",
        "funlib.persistence @ git+https://github.com/kirchhausenlab/funlib.persistence_TKLAB",
        "gunpowder @ git+https://github.com/bentaculum/gunpowder@total_roi_with_nonspatial_array",
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
