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
        "numpy<2",
        "dask",
        "dask[distributed]",
        "zarr",
        "scikit-learn",
        "pyyaml",
        "streamlit",
        "watchdog",
        "tqdm",
        "matplotlib",
        "ruff",
        "wheel",
        "imagecodecs",
        "loguru",
        "scikit-image",
        "quilt3",
        "timm>=0.9.10",
        "torch",
        "torchvision",
        "tifffile",
        "mlpack",
        "tensorboardX",
        "tensorboard",
        "configargparse",
        "neuroglancer",
        "h5py",
        "protobuf",
        "funlib.learn.torch @ git+https://github.com/kirchhausenlab/funlib.learn.torch@5590fb51aef8381eeae99bbe75800ecb186684a1",
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
