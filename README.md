# <ins>A</ins>utomated <ins>S</ins>egmentation of cellular substructures in <ins>E</ins>lectron <ins>M</ins>icroscopy (ASEM)

https://user-images.githubusercontent.com/8866751/200858874-af9220c9-60ac-4b3b-9b45-181d266f82b0.mp4

This repository contains the segmentation pipeline described in

> Benjamin Gallusser, Giorgio Maltese, Giuseppe Di Caprio et al.<br>[_Deep neural network automated segmentation of cellular structures in volume electron microscopy_](https://rupress.org/jcb/article/222/2/e202208005/213736/Deep-neural-network-automated-segmentation-of),<br>Journal of Cell Biology, 2022.

Please cite the publication if you are using this code in your research.

Our semi-automated annotation tool from the same publication is available at [https://github.com/kirchhausenlab/gc_segment](https://github.com/kirchhausenlab/gc_segment).

## Table of Contents

- [Setup](#Setup)
- [Installation](#Installation)
- [Optional: Download our data](#Optional-Download-our-data)
- [Prepare your own data for prediction](#Prepare-your-own-data-for-prediction)
- [Prediction](#Prediction)
- [Prepare your own ground truth annotations for fine-tuning or training](#Prepare-your-own-ground-truth-annotations-for-fine-tuning-or-training)
- [Fine-Tuning](#Fine-tuning)
- [Training](#Training)

### Machine Setup💻

You can use the following machines to run the Cell Interactome pipeline:

1. Ubuntu <img src="https://user-images.githubusercontent.com/25181517/186884153-99edc188-e4aa-4c84-91b0-e2df260ebc33.png" width="15">
2. MacOS <img src="https://user-images.githubusercontent.com/25181517/186884152-ae609cca-8cf1-4175-8d60-1ce1fa078ca2.png" width="15"> [*Please Use Docker*] <img src="https://user-images.githubusercontent.com/25181517/117207330-263ba280-adf4-11eb-9b97-0ac5b40bc3be.png" width="15">
3. Windows - <img src="https://user-images.githubusercontent.com/25181517/186884150-05e9ff6d-340e-4802-9533-2c3f02363ee3.png" width="18"> [*Please Use Docker*]<img src="https://user-images.githubusercontent.com/25181517/117207330-263ba280-adf4-11eb-9b97-0ac5b40bc3be.png" width="18">

### CUDA Installation 🛠

This project requires CUDA version 12.x. Veri️fy the correct version of CUDA installed by running the following command:

```bash
nvcc --version
```

### Installation 🛠️

1. Clone the repository:

```bash
git clone --recursive git@github.com:kirchhausenlab/cell_interactome.git
```

2. 📦 In case of errors, please ensure you have the required dependencies for Python installed:

```bash
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl \
ninja-build cmake libegl1-mesa-dev python3-dev
```

3. Create a conda environment.

```bash
conda create -n incasem python=3.10 --no-default-packages
python -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124
mamba install pyqt qtpy
python -m pip install -e ".[dev]" \ --extra-index-url https://download.pytorch.org/whl/cu124
```

In case of installation issues, please clear your cache as follows:

```bash
rm -rf ~/.cache/pip
```

4. Add required third-party libraries to the project.
5. You can format your files using the following command:

```bash
ruff format
ruff clean
```

The `ruff.toml` file contains the configuration for the formatter.

## Setup

**Incasem** can be setup in one of three ways:

1. **Jupter Notebook**: The easiest way to get started is to use the Jupyter Notebook `incasem.ipynb` in the `notebooks` directory. This notebook provides a step-by-step guide to the main functionalities of the package. It is recommended to use this notebook if you are new to the package. Kindly move your tiff and zarr files to google drive to start using the notebook. Furthermore, google colab provides free GPU access which can be used to train the models, run predictions and visualize the results. The link to the notebook is [here](https://colab.research.google.com/drive/1)

2. **Streamlit UI**: A user-friendly interface, for which you need to install miniforge first. Follow the instructions [here](https://github.com/conda-forge/miniforge) to install miniforge. Once installed, follow the instructions to start using Mamba. Mamba works exactly like conda so you can use the same commands you would use as if it were conda WITHOUT having to change the command. So if you started a virtual environment with conda using `conda activate test-env`, with mamba the command remains the same. If you have conda installed ensure that it doesnt conflict with mamba as show [here](https://mamba.readthedocs.io/en/latest/user_guide/troubleshooting.html#defaults-channels).
   Please run the following:

```bash
python -m pip install -e ".[dev,xformers]" \
--extra-index-url https://download.pytorch.org/whl/cu124 \
pip3 install -e .
cd incasem/automate
streamlit run main.py
```

Follow the instructions on the UI to start using incasem.

3. **Command Line Interface**: Detailed instructions on how to use the command line interface are provided in the `wiki/installation.md` file
