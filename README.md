# <ins>A</ins>utomated <ins>S</ins>egmentation of cellular substructures in <ins>E</ins>lectron <ins>M</ins>icroscopy (ASEM)

https://user-images.githubusercontent.com/8866751/200858874-af9220c9-60ac-4b3b-9b45-181d266f82b0.mp4

This repository contains the segmentation pipeline described in

> Benjamin Gallusser, Giorgio Maltese, Giuseppe Di Caprio et al.<br>[*Deep neural network automated segmentation of cellular structures in volume electron microscopy*](https://rupress.org/jcb/article/222/2/e202208005/213736/Deep-neural-network-automated-segmentation-of),<br>Journal of Cell Biology, 2022.

Please cite the publication if you are using this code in your research.

Our semi-automated annotation tool from the same publication is available at [https://github.com/kirchhausenlab/gc_segment](https://github.com/kirchhausenlab/gc_segment).

## Table of Contents
- [Interactive Demo](#Interactive-Demo)
- [Datasets](#Datasets)
- [Installation](#Installation)
- [Optional: Download our data](#Optional-Download-our-data)
- [Optional: Docker](#Optional-Docker)
- [Prepare your own data for prediction](#Prepare-your-own-data-for-prediction)
- [Prediction](#Prediction)
- [Prepare your own ground truth annotations for fine-tuning or training](#Prepare-your-own-ground-truth-annotations-for-fine-tuning-or-training)
- [Fine-Tuning](#Fine-tuning)
- [Training](#Training)

## Interactive Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sk9UDe4t1qL_iz-nkJJPDVClMxRD8Avh)     
An interactive demo is provided via Google Colab. This notebook can be used to work with sample data in and learn the basics of `incasem`. You can also work with your own data in the notebook, but it will require some modifications as specified in the notebook.

## Datasets 
[Take a look](http://asem-viewer-env.eba-rrnvmfwa.us-east-1.elasticbeanstalk.com/) at the lab's FIB-SEM datasets (raw, labels, predictions) directly in the browser with our simple-to-use cell viewing tool based on [neuroglancer](https://github.com/google/neuroglancer).

## Installation
This package is written for machines with either a Linux or a MacOS operating system.
> This README was written to work with the `bash` console. If you want to use `zsh` (default on newer versions of MacOS) or any other console, please make sure that you adapt things accordingly.

> Newer versions of MacOS (Catalina or newer): the following commands work correctly if you run the Terminal under Rosetta.
In Finder, go to `Applications/Utilities`, right click on `Terminal`, select `Get Info`, tick `Open using Rosetta`.

We offer multiple installation options via different branches:
- `main` installed via CLI (Recommended)
- `main` installed via UI
- `legacy` installed via CLI

`main` and `legacy` have the same core pipeline functionality but `legacy` tracks experiments via `mongodb` which requires
configuring a networked database. Many users voiced frustration with this step, so we have refactored `main` to track experiments 
and configurations in your local file system and view training progress with `tensorboard`.     
 
For most users, this will be preferable. If you intend to run hundreds to thousands of trainings and predictions, we 
recommend installing the `legacy` branch and configuring the experiment tracking server!

**If you are interested in installing `legacy`, please stop following the installation instructions here, change branch, and instead follow the instructions
in that `README`**

#### 1. Clone the main repository.
```bash
git clone https://github.com/kirchhausenlab/incasem.git ~/incasem
```

#### 2. Create a new python environment with conda or mamba (recommended).
If you don't have [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html) or [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html), install first.
```bash
conda create -n incasem --no-default-packages python=3.10
```

#### 3. Pip-install the incasem package contained in this repository into the environment.
Activate the new environment.
```bash
conda activate incasem
```
Install
```bash
pip install -e ./incasem
```

#### 4. Install PyTorch as outlined [here](https://pytorch.org/get-started/locally/). 

## Optional: Download our data
The datasets in the publication are available in an [AWS bucket](https://open.quiltdata.com/b/asem-project/tree/datasets/) and can be downloaded with the [quilt3 API](https://docs.quiltdata.com/api-reference/api).
The cells have been renamed and re-indexed since the publication of _Gallusser, 2022_, so pleae refer to the table here 
if you are interested in a particular cell from that publication.

TODO:
TABLE

#### 1. Download an example dataset from the AWS bucket: 
Navigate to `~/incasem/data`:
```bash
cd ~/incasem/data
```
Open a python session and run the following lines.
> It may take a while until the download starts. Expected download speed is >= 2MB/s.
```python
import quilt3
b = quilt3.Bucket("s3://asem-project")
# download
b.fetch("datasets/example_cell.zarr/", "example_cell/example_cell.zarr/")
```

We provide all datasets as 2d `.tiff` images as well as in [`.zarr` format](https://zarr.readthedocs.io/en/stable/), which is more suitable for deep learning on 3D images. Above we only downloaded the `.zarr` format.


## Prepare your own data for prediction

We assume that the available 3d data is stored as a sequence of 2d `.tif` images in a directory.

#### 0. Copy your data into the project directory

```bash
cp -r old/data/location ~/incasem/data/my_new_data
```
#### 1. Go to the `01_data_formatting` directory

```bash
cd ~/incasem/scripts/01_data_formatting
```

#### 2. Activate the python environment
> In case you have not installed the python environment yet, refer to the [installation instructions](#Installation).

Before running python scripts, activate the ```incasem``` environment

```bash
conda activate incasem
```


#### 3. Conversion from `TIFF` to `zarr` format
Convert the sequence of `.tif` images (3D stack) to [`.zarr` format](https://zarr.readthedocs.io/en/stable/).
```bash
python 01_image_sequences_to_zarr.py -i ~/incasem/data/my_new_data -f ~/incasem/data/my_new_data.zarr
```

> To obtain documentation on how to use a script, run `python <script_name>.py -h`.


#### 4. Equalize intensity histogram of the data
Equalize the raw data with [CLAHE (Contrast limited adaptive histogram equalization)](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization). The default clip limit is `0.02`.
```bash
python 04_equalize_histogram.py -f ~/incasem/data/my_new_data.zarr -d volumes/raw -o volumes/raw_equalized_0.02
```

## Prediction
#### 1. Create a data configuration file
For running a prediction you need to create a configuration file in JSON format that specifies which data should be used.
Here is an example, also available at `~/incasem/scripts/03_predict/data_configs/example_cell.json`:
```json
{
    "example_cell_roi_nickname" : {
        "file": "example_cell/example_cell.zarr",
        "offset": [400, 926, 2512],
        "shape": [241, 476, 528],
        "voxel_size": [5, 5, 5],
        "raw": "volumes/raw_equalized_0.02"
    }
}
```
`offset` and `shape` are specified in voxels and in **z, y, x** format. They have to outline a _region of interest_ (ROI) that lies within the total available ROI of the dataset (as defined in `.zarray` and `.zattrs` files of each zarr volume).
> Note that the offset in each `.zattr` file is defined in nanometers, while the shape in `.zarray` is defined in voxels.

We assume the data to be in `~/incasem/data`, as defined [here](scripts/03_predict/config_prediction.yaml).

#### 2. Choose a model
We provide the following pre-trained models:
- For FIB-SEM data prepared by chemical fixation, `5x5x5` nm<sup>3</sup> resolution:
    - Mitochondria (model ID `1847`)
    - Golgi Apparatus (model ID `1837`)
    - Endoplasmic Reticulum (model ID `1841`)
- For FIB-SEM data prepared by high-pressure freezing, `4x4x4` nm<sup>3</sup> resolution:
    - Mitochondria (model ID `1675`)
    - Endoplasmic Reticulum (model ID `1669`)
- For FIB-SEM data prepared by high-pressure freezing, `5x5x5` nm<sup>3</sup> resolution:
    - Clathrin-Coated Pits (model ID `1986`) 
    - Nuclear Pores (model ID `2000`)

A checkpoint file for each of these models is stored in `~/incasem/models/pretrained_checkpoints/`.


#### 3. Run the prediction
Cell 6 has been prepared by chemical fixation and we will generate predictions for Endoplasmic Reticulum in this example, using model ID `1841`. In the prediction scripts folder,
```bash
cd ~/incasem/scripts/03_predict
```
Run
```bash
python predict.py --run_id 1841 --name example_cell_predict_ER with config_prediction.yaml 'prediction.data=data_configs/example_cell.json' 'prediction.checkpoint=../../models/pretrained_checkpoints/model_checkpoint_1841_er_CF.pt'
```
Note that we need to specify which model to use twice:
- `--run_id 1841` to load the appropriate settings from the models database.
- `'prediction.checkpoint=../../models/pretrained_checkpoints/model_checkpoint_1841_er_CF.pt'` to pass the path to the checkpoint file.


#### Optional:
If you have corresponding ground truth annotations, create a metric exclusion zone as [described below](#Prepare-your-own-ground-truth-annotations-for-fine-tuning-or-training). For the example of predicting Endoplasmic Reticulum in example_cell from above, put the metric exclusion zone in `example_cell/example_cell.zarr/volumes/metric_masks/er` and adapt `data_configs/example_cell.json` to:
```json
{
    "example_cell_roi_nickname" : {
        "file": "example_cell/example_cell.zarr",
        "offset": [400, 926, 2512],
        "shape": [241, 476, 528],
        "voxel_size": [5, 5, 5],
        "raw": "volumes/raw_equalized_0.02",
        "metric_masks": [
            "volumes/metric_masks/er"
        ],
        "labels": {
            "volumes/labels/er": 1
        }
    }
}
```

Now run
```bash
python predict.py --run_id 1841 --name example_prediction_cell6_ER_with_GT with config_prediction.yaml 'prediction.log_metrics=True' 'prediction.data=data_configs/example_cell.json' 'prediction.checkpoint=../../models/pretrained_checkpoints/model_checkpoint_1841_er_CF.pt'
```
, which will print an F1 score for the generated prediction given the ground truth annotations (`labels`).


#### 4. Convert the prediction to 'TIFF' format

Run `cd ~/incasem/scripts/04_postprocessing` to access the postprocessing scripts.

Now adapt and execute the conversion command below. In this example command, we assume that we have used model ID `1841`  to generate Endoplasmic Reticulum predictions for a subset of cell 6, and the automatically assigned prediction ID is `0001`.
```bash
python 20_convert_zarr_to_image_sequence.py --filename ~/incasem/data/example_cell/example_cell.zarr --datasets volumes/predictions/train_1841/predict_0001/segmentation --out_directory ~/incasem/data/example_cell --out_datasets example_er_prediction
```
You can open the resulting TIFF stack for example in ImageJ. Note that since we only made predictions on a subset of example_cell, the prediction TIFF stack is smaller than the raw data TIFF stack.


## Prepare your own ground truth annotations for fine-tuning or training

Example:  Endoplasmic reticulum (ER) annotations.

We assume that the available 3d pixelwise annotations are stored as a sequence of 2d `.tif` images in a directory and that the size of each `.tif` annotation image matches the size of the corresponding electron microscopy `.tif` image.

Furthermore, we assume that you have already prepared the corresponding electron microscopy images as outlined [above](#Prepare-your-own-data-for-prediction).

> The minimal block size that our training pipeline is set up to process is `(204, 204, 204)` voxels.


#### 0. Copy the annotation data into the project directory

```bash
cp -r old/annotations/location ~/incasem/data/my_new_er_annotations
```

#### 1. Go to the `01_data_formatting` directory

```bash
cd ~/incasem/scripts/01_data_formatting
```


#### 2. Activate the python environment
> In case you have not installed the python environment yet, refer to the [installation instructions](#Installation).

Before running python scripts, activate the ```incasem``` environment

```bash
conda activate incasem
```

#### 3. Conversion from `TIFF` to `zarr` format
Convert the sequence of `.tif` annotations (3D stack) to [.`zarr` format](https://zarr.readthedocs.io/en/stable/).
In this example, we use
```bash
python 00_image_sequences_to_zarr.py -i ~/incasem/data/my_new_er_annotations -f ~/incasem/data/my_new_data.zarr -d volumes/labels/er --dtype uint32
```
We assume the `.tif` file names are in the format `name_number.tif`, as encapsulated by the default regular expression `.*_(\d+).*\.tif$`. If you want to change it, add `-r your_regular_expression` to the line above.

If your datasets is hundreds of GB in size, try using the conversion script `01_image_sequences_to_zarr_with_dask.py`. You will need to install a different conda environment to work with `dask`, details directly in the [script](scripts/01_data_formatting/01_image_sequence_to_zarr.py).

```bash
python 01_image_sequences_to_zarr_with_dask.py -i ~/incasem/data/my_new_er_annotations -f ~/incasem/data/my_new_data.zarr -d volumes/labels_er --resolution 5 5 5 --dtype uint32
```

If the position of the labels is wrong, you can correct the offset by directly editing the dataset attributes file on disk:
```
cd ~/incasem/data/my_new_data.zarr/volumes/labels/er
vim .zattrs
```
In this file the offset is expressed in nanometers instead of voxels. So if the voxel size is `(5,5,5) nm`, you need to multiply the previous coordinates (z,y,x) by 5.


#### 4. Create a metric exclusion zone
We create a mask that will be used to calculate the F1 score for predictions, e.g. in the periodic validation during training.
This mask, which we refer to as _exclusion zone_, simply sets the pixels at the object boundaries to 0, as we do not want that small errors close to the object boundaries affect the overall prediction score.

We suggest the following exclusion zones in voxels:
- mito: 4 ```--exclude_voxels_inwards 4 --exclude_voxels_outwards 4```    
- golgi: 2 ```--exclude_voxels_inwards 2 --exclude_voxels_outwards 2```    
- ER: 2 ```--exclude_voxels_inwards 2 --exclude_voxels_outwards 2```    
- NP (nuclear pores): 1 ```--exclude_voxels_inwards 1 --exclude_voxels_outwards 1```    
- CCP (coated pits): 1 ```--exclude_voxels_inwards 2 --exclude_voxels_outwards 2```    

For our example with Endoplasmic Reticulum annotations, we run
```bash
python 06_create_metric_mask.py -f ~/incasem/data/my_new_data.zarr -d volumes/labels/er --out_dataset volumes/metric_masks/er --exclude_voxels_inwards 2 --exclude_voxels_outwards 2
```

## Fine-Tuning

If the prediction quality on a new target cell when using one of our pre-trained models is not satisfactory, you can finetune the model with a very small amount of ground truth from that target cell.

This is an example based on our datasets, which are publicly available in `.zarr` format via Amazon Web Services.
We will fine-tune the mitochondria model ID `1847`, which was trained on data from cells `46` and `58` (cell_1 and cell_2 from _Gallusser, 2022_),
with a small amount of additional data from cell `61` (cell_3 from _Gallusser, 2022_).

#### 0. Download training data
If you haven't done so before, download `61` from our published datasets as outlined in the section [_Download our data_](#Optional-Download-our-data).

#### 1. Create a fine-tuning data configuration file
For fine-tuning a model you need to create a configuration file in `JSON` format that specifies which data should be used.
Here is an example, also available at `~/incasem/scripts/02_train/data_configs/example_finetune_mito.json`:
```json
{
    "61_finetune_mito" : {
        "file": "61/61.zarr",
        "offset": [700, 2000, 6200],
        "shape": [250, 250, 250],
        "voxel_size": [5, 5, 5],
        "raw": "volumes/raw_equalized_0.02",
        "labels" : {
            "volumes/labels/mito": 1
        }
    }
}
```
Refer to the section [_Training_](#Training) for a detailed walk-through of such a configuration file.


#### 2. Launch the fine-tune training
In the training scripts folder,
```bash
cd ~/incasem/scripts/02_train
```
run
```bash
python train.py --name example_finetune --start_from 1847 ~/incasem/models/pretrained_checkpoints/model_checkpoint_1847_mito_CF.pt with config_training.yaml training.data=data_configs/example_finetune_mito.json validation.data=data_configs/example_finetune_mito.json torch.device=0 training.iterations=15000
```

Note that since we do not have extra validation data on the target `61`, we simply pass the training data configuration file to define a dummy validation dataset. 

#### 3. Observe the training

##### Tensorboard
To monitor the training loss in detail, open tensorboard:
```bash
tensorboard --logdir=~/incasem/training_runs/tensorboard/YOUR_RUN_ID/training
```


#### 4. Pick a fine-tuned model for prediction
Since we usually do not have any ground truth on the target cell that we fine-tuned for, 
we cannot rigorously pick the best model iteration.

We find that for example with ground truth in a 2 um<sup>3</sup> region of interest, typically after
5,000 - 10,000 iterations the fine-tuning has converged. The training loss (visible in tensorboard) 
can serve as a proxy for picking a model iteration in said interval.

Now you can use the fine-tuned model to generate predictions on the new target cell, as described in the section [_Prediction_](#Prediction).


## Training
This is an example based on our datasets, which are publicly available in `.zarr` format via Amazon Web Services.

#### 0. Download training data
Download `46` and `58` from our published datasets as outlined in the section [_Download our data_](#Optional-Download-our-data).

#### 1. Prepare the data

We create a mask that will be used to calculate the F1 score for predictions, e.g. in the periodic validation during training.
This mask, which we refer to as _exclusion zone_, simply sets the pixels at the object boundaries to 0, as we do not want that small errors close to the object boundaries affect the overall prediction score.

For our example with Endoplasmic Reticulum annotations on `46` and `58`, we run (from the data formatting directory):
```bash
python 06_create_metric_mask.py -f ~/incasem/data/46/46.zarr -d volumes/labels/er --out_dataset volumes/metric_masks/er --exclude_voxels_inwards 2 --exclude_voxels_outwards 2
```
and 
```bash
python 06_create_metric_mask.py -f ~/incasem/data/58/58.zarr -d volumes/labels/er --out_dataset volumes/metric_masks/er --exclude_voxels_inwards 2 --exclude_voxels_outwards 2
```

#### 2. Create a training data configuration file
For running a training you need to create a configuration file in `JSON` format that specifies which data should be used.
Here is an example, also available at `~/incasem/scripts/02_train/data_configs/example_train_er.json`:
> We assume the data to be in `~/incasem/data`, as defined [here](scripts/02_train/config_training.yaml).
```json
{
    "46_er" : {
        "file": "46/46.zarr",
        "offset": [150, 120, 1295],
        "shape": [600, 590, 1350],
        "voxel_size": [5, 5, 5],
        "raw": "volumes/raw_equalized_0.02",
        "metric_masks": [
            "volumes/metric_masks/er"
        ],
        "labels" : {
            "volumes/labels/er": 1
        }
    },
    "58_er": {
        "file": "58/58.zarr",
        "offset": [100, 275, 700],
        "shape": [500, 395, 600],
        "voxel_size": [5, 5, 5],
        "raw": "volumes/raw_equalized_0.02",
        "metric_masks": [
        	"volumes/metric_masks/er"
        ],
        "labels": {
            "volumes/labels/er": 1
        }
    }
}
```
`offset` and `shape` are specified in voxels and in **z, y, x** format. They have to outline a _region of interest_ (ROI) that lies within the total available ROI of the dataset (as defined in `.zarray` and `.zattrs` files of each zarr volume).
> Note that the offset in each `.zattr` file is defined in nanometers, while the shape in `.zarray` is defined in voxels.

All pixels inside the ROIs that belong to the structure of interest (e.g. endoplasmic reticulum above) in such a data configuration file have to be fully annotated. Additionally, our network architecture requires a context of 47 voxels of raw EM data around each ROI.


#### 3. Create a validation data configuration file
Additionally, you need to create a configuration file in `JSON` format that specifies which data should be used for periodic validation of the model during training.
Here is an example, also available at `~/incasem/scripts/02_train/data_configs/example_validation_er.json`:
```json
{
    "46_er_validation" : {
        "file": "46/46.zarr",
        "offset": [150, 120, 2645],
        "shape": [600, 590, 250],
        "voxel_size": [5, 5, 5],
        "raw": "volumes/raw_equalized_0.02",
        "metric_masks": [
            "volumes/metric_masks/er"
        ],
        "labels" : {
            "volumes/labels/er": 1
        }
    },
    "58_er_validation": {
        "file": "58/58.zarr",
        "offset": [300, 70, 700],
        "shape": [300, 205, 600],
        "voxel_size": [5, 5, 5],
        "raw": "volumes/raw_equalized_0.02",
        "metric_masks": [
            "volumes/metric_masks/er"
        ],
        "labels": {
            "volumes/labels/er": 1
        }
    }
}

```

#### 4. Optional: Adapt the training configuration
The file [`config_training.yaml`](scripts/02_train/config_training.yaml) exposes a lot of parameters of the model training.

Most importantly:
- If you would like to use data with a different resolution, apart from specifying in the data configuration files as outlined above, you need to adapt `data.voxel_size` in `config_training.yaml`.
- We guide the random sampling of blocks by rejecting blocks that consist of less than a given percentage (`training.reject.min_masked`) of foreground voxels with a chosen probability ('training.reject.probability'). If your dataset contains a lot of background, or no background at all, you might want to adapt these parameters accordingly.

#### 5. Launch the training
At the training scripts folder,
```bash
cd ~/incasem/scripts/02_train
```
run
```bash
python train.py --name example_training with config_training.yaml training.data=data_configs/example_train_er.json validation.data=data_configs/example_validation_er.json torch.device=0
```

#### 6. Observe the training

Each training run logs information to disk and to the training database, which can be inspected using Omniboard.
The log files on disk are stored in `~/incasem/training_runs`.

##### Tensorboard
To monitor the training loss in detail, open tensorboard:
```bash
tensorboard --logdir=~/incasem/training_runs/tensorboard/YOUR_RUN_ID/training
```
