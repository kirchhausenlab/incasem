# Intelligent Naive-Cell Automatic Segmentation for Electron Microscopy

## Table of Contents
- [Installation](#Installation)
- [Optional: Download our data](#Optional-Download-our-data)
- [Prepare your own data for prediction](#Prepare-your-own-data-for-prediction)
- [Prediction](#Prediction)
- [Prepare your own ground truth annotations for fine-tuning or training](#Prepare-your-own-ground-truth-annotations-for-fine-tuning-or-training)
- [Fine-Tuning](#Fine-tuning)
- [Training](#Training)

## Installation
This package is written for machines with either a Linux or a MacOS operating system.
> This README was written to work with the `bash` console. If you want to use `zsh` (default on newer versions of MacOS) or any other console, please make sure that you adapt things accordingly.

> Newer versions of MacOS (Catalina or newer): the following commands work correctly if you run the Terminal under Rosetta.
In Finder, go to `Applications/Utilities`, right click on `Terminal`, select `Get Info`, tick `Open using Rosetta`.

#### 1. Install anaconda for creating a conda python environment.
Open a terminal window and download anaconda.
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
Optional: In case of permission issues, run
```bash
chmod +x Miniconda3-latest-Linux-x86_64.sh`
```

Install anaconda.
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

Type `conda` to check whether the installation worked, if not try to reload your `.bashrc` file with
`source ~/.bashrc`.

#### 2. Clone the main repository.
```bash
git clone https://github.com/kirchhausenlab/incasem.git ~/incasem
```

#### 3. Create a new anaconda python environment.
```bash
conda create -n incasem
```

#### 4. Pip-install the incasem package contained in this repository into the environment.
Activate the new environment.
```bash
conda activate incasem
```
Install
```bash
conda install pip
```
and
```bash
pip install -e ./incasem
```

#### 5. Install pytorch as outlined [here](https://pytorch.org/get-started/locally/).

#### 6. Install our neuroglancer scripts
```bash
pip install -e git+git://github.com/kirchhausenlab/funlib.show.neuroglancer.git@more_scripts_v2#egg=funlib.show.neuroglancer
```

#### 7. Set up the experiment tracking databases for training and prediction
- If not already installed on your system (check by running `mongod`), install [MongoDB](https://docs.mongodb.com/manual/administration/install-community/).
- Start up the MongoDB service. For example, if your machine is running Ubuntu (refer to [documentation](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/#run-mongodb-community-edition)), execute
```bash
sudo service mongod start
```
- Run
```bash
cd ~/incasem; python download_models.py
```

#### 9. Install Omniboard for viewing the experiment databases
Install Nodeenv.
```bash
pip install nodeenv
```
Create a new node.js environment (and thereby install node.js, if not already installed).
> This may take a while.
```bash
cd ~; nodeenv omniboard_environment
```

Activate the environment.
```bash
source omniboard_environment/bin/activate
```
Install [omniboard](https://vivekratnavel.github.io/omniboard/#/quick-start).
> This may take a while.
```bash
npm install -g omniboard
```

## Optional: Download our data
The datasets in the publication are available in an [AWS bucket](https://open.quiltdata.com/b/fibsem/tree/dataset/) and can be downloaded with the [quilt3 API](https://docs.quiltdata.com/api-reference/api).

#### 1. Download an example dataset from the AWS bucket: cell 6
Navigate to `~/incasem/data`:
```bash
cd ~/incasem/data
```
Open a python shell and run
```python
import quilt3
b = quilt3.Bucket("s3://fibsem")
# download
b.fetch("dataset/cell_6/", "~/incasem/data/cell_6/")
```

TODO add info about different formats: tiff, zarr, possibly n5.

#### 2. Explore a dataset
Example: Cell 6 raw electron microscopy data, Endoplasmic Reticulum prediction and corresponding Endoplasmic Reticulum ground-truth annotation.
```bash
neuroglancer -f cell_6/cell_6.zarr -d volumes/raw_equalized_0.02 volumes/predictions/er/segmentation volumes/labels/er
```
Navigate to position `520, 1164, 2776` (z,y,x) to focus on the Endoplasmic Reticulum predictions. You can simply overwrite the coordinates on the top left to do so.

If you are not familiar with inspecting 3D data with neuroglancer, you might want to have a look at this [video tutorial](https://youtu.be/TwBTyWWnbxc?t=75).

> Note: `neuroglancer` might not work in Safari. In this case, simply copy the link given by `neuroglancer` to Chrome or Firefox.


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
python 00_image_sequences_to_zarr.py -i ~/incasem/data/my_new_data -f ~/incasem/data/my_new_data.zarr
```

> To obtain documentation on how to use a script, run `python <script_name>.py -h`.


#### 4. Equalize intensity histogram of the data
Equalize the raw data with [CLAHE (Contrast limited adaptive histogram equalization)](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization). The default clip limit is `0.02`.
```bash
python 40_equalize_histogram.py -f ~/incasem/data/my_new_data.zarr -d volumes/raw -o volumes/raw_equalized
```
#### 5. Inspect the converted data with `neuroglancer`:
```bash
neuroglancer -f ~/incasem/data/my_new_data.zarr -d /volumes/raw
```
Refer to our [instructions](#2.-Explore-a-dataset) on how to use neuroglancer.


## Prediction
#### 1. Create a data configuration file
For running a prediction you need to create a configuration file in JSON format that specifies which data should be used.
Here is an example, also available at `~/incasem/scripts/03_predict/data_configs/example_cell6.json`:
```json
{
    "Cell_6_example_roi_nickname" : {
        "file": "cell_6/cell_6.zarr",
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

##### Optional: For detailed information about the trained modes, refer to the database downloaded above:
Activate the omniboard environment.
```bash
source ~/omniboard_environment/bin/activate
```
Run 
```bash
omniboard -m localhost:27017:incasem_trainings
```
and paste `localhost:9000` into your browser.

#### 3. Run the prediction
Cell 6 has been prepared by chemical fixation and we will generate predictions for Endoplasmic Reticulum in this example, using model ID `1841`. In the prediction scripts folder,
```bash
cd ~/incasem/scripts/03_predict
```
Run
```bash
python predict.py --run_id 1841 --name example_prediction_cell6_ER with config_prediction.yaml 'prediction.data=data_configs/example_cell6.json' 'prediction.checkpoint=../../models/pretrained_checkpoints/model_checkpoint_1841_er_CF.pt'
```
Note that we need to specify which model to use twice:
- `--run_id 1841` to load the appropriate settings from the models database.
- `'prediction.checkpoint=../../models/pretrained_checkpoints/model_checkpoint_1841_er_CF.pt'` to pass the path to the checkpoint file.

#### 4. Visualize the prediction
Every prediction is stored with a unique identifier (increasing number). If the example above was your first prediction run, you will see a folder `~/incasem/data/cell_6/cell_6.zarr/volumes/predictions/train_1841/predict_0001/segmentation`. To inspect these predictions, together with the corresponding EM data and ground truth, use the following command:
```bash
neuroglancer -f ~/incasem/data/cell_6/cell_6.zarr -d volumes/raw_equalized_0.02 volumes/labels/er volumes/predictions/train_1841/predict_0001/segmentation
```

#### 5. Convert the prediction to 'TIFF' format

Run `cd ~/incasem/scripts/04_postprocessing` to access the postprocessing scripts.

Now adapt and execute the conversion command below. In this example command, we assume that we have used model ID `1841`  to generate Endoplasmic Reticulum predictions for a subset of cell 6, and the automatically assigned prediction ID is `0001`.
```bash
python 20_convert_zarr_to_image_sequence.py --filename ~/incasem/data/cell_6/cell_6.zarr --datasets volumes/predictions/train_1841/predict_0001/segmentation --out_directory ~/incasem/data/cell_6 --out_datasets example_er_prediction
```
You can open the resulting TIFF stack for example in ImageJ. Note that since we only made predictions on a subset of cell 6, the prediction TIFF stack is smaller than the raw data TIFF stack.


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
In this example, we assume
```bash
python 00_image_sequences_to_zarr.py -i ~/incasem/data/my_new_er_annotations -f ~/incasem/data/my_new_data.zarr -d volumes/labels/er --dtype uint32
```

Inspect the converted data with `neuroglancer`:
```bash
neuroglancer -f ~/incasem/data/my_new_data.zarr -d volumes/raw volumes/labels/er
```
Refer to our [instructions](#2.-Explore-a-dataset) on how to use neuroglancer.

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
python 60_create_metric_mask.py -f ~/incasem/data/my_new_data.zarr -d volumes/labels/er --out_dataset volumes/metric_masks/er --exclude_voxels_inwards 2 --exclude_voxels_outwards 2
```


## Fine-Tuning

If the prediction quality on a new target cell when using one of our pre-trained models is not satisfactory, you can finetune the model with a very small amount of ground truth from that target cell.

This is an example based on our datasets, which are publicly available in `.zarr` format via Amazon Web Services.
We will fine-tune the mitochondria model ID `1847`, which was trained on data from cells 1 and 2, with a small amount of additional data from cell 3.

#### 0. Download training data
If you haven't done so before, download `cell_3` from our published datasets as outlined in the section [_Download our data_](#Download-our-data).

#### 1. Create a fine-tuning data configuration file
For fine-tuning a model you need to create a configuration file in `JSON` format that specifies which data should be used.
Here is an example, also available at `~/incasem/scripts/02_train/data_configs/example_finetune_mito.json`:
```json
{
    "cell_3_finetune_mito" : {
        "file": "cell_3/cell_3.zarr",
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

Note that since we do not have extra validation data on the target cell 3, we simply pass the training data configuration file to define a dummy validation dataset. 

#### 3. Observe the training

Each training run logs information to disk and to the training database, which can be inspected using Omniboard.
The log files on disk are stored in `~/incasem/training_runs`.

##### Tensorboard
To monitor the training loss in detail, open tensorboard:
```bash
tensorboard --logdir=~/incasem/training_runs/tensorboard
```

##### Omniboard (training database)
To observe the training and validation F1 scores, as well as the chosen experiment configuration, we use Omniboard.

Activate the omniboard environment:
```bash
source ~/omniboard_environment/bin/activate
```
Run 
```bash
omniboard -m localhost:27017:incasem_trainings
```
and paste `localhost:9000` into your browser.


#### 4. Pick a fine-tuned model for prediction
Since we usually do not have any ground truth on the target cell that we fine-tuned for, we cannot rigorously pick the best model iteration.

We find that for example with ground truth in a 2 um<sup>3</sup> region of interest, typically after 5,000 - 10,000 iterations the fine-tuning has converged. The training loss (visible in tensorboard) can serve as a proxy for picking a model iteration in said interval.

Now you can use the fine-tuned model to generate predictions on the new target cell, as described in the section [_Prediction_](#Prediction).


## Training
This is an example based on our datasets, which are publicly available in `.zarr` format via Amazon Web Services.

#### 0. Download training data
Download `cell_1` and `cell_2` from our published datasets as outlined in the section [_Download our data_](#Download-our-data).

#### 1. Prepare the data

We create a mask that will be used to calculate the F1 score for predictions, e.g. in the periodic validation during training.
This mask, which we refer to as _exclusion zone_, simply sets the pixels at the object boundaries to 0, as we do not want that small errors close to the object boundaries affect the overall prediction score.

For our example with Endoplasmic Reticulum annotations on `cell_1` and `cell_2`, we run (from the data formatting directory):
```bash
python 60_create_metric_mask.py -f ~/incasem/data/cell_1/cell_1.zarr -d volumes/labels/er --out_dataset volumes/metric_masks/er --exclude_voxels_inwards 2 --exclude_voxels_outwards 2
```
and 
```bash
python 60_create_metric_mask.py -f ~/incasem/data/cell_2/cell_2.zarr -d volumes/labels/er --out_dataset volumes/metric_masks/er --exclude_voxels_inwards 2 --exclude_voxels_outwards 2
```

#### 2. Create a training data configuration file
For running a training you need to create a configuration file in `JSON` format that specifies which data should be used.
Here is an example, also available at `~/incasem/scripts/02_train/data_configs/example_train_er.json`:
> We assume the data to be in `~/incasem/data`, as defined [here](scripts/02_train/config_training.yaml).
```json
{
    "cell_1_er" : {
        "file": "cell_1/cell_1.zarr",
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
    "cell_2_er": {
        "file": "cell_2/cell_2.zarr",
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
    "cell_1_er_validation" : {
        "file": "cell_1/cell_1.zarr",
        "offset": [150, 120, 2645],
        "shape": [600, 590, 250],
        "voxel_size": [5, 5, 5],
        "raw": "volumes/raw_equalized_0.02",
        "mask": "volumes/mask",
        "metric_masks": [
            "volumes/metric_masks/er"
        ],
        "labels" : {
            "volumes/labels/er": 1
        }
    },
    "cell_2_er_validation": {
        "file": "cell_2/cell_2.zarr",
        "offset": [300, 70, 700],
        "shape": [300, 205, 600],
        "voxel_size": [5, 5, 5],
        "raw": "volumes/raw_equalized_0.02",
        "mask": "volumes/mask",
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
tensorboard --logdir=~/incasem/training_runs/tensorboard
```

##### Omniboard (training database)
To observe the training and validation F1 scores, as well as the chosen experiment configuration, we use Omniboard.

Activate the omniboard environment:
```bash
source ~/omniboard_environment/bin/activate
```
Run 
```bash
omniboard -m localhost:27017:incasem_trainings
```
and paste `localhost:9000` into your browser.


#### 7. Pick a model for prediction
Using Omniboard, pick a model iteration where the validation loss and the validation F1 score have converged. Now use this model to generate predictions on new data, as described in the section [_Prediction_](#Prediction).
