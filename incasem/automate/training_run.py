import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
import streamlit as st
from utils import create_config_file
from incasem.automate.fine_tuning import IncasemFineTuning

path_to_scripts_config: Path = Path(__name__).resolve().parents[2]
DATA_CONFIG_PATH: Path = path_to_scripts_config / "scripts/02_train/data_configs"


@dataclass
class ConfigEntry:
    name: str = ""
    file_name: str = ""
    offset: List[int] = field(default_factory=lambda: [0, 0, 0])
    shape: List[int] = field(default_factory=lambda: [0, 0, 0])
    voxel_size: List[int] = field(default_factory=lambda: [5, 5, 5])
    raw: str = ""
    metric_masks: List[str] = field(default_factory=list)
    labels: Dict[str, int] = field(default_factory=dict)


@st.cache_data
def init_session_state():
    if "step" not in st.session_state:
        st.session_state.step = "create_metric_masks"
    if "train_entries" not in st.session_state:
        st.session_state.train_entries = []
    if "val_entries" not in st.session_state:
        st.session_state.val_entries = []
    if "cell_paths" not in st.session_state:
        st.session_state.cell_paths = []
    if "autofill_raw" not in st.session_state:
        st.session_state.autofill_raw = ""
    if "autofill_metric_masks" not in st.session_state:
        st.session_state.autofill_metric_masks = []
    if "auto_fill_label_name" not in st.session_state:
        st.session_state.auto_fill_label_name = ""
    if "autofill_labels" not in st.session_state:
        st.session_state.autofill_labels = {}


@st.cache_resource
def show_workflow_overview():
    with st.expander("Workflow Overview 📋", expanded=True):
        st.markdown(
            """
        ## Incasem Training Workflow
        1. **Create Metric Masks** 🎭
        2. **Create Configuration Files** ⚙️
        3. **Run Training** 🚀
        
        Follow each step to set up and run your training process.
        """
        )


@st.cache_data
def create_metric_masks():
    """_summary_
    Code to get the paths for the cells, and generate metric masks for the zarr file. It will be stored in the cell/cell.zarr/volumes/metric_masks folder.
    A metric mask is a mask that will be used to calculate the F1 score for predictions, e.g. in the periodic validation during training. We look at the input
    cell path and create the mask
    """
    with st.container():
        st.subheader("Step 1: Create Metric Masks 🎭")
        st.write("Create masks for calculating the F1 score during training.")
        st.write(
            "We will generate a metric mask for each cell and store it in the cell/cell.zarr/volumes/metric_masks folder."
        )
        st.markdown(
            """You can check the mask in the `/incasem/incasem/data/cell_1/cell_1.zarr/volumes/metric_masks` folder"""
        )

        cell_paths = st.text_area("Enter the paths for the cells (one per line)").split(
            "\n"
        )
        cell_paths = [path.strip() for path in cell_paths if path.strip()]

        if st.button("Create Metric Masks"):
            st.session_state.cell_paths = cell_paths
            path_to_scripts = Path(__name__).resolve().parents[2] / "scripts"
            for cell_path in cell_paths:
                IncasemFineTuning.create_metric_mask(
                    output_path=cell_path, path_to_scripts=path_to_scripts
                )
            st.success("Metric masks created successfully")
            st.session_state.step = "create_configs"
        st.warning(
            "Incase of failures, please check the paths first. If there's no path error, the error might be due to the fact that the metric mask already exists."
        )


def create_config_entry(entry_type, index):
    """_summary_

    Parameters
    ----------
    entry_type : _type_
        Create a configuration entry for training or validation
    index : _type_
        Index of the entry

    Returns
    -------
    _type_
        Entry type to be used in the configuration file
    """
    entry = ConfigEntry()
    with st.container():
        st.subheader(f"{entry_type.capitalize()} Entry {index + 1} 📝")
        entry.name = st.text_input(
            f"Name for {entry_type} entry {index + 1}", f"{entry_type}_{index + 1}"
        )
        entry.file_name = st.text_input(
            f"File path for {entry_type} entry {index + 1}", entry.file_name
        )
        entry.offset = [
            int(
                st.number_input(
                    f"Offset Z for {entry_type} entry {index + 1}",
                    value=entry.offset[0],
                )
            ),
            int(
                st.number_input(
                    f"Offset Y for {entry_type} entry {index + 1}",
                    value=entry.offset[1],
                )
            ),
            int(
                st.number_input(
                    f"Offset X for {entry_type} entry {index + 1}",
                    value=entry.offset[2],
                )
            ),
        ]
        entry.shape = [
            int(
                st.number_input(
                    f"Shape Z for {entry_type} entry {index + 1}", value=entry.shape[0]
                )
            ),
            int(
                st.number_input(
                    f"Shape Y for {entry_type} entry {index + 1}", value=entry.shape[1]
                )
            ),
            int(
                st.number_input(
                    f"Shape X for {entry_type} entry {index + 1}", value=entry.shape[2]
                )
            ),
        ]
        entry.raw = st.text_input(
            f"Raw data path for {entry_type} entry {index + 1}",
            st.session_state.autofill_raw
            if st.session_state.autofill_raw
            else entry.raw,
        )
        if index == 0 and entry.raw:
            st.session_state.autofill_raw = entry.raw
        entry.metric_masks = st.text_input(
            f"Metric masks for {entry_type} entry {index + 1} (comma-separated)",
            ",".join(
                st.session_state.autofill_metric_masks
                if st.session_state.autofill_metric_masks
                else entry.metric_masks
            ),
        ).split(",")
        if index == 0 and entry.metric_masks:
            st.session_state.autofill_metric_masks = entry.metric_masks
        label_key = st.text_input(
            f"Label key for {entry_type} entry {index + 1}",
            st.session_state.auto_fill_label_name
            if st.session_state.auto_fill_label_name
            else "",
        )
        if index == 0 and label_key:
            st.session_state.auto_fill_label_name = label_key

        label_value = st.number_input(
            f"Label value for {entry_type} entry {index + 1}",
            value=1,
            help="Keep the value to be 1 for the same label",
        )
        if label_key:
            entry.labels[label_key] = int(label_value)
            if index == 0:
                st.session_state.autofill_labels[label_key] = int(label_value)
        else:
            entry.labels = st.session_state.autofill_labels
    return entry


@st.cache_data
def sample_json() -> None:
    st.json(
        {
            "cell_1_er": {
                "file": "cell_1/cell_1.zarr",
                "offset": [150, 120, 1295],
                "shape": [600, 590, 1350],
                "voxel_size": [5, 5, 5],
                "raw": "volumes/raw_equalized_0.02",
                "metric_masks": ["volumes/metric_masks/er"],
                "labels": {"volumes/labels/er": 1},
            },
            "cell_2_er": {
                "file": "cell_2/cell_2.zarr",
                "offset": [100, 275, 700],
                "shape": [500, 395, 600],
                "voxel_size": [5, 5, 5],
                "raw": "volumes/raw_equalized_0.02",
                "metric_masks": ["volumes/metric_masks/er"],
                "labels": {"volumes/labels/er": 1},
            },
        }
    )


def create_configs():
    st.code("Sample configuration file:")
    sample_json()

    with st.container():
        st.subheader("Step 2: Create Configuration Files ⚙️")
        # Find existing configuration files
        config_files = list(DATA_CONFIG_PATH.glob("*.json"))
        existing_configs = [config.name for config in config_files]

        # Option to use existing configs or create new ones
        use_existing = st.radio(
            label="Do you want to use existing configuration files?",
            options=("Yes", "No"),
            key="use_existing",
            help="Select 'Yes' to use existing configuration files, or 'No' to create new ones.",
            index=0,
        )

        if use_existing == "Yes":
            # Select existing training config
            train_config_name = st.selectbox(
                "Select the training config file",
                options=existing_configs,
                key="train_config_select",
            )
            st.session_state.train_config_path = str(
                DATA_CONFIG_PATH.joinpath(train_config_name)  # type: ignore
            )

            # Select existing validation config
            val_config_name = st.selectbox(
                "Select the validation config file",
                options=existing_configs,
                key="val_config_select",
            )
            if val_config_name:
                st.session_state.val_config_path = str(
                    DATA_CONFIG_PATH.joinpath(val_config_name)
                )
            else:
                st.session_state.val_config_path = ""

            st.success("Existing configuration files selected!")
            st.session_state.step = "run_training"
            st.write("Set up your training and validation configurations.")

        else:
            st.warning("Create new configuration files.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Add training entry"):
                    st.session_state.train_entries.append(ConfigEntry())
            with col2:
                if st.button("Add validation entry"):
                    st.session_state.val_entries.append(ConfigEntry())

            with st.container(border=True):
                st.subheader("Training Entries")
                st.write("Enter the configuration for the training data.")
                for i in range(len(st.session_state.train_entries)):
                    st.session_state.train_entries[i] = create_config_entry(
                        "training", i
                    )

            with st.container(border=True):
                st.subheader("Validation Entries")
                st.write("Enter the configuration for the validation data.")
                for i in range(len(st.session_state.val_entries)):
                    st.session_state.val_entries[i] = create_config_entry(
                        "validation", i
                    )

            with st.container(border=True):
                st.info("Name your configuration files:")

                train_config_name = st.text_input(
                    "Enter the name for the training config JSON file", ""
                )
                val_config_name = st.text_input(
                    "Enter the name for the validation config JSON file", ""
                )

                if st.button("Create Configurations"):
                    train_config = {
                        entry.name: entry.__dict__
                        for entry in st.session_state.train_entries
                    }
                    val_config = {
                        entry.name: entry.__dict__
                        for entry in st.session_state.val_entries
                    }

                    st.code(train_config)
                    st.code(val_config)

                    train_config_path = create_config_file(
                        str(DATA_CONFIG_PATH), train_config, train_config_name
                    )
                    val_config_path = create_config_file(
                        str(DATA_CONFIG_PATH), val_config, val_config_name
                    )

                    st.session_state.train_config_path = train_config_path
                    st.session_state.val_config_path = val_config_path
                    st.success("Configuration files created successfully!")
                    st.session_state.step = "run_training"

            st.write(f"Training config path: {st.session_state.train_config_path}")
            st.write(f"Validation config path: {st.session_state.val_config_path}")


def run_training():
    st.write("Start the training process with your configured settings.")

    model_name = st.text_input(
        label="Enter the name of the model",
        value="example_training_",
        help="This will be used to name the output directory for the model.",
    )

    path_to_train_script_dir = Path(__name__).resolve().parents[2] / "scripts/02_train"
    yaml_config_path = st.text_input(
        label="Path to YAML configuration file",
        value=f"{path_to_train_script_dir}/config_training.yaml",
        help="Path to the YAML configuration file for training.",
    )

    training_data_path = st.text_input(
        label="Path to training data config",
        value=st.session_state.get("train_config_path", ""),
        help="Path to the training data configuration file.",
    )

    validation_data_path = st.text_input(
        label="Path to validation data config",
        value=st.session_state.get("val_config_path", ""),
        help="Path to the validation data configuration file.",
    )

    # update yaml config

    with open(f"{path_to_train_script_dir}/config_training.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    path_to_parent_dir = Path(__name__).resolve().parents[3]
    runs_dir = st.text_input(
        label="Enter the paths to the runs directory where you will store the configuration for every run. (Whenever you run a training, you are supposed to create a new directory for it.)",
        value=path_to_parent_dir.joinpath("runs"),
        key="runs_dir",
    )
    data_dir = st.text_input(
        "Enter the path to the data directory where the data you want to train on is stored.",
        value=path_to_parent_dir.joinpath("incasem", "data"),
        key="data_dir",
    )

    database_dir = st.text_input(
        "*(In most cases you do not need to modify this)* Enter the path to the database directory where the database is stored.",
        value=path_to_parent_dir.joinpath("incasem", "mock_db"),
        key="database_dir",
    )

    iterations = st.number_input(
        "Enter the number of iterations for training.",
        value=200000,
        key="iterations",
    )

    if st.button("Update Yaml"):
        config["directories"]["runs"] = runs_dir
        config["directories"]["data"] = data_dir
        config["directories"]["db"] = database_dir
        config["training"]["iterations"] = iterations
        with open(f"{path_to_train_script_dir}/config_training.yaml", "w") as f:
            _ = yaml.dump(config, stream=f, default_flow_style=False, sort_keys=False)

    train_cmd = f"python {path_to_train_script_dir}/train.py --name {model_name} with {yaml_config_path} training.data={training_data_path} validation.data={validation_data_path} torch.device=0"
    st.code(train_cmd, language="bash")
    st.write("Run the training script from your terminal with the above command.")


def main():
    st.title("Incasem Training Workflow")

    init_session_state()
    show_workflow_overview()

    with st.expander("Step 1: Create Metric Masks 🎭", expanded=False):
        st.markdown(
            """
                    We create a mask that will be used to calculate the F1 score for predictions, e.g. in the periodic validation during training. 
                    This mask, which we refer to as exclusion zone, simply sets the pixels at the object boundaries to 0, 
                    as we do not want that small errors close to the object boundaries affect the overall prediction score.
                    For our example with Endoplasmic Reticulum annotations on cell_1 and cell_2, we run (from the data formatting directory):

                    ```python 60_create_metric_mask.py -f ~/incasem/data/cell_1/cell_1.zarr -d volumes/labels/er --out_dataset volumes/metric_masks/er --exclude_voxels_inwards 2 --exclude_voxels_outwards 2```
                    and 
                    
                    ```python 60_create_metric_mask.py -f ~/incasem/data/cell_2/cell_2.zarr -d volumes/labels/er --out_dataset volumes/metric_masks/er --exclude_voxels_inwards 2 --exclude_voxels_outwards 2```
                    """
        )
        path_to_data = Path(__name__).resolve().parents[1] / "data"
        path_to_scripts = Path(__name__).resolve().parents[2] / "scripts"
        IncasemFineTuning.create_metric_mask(path_to_data, path_to_scripts)

    with st.expander("Step 2: Create Configuration Files ⚙️", expanded=False):
        st.write("Configuration files have been created.")
        create_configs()

    with st.expander("Step 3: Run Training 🚀", expanded=False):
        run_training()


if __name__ == "__main__":
    main()