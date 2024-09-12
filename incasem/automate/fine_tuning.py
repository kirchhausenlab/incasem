import subprocess
from incasem.automate.directory_navigator import get_dir
import streamlit as st
from incasem.logger.config import logger
from incasem.automate.utils import handle_exceptions
from utils import (
    convert_tiff_to_zarr,
    create_config_file,
    run_command,
    validate_tiff_filename,
    validate_path,
)
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path

DEFAULT_MODEL_ID = "1841"


@dataclass
class ConfigEntry:
    """Data class for storing configuration entry details."""

    path: str
    name: str
    raw: str
    labels: Dict[str, int] = field(default_factory=dict)
    offset: List[int] = field(default_factory=list)
    shape: List[int] = field(default_factory=list)
    voxel_size: List[int] = field(default_factory=list)


@dataclass
class FineTuneConfig:
    """Data class for storing fine-tuning configuration details."""

    input_path: str
    path_to_scripts: Path = Path(__name__).resolve().parents[2] / "scripts"
    path_to_data: Path = (
        Path(__name__)
        .resolve()
        .parents[2]
        .joinpath("scripts", "02_train", "data_configs")
    )
    file_name: str = "fine_tuning_sample"
    config_entries: List[ConfigEntry] = field(default_factory=list)
    output_path: str = ""

    def create_config_file(self) -> None:
        try:
            config_dict = {
                entry.name: {
                    "file": entry.path,
                    "offset": entry.offset,
                    "shape": entry.shape,
                    "voxel_size": entry.voxel_size,
                    "raw": entry.raw,
                    "labels": entry.labels,
                }
                for entry in self.config_entries
            }
            self.output_path = str(self.path_to_data)
            conf_path = create_config_file(
                output_path=self.output_path,
                config=config_dict,
                file_name=self.file_name,
            )
            logger.info(f"Configuration file created at {conf_path}")
        except Exception as e:
            logger.error(f"Error creating configuration file: {e}")
            st.error(f"Error creating configuration file: {e}")


@dataclass
class IncasemFineTuning:
    """Class to manage the Incasem Fine-Tuning workflow."""

    path_to_scripts: Path = Path(__name__).resolve().parents[2] / "scripts"
    path_to_data: Path = Path(__name__).resolve().parents[1] / "data"

    @staticmethod
    @handle_exceptions
    def run_fine_tuning(
        config: FineTuneConfig,
        model_id: str,
        checkpoint_path: str,
        output_path: str,
    ) -> None:
        """Run fine-tuning and allow users to view results on Omniboard and TensorBoard."""
        st.write("Running fine-tuning...")
        name = st.text_input(
            "Enter the name of the fine-tuned file",
            "example_finetune",
            help="Default name set to 'example_finetune'.",
        )
        iterations = st.number_input(
            label="Enter the number of iterations (default: 15000)",
            value=15000,
            help="Leave blank to use the default number of iterations (15000).",
        )
        st.write("Sample command looks like this:")
        st.code(
            "python train.py --name example_finetune --start_from 1847 ~/incasem/models/pretrained_checkpoints/model_checkpoint_1847_mito_CF.pt with config_training.yaml training.data=data_configs/example_finetune_mito.json validation.data=data_configs/example_finetune_mito.json torch.device=0 training.iterations=15000"
        )
        path_to_train_script = config.path_to_scripts.joinpath("02_train", "train.py")
        path_training_data = st.text_input(
            label="Enter the path to the training data",
            value=config.path_to_data.joinpath(
                "data_configs", "example_finetune_mito.json"
            ),
        )
        path_to_validation_data = st.text_input(
            label="Enter the path to the validation data",
            value=config.path_to_data.joinpath(
                "data_configs", "example_finetune_mito.json"
            ),
        )
        fine_tune_cmd = f"python {path_to_train_script} --name {name} --start_from {model_id} {checkpoint_path} with config_training.yaml training.data={path_training_data} validation.data={path_to_validation_data} torch.device=0 training.iterations={iterations}"
        st.write("Your command to run is: ")
        st.code(fine_tune_cmd)

        st.write("Starting TensorBoard...")
        tensorboard_cmd = (
            f"tensorboard --logdir={output_path}/tensorboard --host 0.0.0.0 --port 6006"
        )
        subprocess.Popen(tensorboard_cmd, shell=True)
        st.write("TensorBoard running at http://localhost:6006")

    @staticmethod
    @handle_exceptions
    def create_metric_mask(output_path: str, path_to_scripts: Path) -> None:
        """Create a metric mask based on user inputs for the exclusion zone."""
        st.subheader("Exclusion Zone Configuration")
        st.markdown("""
            Define the exclusion zone for the metric mask. This zone determines which pixels
            near object boundaries will be excluded from the F1 score calculation during validation.

            **Recommended values:**
            - Mitochondria: 4 inwards, 4 outwards
            - Golgi & ER: 2 inwards, 2 outwards
            - Nuclear Pores: 1 inwards, 1 outwards
            - Coated Pits: 2 inwards, 2 outwards
        """)
        f_output_path = st.text_input(
            "Enter the output path",
            value=output_path,
            help="Specify the output path for the metric mask.",
        )

        # lookup output path
        if not Path(f_output_path).exists():
            st.error("Output path does not exist. Please ensure the path is correct.")

        dataset = st.text_input(
            "Enter the dataset path",
            value="volumes/labels/er",
            help="Specify the path to the dataset for which the mask will be created.",
        )

        out_dataset = st.text_input(
            "Enter the output dataset name",
            value="volumes/metric_masks/er",
            help="Specify the name for the output dataset.",
        )
        col1, col2 = st.columns(2)
        with col1:
            exclude_voxels_inwards = st.number_input(
                "Exclude voxels inwards",
                value=2,
                help="Specify the number of voxels to exclude inwards.",
            )
        with col2:
            exclude_voxels_outwards = st.number_input(
                "Exclude voxels outwards",
                value=2,
                help="Specify the number of voxels to exclude outwards.",
            )

        if st.button("Create Metric Mask"):
            path_to_create_metric_mask = (
                path_to_scripts / "01_data_formatting" / "60_create_metric_mask.py"
            )
            exclusion_cmd = (
                f"python {path_to_create_metric_mask} "
                f"-f {f_output_path} "
                f"-d {dataset} "
                f"--out_dataset {out_dataset} "
                f"--exclude_voxels_inwards {exclude_voxels_inwards} "
                f"--exclude_voxels_outwards {exclude_voxels_outwards}"
            )
            run_command(exclusion_cmd, "Metric exclusion zone created successfully!")
            # verify the output path
            st.write("For verification, check the output path:", f_output_path)
            # print out the directory structure for the output path


@handle_exceptions
def fine_tuning_workflow() -> None:
    """Manage the entire fine-tuning workflow through Streamlit."""
    st.title("Incasem Fine-Tuning Interface")

    with st.expander("Incasem Fine-Tuning Workflow", expanded=True, icon="🚀"):
        st.markdown("""
        ## Welcome to the Incasem fine-tuning workflow

        This interface guides you through the process of fine-tuning Incasem models.

        **_Please follow the steps below:_**
        ### Step 1: Input paths
        - Choose the input file type (TIFF or ZARR).
        - Enter the input path for annotations.
        - Convert TIFF to Zarr if necessary.

        **_If you choose TIFF:_**
        - We will check the size of the TIFF file and convert it to Zarr using Dask if it is larger than 10 GB.
        - We will move the file to the necessary location and automatically set up the file structure along with the environment.

        ### Step 2: Mask
        - Create a metric mask.

        ### Step 3: Create fine-tuning configuration entries
        - Create configuration entries for fine-tuning.

        ### Step 4: Choose a model
        - Choose a model for fine-tuning.

        ### Step 5: Run Fine-Tuning
        - Run fine-tuning using the specified model and configuration.
        """)
    with st.expander("File Navigation", expanded=False, icon="📂"):
        get_dir()
    try:
        path_to_data = Path(__name__).resolve().parents[1].joinpath("data")
        # Step 1: Input paths
        with st.expander("1. Input Paths", expanded=False, icon="🔨"):
            file_type = st.radio("Select file type", ("TIFF", "ZARR"))
            input_path = st.text_input(
                label="Enter the input path for annotations",
                value=f"{path_to_data}/cell_1",
                help="Specify the input path for annotations (local or cloud).",
            )
            # if file_type == "TIFF" and not validate_tiff_filename(input_path):
            #     st.error(
            #         "Invalid TIFF filename format. Please ensure the filename follows the pattern: .*_(\\d+).*\\.tif$"
            #     )
            if file_type == "TIFF":
                if st.button("Convert TIFF to Zarr"):
                    convert_tiff_to_zarr(input_path=input_path)
                    st.success("Valid TIFF filename format.")
            validate_path(input_path)
            output_path = f"{input_path}" if file_type == "TIFF" else input_path
            st.write(
                "*If the prediction quality on a new dataset is not satisfactory, consider fine-tuning.*"
            )
        path_to_scripts = Path(__name__).resolve().parents[2] / "scripts"
        # Step 2: Metric Mask
        with st.expander("2. Mask", expanded=False, icon="🛠️"):
            IncasemFineTuning.create_metric_mask(
                output_path=str(output_path), path_to_scripts=path_to_scripts
            )

        with st.expander("3. Show Configuration Entries", expanded=False, icon="📝"):
            """Print a sample JSON file."""
            st.write("This is a sample JSON file for fine-tuning.")
            st.json(
                {
                    "cell_3_finetune_mito": {
                        "file": "cell_3/cell_3.zarr",
                        "offset": [700, 2000, 6200],
                        "shape": [250, 250, 250],
                        "voxel_size": [5, 5, 5],
                        "raw": "volumes/raw_equalized_0.02",
                        "labels": {"volumes/labels/mito": 1},
                    }
                }
            )

        # Step 3: Fine-Tuning Configuration Entries
        with st.expander(
            "4. Create Fine-Tuning Configuration Entries", expanded=False, icon="📝"
        ):
            st.write(
                "Fill in the details for your dataset. You can define the location of labels, offsets, and other parameters."
            )
            if "config_entries" not in st.session_state:
                st.session_state["config_entries"] = []
            if st.button("Add Entry"):
                st.session_state["config_entries"].append(ConfigEntry("", "", ""))
            for i, entry in enumerate(st.session_state["config_entries"]):
                with st.container():
                    st.subheader(f"Configuration Entry {i+1}")
                    entry.name = st.text_input(
                        f"Enter name for entry {i+1}",
                        value=f"Cell_{i+1}",
                        placeholder="Cell_6_example_roi_nickname",
                        help="Enter the name for the entry",
                    )
                    entry.path = st.text_input(
                        f"Enter file path for entry {i+1}",
                        value="cell_6/cell_6.zarr",
                        help="Enter the path to the file, eg = 'cell_6/cell_6.zarr'",
                    )
                    # Offset
                    offset = st.text_input(
                        f"Enter offset for entry {i+1} (z, y, x)",
                        value="400, 926, 2512",
                        help="Enter the offset for the file, eg = '400, 926, 2512'",
                    )
                    if offset:
                        entry.offset = [int(x) for x in offset.split(",")]

                    # Shape
                    shape = st.text_input(
                        f"Enter shape for entry {i+1} (z, y, x)",
                        value="241, 476, 528",
                        help="Enter the shape for the file, eg = '241, 476, 528'",
                    )
                    if shape:
                        entry.shape = [int(x) for x in shape.split(",")]

                    # Voxel Size
                    voxel_size = st.text_input(
                        f"Enter voxel size for entry {i+1} (z, y, x)",
                        value="5, 5, 5",
                        help="Enter the voxel size for the file, eg = '5, 5, 5'",
                    )
                    if voxel_size:
                        entry.voxel_size = [int(x) for x in voxel_size.split(",")]
                    raw = st.text_input(
                        f"Enter raw key for entry {i+1}",
                        value="volumes/raw",
                        help="Enter the raw key for the file eg = 'volumes/raw'",
                    )
                    entry.raw = raw

                    labels = st.text_input(
                        f"Enter labels for entry {i+1}",
                        value="volumes/labels",
                        help="Enter the labels for the file eg = 'volumes/labels'",
                    )
                    if labels:
                        entry.labels = {labels: 1}
            config = FineTuneConfig(
                input_path=str(input_path),
                config_entries=st.session_state["config_entries"],
            )
            file_name = st.text_input(
                "Enter the file name",
                value="fine_tuning_sample",
                help="Specify the file name for the configuration.",
            )

            if st.button("Create Configuration File"):
                config.file_name = file_name
                config.create_config_file()
                st.write("Configuration file created successfully!")
            if "config_entries" in st.session_state:
                # Step 4: Model Selection
                st.write("**Choose a model**")
                model_options = {
                    "FIB-SEM Chemical Fixation Mitochondria (CF, 5x5x5)": "1847",
                    "FIB-SEM Chemical Fixation Golgi Apparatus (CF, 5x5x5)": "1837",
                    "FIB-SEM Chemical Fixation Endoplasmic Reticulum (CF, 5x5x5)": "1841",
                    "FIB-SEM High-Pressure Freezing Mitochondria (HPF, 4x4x4)": "1675",
                    "FIB-SEM High-Pressure Freezing Endoplasmic Reticulum (HPF, 4x4x4)": "1669",
                    "FIB-SEM High-Pressure Freezing Clathrin-Coated Pits (HPF, 5x5x5)": "1986",
                    "FIB-SEM High-Pressure Freezing Nuclear Pores (HPF, 5x5x5)": "2000",
                }
                model_choice = st.selectbox(
                    "Select a model", list(model_options.keys())
                )
                model_id = DEFAULT_MODEL_ID
                if model_choice:
                    model_id = model_options[model_choice]
                checkpoint_path = f"../models/pretrained_checkpoints/model_checkpoint_{model_id}_er_CF.pt"
        with st.expander("5. Run Fine-Tuning", expanded=False, icon="🚀"):
            st.write("Start the fine-tuning process.")

            IncasemFineTuning.run_fine_tuning(
                config=config,
                model_id=model_id,  # type: ignore
                checkpoint_path=checkpoint_path,  # type: ignore
                output_path=str(output_path),
            )

    except Exception as e:
        st.error(f"Error: {e}")
