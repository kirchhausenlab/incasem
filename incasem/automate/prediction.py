from incasem.automate.utils import handle_exceptions
import json
import yaml
import subprocess
from incasem.logger.config import logger
from incasem.automate.directory_navigator import get_dir
from dataclasses import dataclass, field
from typing import Dict, List
import streamlit as st
from utils import (
    convert_tiff_to_zarr,
    create_config_file,
    run_command,
    validate_tiff_filename,
    validate_path,
)
from pathlib import Path

DEFAULT_MODEL_ID = "1847"


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
class PredictionConfig:
    """Data class for storing the overall prediction configuration."""

    input_path: str
    config_entries: List[ConfigEntry]
    file_name: str = "prediction_sample_file_"
    path_to_scripts: Path = Path(__name__).resolve().parents[2] / "scripts"
    path_to_store_conf: Path = Path(__name__).resolve().parents[1] / "data"
    path_to_store_run: Path = Path(__name__).resolve().parents[2] / "runs"

    def create_config_file(self) -> None:
        """Creates and saves the configuration file."""

        try:
            config_dict = {
                entry.name: {
                    "file": entry.path,
                    "raw": entry.raw,
                    "shape": entry.shape,
                    "voxel_size": entry.voxel_size,
                    "offset": entry.offset,
                }
                for entry in self.config_entries
            }
            self.output_path = self.path_to_store_conf
            conf_path = create_config_file(
                output_path=str(self.output_path),
                config=config_dict,
                file_name=self.file_name,
            )
            logger.info(f"Configuration file created at {conf_path}")
        except Exception as e:
            logger.error(f"Error creating configuration file: {e}")
            st.error(f"Error creating configuration file: {e}")


class PredictionRunner:
    """Class to handle the entire prediction process."""

    def __init__(
        self,
        config: PredictionConfig,
        model_id: str,
        checkpoint_path: str,
        is_tiff: bool,
        yaml_config: Dict[str, Dict[str, str]] = {},
    ):
        self.config = config
        self.config_file = config.file_name
        self.model_id = model_id
        self.checkpoint_path = checkpoint_path
        self.is_tiff = is_tiff
        self.curr_path = Path(__file__).parent
        self.path_to_scripts = self.curr_path.parent / "scripts"

    def equalize_histogram(self):
        """Equalize the intensity histogram of the data."""
        path_to_histogram_script = (
            self.path_to_scripts / "01_data_formatting/40_equalize_histogram.py"
        )
        logger.info("Equalizing intensity histogram of the data...")
        equalize_cmd = (
            f"python {path_to_histogram_script}"
            f"-f {self.config.output_path} -d volumes/raw -o volumes/raw_equalized_0.02"
        )
        st.write("Equalizing histogram...")
        st.code(equalize_cmd)
        run_command(equalize_cmd, "Histogram equalization complete!")

    def preprocess_data(self):
        """Convert TIFF to zarr format if required, and equalize histogram."""
        if self.is_tiff:
            logger.info(f"Converting TIFF to zarr: {self.config.input_path}")
            if st.button("Convert TIFF to Zarr"):
                convert_tiff_to_zarr(
                    input_path=self.config.input_path,
                )
        # ask the user if they want to equalize the histogram
        if st.checkbox("Do you want to equalize the histogram?"):
            self.equalize_histogram()

    def visualize_in_neuroglancer(self):
        """Visualize the data in Neuroglancer."""
        logger.info("Opening Neuroglancer...")
        neuroglancer_cmd = (
            f"neuroglancer -f {self.config.output_path} -d volumes/raw_equalized_0.02"
        )
        run_command(neuroglancer_cmd, "Neuroglancer opened!")

    def run_prediction(self):
        """Run prediction using the specified model and configuration."""
        self.model_id = st.text_input("Enter the model run ID", DEFAULT_MODEL_ID)
        self.checkpoint_path = st.text_input(
            "Enter the checkpoint path",
            f"../models/pretrained_checkpoints/model_checkpoint_{self.model_id}_er_CF.pt",
        )
        logger.info(f"Running prediction with model ID {self.model_id}...")
        # predict_script = PredictionConfig.path_to_scripts / "03_predict/predict.py"

        prediction_config_path = st.text_input(
            "Enter the path to the prediction configuration file",
            f"{PredictionConfig.path_to_store_conf}/{self.config_file}.json",
        )
        # pathlib code to show all files in the directory
        for file in Path(PredictionConfig.path_to_store_conf).iterdir():
            if file.is_file():
                st.write(file)

        predict_cmd = (
            f"python "
            f"../../scripts/predictions/predict.py "
            f"--run_id {self.model_id} --name example_prediction "
            f"with ../../scripts/predictions/config_prediction.yaml 'prediction.data={prediction_config_path}' "
            f"'prediction.checkpoint={self.checkpoint_path}'"
        )
        st.markdown("""
        We will highly recommend that you look at the configuration file before running the prediction.
        A test configuration file looks like:
            {
                "directories": {
                    "data": "../../incasem/data"
                },
                "prediction": {
                    "pipeline": "baseline",
                    "data": null,
                    "run_id_training": null,
                    "checkpoint": null,
                    "directories": {
                        "prefix": "../../incasem/data"
                    },
                    "input_size_voxels": [
                        204,
                        204,
                        204
                    ],
                    "output_size_voxels": [
                        110,
                        110,
                        110
                    ],
                    "num_workers": 8,
                    "log_metrics": false,
                    "torch": {
                        "device": 0
                    }
                }
            } 
            Generate your own configuration file and run the prediction command below based on what your run id, model checkpoint path and configuration file path is.
            """)

        if st.button("generate prediction configuration"):
            # Check if the model run ID exists
            st.write("Checking if the model run ID exists...")
            run_exists = False
            for file in Path(PredictionConfig.path_to_store_run).iterdir():
                if file.is_dir() and self.model_id in file.name:
                    run_exists = True
                    st.write(file)
                    # Open the config.json file from the existing run
                    config_file_path = file / "config.json"
                    with open(config_file_path) as f:
                        config_data = json.load(f)
                    st.write("Existing configuration file:")
                    st.json(config_data)
                    break

            if not run_exists:
                st.write("Model run ID does not exist. Please create a new run.")
                # Create a new experiment
                subprocess.run(predict_cmd, shell=True)
                st.write(
                    "Please run the prediction command below to generate the prediction."
                )
                st.code(f"{predict_cmd} --run_prediction")


@handle_exceptions
def take_input_and_run_predictions():
    """Gather user inputs and run predictions based on the provided configuration."""
    # UI Setup
    st.title("Incasem Prediction")
    st.write("Welcome to the Incasem prediction interface")
    st.markdown(
        """
        For running a prediction you need to create a configuration file in JSON format that specifies which data should be used. Here is an example, also available at ~/incasem/scripts/03_predict/data_configs/example_cell6.json:
        """
    )
    with st.expander("File Navigation", expanded=True, icon="📂"):
        get_dir()

    with st.expander("Prediction Configuration", expanded=True, icon="⚙️"):
        current_data_path = Path(__name__).resolve().parents[1] / "data"
        # Gather inputs
        file_type = st.radio("Select file type", ("TIFF", "ZARR"))
        input_path = st.text_input(
            label="Enter the input path",
            value=f"{current_data_path}/cell_6",
            placeholder="Enter the input path",
            help="Enter the path to the file, eg = '/nfs/scratch/username/Incasem_v2/incasem/data/cell_6/cell_6.zarr' or \
            '/User/yourname/Desktop/Incasem_v2/incasem/data/cell_6/cell_6.zarr'\
            ",
        )

        if file_type == "TIFF" and not validate_tiff_filename(input_path):
            st.error(
                "Invalid TIFF filename format. Please ensure the filename follows the pattern: .*_(\\d+).*\\.tif$"
            )
            return
        else:
            st.success("Valid TIFF filename format.")

        validate_path(input_path)

    with st.expander("Configuration File", expanded=True, icon="📄"):
        show_sample_config()
    with st.expander("Configuration Entries", expanded=True, icon="📝"):
        # Config Entries
        st.write("Create prediction configuration entries")
        if "config_entries" not in st.session_state:
            st.session_state["config_entries"] = []

        if st.button("Add configuration entry"):
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

        # Create Config File
        config = PredictionConfig(
            input_path=input_path,
            config_entries=st.session_state["config_entries"],
        )
        file_name = st.text_input(
            "Enter the name of the inference file", config.file_name
        )
        # PredictionConfig.file_name = file_name

        if st.button("Create Configuration"):
            config.file_name = file_name
            config.create_config_file()
            st.write("Configuration file created successfully!")

        if "config_entries" in st.session_state:
            # Model Selection
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
            model_choice = st.selectbox("Select a model", list(model_options.keys()))
            model_id = DEFAULT_MODEL_ID
            if model_choice:
                model_id = model_options[model_choice]
            checkpoint_path = (
                f"../models/pretrained_checkpoints/model_checkpoint_{model_id}_er_CF.pt"
            )
            st.write(f"Selected model: {model_choice}")

        st.text(
            "Please only run a prediction if you have completed the previous steps."
        )
    with st.expander("Run Prediction", expanded=True, icon="🚀"):
        # Run Prediction
        runner = PredictionRunner(
            config=config,
            model_id=model_id,  # type: ignore
            checkpoint_path=checkpoint_path,  # type: ignore
            is_tiff=(file_type == "TIFF"),
        )
        runner.preprocess_data()
        if st.checkbox("Do you want to visualize the data in Neuroglancer?"):
            runner.visualize_in_neuroglancer()
        runner.run_prediction()


def show_sample_config():
    st.write("This is what a sample configuration file looks like:")
    st.json(
        {
            "Cell_6_example_roi_nickname": {
                "file": "cell_6/cell_6.zarr",
                "offset": [400, 926, 2512],
                "shape": [241, 476, 528],
                "voxel_size": [5, 5, 5],
                "raw": "volumes/raw_equalized_0.02",
            }
        }
    )


def main():
    take_input_and_run_predictions()
