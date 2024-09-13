import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import streamlit as st
import zarr
from incasem.automate.utils import handle_exceptions


@dataclass
class ZarrFileNavigator:
    data_dir: Path

    def find_subdirectories(self) -> List[Path]:
        """
        Find all subdirectories within the specified `data_dir`.
        """
        return [d for d in self.data_dir.iterdir() if d.is_dir()]

    def find_zarr_files(self, sub_dir: Path) -> List[Path]:
        """
        Recursively find all Zarr files within the specified `sub_dir`.
        """
        return list(sub_dir.rglob("*.zarr"))

    def list_zarr_components(self, zarr_file_path: Path) -> List[str]:
        """
        List components (e.g., 'volumes/labels', 'volumes/predictions') within a Zarr file.
        """
        root = zarr.open(str(zarr_file_path), mode="r")
        components = []

        if "volumes" in root:
            volumes = root["volumes"]
            for key in volumes:
                if key in ["labels", "predictions"] or key.startswith("raw"):  # type: ignore
                    group = volumes[key]  # type: ignore
                    if isinstance(group, zarr.Group):  # type: ignore
                        for sub_key in group:  # type: ignore
                            components.append(f"volumes/{key}/{sub_key}")
                    else:
                        components.append(f"volumes/{key}")
        return components

    # def find_segmentation_folder(
    #     self, selected_components: List[str], selected_file: Path
    # ) -> str:
    #     """
    #     Find the segmentation folder path inside the predictions folder.
    #     """
    #     for component in selected_components:
    #         if component.startswith("volumes/predictions/"):
    #             prediction_key = component.split("/")[
    #                 2
    #             ]  # Extract the key after 'predictions'

    #             # Traverse into the 'predict' folder for segmentation
    #             prediction_path = (
    #                 selected_file / f"volumes/predictions/{prediction_key}"
    #             )
    #             for predict_folder in prediction_path.iterdir():
    #                 if predict_folder.is_dir() and "predict" in predict_folder.name:
    #                     segmentation_path = predict_folder / "segmentation"
    #                     if segmentation_path.exists():
    #                         return f"volumes/predictions/{prediction_key}/{predict_folder.name}/segmentation"
    #     return ""

    def find_segmentation_folders(
        self, selected_components: List[str], selected_file: Path
    ) -> List[str]:
        """
        Find all possible segmentation folder paths inside the predictions folder.
        """
        segmentation_folders = []
        for component in selected_components:
            if component.startswith("volumes/predictions/"):
                prediction_key = component.split("/")[
                    2
                ]  # Extract the key after 'predictions'

                # Traverse into the 'predict' folder for segmentation
                prediction_path = (
                    selected_file / f"volumes/predictions/{prediction_key}"
                )
                for predict_folder in prediction_path.iterdir():
                    if predict_folder.is_dir() and "predict" in predict_folder.name:
                        segmentation_path = predict_folder / "segmentation"
                        if segmentation_path.exists():
                            segmentation_folders.append(
                                f"volumes/predictions/{prediction_key}/{predict_folder.name}/segmentation"
                            )
        return segmentation_folders

    def construct_neuroglancer_command(
        self, selected_file: Path, selected_components: List[str]
    ) -> str:
        """
        Construct the Neuroglancer command based on selected components and the Zarr file.
        """
        return f"neuroglancer -f {selected_file} -d " + " ".join(selected_components)


@handle_exceptions
def view_cells_and_flatten_them():
    """
    Streamlit application to explore Zarr files, select components, and view them in Neuroglancer.
    """
    st.title("Zarr File Explorer")

    path_to_data_dir = Path(__name__).resolve().parents[1] / "data"

    data_dir = st.text_input(
        "Enter the data directory:",
        value=str(path_to_data_dir),
        key="data_dir",
        help="Enter the path to the data directory containing Zarr files to explore.",
        placeholder="Enter the path to the data directory",
    )

    navigator = ZarrFileNavigator(Path(data_dir))

    if navigator.data_dir.is_dir():
        subdirs = navigator.find_subdirectories()
        if subdirs:
            selected_subdir = st.selectbox("Select a subdirectory:", subdirs)
            if selected_subdir:
                sub_dir_path = Path(selected_subdir)
                zarr_files = navigator.find_zarr_files(sub_dir_path)
                if zarr_files:
                    st.write("Found Zarr files:")
                    selected_file = st.selectbox("Select a Zarr file:", zarr_files)
                    if selected_file:
                        components = navigator.list_zarr_components(Path(selected_file))
                        if components:
                            selected_components = st.multiselect(
                                "Select components:", components
                            )
                            if selected_components:
                                st.write("Selected components:")
                                st.write(selected_components)

                                # Find all segmentation folders in predictions
                                segmentation_folders = (
                                    navigator.find_segmentation_folders(
                                        selected_components, Path(selected_file)
                                    )
                                )

                                if segmentation_folders:
                                    # Allow user to select which segmentation folder they want
                                    selected_segmentation_folder = st.selectbox(
                                        "Select a segmentation folder:",
                                        segmentation_folders,
                                    )
                                    selected_components.append(
                                        selected_segmentation_folder
                                    )

                                neuroglancer_cmd = (
                                    navigator.construct_neuroglancer_command(
                                        Path(selected_file), selected_components
                                    )
                                )

                                st.write("Neuroglancer Command:")
                                st.code(neuroglancer_cmd, language="bash")

                                st.write(
                                    "Run the command in your terminal to view the data in Neuroglancer."
                                )
