import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import streamlit as st
import zarr
from incasem.automate.utils import handle_exceptions
from pathlib import Path


@dataclass
class ZarrFileNavigator:
    data_dir: Path

    def find_cells(self) -> List[str]:
        """
        Find directories within the specified `data_dir` that start with 'cell'.
        """
        cells = [
            d.name
            for d in self.data_dir.iterdir()
            if d.is_dir() and d.name.startswith("cell")
        ]
        return cells

    def find_zarr_files(self, cell_dir: Path) -> List[Path]:
        """
        Recursively find all Zarr files within the specified `cell_dir`.
        """
        return list(cell_dir.rglob("*.zarr"))

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

    def find_segmentation_folder(
        self, selected_components: List[str], selected_file: Path
    ) -> str:
        """
        Find the segmentation folder path based on selected components and Zarr file.
        """
        for component in selected_components:
            if component.startswith("volumes/labels/") or component.startswith(
                "volumes/predictions/"
            ):
                label_key = component.split("/")[2]
                prediction_path = (
                    selected_file / f"volumes/predictions/{label_key}/segmentation"
                )
                if prediction_path.exists():
                    return f"volumes/predictions/{label_key}/segmentation"
        return ""

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
        cells = navigator.find_cells()
        if cells:
            selected_cell = st.selectbox("Select a cell:", cells)
            if selected_cell:
                cell_dir = navigator.data_dir / selected_cell
                zarr_files = navigator.find_zarr_files(cell_dir)
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

                                segmentation_folder = (
                                    navigator.find_segmentation_folder(
                                        selected_components, Path(selected_file)
                                    )
                                )
                                if segmentation_folder:
                                    selected_components.append(segmentation_folder)

                                neuroglancer_cmd = (
                                    navigator.construct_neuroglancer_command(
                                        Path(selected_file), selected_components
                                    )
                                )
                                st.write("Neuroglancer Command:")
                                st.code(neuroglancer_cmd, language="bash")

                                if st.button("Run Neuroglancer"):
                                    subprocess.run(neuroglancer_cmd, shell=True)
                                    st.success("Neuroglancer command executed!")
