from pathlib import Path
from typing import Optional
import streamlit as st
from incasem.logger.config import logger


class DirectoryNavigator:
    def __init__(self, start_dir: Path = Path.cwd()):
        """
        Initialize the DirectoryNavigator with the starting directory.

        Args:
        - start_dir (Path): The starting directory for the navigator, defaults to the current working directory.
        """
        self.current_directory = start_dir

    def list_directories_in_directory(self) -> list:
        """
        List all subdirectories in the current directory.

        Returns:
        - list: A list of subdirectory names.
        """
        try:
            st.text(f"Listing directories in {self.current_directory}")
            return [
                entry.name
                for entry in self.current_directory.iterdir()
                if entry.is_dir()
            ]
        except PermissionError:
            st.error("Permission denied to access this directory.")
            logger.error(
                "Permission denied to access directory: %s", self.current_directory
            )
            return []

    def navigate_directory(self, direction: str, selected_subdir: Optional[str] = None):
        """
        Navigate through the directory structure.

        Args:
        - direction (str): Direction of navigation, either 'up' or 'down'.
        - selected_subdir (str): The selected subdirectory name for downward navigation.
        """
        if direction == "up":
            st.write("Going up one level...")
            self.current_directory = self.current_directory.parent
        elif direction == "down" and selected_subdir:
            st.write(f"Going down to {selected_subdir}...")
            new_dir = self.current_directory / selected_subdir
            if new_dir.is_dir():
                self.current_directory = new_dir

    def display_current_directory(self):
        """
        Display the current directory path in the Streamlit UI.
        """
        st.write(f"Current Directory: {self.current_directory}")

    def display_selected_directory(self):
        """
        Display the selected directory path in the Streamlit UI.
        """
        st.write("Selected Directory:")
        st.write(self.current_directory)


def get_dir():
    # Initialize the DirectoryNavigator with the default directory
    if "navigator" not in st.session_state:
        st.session_state.navigator = DirectoryNavigator()

    st.title("Directory Navigator")
    st.write(
        "Navigate through the directory structure. This navigator allows you to list subdirectories, go up and down the directory levels, and submit the selected path."
    )
    navigator = st.session_state.navigator

    # Display the current directory
    navigator.display_current_directory()

    # List and select subdirectories
    selected_subdir = st.selectbox(
        "Subdirectories:",
        [""] + navigator.list_directories_in_directory(),
    )

    # Buttons to navigate up and down the directory levels
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Go Up"):
            navigator.navigate_directory("up")

    with col2:
        if st.button("Go Down") and selected_subdir:
            navigator.navigate_directory("down", selected_subdir)

    if st.button("Submit Selected Path"):
        navigator.display_selected_directory()
