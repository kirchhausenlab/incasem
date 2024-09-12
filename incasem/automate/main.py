import streamlit as st
from dataclasses import dataclass
from typing import Callable
from data_download import DataDownloader
from fine_tuning import fine_tuning_workflow
from incasem_setup import CondaEnvironmentManager
from incasem.automate.utils import handle_exceptions
from prediction import take_input_and_run_predictions
from training_run import main as take_input_and_create_configs
from view import view_cells_and_flatten_them


@dataclass
class AppMode:
    title: str
    handler: Callable


class IncasemApp:
    def __init__(self):
        self.app_modes = {
            "Setup": AppMode(title="Setup", handler=self.setup),
            "Data Download": AppMode(title="Data Download", handler=self.data_download),
            "View Cells": AppMode(title="View Cells", handler=self.view_cells),
            "Training": AppMode(title="Training", handler=self.training),
            "Prediction": AppMode(title="Prediction", handler=self.prediction),
            "Fine Tuning": AppMode(title="Fine Tuning", handler=self.fine_tuning),
        }
        self.data_downloader = DataDownloader()
        self.conda_manager = CondaEnvironmentManager()

    @handle_exceptions
    def __call__(self):
        st.sidebar.title("Incasem Navigation")
        app_mode = st.sidebar.selectbox(
            "Choose the app mode",
            [mode.title for mode in self.app_modes.values()],
        )
        if app_mode in self.app_modes:
            self.app_modes[app_mode].handler()

        self.display_workflow()

    def display_workflow(self):
        st.sidebar.title("Workflow Overview")
        st.sidebar.markdown("### Step-by-Step Guide")
        workflow_steps = [
            "1) Setup Conda, Setup Environment, Export Environment",
            "2) Data Download",
            "3) View Cells",
            "4) Run Training",
            "5) Prediction",
            "6) Fine Tuning",
        ]
        for step in workflow_steps:
            st.sidebar.write(step)

    def setup(self):
        st.title("Incasem Setup")
        st.write("Welcome to the Incasem setup")

        st.write(
            "Incasem requires one to use a virtual environment to manage dependencies. We can either use Conda or virtualenv."
        )

        st.markdown("""
        This package is written for machines with either a Linux or a MacOS operating system.
        Conda lets you isolate your environment and get the necessary packages for your project.
        We do the following here:
        1. Install anaconda for creating a conda python environment.
        2. Create a conda environment.
        3. Export the conda environment.    
        """)
        # write some instructions and explanations for what the code does below
        st.write(
            "This setup will guide you through the installation of Conda, setting up a Conda environment, and downloading the necessary data."
        )

        st.write("Firstly, we will setup Conda.")
        st.markdown("""
        Conda is a package manager that will help us manage our Python environment. One can create, save, load, and switch between different environments on your local machine.
                    """)
        with st.expander("Setup Conda", expanded=True, icon="🐍"):
            if st.button("Setup Conda"):
                st.write(
                    "Setting up Conda..., please wait. Conda is a package manager that will help us manage our Python environment."
                )
                self.conda_manager.setup_conda()
        st.markdown("""
        Once we have conda installed, now we will create a new environment.
        """)
        with st.expander("Setup Environment", expanded=False, icon="🛠"):
            env_name = st.text_input("Enter environment name", "incasem")
            if st.button("Setup Environment"):
                st.write(
                    "Setting up the Conda environment..., please wait. This will create a new Conda environment with the specified name."
                )
                self.conda_manager.setup_environment(env_name)
        st.markdown("""
        Finally, we will export the environment.
        """)
        with st.expander("Export Environment", expanded=False, icon="📦"):
            if st.button("Export Environment"):
                st.write(
                    "Exporting the Conda environment..., please wait. This will create an environment file that can be shared and used to recreate the environment."
                )
                st.write("Note that this step is optional.")
                self.conda_manager.export_env()

    def data_download(self):
        st.title("Data Download")
        curr_name = st.text_input(
            label="Enter dataset name",
            value="cell_6_example",
            key="dataset_name",
            placeholder="Enter dataset name",
            help="You can find the dataset names in the AWS bucket.",
        )
        self.data_downloader.input_dataset_name(curr_name)
        self.data_downloader.download_data()

    def view_cells(self):
        view_cells_and_flatten_them()

    def prediction(self):
        take_input_and_run_predictions()

    def fine_tuning(self):
        fine_tuning_workflow()

    def training(self):
        take_input_and_create_configs()


app = IncasemApp()
app()
