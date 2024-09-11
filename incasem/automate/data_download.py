import subprocess
import os
import streamlit as st
from incasem.automate.utils import handle_exceptions
import quilt3
from incasem.logger.config import logger
from pathlib import Path


class DataDownloader:
    def __init__(
        self, bucket_name: str = "s3://asem-project", dataset_name: str = "cell_6"
    ):
        self.bucket_name = bucket_name
        self.dataset_name = dataset_name

    def input_dataset_name(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        logger.info(f"Dataset name set to {self.dataset_name}")

    @staticmethod
    def _run_subprocess(command: str) -> None:
        """Run a shell command with subprocess, ensuring errors are caught."""
        try:
            result = subprocess.run(
                f"python -c {command}",
                shell=True,
            )
            if result.returncode == 0:
                st.write(f"Command executed successfully: {command}")
                logger.info(f"Command executed successfully: {command}")
            else:
                st.error(f"Command failed: {command}")
                logger.error(f"Command failed: {command}")
        except subprocess.CalledProcessError as e:
            st.error(f"Subprocess error: {e}")
            logger.error("Subprocess error: %s" % str(e))
            raise e

    @handle_exceptions
    def download_data(self) -> None:
        """Download the specified dataset from the AWS bucket."""
        st.subheader("Download Data")

        with st.expander("About Data Download", expanded=True, icon="☁️"):
            st.write(
                "The datasets in the publication are available in an AWS bucket(https://open.quiltdata.com/b/asem-project/tree/datasets/) and can be downloaded with the quilt3 API.(https://docs.quiltdata.com/api-reference/api)"
            )
            st.write(f"Downloading example dataset from AWS bucket: {self.dataset_name}. In the \
                background, we shall navigate a level outside of the current folder and run a Python script to download the data from the s3 bucket.")

            path_to_data = Path(__file__).parent.parent / "data"
            path_to_data.mkdir(parents=True, exist_ok=True)
            cell_name = self.dataset_name
            if (path_to_data / cell_name).exists():
                st.write(f"Data already downloaded to {path_to_data}/{cell_name}")

            st.write(f"**Note**: Data download may take a few minutes.")
            if st.button("Download Data"):
                b = quilt3.Bucket("s3://asem-project")
                b.fetch(
                    f"datasets/{cell_name}/{cell_name}.zarr/",
                    f"{path_to_data}/{cell_name}/{cell_name}.zarr/",
                )
                # load database dump into local mongodb
                st.write(
                    f"Data downloaded to {path_to_data}/{cell_name}/{cell_name}.zarr/"
                )
                st.write("Data download complete!")
