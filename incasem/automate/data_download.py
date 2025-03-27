import subprocess
from pathlib import Path

import boto3
import quilt3
import streamlit as st
from automate.utils import handle_exceptions
from logger.config import logger


class DataDownloader:
    def __init__(
        self, bucket_name: str = "s3://asem-project", dataset_name: str = "cell_6"
    ):
        self.bucket_name = bucket_name
        self.dataset_name = dataset_name
        self.s3_client = boto3.client("s3")
        # Extract bucket name without s3:// prefix for boto3
        self.bucket_name_clean = bucket_name.replace("s3://", "")
        self.b = quilt3.Bucket(bucket_name)
        self.available_datasets = []

    def input_dataset_name(self, dataset_name: str) -> None:
        """Set the dataset name and verify its existence."""
        previous_name = self.dataset_name
        self.dataset_name = dataset_name
        logger.info(f"Dataset name set to {self.dataset_name}")

        # If dataset name changed, verify existence
        if previous_name != dataset_name:
            self.verify_dataset_existence()

    def verify_dataset_existence(self) -> bool:
        """Verify if the dataset exists in the AWS bucket and get its size."""
        try:
            # First refresh available datasets if needed
            if not self.available_datasets:
                self.load_available_datasets()
            # Check using two possible formats of the path
            dataset_path = f"datasets/{self.dataset_name}/"
            dataset_path_no_slash = f"datasets/{self.dataset_name}"
            dataset_path_zarr = f"datasets/{self.dataset_name}.zarr/"

            # Check if any of the formats exist in available_datasets
            if (
                dataset_path in self.available_datasets
                or dataset_path_no_slash in self.available_datasets
                or dataset_path_zarr in self.available_datasets
            ):
                print("Dataset exists")
                # Get size information
                st.success(
                    f"✅ Dataset '{self.dataset_name}' exists! 🎉 in the AWS bucket, please proceed to download"
                )
                return True
            else:
                st.error(f"❌ Dataset '{self.dataset_name}' not found in AWS bucket")
                if self.available_datasets:
                    st.info(
                        "💡 Tip: Click 'List Available Datasets' to see valid options"
                    )
                return False
        except Exception as e:
            logger.error(f"Error verifying dataset: {str(e)}")
            st.error(f"Error verifying dataset: {str(e)}")
            return False

    def load_available_datasets(self) -> None:
        """Load available datasets from the bucket."""
        try:
            result = self.b.ls("datasets/")
            # Extract the dataset names from the complex structure
            self.available_datasets = []

            # Process the first element which contains the prefixes
            if result and len(result) > 0 and isinstance(result[0], list):
                for item in result[0]:
                    if "Prefix" in item:
                        self.available_datasets.append(item["Prefix"])

            logger.info(f"Successfully loaded {len(self.available_datasets)} datasets")
        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")
            st.error(f"Error loading datasets: {str(e)}")

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

        # Helpful description
        with st.expander("About Data Download", expanded=True, icon="☁️"):
            st.markdown("""
            ### Data Download Process
            
            The datasets from the publication are stored in an AWS S3 bucket and can be accessed using the Quilt3 API.
            
            #### How to use this tool:
            1. Enter a dataset number (e.g. `100`) in the input field above
            2. Click "Verify Dataset" to check if it exists and see its size
            3. Click "List Available Datasets" to see all options
            4. Click "Download Data" to begin the download process
            
            **Note:** If you see `datasets/100` listed, simply enter `100` as the dataset name.
            """)

        st.write(
            "The datasets in the publication are available in an AWS bucket(https://open.quiltdata.com/b/asem-project/tree/datasets/) and can be downloaded with the quilt3 API.(https://docs.quiltdata.com/api-reference/api)"
        )
        st.write(
            f"Downloading example dataset from AWS bucket: {self.dataset_name}. In the \
            background, we shall navigate a level outside of the current folder and run a Python script to download the data from the s3 bucket."
        )

        # Dataset verification section
        st.markdown("### Dataset Verification")
        if st.button("🔍 Verify Dataset Existence on AWS cloud"):
            self.verify_dataset_existence()

        # List available datasets
        if st.button("📋 List Available Datasets"):
            with st.spinner("Loading available datasets..."):
                self.load_available_datasets()

                if self.available_datasets:
                    # Create a more user-friendly display with just the dataset numbers
                    clean_names = []
                    for dataset in self.available_datasets:
                        name = dataset.replace("datasets/", "").replace("/", "")
                        if name:  # Skip empty strings
                            clean_names.append(name)

                    # Group datasets into columns for better display
                    col1, col2, col3 = st.columns(3)
                    chunks = [
                        clean_names[i : i + len(clean_names) // 3 + 1]
                        for i in range(0, len(clean_names), len(clean_names) // 3 + 1)
                    ]

                    for i, chunk in enumerate([
                        chunks[0],
                        chunks[1] if len(chunks) > 1 else [],
                        chunks[2] if len(chunks) > 2 else [],
                    ]):
                        col = [col1, col2, col3][i]
                        with col:
                            for name in chunk:
                                st.code(name, language=None)
                else:
                    st.warning(
                        "No datasets found in the bucket or unable to retrieve the list."
                    )

        # Download section
        st.markdown("### Download Section")
        path_to_data = Path(__file__).parent.parent / "data"
        path_to_data.mkdir(parents=True, exist_ok=True)
        cell_name = self.dataset_name

        # Check if already downloaded
        if (path_to_data / cell_name).exists():
            st.success(
                f"✅ Data already downloaded to {path_to_data}/{cell_name}, please verify the size, it should be around 3-10 GB minimum, otherwise download again."
            )
            st.info("You can proceed to the next step or download a different dataset.")

        # Download button with progress tracking
        if st.button("⬇️ Download Data"):
            if self.verify_dataset_existence():
                download_path = f"{path_to_data}/{cell_name}/{cell_name}.zarr/"

                with st.spinner(
                    f"Downloading dataset {cell_name}... This may take several minutes depending on size."
                ):
                    # Create a progress bar
                    try:
                        # Fetch with progress tracking if possible
                        self.b.fetch(
                            f"datasets/{cell_name}/{cell_name}.zarr/",
                            download_path,
                        )
                        st.success(
                            f"✅ Data successfully downloaded to {download_path}"
                        )
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error during download: {str(e)}")
                        logger.error(f"Download error: {str(e)}")
