import subprocess
import os
import streamlit as st
from automate.utils import handle_exceptions
import quilt3
from logger.config import logger
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

class DataDownloader:
    def __init__(
        self, bucket_name: str = "s3://asem-project", dataset_name: str = "cell_6"
    ):
        self.bucket_name = bucket_name
        self.dataset_name = dataset_name
        self.s3_client = boto3.client('s3')
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
            # First refresh available datasets
            if not self.available_datasets:
                self.load_available_datasets()
            
            dataset_path = f"datasets/{self.dataset_name}"
            if dataset_path in self.available_datasets or f"{dataset_path}/" in self.available_datasets:
                # Get size information
                size_bytes = self.get_dataset_size(dataset_path)
                size_mb = size_bytes / (1024 * 1024)
                size_gb = size_bytes / (1024 * 1024 * 1024)
                
                if size_gb >= 1:
                    size_str = f"{size_gb:.2f} GB"
                else:
                    size_str = f"{size_mb:.2f} MB"
                
                st.success(f"✅ Dataset '{self.dataset_name}' exists! Size: {size_str}")
                return True
            else:
                st.error(f"❌ Dataset '{self.dataset_name}' not found in AWS bucket")
                if self.available_datasets:
                    st.info("💡 Tip: Click 'List Available Datasets' to see valid options")
                return False
        except Exception as e:
            logger.error(f"Error verifying dataset: {str(e)}")
            st.error(f"Error verifying dataset: {str(e)}")
            return False
            
    def get_dataset_size(self, dataset_path: str) -> int:
        """Get the total size of a dataset folder in bytes."""
        try:
            total_size = 0
            # List all objects with the dataset prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.bucket_name_clean, 
                Prefix=dataset_path
            )
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        total_size += obj['Size']
            
            return total_size
        except ClientError as e:
            logger.error(f"Error getting dataset size: {str(e)}")
            return 0
    
    def load_available_datasets(self) -> None:
        """Load available datasets from the bucket."""
        try:
            self.available_datasets = self.b.ls("datasets/")
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
        
        # Dataset verification section
        st.markdown("### Dataset Verification")
        if st.button("🔍 Verify Dataset"):
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
                    chunks = [clean_names[i:i+len(clean_names)//3+1] for i in range(0, len(clean_names), len(clean_names)//3+1)]
                    
                    for i, chunk in enumerate([chunks[0], chunks[1] if len(chunks) > 1 else [], chunks[2] if len(chunks) > 2 else []]):
                        col = [col1, col2, col3][i]
                        with col:
                            for name in chunk:
                                st.code(name, language=None)
                else:
                    st.warning("No datasets found in the bucket or unable to retrieve the list.")
        
        # Download section
        st.markdown("### Download Section")
        path_to_data = Path(__file__).parent.parent / "data"
        path_to_data.mkdir(parents=True, exist_ok=True)
        cell_name = self.dataset_name
        
        # Check if already downloaded
        if (path_to_data / cell_name).exists():
            st.success(f"✅ Data already downloaded to {path_to_data}/{cell_name}")
            st.info("You can proceed to the next step or download a different dataset.")
        
        # Download button with progress tracking
        if st.button("⬇️ Download Data"):
            if self.verify_dataset_existence():
                download_path = f"{path_to_data}/{cell_name}/{cell_name}.zarr/"
                
                with st.spinner(f"Downloading dataset {cell_name}... This may take several minutes depending on size."):
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    
                    # Define a progress callback function
                    def progress_callback(bytes_transferred):
                        # Get total size for percentage calculation
                        total_size = self.get_dataset_size(f"datasets/{cell_name}")
                        if total_size > 0:
                            progress = min(bytes_transferred / total_size, 1.0)
                            progress_bar.progress(progress)
                    
                    try:
                        # Fetch with progress tracking if possible
                        self.b.fetch(
                            f"datasets/{cell_name}/{cell_name}.zarr/",
                            download_path,
                            callback=progress_callback if hasattr(self.b, 'fetch_with_callback') else None
                        )
                        progress_bar.progress(1.0)
                        st.success(f"✅ Data successfully downloaded to {download_path}")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error during download: {str(e)}")
                        logger.error(f"Download error: {str(e)}")