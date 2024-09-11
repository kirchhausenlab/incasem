import subprocess
import os

import quilt3
import streamlit as st
from incasem_setup import handle_exceptions


def is_mongodb_installed() -> bool:
    """Run a subprocess to see if MongoDB is installed or not"""
    return (
        subprocess.run(
            ["command", "-v", "mongod"],
            capture_output=True,
            text=True,
            shell=True,
            check=True,
        ).returncode
        == 0
    )


def setup_mongodb():
    st.subheader("Setup MongoDB")

    with st.expander("Setup MongoDB", expanded=True, icon="📒"):
        if is_mongodb_installed():
            st.write("MongoDB is already installed.")
        else:
            with st.echo():
                st.write("MongoDB is not installed. Installing MongoDB ....")
                if st.button("Install MongoDB"):
                    subprocess.run("brew tap mongodb/brew", shell=True)
                    subprocess.run("brew install mongodb-community@4.4", shell=True)

    with st.expander("Start MongoDB", expanded=False, icon="🚀"):
        st.write("Starting MongoDB service...")
        with st.echo():
            subprocess.run("brew services start mongodb-community", shell=True)

    with st.expander("Download models", expanded=False, icon="📦"):            
        if st.button("Download models"):
            download_models()

def download_models(bucket: quilt3.Bucket=quilt3.Bucket("s3://asem-project")) -> None:
    st.write("Downloading models from the AWS bucket...")
    bucket.fetch("models/", "./models/")
    os.popen('mongorestore --archive="models/fiborganelle_trainings" --nsFrom="public_fiborganelle.*" --nsTo="incasem_trainings.*"').read()