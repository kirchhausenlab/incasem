
import subprocess
from pathlib import Path
import streamlit as st
from incasem_setup import handle_exceptions


@handle_exceptions
def setup_omniboard():
    with st.expander("Setup Omniboard", expanded=True, icon="📊"):
        path_to_project = Path(__name__).resolve().parents[2]
        st.subheader("Setup Omniboard")
        st.write("Setting up Node.js environment and installing Omniboard...")
        subprocess.run("pip install nodeenv", shell=True )
        subprocess.run(f"cd {path_to_project}; nodeenv omniboard_environment", shell=True)

        with st.echo():
            st.write(f"Please navigate to {path_to_project} and run the following commands:")
        
            st.code("source omniboard_environment/bin/activate")
            st.code("npm install -g omniboard")
            st.write("Omniboard setup complete!")
