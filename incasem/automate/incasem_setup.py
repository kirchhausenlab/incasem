import subprocess
from pathlib import Path
from incasem.automate.utils import handle_exceptions
import streamlit as st


class CondaEnvironmentManager:
    """Class to manage Conda environment setup and handling in a Streamlit app."""

    def __init__(self):
        self.env_name = "incasem"
        self.curr_path = Path(__file__).parent
        self.path_to_reqs = self.curr_path.parent

    @staticmethod
    def is_conda_installed() -> bool:
        """Check if Conda is installed on the system by running a subprocess command."""
        st.text(
            "Checking Conda Installation. We are checking your system for Conda installation.",
            help="This may take a few seconds.",
        )
        return (
            subprocess.run(
                ["command", "-v", "conda"],
                capture_output=True,
                text=True,
                shell=True,
                check=True,
            ).returncode
            == 0
        )

    @staticmethod
    def is_env_active(env_name: str) -> bool:
        """Check if a Conda environment is active by listing environments."""
        st.write(f"Checking if the environment '{env_name}' is active...")
        cmd = "conda env list"
        result = subprocess.run(
            cmd, capture_output=True, text=True, shell=True, check=True
        )
        return f"{env_name}" in result.stdout

    @handle_exceptions
    def setup_conda(self):
        """Set up Conda by checking installation status and guiding the user through installation."""
        st.subheader("Conda Installation")
        if self.is_conda_installed():
            st.write("Conda is already installed.")
        else:
            st.write(
                "Conda is not installed. Please install Conda using the commands below. Essentially, we shall download the Miniconda installer, make it executable, and run it. \
                After installation, we shall add the Conda binary to the PATH and source the bashrc file. \
                Then we will export the PATH to the Conda binary \
                so that we can use Conda commands in the current shell. \
                finally we will source the bashrc file to make the changes permanent."
            )
            install_cmds = [
                "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh",
                "chmod +x Miniconda3-latest-Linux-x86_64.sh",
                "bash Miniconda3-latest-Linux-x86_64.sh",
                "export PATH=~/miniconda3/bin:$PATH",
                "source ~/.bashrc",
            ]
            for cmd in install_cmds:
                st.code(cmd, language="bash")

    @handle_exceptions
    def setup_environment(self, env_name: str) -> None:
        env_name = env_name or self.env_name
        """Set up a Conda environment, activating it if it exists, or creating and setting it up."""
        st.write(
            "We shall check if the environment exists, if it does not, we shall create it for you. The benefit of having a conda environment is that it allows you to manage different versions of Python and packages to be run \
            in different projects. This ensures that the dependencies of one project do not interfere with those of another project."
        )
        st.subheader(f"Creating a new Conda environment: {env_name}")
        if self.is_env_active(env_name):
            st.write(
                f"Environment '{env_name}' already exists. You can activate it using:"
            )
            st.code(f"conda activate {env_name}", language="bash")
        else:
            st.write(f"Creating and setting up the Conda environment '{env_name}'...")
            setup_cmds = [
                f"conda create --name {env_name} python=3.9 -y",
                f"conda activate {env_name}",
                f"pip install -e {self.path_to_reqs}",
            ]
            st.write(
                "Essentially, you will run pip install -e . if you were in the root directory of the project."
            )
            for cmd in setup_cmds:
                st.code(cmd, language="bash")

    @handle_exceptions
    def export_env(self):
        """Export the current Conda environment to a YAML file."""
        export_cmd = "conda env export > environment.yml && mv environment.yml ../"
        st.write("To export your Conda environment, run the following command:")
        st.code(export_cmd, language="bash")
        st.write("Environment setup is complete!")


# Usage Example:
if __name__ == "__main__":
    conda_manager = CondaEnvironmentManager()
    conda_manager.setup_conda()
    conda_manager.setup_environment("incasem")
    conda_manager.export_env()
