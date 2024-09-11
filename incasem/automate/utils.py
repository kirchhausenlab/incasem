import json
import os
import re
import subprocess
import streamlit as st
from functools import wraps
from incasem.logger.config import logger
import inspect
from typing import Callable
from pathlib import Path


def handle_exceptions(input_func: Callable) -> Callable:
    """Decorator for handling exceptions and logging errors."""

    @wraps(input_func)
    def wrapper(*args, **kwargs):
        try:
            return input_func(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(
                "File not found in function '{}' at line {} in '{}': {}",
                input_func.__name__,
                inspect.getsourcelines(input_func)[1],
                inspect.getfile(input_func),
                str(e),
            )
            raise FileNotFoundError(
                "File not found in function '{}' at line {} in '{}': {}".format(
                    input_func.__name__,
                    inspect.getsourcelines(input_func)[1],
                    inspect.getfile(input_func),
                    str(e),
                )
            )
        except json.JSONDecodeError as e:
            logger.error(
                "JSON decode error in function '{}' at line {} in '{}': {}",
                input_func.__name__,
                inspect.getsourcelines(input_func)[1],
                inspect.getfile(input_func),
                str(e),
            )
            raise json.JSONDecodeError(
                "JSON decode error in function '{}'".format(input_func.__name__),
                doc="",
                pos=0,
            )
        except ValueError as e:
            logger.error(
                "Value Error in function '{}' at line {} in '{}': {}",
                input_func.__name__,
                inspect.getsourcelines(input_func)[1],
                inspect.getfile(input_func),
                str(e),
            )
            raise ValueError(
                "value error in function '{}': {}".format(input_func.__name__, str(e))
            )
        except Exception as e:
            logger.error(
                "Error in function '{}' at line {} in '{}': {}",
                input_func.__name__,
                inspect.getsourcelines(input_func)[1],
                inspect.getfile(input_func),
                str(e),
            )
            raise RuntimeError(
                "The function '{}' failed with the error {} ".format(
                    input_func.__name__,
                    str(e),
                )
            )

    return wrapper


@handle_exceptions
def run_command(command: str, success_message: str) -> None:
    """Wrapper function to run commands"""
    try:
        subprocess.run(command, shell=True, check=True)
        logger.info(success_message)
        # st.success(success_message)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {command}\n{e}")
        st.error(f"Error executing command: {command}")


def create_config_file(output_path: str, config: dict, file_name: str) -> str:
    """Wrapper to make configuration files at the given path"""
    try:
        config_path = os.path.join(output_path, file_name + ".json")
        logger.info(f"Creating configuration file at {config_path}")
        st.write(f"Creating configuration file at {config_path}")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        return config_path
    except Exception as e:
        logger.error(f"Error creating configuration file: {e}")
        st.error(f"Error creating configuration file: {e}")
        return ""


def validate_tiff_filename(filename: str) -> bool:
    """Validate Tiff names according to the regex"""
    st.write(
        "We are validating the filename using the proper naming convensions for our code base, if your file has the wrong name, please check our documentation and change it otherwise it will fail to run"
    )
    return bool(re.match(r".*_(\d+).*\.tif$", filename))


def convert_tiff_to_zarr(input_path: str) -> None:
    try:
        st.write("Checking the size of the TIFF file...")
        file_size = os.path.getsize(input_path)
        st.write(
            f"File size: {file_size} Gigabytes [Please Enter in Gigabytes otherwise you may have issues, 1 GB = 1000MB]"
        )
        path_to_run_conversion_script = Path(__name__).resolve().parents[2]
        script_to_run_without_dask = (
            path_to_run_conversion_script
            / "scripts/01_data_formatting/00_image_sequences_to_zarr.py"
        )
        script_to_run_with_dask = (
            path_to_run_conversion_script
            / "scripts/01_data_formatting/01_image_sequences_to_zarr_with_dask.py"
        )
        if (
            not script_to_run_without_dask.exists()
            or not script_to_run_with_dask.exists()
        ):
            logger.error("Path to run conversion script does not exist.")
            st.error("Path to run conversion script does not exist.")
            return

        output_path = input_path + ".zarr"
        if file_size > 10:  # 10 GB
            st.write(
                "Large TIFF file detected. Using Dask for conversion...The command looks like follows:"
            )
            st.markdown(
                "python 01_image_sequences_to_zarr_with_dask.py -i ~/incasem/data/my_new_data -f ~/incasem/data/my_new_data.zarr -d volumes/raw --resolution 5 5 5 --dtype uint32"
            )
            st.write(
                "We will store the resulting output at the same location for your input_path as {input_path}.zarr"
            )
            convert_cmd = f"python {str(script_to_run_with_dask)} -i {input_path} -f {output_path} -d volumes/raw --resolution 5 5 5 --dtype uint32"
        else:
            st.write("Converting TIFF to zarr format...")
            convert_cmd = f"python {str(script_to_run_with_dask)} -i {input_path} -f {output_path}"

        run_command(convert_cmd, "Conversion to zarr format complete!")
        st.success(
            "To view the resulting file on neuroglancer, you can go under the cells folder and select our required files"
        )
    except Exception as e:
        logger.error("Error converting TIFF to Zarr: %s" % e)
        st.error(f"Error converting TIFF to Zarr: {e}")


def validate_path(input_path: str) -> None:
    """Validate the path and check if it exists"""
    try:
        verify_path = Path(input_path)
        if not verify_path.exists():
            logger.error("The specified input path does not exist.")
            return
        logger.info("The specified input path exists.")
    except (FileNotFoundError, Exception) as e:
        logger.error("Error validating path: %s" % e)
