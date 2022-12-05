import os
from typing import Dict

import dill
import numpy as np
import yaml

from sensor.exceptions import SensorException
from sensor.logger import logging


def read_yaml_file(file_path: str) -> Dict:
    """
    Reads the yaml file.

    Args:
        file_path (str):
            The filepath of the yaml file.

    Returns:
        dict:
            The schema file
    """
    try:
        logging.info("Reading schema.yaml file. .")
        with open(file_path, "rb") as schema_file:
            return yaml.safe_load(schema_file)
    except Exception as error:
        raise SensorException(error)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes the yaml file to the path specified.

    Args:
        file_path (str):
            The filepath where the schema file is to be written.
        content (object):
            The contents of the yaml file.
        replace (boolean, optional):
            If True, Replaces the existing schema file in the filepath.

    Returns:
        None:
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as schema_file:
            yaml.dump(content, schema_file)
        logging.info("Writing drift report to report.yaml")
    except Exception as error:
        raise SensorException(error)


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array in a file

    Args:
        file_path:
            Location where the file to be saved.
        array:
            NumPy array to save.

    Returns:
        None:
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_writer:
            np.save(file_writer, array)
    except Exception as error:
        raise SensorException(error)


def load_numpy_array_data(file_path: str) -> np.array:
    """
    Loads the numpy array data from a given file path.

    Args:
        file_path:
            Location of the file where there is numpy array.

    Returns:
        np.array:
            A NumPy array.
    """
    try:
        with open(file_path, "rb") as file_reader:
            return np.load(file_reader, allow_pickle=True)
    except Exception as error:
        raise SensorException(error)


def save_object(file_path: str, obj: object) -> None:
    """
    Saves the object to the specified file path.

    Args:
        file_path:
            The file path where the object to be saved
        obj:
            The object to be saved in the file path.

    Returns:
        None:
    """
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "wb") as file_writer:
            dill.dump(obj, file_writer)
        file_writer.close()
    except Exception as error:
        raise SensorException(error)


def load_object(file_path: str) -> object:
    """
    Loads the object to the specified file path.

    Args:
        file_path:
            The file path where the object to be loaded.

    Returns:
        object:
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file path: {file_path} does not exist.")
        with open(file_path, "rb") as file_reader:
            dill.load(file_reader)
            return dill
    except Exception as error:
        raise SensorException(error)
