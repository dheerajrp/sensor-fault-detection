import os
from typing import Dict

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
