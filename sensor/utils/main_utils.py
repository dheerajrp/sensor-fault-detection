import os
from typing import Dict

import yaml

from sensor.exceptions import SensorException
from sensor.logger import logging


def read_yaml_file(file_path) -> Dict:
    try:
        logging.info('Reading schema.yaml file. .')
        with open(file_path, 'rb') as schema_file:
            return yaml.safe_load(schema_file)
    except Exception as error:
        raise SensorException(error)


def write_yaml_file(file_path: str,
                    content: object,
                    replace: bool = False
                    ) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'rb') as schema_file:
            yaml.dump(content, schema_file)
    except Exception as error:
        raise SensorException(error)
