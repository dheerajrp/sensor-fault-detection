import logging
import sys
from typing import Optional

import numpy as np
import pandas
from pandas import DataFrame

from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.constants.database import DATABASE_NAME
from sensor.exceptions import SensorException


class SensorData:
    def __init__(self):
        """
        Establishes connection to mongodb client.
        """
        try:
            self.mongodb_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as error:
            raise SensorException(error, sys)

    def export_collection_as_dataframe(
        self,
        collection_name: str,
        database_name: Optional[str] = None,
    ) -> DataFrame:
        """
        Exports the collection from mongodb to a pandas dataframe.

        Args:
            collection_name (str):
                Name of the collection.
            database_name (optional, str):
                Name of the database.

        Returns:
            DataFrame:
                Pandas DataFrame
        """
        if database_name is None:
            collection = self.mongodb_client.database[collection_name]
        else:
            collection = self.mongodb_client[database_name][collection_name]

        df = pandas.DataFrame(list(collection.find()))

        if "_id" in df.columns.tolist():
            df = df.drop(columns="_id", axis=1)
            logging.info("dropped id column. .")

        df.replace({"na": np.nan}, inplace=True)
        logging.info("replaced na by nan. .")
        return df
