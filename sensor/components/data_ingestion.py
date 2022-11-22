import os.path
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from sensor.data_access.sensor_data import SensorData
from sensor.entity.artifact_entity import DataIngestionArtifact
from sensor.entity.config_entity import DataIngestionConfig
from sensor.exceptions import SensorException
from sensor.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Export mongodb collection record as dataframe into feature store

        Returns:
            DataFrame
        """
        try:
            logging.info('Exporting data from mongo to feature store')
            sensor_data = SensorData()
            dataframe = sensor_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name)
            # create folder for feature store
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_name = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_name, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as error:
            raise SensorException(error, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Feature store dataset is split into train and test

        Args:
            dataframe (DataFrame):
                Feature store dataframe

        Returns:
            None:
        """
        try:
            train_csv, test_csv = train_test_split(dataframe,
                                                   test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info('Performed train test split. .')
            dir_path = os.path.dirname(
                self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path)
            logging.info('Exporting train csv. .')
            train_csv.to_csv(self.data_ingestion_config.training_file_path,
                             index=False, header=True)
            test_csv.to_csv(self.data_ingestion_config.testing_file_path,
                            index=False, header=True)
            logging.info('Train test split completed successfully. .')
        except Exception as error:
            raise SensorException(error, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_data_into_feature_store()
            self.split_data_as_train_test(dataframe=dataframe)
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config
                .training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path)
            return data_ingestion_artifact
        except Exception as error:
            raise SensorException(error, sys)
