import sys

from sensor.components.data_ingestion import DataIngestion
from sensor.entity.artifact_entity import DataIngestionArtifact
from sensor.entity.config_entity import (TrainingPipelineConfig,
                                         DataIngestionConfig)
from sensor.exceptions import SensorException
from sensor.logger import logging


class TrainPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.data_ingestion_config = DataIngestionConfig(
            training_pipeline_config=self.training_pipeline_config)

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion. .")
            self.data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Completed data ingestion. . Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as error:
            raise SensorException(error, sys)

    def start_data_validation(self):
        try:
            pass
        except Exception as error:
            raise SensorException(error, sys)

    def start_data_transformation(self):
        try:
            pass
        except Exception as error:
            raise SensorException(error, sys)

    def start_model_trainer(self):
        try:
            pass
        except Exception as error:
            raise SensorException(error, sys)

    def start_model_evaluation(self):
        try:
            pass
        except Exception as error:
            raise SensorException(error, sys)

    def start_model_pusher(self):
        try:
            pass
        except Exception as error:
            raise SensorException(error, sys)

    def run_pipeline(self):
        try:
            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
        except Exception as error:
            raise SensorException(error, sys)
