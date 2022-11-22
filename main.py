from pprint import pprint

import sensor.constants.database
from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.entity.config_entity import (TrainingPipelineConfig,
                                         DataIngestionConfig,
                                         DataValidationConfig)
from sensor.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    training_pipeline_config = TrainingPipelineConfig()
    # data_ingestion_config = DataIngestionConfig(training_pipeline_config)
    # print(data_ingestion_config.__dict__)
    # train_pipeline = TrainPipeline()
    # train_pipeline.run_pipeline()
    # d = DataValidationConfig(training_pipeline_config)
    # pprint(d.__dict__)
