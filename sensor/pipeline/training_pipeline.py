from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_transformation import DataTransformation
from sensor.components.data_validation import DataValidation
from sensor.components.model_evaluation import ModelEvaluation
from sensor.components.model_pusher import ModelPusher
from sensor.components.model_trainer import ModelTrainer
from sensor.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact, ModelPusherArtifact,
)
from sensor.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig, ModelPusherConfig,
)
from sensor.exceptions import SensorException
from sensor.logger import logging


class TrainPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.data_ingestion_config = DataIngestionConfig(
            training_pipeline_config=self.training_pipeline_config
        )

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion. .")
            self.data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(
                f"Completed data ingestion. . Artifact: {data_ingestion_artifact}"
            )
            return data_ingestion_artifact
        except Exception as error:
            raise SensorException(error)

    def start_data_validation(
            self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config,
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(
                f"Completed data validation. . Artifact: " f"{data_validation_artifact}"
            )
            return data_validation_artifact
        except Exception as error:
            raise SensorException(error)

    def start_data_transformation(
            self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_transformation = DataTransformation(
                data_validation_artifact, data_transformation_config
            )
            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )
            return data_transformation_artifact
        except Exception as error:
            raise SensorException(error)

    def start_model_trainer(
            self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
            model_trainer = ModelTrainer(
                data_transformation_artifact, model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as error:
            raise SensorException(error)

    def start_model_evaluation(
        self,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            model_eval_config = ModelEvaluationConfig(self.training_pipeline_config)
            model_evaluator = ModelEvaluation(
                model_eval_config, data_validation_artifact, model_trainer_artifact
            )
            model_eval_artifact = model_evaluator.initiate_model_evaluation()
            return model_eval_artifact
        except Exception as error:
            raise SensorException(error)

    def start_model_pusher(self,
                           model_eval_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        try:
            model_pusher_config = ModelPusherConfig(self.training_pipeline_config)
            model_pusher = ModelPusher(model_pusher_config, model_eval_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as error:
            raise SensorException(error)

    def run_pipeline(self):
        try:
            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
            data_validation_artifact: DataValidationArtifact = (
                self.start_data_validation(
                    data_ingestion_artifact=data_ingestion_artifact
                )
            )
            data_transformation_artifact: DataTransformationArtifact = (
                self.start_data_transformation(data_validation_artifact)
            )
            model_trainer_artifact: ModelTrainerArtifact = self.start_model_trainer(
                data_transformation_artifact
            )
            model_eval_artifact: ModelEvaluationArtifact = self.start_model_evaluation(
                data_validation_artifact, model_trainer_artifact
            )
            if not model_eval_artifact.is_model_accepted:
                raise Exception("Trained model is not better than the best model.")
            model_pusher_artifact: ModelPusherArtifact = self.start_model_pusher(model_eval_artifact)
        except Exception as error:
            raise SensorException(error)
