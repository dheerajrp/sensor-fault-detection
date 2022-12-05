import pandas

from sensor.constants.training_pipeline import TARGET_COLUMN
from sensor.entity.artifact_entity import (
    DataValidationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)
from sensor.entity.config_entity import ModelEvaluationConfig
from sensor.exceptions import SensorException
from sensor.logger import logging
from sensor.ml.metrics.classification_metric import get_classification_score
from sensor.ml.model.estimator import ModelResolver
from sensor.utils.main_utils import load_object, write_yaml_file


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_eval_config = model_evaluation_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as error:
            raise SensorException(error)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

            # valid train and test dataframe
            train_df = pandas.read_csv(valid_train_file_path)
            test_df = pandas.read_csv(valid_test_file_path)

            df = pandas.concat([train_df, test_df])

            model_resolver = ModelResolver()

            trained_model_file_path = (
                self.model_trainer_artifact.trained_model_file_path
            )
            is_model_accepted = True
            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    trained_model_path=trained_model_file_path,
                    trained_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact,
                    best_model_metric_artifact=None,
                    best_model_path=None,
                    improved_accuracy=None,
                    is_model_accepted=is_model_accepted,
                )
                logging.info(f"Model Evaluation Artifact: {ModelEvaluationArtifact}")
                return model_evaluation_artifact

            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)
            train_model = load_object(file_path=trained_model_file_path)

            y_true = df[TARGET_COLUMN]
            y_train_pred = train_model.predict(df)
            y_latest_pred = train_model.predict(latest_model)

            train_model_metric = get_classification_score(y_true, y_train_pred)
            latest_metric = get_classification_score(y_true, y_latest_pred)

            improved_accuracy = train_model_metric.f1_score - latest_metric.f1_score
            if self.model_eval_config.model_evaluation_threshold < improved_accuracy:
                is_model_accepted = True
            else:
                is_model_accepted = False

            model_evaluation_artifact = ModelEvaluationArtifact(
                trained_model_path=trained_model_file_path,
                trained_model_metric_artifact=train_model_metric,
                best_model_metric_artifact=latest_metric,
                best_model_path=latest_model_path,
                improved_accuracy=improved_accuracy,
                is_model_accepted=is_model_accepted,
            )
            logging.info(f"Model Evaluation Artifact: {ModelEvaluationArtifact}")
            model_eval_report = model_evaluation_artifact.__dict__
            write_yaml_file(
                file_path=self.model_eval_config.model_evaluation_file_path,
                content=model_eval_report,
            )
            return model_evaluation_artifact
        except Exception as error:
            raise SensorException(error)
