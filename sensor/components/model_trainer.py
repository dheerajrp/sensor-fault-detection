import os

from xgboost import XGBClassifier

from sensor.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from sensor.entity.config_entity import ModelTrainerConfig
from sensor.exceptions import SensorException
from sensor.logger import logging
from sensor.ml.metrics.classification_metric import get_classification_score
from sensor.ml.model.estimator import SensorModel
from sensor.utils.main_utils import load_numpy_array_data, load_object, save_object


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as error:
            raise SensorException(error)

    def train_model(self, x_train, y_train):
        try:
            xg_boost_classifier = XGBClassifier()
            model = xg_boost_classifier.fit(x_train, y_train)
            return model
        except Exception as error:
            raise SensorException(error)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            # load train array and test array
            training_array_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            testing_array_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            training_array = load_numpy_array_data(file_path=training_array_file_path)
            testing_array = load_numpy_array_data(file_path=testing_array_file_path)

            x_train, y_train, x_test, y_test = (
                training_array[:, :-1],
                training_array[:, -1],
                testing_array[:, :-1],
                testing_array[:, -1],
            )
            model = self.train_model(x_train, y_train)

            y_train_pred = model.predict(x_train)
            train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            if train_metric.f1_score <= self.model_trainer_config.expected_accuracy:
                raise Exception("Model is not satisfying expected score.")

            y_test_pred = model.predict(x_test)
            test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # checking for overfitting/underfitting
            diff = abs(train_metric.f1_score - test_metric.f1_score)

            if diff >= self.model_trainer_config.trained_model_acceptance_criterion:
                raise Exception("Model is not good. Please redo the experimentation. .")

            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            sensor_model = SensorModel(preprocessor=preprocessor, model=model)

            # saving the sensor model
            model_dir_path = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )
            os.makedirs(model_dir_path, exist_ok=True)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=sensor_model,
            )

            # Trained model artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as error:
            raise SensorException(error)
