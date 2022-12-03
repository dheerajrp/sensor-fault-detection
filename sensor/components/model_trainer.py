from xgboost import XGBClassifier

from sensor.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from sensor.entity.config_entity import ModelTrainerConfig
from sensor.exceptions import SensorException
from sensor.ml.metrics.classification_metric import get_classification_score
from sensor.utils.main_utils import load_numpy_array_data


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

            y_test_pred = model.predict(x_test)
            test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        except Exception as error:
            raise SensorException(error)
