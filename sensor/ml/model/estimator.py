import os

from sensor.constants.training_pipeline import (
    SAVED_MODEL_DIR,
    MODEL_TRAINER_TRAINED_MODEL_NAME,
)


class TargetValueMapping:
    def __init__(self):
        self.pos: int = 1
        self.neg: int = 0

    def to_dict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class SensorModel:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as error:
            raise error


class ModelResolver:
    def __init__(self, saved_model_dir: str = SAVED_MODEL_DIR):
        try:
            self.saved_model_dir = saved_model_dir
        except Exception as error:
            raise error

    def get_best_model_path(self) -> str:
        try:
            timestamps = list(map(int, os.listdir(self.saved_model_dir)))
            latest_timestamp = max(timestamps)
            latest_model_path = os.path.join(
                self.saved_model_dir,
                str(latest_timestamp),
                MODEL_TRAINER_TRAINED_MODEL_NAME,
            )
            return latest_model_path
        except Exception as error:
            raise error

    def is_model_exists(self) -> bool:
        try:
            if not os.path.exists(self.saved_model_dir):
                return False

            timestamps = os.listdir(self.saved_model_dir)
            if len(timestamps) == 0:
                return False

            latest_model_path = self.get_best_model_path()
            if not os.path.exists(latest_model_path):
                return False

            return True
        except Exception as error:
            raise error
