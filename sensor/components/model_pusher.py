import os
import shutil

from sensor.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from sensor.entity.config_entity import ModelPusherConfig
from sensor.exceptions import SensorException


class ModelPusher:
    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_eval_artifact: ModelEvaluationArtifact,
    ):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_eval_artifact = model_eval_artifact
        except Exception as error:
            raise SensorException(error)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            trained_model_file_path = self.model_eval_artifact.trained_model_path

            # creating model pusher directory to save the model
            model_file_path = self.model_pusher_config.model_file_path
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
            shutil.copy(src=trained_model_file_path, dst=model_file_path)

            # saved model
            saved_model_path = self.model_pusher_config.saved_model_dir
            os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
            shutil.copy(src=trained_model_file_path, dst=saved_model_path)

            # preparing artifact
            model_pusher_artifact = ModelPusherArtifact(
                saved_model_path=saved_model_path, model_file_path=model_file_path
            )
            return model_pusher_artifact
        except Exception as error:
            raise SensorException(error)
