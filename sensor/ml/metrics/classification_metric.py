from sklearn.metrics import f1_score, recall_score, precision_score

from sensor.entity.artifact_entity import ClassificationMetricArtifact
from sensor.exceptions import SensorException


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)
        cls_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            recall_score=model_recall_score,
            precision_score=model_precision_score,
        )
        return cls_metric
    except Exception as error:
        raise SensorException(error)
