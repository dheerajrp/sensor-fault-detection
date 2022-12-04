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
