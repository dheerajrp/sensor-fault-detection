import logging

import numpy as np
import pandas
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from sensor.constants.training_pipeline import TARGET_COLUMN
from sensor.entity.artifact_entity import (DataValidationArtifact,
                                           DataTransformationArtifact)
from sensor.entity.config_entity import DataTransformationConfig
from sensor.exceptions import SensorException
from sensor.ml.model.estimator import TargetValueMapping
from sensor.utils.main_utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as error:
            raise SensorException(error)

    @staticmethod
    def read_data(file_path: str) -> pandas.DataFrame:
        try:
            return pandas.read_csv(file_path)
        except Exception as error:
            raise SensorException(error)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        try:
            robust_scaler = RobustScaler()
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0)
            preprocessor = Pipeline(steps=[('Imputer', simple_imputer),
                                           ('RobustScaler', robust_scaler)])
            return preprocessor
        except Exception as error:
            raise SensorException(error)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path)
            preprocessor = self.get_data_transformer_object()

            # training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df.replace(TargetValueMapping().to_dict())

            # testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df.replace(TargetValueMapping().to_dict())

            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(
                input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(
                input_feature_test_df)

            smt = SMOTETomek(sampling_strategy='minority')
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df)
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                transformed_input_test_feature, target_feature_test_df)

            train_array = np.c_[input_feature_train_final,
                                np.array(target_feature_train_final)]
            test_array = np.c_[input_feature_test_final,
                               np.array(target_feature_test_final)]

            # saving array and preprocessing object
            save_numpy_array_data(
                file_path=self.data_transformation_config
                .data_transformed_train_file_path, array=train_array)
            save_numpy_array_data(
                file_path=self.data_transformation_config
                .data_transformed_test_file_path, array=test_array)

            save_object(file_path=self.data_transformation_config
                        .data_transformation_transformed_object_file_path,
                        obj=preprocessor)
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config
                .data_transformation_transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config
                .data_transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config
                .data_transformed_test_file_path
            )
            logging.info(f'Data Transformation artifact: {data_transformation_artifact}')
            return data_transformation_artifact
        except Exception as error:
            raise SensorException(error)
