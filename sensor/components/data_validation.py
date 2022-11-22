import pandas

from sensor.constants.training_pipeline import SCHEMA_FILE_PATH
from sensor.entity.artifact_entity import DataValidationArtifact, \
    DataIngestionArtifact
from sensor.entity.config_entity import DataValidationConfig
from sensor.exceptions import SensorException
from sensor.logger import logging
from sensor.utils.main_utils import read_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as error:
            raise SensorException(error)

    def validate_number_of_columns(self, dataframe: pandas.DataFrame) -> bool:
        try:
            number_of_columns = self._schema_config['columns']
            if number_of_columns == len(dataframe.columns):
                return True
            return False
        except Exception as error:
            raise SensorException(error)

    def is_numerical_column_exists(self, dataframe: pandas.DataFrame) -> bool:
        try:
            numerical_columns = self._schema_config['numerical_columns']
            dataframe_columns = dataframe.columns

            num_column_status = True
            missing_numerical_columns = []
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    num_column_status = False
                    missing_numerical_columns.append(num_column)
            logging.info(f'Missing numerical columns: [{missing_numerical_columns}]')
            return num_column_status
        except Exception as error:
            raise SensorException(error)

    @staticmethod
    def read_data(file_path) -> pandas.DataFrame:
        try:
            logging.info('Reading data for validation. .')
            return pandas.read_csv(file_path)
        except Exception as error:
            raise SensorException(error)

    def detect_data_drift(self):
        try:
            pass
        except Exception as error:
            raise SensorException(error)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            error_message = ''
            logging.info('Initiating data validation. .')
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # reading data from the file location
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # validating columns
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message = f'{error_message} Train data doesn\'t contain'
                f'all the columns'
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message = f'{error_message} Test data doesn\'t contain'
                f'all the columns'

            # numerical column validation
            status = self.is_numerical_column_exists(train_dataframe)
            if not status:
                error_message = f'{error_message} Train data doesn\'t contain'
                f'all the numerical columns'

            status = self.is_numerical_column_exists(test_dataframe)
            if not status:
                error_message = f'{error_message} Test data doesn\'t contain'
                f'all the numerical columns'
            if len(error_message) > 0:
                raise Exception(error_message)


        except Exception as error:
            raise SensorException(error)
