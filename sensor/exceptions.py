"""
Custom Exception class
"""
import sys


def error_message_detail(error, details: sys):
    _, _, execution_info = details.exc_info()

    file_name = execution_info.tb_frame.f_code.co_filename
    line_number = execution_info.tb_lineno

    error_message = (
        f"Error occurred python script name: [{file_name}] line "
        f"number: [{line_number}] error message ["
        f"{str(error)}]"
    )

    return error_message


class SensorException(Exception):
    def __init__(self, error_message, error_details: sys = sys):
        """
        :param error_message: error message in string format
        """
        super().__init__(error_message)

        self.error_message = error_message_detail(error_message, details=error_details)

    def __str__(self):
        return self.error_message
