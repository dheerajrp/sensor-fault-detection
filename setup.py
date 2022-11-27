"""
To create a library
"""
from typing import List

from setuptools import find_packages, setup


def get_requirements() -> List[str]:
    """
    This function will return list of requirements.

    Returns:
        List of requirements.
    """
    with open("requirements.txt", "r") as file:
        requirements_list = file.readlines()
        file.close()
    return requirements_list


setup(
    name="sensor-fault-detection",
    version="0.0.1",
    author="Dheeraj RP",
    author_email="dheerajrp66@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)
