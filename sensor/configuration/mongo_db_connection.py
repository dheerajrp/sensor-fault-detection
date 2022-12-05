import os

import certifi
import pymongo
from dotenv import load_dotenv

from sensor.constants.database import DATABASE_NAME
from sensor.exceptions import SensorException

ca = certifi.where()
load_dotenv()


class MongoDBClient:
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv("MONGODB_URL")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as error:
            raise SensorException(error)
