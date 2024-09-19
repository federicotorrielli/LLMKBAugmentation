from pymongo import MongoClient

__USER = "gioandro91"
__PASSWORD = "wCpJPBCHIGzppKXG"
__DBNAME = "Semagram"

CONNECTION_STRING = f"mongodb+srv://{__USER}:{__PASSWORD}@csemagram.a4kafks.mongodb.net/{__DBNAME}?retryWrites=true&w=majority"

client = MongoClient(CONNECTION_STRING)

dbname = client["semagram"]
sem_collection = dbname["concepts"]
category_collection = dbname["categories"]
