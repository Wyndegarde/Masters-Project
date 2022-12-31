from pymongo import MongoClient

"""
Questions: 
    - How to create ID for history? Must haves: model type, training parameters, date and time
    - Should I have a seperate collection for the best performing model of each type? 
    - How best to retrieve each model?
"""


class DatabaseHandler:
    def __init__(self):
        self.client = MongoClient("mongodb://root:example@localhost:27017")

        self.history_db = self.client.history
        self.history_collection = self.history_db.history

    def insert_history(self, history: dict):
        # TODO: create ID for history
        self.history_collection.insert_one(history)

    def clear_history(self):
        self.history_collection.delete_many({})

    def retrieve_history(self):
        return self.history_collection.find()
