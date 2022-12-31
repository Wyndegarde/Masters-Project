from pymongo import MongoClient

# Connect to the MongoDB instance
client: MongoClient = MongoClient("mongodb://root:example@localhost:27017")

# # Get a reference to the "test" database
# db = client.test

# # Get a reference to the "mycollection" collection
# mycollection = db.mycollection

# # Insert a document into the "mycollection" collection
# mycollection.insert_one({"name": "John", "age": 30})

# # Find all documents in the "mycollection" collection
# documents = mycollection.find()

# # Print the documents
# for doc in documents:
#     print(doc)

db = client.history
mycollection = db.history

documents = mycollection.find()

for doc in documents:
    print(doc)
