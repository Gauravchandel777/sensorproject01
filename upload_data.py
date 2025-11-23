from pymongo.mongo_client import MongoClient
import pandas as pd
import json
# my URL
uri="mongodb+srv://gauravchandel2005_db_user:Rq3V3kyi2wKdEBjP@cluster0.mmzjxqi.mongodb.net/?appName=Cluster0"


#create anew client and connect
client=MongoClient(uri)

#crfeate data base name
DATABASE_NAME="PROJECT"
COLLECTION_NAME="WAFER_FAULT"

df=pd.read_csv("C:\Users\asus\Downloads\sensorproject\notebooks\wafer_23012020_041211.csv")


json_record=list(json.loads(df.T.to_json()).values())

type(json_record)

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)