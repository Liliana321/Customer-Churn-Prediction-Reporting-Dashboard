from data_cleaning import preprocess_churn_data
import pandas as pd
from pymongo import MongoClient

def connect_to_mongodb(username, password):
    """
    Connette a MongoDB Atlas usando username e password.
    Ritorna l'oggetto client, database e collezione.
    """
    uri = f"mongodb+srv://{username}:{password}@churn.qxjbx4w.mongodb.net/?retryWrites=true&w=majority"
    # Connessione al cluster
    client = MongoClient(uri)
    db = client["churn"]
    collection = db["customers"]

    # Verifica se la collezione è vuota
    if collection.estimated_document_count() == 0:
        print("⚠️ Collezione vuota. Caricamento dati da CSV in corso...")
        df = df = preprocess_churn_data(pd.read_csv("Churn.csv"))
        collection.insert_many(df.to_dict("records"))
        print(f"✅ Inseriti {len(df)} documenti nella collezione 'customers'.")
    else:
        print("✅ Collezione già popolata.")

    return client, db, collection

username = input("Inserisci il tuo nome utente MongoDB Atlas: ")
password = input("Inserisci la tua password: ")

client, db, collection = connect_to_mongodb(username, password)

# Ora puoi eseguire query
print(collection.find_one())

# Aggregazione per contare i clienti per ciascun valore di Churn
pipeline = [
    {
        "$group": {
            "_id": "$Churn",
            "total": {"$sum": 1}
        }
    }
]

# Esecuzione dell'aggregazione
results = collection.aggregate(pipeline)

# Stampa dei risultati
for result in results:
    print(f"Churn: {result['_id']}, Totale clienti: {result['total']}")

