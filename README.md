# 📊 Customer Churn Prediction & Reporting Dashboard

Questo progetto Python offre una soluzione completa per l'analisi, la gestione e la previsione del "churn" (abbandono) dei clienti. Integra fasi di pulizia dei dati, analisi esplorativa, interazione con un database MongoDB, addestramento di un modello di Machine Learning (XGBoost) e un'interfaccia utente grafica (GUI) per facilitare l'interazione.

---

## 🧠 Obiettivi

- Prevedere la probabilità di abbandono di ciascun cliente.
- Visualizzare pattern comportamentali tramite EDA.
- Salvare i dati su MongoDB per accesso remoto e reporting.
- Valutare il modello con metriche robuste.
- Fornire una base solida per una dashboard analitica futura.

---

## 📈 Funzionalità Principali
- **Pulizia e Preprocessing Dati** : Trasformazione di variabili categoriche, gestione dei valori mancanti e conversione dei tipi di dati per il dataset di churn.

- **Analisi Esplorativa dei Dati (EDA)**: Generazione di visualizzazioni per comprendere i fattori che influenzano il churn dei clienti.

- **Integrazione MongoDB Atlas**: Connessione a un cluster MongoDB Atlas per il caricamento e la gestione dei dati pre-processati.

- **Addestramento Modello XGBoost**: Costruzione, ottimizzazione degli iperparametri (tramite GridSearchCV) e valutazione di un modello predittivo di churn basato su XGBoost.

- **Interfaccia Utente Grafica (GUI)**: Un'applicazione desktop basata su PyQt5 che unifica tutte le funzionalità, rendendo il progetto accessibile e facile da usare.
  
## 📁 Struttura del progetto

```
customer-churn-dashboard/
│
├── data/                 # CSV originali (es. Churn.csv) e dati eventualmente elaborati
│
├── models/               # File dei modelli salvati (.joblib)
│   └── best_xgboost_churn_model.joblib
│
├── src/                  # Codice principale del progetto
│   ├── data_cleaning.py  # Funzione preprocess_churn_data(): codifica, imputazione, feature engineering
│   ├── data_exploration.py  # Grafici EDA: churn per genere, contratto, internet, tenure, ecc.
│   ├── mongo_integration.py  # Connessione a MongoDB Atlas, caricamento dati e aggregazioni
│   ├── xgboost_churn.py  # Training, GridSearchCV, valutazione e salvataggio modello XGBoost
│   └── churn_app.py      # Applicazione GUI principale
│
├── requirements.txt      # (opzionale) Librerie necessarie per eseguire il progetto
│
└── README.md             # Documentazione del progetto
```

---

## 🚀 Come eseguire il progetto

Per far partire il progetto, segui questi semplici passaggi. Ti serviranno Python (versione 3.7 o superiore consigliata) e pip.

1. **Clona il repository**:
   ```bash
   git clone https://github.com/tuo-username/customer-churn-dashboard.git
   cd customer-churn-dashboard
   ```

2. **Installa le dipendenze**:
   Installiamo tutte le librerie Python necessarie. Esegui questo comando nel terminale:
   
   ```bash
   pip install pandas scikit-learn xgboost matplotlib seaborn pymongo PyQt5 joblib
   ```

4. **Esegui il preprocessing**:
   
   ```python
   from src.data_cleaning import preprocess_churn_data
   import pandas as pd
   df = pd.read_csv("data/Churn.csv")
   df = preprocess_churn_data(df)
   ```

6. **Visualizza le analisi esplorative**:
   
   ```python
   from src.exploration import plot_churn_by_features
   plot_churn_by_features(df)
   ```

8. **Connetti e carica su MongoDB**:
   ```bash
   python src/mongo_integration.py
   ```

9. **Addestra e valuta il modello**:
   ```bash
   python src/xgboost_churn.py
   ```
10. **Avvia l'applicazione GUI (Consigliato)**:
    Il modo più semplice per usare il progetto è tramite l'interfaccia grafica. Da qui potrai gestire la pulizia dei dati, l'esplorazione, la connessione a MongoDB e le previsioni.
     ```bash
     python src/churn_app.py
   

---

## 📈 Output del progetto

Durante l'esecuzione del progetto, verranno generati o aggiornati i seguenti file nella rispettiva struttura delle cartelle:

- `data/Churn_cleaned.csv`: Questo file conterrà il dataset originale (Churn.csv) dopo aver applicato il processo di pulizia e preprocessing dei dati.
- `models/best_xgboost_churn_model.joblib`: Qui verrà salvato il modello XGBoost addestrato e ottimizzato (tramite GridSearchCV). Questo file è essenziale per poter caricare e utilizzare il modello per future previsioni senza doverlo addestrare nuovamente.
- `predictions.csv`: Un file CSV che registrerà tutte le previsioni di churn effettuate tramite l'interfaccia a riga di comando (`predict_churn.py`) o l'applicazione GUI (`churn_app.py`). Ogni riga aggiunta conterrà i dati del cliente e la relativa probabilità e predizione di churn.

---

## 🧩 Contribuzione
Sentiti libero di contribuire al progetto aprendo issue o pull request.

---

## 👩‍💻 Autore

Progetto realizzato da **Liliana Gilca**  
Dataset: [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
Università degli Studi di Roma "La Sapienza"  
Corso di **Data Analytics**, Anno Accademico 2025  
📧 Email: lilianagilca0@gmail.com

✉️ *Sono ben accetti consigli, suggerimenti o proposte di miglioramento. Sentiti libero/a di contattarmi!*


