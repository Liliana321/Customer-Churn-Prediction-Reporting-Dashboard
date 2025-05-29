# 📊 Customer Churn Prediction & Reporting Dashboard

Un progetto completo di machine learning per prevedere l'abbandono dei clienti (churn) in un'azienda di telecomunicazioni. Include fasi di pulizia dei dati, esplorazione visiva, modellazione predittiva con **XGBoost**, integrazione con **MongoDB Atlas** e salvataggio del modello migliore per uso futuro.

---

## 🧠 Obiettivi

- Prevedere la probabilità di abbandono di ciascun cliente.
- Visualizzare pattern comportamentali tramite EDA.
- Salvare i dati su MongoDB per accesso remoto e reporting.
- Valutare il modello con metriche robuste.
- Fornire una base solida per una dashboard analitica futura.

---

## 📁 Struttura del progetto

```
customer-churn-dashboard/
│
├── data/                          # CSV originali (es. Churn.csv) e dati eventualmente elaborati
│
├── models/                        # File dei modelli salvati (.joblib)
│   └── best_xgboost_churn_model.joblib
│
├── src/                           # Codice principale del progetto
│   ├── data_cleaning.py           # Funzione preprocess_churn_data(): codifica, imputazione, feature engineering
│   ├── data_exploration.py        # Grafici EDA: churn per genere, contratto, internet, tenure, ecc.
│   ├── mongo_integration.py       # Connessione a MongoDB Atlas, caricamento dati e aggregazioni
│   └── xgboost_churn.py           # Training, GridSearchCV, valutazione e salvataggio modello XGBoost
│
├── requirements.txt               # (opzionale) Librerie necessarie per eseguire il progetto
│
└── README.md                      # Documentazione del progetto
```

---

## 🚀 Come eseguire il progetto

1. **Clona il repository**:
   ```bash
   git clone https://github.com/tuo-username/customer-churn-dashboard.git
   cd customer-churn-dashboard
   ```

2. **Installa le dipendenze**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Esegui il preprocessing**:
   ```python
   from src.data_cleaning import preprocess_churn_data
   import pandas as pd
   df = pd.read_csv("data/Churn.csv")
   df = preprocess_churn_data(df)
   ```

4. **Visualizza le analisi esplorative**:
   ```python
   from src.data_exploration import plot_churn_by_features
   plot_churn_by_features(df)
   ```

5. **Connetti e carica su MongoDB**:
   ```bash
   python src/mongo_integration.py
   ```

6. **Addestra e valuta il modello**:
   ```bash
   python src/xgboost_churn.py
   ```

---

## 📈 Output del progetto

- Modello XGBoost salvato come `.joblib`
- Visualizzazione dei principali fattori predittivi
- Classificazione accurata di clienti a rischio
- Aggregazioni da database MongoDB

---

## 🧩 Stato attuale

🔧 **Il progetto è in fase di completamento.** Sono previste ulteriori estensioni con:
- Dashboard interattiva (es. Streamlit)
- Report PDFì

---

## 👩‍💻 Autore

Progetto realizzato da **Liliana Gilca**  
Dataset: [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
Università degli Studi di Roma "La Sapienza"  
Corso di **Data Analytics**, Anno Accademico 2025  
📧 Email: lilianagilca0@gmail.com

✉️ *Sono ben accetti consigli, suggerimenti o proposte di miglioramento. Sentiti libero/a di contattarmi!*


