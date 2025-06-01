import joblib
import pandas as pd
import os
from data_cleaning import preprocess_churn_data

def predict_churn_from_input():
    print("Inserisci le informazioni del cliente per la previsione di churn:\n")

    # Input utente
    gender = input("Genere (Male/Female): ")
    SeniorCitizen = int(input("Senior Citizen? (0 = No, 1 = Yes): "))
    Partner = input("Ha un partner? (Yes/No): ")
    Dependents = input("Ha persone a carico? (Yes/No): ")
    tenure = int(input("Anzianità in mesi (es. 5): "))
    PhoneService = input("Ha servizio telefonico? (Yes/No): ")
    MultipleLines = input("Ha linee multiple? (No / Yes / No phone service): ")
    InternetService = input("Tipo di connessione internet (No / DSL / Fiber optic): ")
    OnlineSecurity = input("Online Security attivo? (Yes/No/No internet service): ")
    OnlineBackup = input("Online Backup attivo? (Yes/No/No internet service): ")
    DeviceProtection = input("Device Protection attivo? (Yes/No/No internet service): ")
    TechSupport = input("Supporto tecnico attivo? (Yes/No/No internet service): ")
    StreamingTV = input("Streaming TV attivo? (Yes/No/No internet service): ")
    StreamingMovies = input("Streaming Movies attivo? (Yes/No/No internet service): ")
    Contract = input("Tipo di contratto (Month-to-month / One year / Two year): ")
    PaperlessBilling = input("Fatturazione elettronica? (Yes/No): ")
    PaymentMethod = input("Metodo di pagamento (Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic)): ")
    MonthlyCharges = float(input("Addebiti mensili (€): "))
    TotalCharges = float(input("Addebiti totali (€): "))

    # Costruzione DataFrame singolo cliente
    new_customer = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }])

    # Applica il preprocessing
    new_customer_processed = preprocess_churn_data(new_customer)

    # Caricamento modello
    model = joblib.load("best_xgboost_churn_model.joblib")

    # Predizione
    proba = model.predict_proba(new_customer_processed)[0][1]
    label = int(proba >= 0.5)

    # Aggiunta dei risultati
    new_customer["Churn_Probability"] = round(proba, 4)
    new_customer["Churn_Prediction"] = label

    # Salva o aggiorna il file cumulativo
    file_path = "predictions.csv"
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_updated = pd.concat([df_existing, new_customer], ignore_index=True)
    else:
        df_updated = new_customer

    df_updated.to_csv(file_path, index=False)

    # Output
    print("\n Probabilità di Churn: {:.2%}".format(proba))
    print("Predizione finale:", "CHURN (1)" if label == 1 else "NO CHURN (0)")
    print(f"Cliente aggiunto a: {file_path}")

    return label, proba

predict_churn_from_input()