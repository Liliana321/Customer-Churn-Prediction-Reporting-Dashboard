### importazione librerie 

import pandas as pd
## data cleaning


def preprocess_churn_data(df):
    """
    Preprocessing per il dataset Churn: codifica variabili categoriche, conversione tipi, gestione valori speciali.
    """

    # GENDER --> Male: 0, Female: 1
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

    # PARTNER e DEPENDENTS --> No: 0, Yes: 1
    df['Partner'] = df['Partner'].map({'No': 0, 'Yes': 1})
    df['Dependents'] = df['Dependents'].map({'No': 0, 'Yes': 1})

    
    # PhoneService --> No: 0, Yes: 1
    df['PhoneService'] = df['PhoneService'].map({'No': 0, 'Yes': 1})

    # MultipleLines --> 'No phone service' e 'No': 0, 'Yes': 1
    df['MultipleLines'] = df['MultipleLines'].map({'No phone service': 0, 'No': 0, 'Yes': 1})

    # InternetService --> 'No': 0, 'DSL': 1, 'Fiber optic': 2
    df['InternetService'] = df['InternetService'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})

    # Servizi Internet (OnlineSecurity, OnlineBackup, ecc.) --> 'No internet service' e 'No': 0, 'Yes': 1
    internet_features = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in internet_features:
        df[col] = df[col].map({'No internet service': 0, 'No': 0, 'Yes': 1})

    # Contract --> 'Month-to-month': 0, 'One year': 1, 'Two year': 2
    df['Contract'] = df['Contract'].map({
        'Month-to-month': 0,
        'One year': 1,
        'Two year': 2
    })

    # PaperlessBilling --> No: 0, Yes: 1
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'No': 0, 'Yes': 1})

    # PaymentMethod --> codifica a 4 valori distinti
    df['PaymentMethod'] = df['PaymentMethod'].map({
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    })

    # TotalCharges --> conversione da object a float64, eventuali errori diventano NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # tenure anzianità del cliente in mesi divisi in gruppi/fasce
    # - '0–12' diventa 0
    # - '13–24' diventa 1
    # - '25–48' diventa 2
    # - '49–72' diventa 3
    df['tenure_group'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 72], labels=['0–12', '13–24', '25–48', '49–72'])
    df['tenure_group'] = df['tenure_group'].map({
    '0–12': 0,
    '13–24': 1,
    '25–48': 2,
    '49–72': 3}).astype(int)
    
    # Imputa con 0 solo se tenure == 0
    df.loc[(df['tenure'] == 0) & (df['TotalCharges'].isna()), 'TotalCharges'] = 0
    
    # Churn --> No: 0, Yes: 1
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    return df


