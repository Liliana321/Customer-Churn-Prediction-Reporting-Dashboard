
import pandas as pd

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

    # MultipleLines
    df['MultipleLines'] = df['MultipleLines'].map({'No phone service': 0, 'No': 0, 'Yes': 1})

    # InternetService
    df['InternetService'] = df['InternetService'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})

    # Internet-related services
    internet_features = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in internet_features:
        df[col] = df[col].map({'No internet service': 0, 'No': 0, 'Yes': 1})

    # Contract
    df['Contract'] = df['Contract'].map({
        'Month-to-month': 0,
        'One year': 1,
        'Two year': 2
    })

    # PaperlessBilling
    df['PaperlessBilling'] = df['PaperlessBilling'].map({'No': 0, 'Yes': 1})

    # PaymentMethod
    df['PaymentMethod'] = df['PaymentMethod'].map({
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    })

    # TotalCharges conversione
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # tenure_group
    df['tenure_group'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 72], labels=['0–12', '13–24', '25–48', '49–72'])
    df['tenure_group'] = df['tenure_group'].map({
        '0–12': 0, '13–24': 1, '25–48': 2, '49–72': 3
    }).astype(int)

    # Imputazione speciale TotalCharges
    df.loc[(df['tenure'] == 0) & (df['TotalCharges'].isna()), 'TotalCharges'] = 0

    # Codifica Churn SOLO se presente
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    return df



