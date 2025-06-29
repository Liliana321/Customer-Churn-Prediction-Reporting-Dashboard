import matplotlib.pyplot as plt
import seaborn as sns

def plot_churn_by_features(df):
    features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
    titles = ['Gender (0=Male, 1=Female)', 'Senior Citizen (0=No, 1=Yes)',
              'Partner (0=No, 1=Yes)', 'Dependents (0=No, 1=Yes)']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        sns.countplot(x=df[feature].astype(str), hue=df['Churn'], ax=axes[i], palette='pastel')
        axes[i].set_title(f'Churn by {titles[i]}', fontsize=12)
        axes[i].legend(title='Churn', labels=['No (0)', 'Yes (1)'])
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Count')

    plt.tight_layout()
    return fig

def plot_churn_by_tenure(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(data=df, x='Churn', y='tenure', palette='pastel', ax=axes[0])
    axes[0].set_title('Boxplot of Tenure by Churn', fontsize=14)
    axes[0].set_xlabel('Churn (0 = No, 1 = Yes)')
    axes[0].set_ylabel('Tenure (months)')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['No (0)', 'Yes (1)'])

    sns.countplot(x=df['tenure_group'].astype(str), hue=df['Churn'], palette='pastel', ax=axes[1])
    axes[1].set_title('Churn by Tenure Group', fontsize=14)
    axes[1].set_xlabel('Tenure Group')
    axes[1].set_ylabel('Number of Customers')
    axes[1].legend(title='Churn', labels=['No (0)', 'Yes (1)'])

    plt.tight_layout()
    return fig

def plot_churn_by_services(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.countplot(x=df['PhoneService'].astype(str), hue=df['Churn'], palette='pastel', ax=axes[0])
    axes[0].set_title('Churn by Phone Service')
    axes[0].set_xlabel('Phone Service (0 = No, 1 = Yes)')
    axes[0].set_ylabel('Number of Customers')
    axes[0].legend(title='Churn', labels=['No (0)', 'Yes (1)'])

    sns.countplot(x=df['MultipleLines'].astype(str), hue=df['Churn'], palette='pastel', ax=axes[1])
    axes[1].set_title('Churn by Multiple Lines')
    axes[1].set_xlabel('Multiple Lines (0 = No/No service, 1 = Yes)')
    axes[1].set_ylabel('')

    sns.countplot(x=df['InternetService'].astype(str), hue=df['Churn'], palette='pastel', ax=axes[2])
    axes[2].set_title('Churn by Internet Service')
    axes[2].set_xlabel('Internet Service (0 = No, 1 = DSL, 2 = Fiber optic)')
    axes[2].set_ylabel('')

    plt.tight_layout()
    return fig

def plot_churn_by_internet_services(df):
    internet_features = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, feature in enumerate(internet_features):
        sns.countplot(x=df[feature].astype(str), hue=df['Churn'], palette='pastel', ax=axes[i])
        axes[i].set_title(f'Churn by {feature}')
        axes[i].set_xlabel(f'{feature} (0 = No/No internet service, 1 = Yes)')
        axes[i].set_ylabel('Customers')
        axes[i].legend(title='Churn', labels=['No (0)', 'Yes (1)'])

    plt.tight_layout()
    return fig

def plot_churn_by_contract_and_payment(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.countplot(x=df['Contract'].astype(str), hue=df['Churn'], palette='pastel', ax=axes[0])
    axes[0].set_title('Churn by Contract Type')
    axes[0].set_xlabel('Contract (0 = Monthly, 1 = 1 Year, 2 = 2 Years)')
    axes[0].set_ylabel('Customers')
    axes[0].legend(title='Churn', labels=['No (0)', 'Yes (1)'])

    sns.countplot(x=df['PaymentMethod'].astype(str), hue=df['Churn'], palette='pastel', ax=axes[1])
    axes[1].set_title('Churn by Payment Method')
    axes[1].set_xlabel('Payment Method (0=Elec. Check, 1=Mailed, 2=Bank Transf., 3=Credit Card)')
    axes[1].set_ylabel('')
    axes[1].legend(title='Churn', labels=['No (0)', 'Yes (1)'])

    plt.tight_layout()
    return fig
