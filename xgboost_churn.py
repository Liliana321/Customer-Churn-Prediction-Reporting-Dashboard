import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from data_cleaning import preprocess_churn_data
import joblib
# Caricamento e preprocessing
df = pd.read_csv("Churn.csv")
df = preprocess_churn_data(df)

# Feature set e target
X = df.drop(columns=["customerID", "Churn"], errors='ignore')
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Peso per classi sbilanciate
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Modello base
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)

# Griglia iperparametri
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01],
    'n_estimators': [100, 300, 500],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1],
    'reg_lambda': [1, 3, 5]
}

# GridSearchCV
grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

# Valutazione
best_model = grid.best_estimator_
print("\n✅ Migliori parametri trovati:")
print(grid.best_params_)

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

print("📈 ROC AUC Score:", roc_auc_score(y_test, y_proba))

print("🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Metriche principali
print("\n🔍 METRICHE DI VALUTAZIONE:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Feature importance
xgb.plot_importance(
    best_model,
    importance_type='gain',
    max_num_features=10,
    height=0.5,
    title='Top 10 Feature Importance (Gain)',
    xlabel='Gain',
    ylabel='Feature'
)
plt.tight_layout()
plt.show()



# Plot importanza delle feature (in base al GAIN: quanto migliora ogni split)
xgb.plot_importance(best_model, 
                    importance_type='gain',  # puoi anche usare 'weight' o 'cover'
                    max_num_features=10,     # mostra solo le 10 più importanti
                    height=0.5, 
                    title='Top 10 Feature Importance (Gain)',
                    xlabel='Gain',
                    ylabel='Feature')
plt.tight_layout()
plt.show()

# Salvataggio  modello
joblib.dump(best_model, "best_xgboost_churn_model.joblib")
print("\n Modello salvato in: best_xgboost_churn_model.joblib")