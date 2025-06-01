import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox,
    QTextEdit, QComboBox, QScrollArea, QMainWindow
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from data_cleaning import preprocess_churn_data
from pymongo import MongoClient
import exploration as eda
import joblib

class GraphsWindow(QWidget):
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle("Grafici EDA")
        layout = QVBoxLayout(self)
        self.df = df

        grafici = [
            ("Churn by Demographics", eda.plot_churn_by_features),
            ("Churn by Tenure", eda.plot_churn_by_tenure),
            ("Churn by Services", eda.plot_churn_by_services),
            ("Churn by Internet Services", eda.plot_churn_by_internet_services),
            ("Churn by Contract and Payment", eda.plot_churn_by_contract_and_payment),
        ]

        for label, func in grafici:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, f=func: self.plot_graph(f))
            layout.addWidget(btn)

        self.canvas_layout = QVBoxLayout()
        layout.addLayout(self.canvas_layout)

    def plot_graph(self, func):
        plt.close('all')
        fig = func(self.df)
        for i in reversed(range(self.canvas_layout.count())):
            widget = self.canvas_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        canvas = FigureCanvas(fig)
        self.canvas_layout.addWidget(canvas)

class ChurnCleaner(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pulizia Dati Churn + MongoDB")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)

        self.cleaned_file_path = None
        self.cleaned_df = None

        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        browse_btn = QPushButton("Sfoglia")
        path_layout.addWidget(QLabel("File CSV:"))
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)

        self.clean_btn = QPushButton("Esegui Preprocessing")
        layout.addWidget(self.clean_btn)

        mongo_layout = QHBoxLayout()
        self.mongo_user = QLineEdit()
        self.mongo_user.setPlaceholderText("Username MongoDB Atlas")
        self.mongo_pass = QLineEdit()
        self.mongo_pass.setEchoMode(QLineEdit.Password)
        self.mongo_pass.setPlaceholderText("Password MongoDB Atlas")
        self.mongo_btn = QPushButton("Connetti a MongoDB")
        mongo_layout.addWidget(self.mongo_user)
        mongo_layout.addWidget(self.mongo_pass)
        mongo_layout.addWidget(self.mongo_btn)
        layout.addLayout(mongo_layout)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.graph_btn = QPushButton("\U0001F4CA Apri Grafici EDA in Nuova Finestra")
        layout.addWidget(self.graph_btn)
        self.graph_btn.clicked.connect(self.open_graph_window)

        self.predictor = ChurnPredictor()
        layout.addWidget(self.predictor)

        browse_btn.clicked.connect(self.browse_csv)
        self.clean_btn.clicked.connect(self.run_preprocessing)
        self.mongo_btn.clicked.connect(self.connect_to_mongodb)

    def browse_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleziona file CSV", ".", "CSV files (*.csv)")
        if file_path:
            self.path_edit.setText(file_path)

    def run_preprocessing(self):
        path = self.path_edit.text().strip()
        if not os.path.exists(path):
            QMessageBox.warning(self, "Errore", "File non trovato.")
            return

        try:
            self.log.append(f"📂 Caricamento da: {os.path.basename(path)}...")
            df = pd.read_csv(path)
            self.log.append("🔧 Pulizia in corso...")
            df_clean = preprocess_churn_data(df)

            out_path = path.replace(".csv", "_cleaned.csv")
            df_clean.to_csv(out_path, index=False)
            self.cleaned_file_path = out_path
            self.cleaned_df = df_clean

            self.log.append(f"✅ Preprocessing completato. File salvato in: {out_path}")
        except Exception as e:
            self.log.append(f"❌ Errore durante la pulizia: {str(e)}")

    def open_graph_window(self):
        if self.cleaned_df is not None:
            self.graph_window = GraphsWindow(self.cleaned_df)
            self.graph_window.show()
        else:
            QMessageBox.warning(self, "Errore", "Esegui prima il preprocessing dei dati.")

    def connect_to_mongodb(self):
        username = self.mongo_user.text().strip()
        password = self.mongo_pass.text().strip()

        if not username or not password:
            QMessageBox.warning(self, "Errore", "Inserisci username e password.")
            return

        try:
            self.log.append("🔗 Connessione a MongoDB in corso...")
            uri = f"mongodb+srv://{username}:{password}@churn.qxjbx4w.mongodb.net/?retryWrites=true&w=majority"
            client = MongoClient(uri)
            db = client["churn"]
            collection = db["customers"]

            if not self.cleaned_file_path or not os.path.exists(self.cleaned_file_path):
                raise Exception("File pulito non trovato. Esegui prima il preprocessing.")

            df = pd.read_csv(self.cleaned_file_path)
            df = preprocess_churn_data(df)

            if collection.estimated_document_count() == 0:
                self.log.append("⚠️ Collezione vuota. Caricamento dati in corso...")
                collection.insert_many(df.to_dict("records"))
                self.log.append(f"✅ Inseriti {len(df)} documenti.")
            else:
                self.log.append("✅ Collezione già popolata.")

            pipeline = [{"$group": {"_id": "$Churn", "total": {"$sum": 1}}}]
            results = collection.aggregate(pipeline)
            self.log.append("📊 Distribuzione Churn:")
            for r in results:
                self.log.append(f"  → Churn: {r['_id']} → Totale: {r['total']}")

        except Exception as e:
            self.log.append(f"❌ Errore MongoDB: {str(e)}")

class ChurnPredictor(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.inputs = {}

        fields = [
            ("gender", "Male/Female"),
            ("SeniorCitizen", "0/1"),
            ("Partner", "Yes/No"),
            ("Dependents", "Yes/No"),
            ("tenure", "es. 5"),
            ("PhoneService", "Yes/No"),
            ("MultipleLines", "Yes/No/No phone service"),
            ("InternetService", "No/DSL/Fiber optic"),
            ("OnlineSecurity", "Yes/No/No internet service"),
            ("OnlineBackup", "Yes/No/No internet service"),
            ("DeviceProtection", "Yes/No/No internet service"),
            ("TechSupport", "Yes/No/No internet service"),
            ("StreamingTV", "Yes/No/No internet service"),
            ("StreamingMovies", "Yes/No/No internet service"),
            ("Contract", "Month-to-month/One year/Two year"),
            ("PaperlessBilling", "Yes/No"),
            ("PaymentMethod", "Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic)"),
            ("MonthlyCharges", "float"),
            ("TotalCharges", "float")
        ]

        for field, placeholder in fields:
            box = QHBoxLayout()
            label = QLabel(field)

            if field == "gender":
                widget = QComboBox(); widget.addItems(["Male", "Female"])
            elif field == "SeniorCitizen":
                widget = QComboBox(); widget.addItems(["0", "1"])
            elif field in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
                widget = QComboBox(); widget.addItems(["Yes", "No"])
            elif field == "MultipleLines":
                widget = QComboBox(); widget.addItems(["No", "Yes", "No phone service"])
            elif field == "InternetService":
                widget = QComboBox(); widget.addItems(["No", "DSL", "Fiber optic"])
            elif field in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
                widget = QComboBox(); widget.addItems(["Yes", "No", "No internet service"])
            elif field == "Contract":
                widget = QComboBox(); widget.addItems(["Month-to-month", "One year", "Two year"])
            elif field == "PaymentMethod":
                widget = QComboBox(); widget.addItems([
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
            elif field in ["MonthlyCharges", "TotalCharges", "tenure"]:
                widget = QLineEdit(); widget.setPlaceholderText(placeholder)
            else:
                widget = QLineEdit(); widget.setPlaceholderText(placeholder)

            self.inputs[field] = widget
            box.addWidget(label)
            box.addWidget(widget)
            layout.addLayout(box)

        self.predict_btn = QPushButton("\U0001F52E Prevedi Churn")
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        layout.addWidget(self.predict_btn)
        layout.addWidget(self.result_box)
        self.predict_btn.clicked.connect(self.predict)

    def predict(self):
        try:
            data = {}
            for key, widget in self.inputs.items():
                val = widget.currentText() if isinstance(widget, QComboBox) else widget.text().strip()
                if key in ['MonthlyCharges', 'TotalCharges']:
                    val = float(val)
                elif key in ['tenure', 'SeniorCitizen']:
                    val = int(val)
                data[key] = val

            new_customer = pd.DataFrame([data])
            processed = preprocess_churn_data(new_customer)
            model = joblib.load("best_xgboost_churn_model.joblib")
            proba = model.predict_proba(processed)[0][1]
            label = int(proba >= 0.5)
            result = f"\U0001F4C8 Probabilità di Churn: {proba:.2%}\n\U0001F4CC Predizione: {'CHURN (1)' if label == 1 else 'NO CHURN (0)'}"
            self.result_box.setText(result)
            new_customer["Churn_Probability"] = round(proba, 4)
            new_customer["Churn_Prediction"] = label
            file_path = "predictions.csv"
            if os.path.exists(file_path):
                df_existing = pd.read_csv(file_path)
                df_updated = pd.concat([df_existing, new_customer], ignore_index=True)
            else:
                df_updated = new_customer
            df_updated.to_csv(file_path, index=False)

        except Exception as e:
            self.result_box.setText(f"\u274C Errore nella predizione:\n{str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChurnCleaner()
    window.resize(1000, 800)
    window.show()
    sys.exit(app.exec_())
