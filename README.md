# 🚦 Road Traffic Accident (RTA) Prediction System

## 📌 Overview

This project is a **complete end-to-end Machine Learning pipeline** designed to predict the likelihood of road traffic accidents.

It includes:

* 📊 Exploratory Data Analysis (EDA)
* ⚖️ Imbalanced data handling using **SMOTE**
* 🤖 Multiple ML models comparison
* 🏆 Best model selection (Gradient Boosting)
* 💾 Model & pipeline saving
* 🌐 Streamlit deployment with interactive dashboard

---

## 🚀 Features

* Automatic data cleaning & preprocessing
* Handles missing values and duplicates
* Class imbalance handling (SMOTE + class weights)
* Multiple models:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Gradient Boosting ✅ (Best)
  * SVM
  * KNN
* Model comparison using F1-score
* Feature importance visualization
* Interactive Streamlit dashboard
* CSV/XLSX upload support
* Download predictions

---

## 📊 Model Performance

**Best Model:** Gradient Boosting

* Accuracy: **99.5%**
* Precision: **100%**
* Recall: **89%**
* F1 Score: **0.94**
* ROC-AUC: **0.98**

---

## 🏗️ Project Structure

```id="k0j1w5"
Smart_Accident_Risk_Classification/
│
├── main.py                         # Training pipeline (EDA + ML)
├── app.py                          # Streamlit deployment
│
├── data_set/
│   ├── Project_dataset.csv
│   ├── rta_best_pipeline.pkl
│   ├── rta_feature_schema.json
│   ├── rta_model_metrics.csv
│   ├── rta_model_comparison_f1.png
│   ├── EDA_Cleaned_Data.xlsx
│   └── RTA_Train_Preprocessed.xlsx
│
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone repository

```bash id="x2f91m"
git clone https://github.com/your-username/rta-prediction.git
cd rta-prediction
```

### 2️⃣ Create virtual environment

```bash id="8x8t5l"
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```bash id="1z7r4a"
pip install -r requirements.txt
```

---

## 📊 Run Training Pipeline

```bash id="jlfvsm"
python main.py
```

### This will:

* Perform EDA
* Train multiple models
* Select best model
* Save:

  * Model pipeline
  * Metrics
  * Visualizations
  * Feature schema

---

## 🌐 Run Streamlit App

```bash id="s8ql0z"
streamlit run app.py
```

Open in browser:

```id="r39h7q"
http://localhost:8501
```

---

## 📂 Input Format

Upload CSV/XLSX file with required features.
Missing columns are automatically handled.

---

## 📈 Dashboard Features

* Prediction summary (table + pie chart)
* Feature importance visualization
* Interactive histograms
* Data filtering by prediction
* Download results

---

## 🧠 ML Pipeline Details

* Preprocessing:

  * Median imputation (numeric)
  * Mode imputation (categorical)
  * Standard scaling
  * One-hot encoding
* Imbalance Handling:

  * SMOTE (activated when imbalance < 20%)
* Evaluation:

  * Cross-validation (F1-score)
  * Accuracy, Precision, Recall, ROC-AUC

---

## ⚠️ Important Notes

* Designed for highly imbalanced datasets
* Works best with structured traffic data
* Ensure correct feature names for predictions

---

## 👨‍💻 Author

**Munnam Balavamsi**

---

## ⭐ Future Improvements

* Real-time traffic data integration
* Deep learning models
* API deployment (FastAPI)
* Cloud deployment (AWS / Azure)

---

## 📜 License

MIT License
