

# =========================================================
# ROAD TRAFFIC ACCIDENTS (RTA) - END-TO-END PIPELINE (EDA + ML + DEPLOYMENT)
# With Imbalance Handling (SMOTE + Class Weights)
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, classification_report)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.simplefilter(action='ignore', category=FutureWarning)

# =========================================================
# CONFIG
# =========================================================
CSV_PATH = r"C:\Smart_Accident_Risk_Classification\data_set\Project_dataset.csv"
OUTPUT_DIR = Path(r"C:\Smart_Accident_Risk_Classification\data_set")
EDA_FLAG = True   # Save EDA snapshot

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(CSV_PATH)
print("Loaded dataset with shape:", df.shape)

# =========================================================
# BASIC CLEANING
# =========================================================
df.columns = df.columns.str.strip()
before = df.shape[0]
df = df.drop_duplicates()
print(f"Dropped {before - df.shape[0]} duplicates")

# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Save cleaned snapshot
if EDA_FLAG:
    snapshot_path = OUTPUT_DIR / "EDA_Cleaned_Data.xlsx"
    df.to_excel(snapshot_path, index=False)
    print(f"Saved cleaned snapshot to: {snapshot_path}")

# =========================================================
# EDA VISUALIZATIONS
# =========================================================
if EDA_FLAG:
    print("\nGenerating EDA visualizations...")

    # 1. Target variable distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x="accident_occurred", data=df, palette="Set2")
    plt.title("Target Distribution (Accident Occurred: 0=No, 1=Yes)")
    target_plot_path = OUTPUT_DIR / "eda_target_distribution.png"
    plt.savefig(target_plot_path)
    plt.show()

    # 2. Numeric distributions
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_features:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        path = OUTPUT_DIR / f"eda_numeric_{col}.png"
        plt.savefig(path)
        plt.show()

    # 3. Categorical distributions
    categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    for col in categorical_features:
        plt.figure(figsize=(7,4))
        sns.countplot(x=col, data=df, order=df[col].value_counts().index, palette="Set3")
        plt.xticks(rotation=45)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        path = OUTPUT_DIR / f"eda_categorical_{col}.png"
        plt.savefig(path)
        plt.show()

    # 4. Correlation heatmap
    if len(numeric_features) > 1:
        plt.figure(figsize=(10,7))
        corr = df[numeric_features].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap (Numeric Features)")
        corr_path = OUTPUT_DIR / "eda_correlation_heatmap.png"
        plt.savefig(corr_path)
        plt.show()

# =========================================================
# FEATURES & TARGET
# =========================================================
TARGET = 'accident_occurred'
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# =========================================================
# PREPROCESSING
# =========================================================
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# =========================================================
# IMBALANCE HANDLING
# =========================================================
imbalance_ratio = y_train.value_counts(normalize=True).min()
print(f"Minority class ratio: {imbalance_ratio:.3f}")
use_smote = imbalance_ratio < 0.20
print("SMOTE enabled?", use_smote)

# =========================================================
# MODELS
# =========================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# =========================================================
# TRAINING & EVALUATION
# =========================================================
results = []

for name, model in models.items():
    print("-"*70)
    print(f"Training {name}")

    if use_smote:
        pipe = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('clf', model)
        ])
    else:
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', model)
        ])

    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"CV F1: {cv_f1:.4f}")
    print(f"Holdout Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    results.append({
        "Model": name,
        "CV_F1": cv_f1,
        "Test_Accuracy": acc,
        "Test_Precision": prec,
        "Test_Recall": rec,
        "Test_F1": f1,
        "Test_ROC_AUC": roc,
        "Pipeline": pipe
    })

# =========================================================
# SAVE METRICS
# =========================================================
metrics_df = pd.DataFrame(results).drop(columns=["Pipeline"])
metrics_path = OUTPUT_DIR / "rta_model_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"Saved metrics to: {metrics_path}")

plt.figure(figsize=(8,5))
sns.barplot(data=metrics_df, x="Model", y="Test_F1", palette="viridis")
plt.xticks(rotation=45)
plt.title("Model Comparison (Test F1)")
plt.tight_layout()
plot_path = OUTPUT_DIR / "rta_model_comparison_f1.png"
plt.savefig(plot_path)
plt.show()
print(f"Saved comparison plot to: {plot_path}")

# =========================================================
# SAVE BEST MODEL
# =========================================================
best_row = metrics_df.loc[metrics_df['Test_F1'].idxmax()]
best_model_name = best_row['Model']
best_pipeline = [r["Pipeline"] for r in results if r["Model"] == best_model_name][0]

model_path = OUTPUT_DIR / f"rta_best_pipeline_{best_model_name.replace(' ', '_')}.pkl"
with open(model_path, "wb") as f:
    pickle.dump(best_pipeline, f)

with open(OUTPUT_DIR / "rta_best_pipeline.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)

print(f" Best pipeline saved at: {model_path}")
print(f" Also saved as: {OUTPUT_DIR / 'rta_best_pipeline.pkl'}")

# =========================================================
# SAVE SCHEMA
# =========================================================
schema = {"numeric": numeric_features, "categorical": categorical_features, "target": TARGET}
schema_path = OUTPUT_DIR / "rta_feature_schema.json"
with open(schema_path, "w") as f:
    json.dump(schema, f, indent=4)
print(f"Saved schema to: {schema_path}")

# =========================================================
# SAVE PREPROCESSED TRAIN DATA
# =========================================================
preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)

if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()

processed_columns = numeric_features.copy()
cat_transformer = preprocessor.named_transformers_['cat'].named_steps['onehot']
for col, categories in zip(categorical_features, cat_transformer.categories_):
    processed_columns.extend([f"{col}_{cat}" for cat in categories])

df_train_processed = pd.DataFrame(X_train_processed, columns=processed_columns)
df_train_processed[TARGET] = y_train.values

preprocessed_path = OUTPUT_DIR / "RTA_Train_Preprocessed.xlsx"
df_train_processed.to_excel(preprocessed_path, index=False)
print(f"Saved preprocessed training data to: {preprocessed_path}")

# =========================================================
print("\nAll done  — artifacts ready for deployment:")
print(" - Best model pipeline:", OUTPUT_DIR / "rta_best_pipeline.pkl")
print(" - Metrics CSV:", metrics_path)
print(" - F1 plot:", plot_path)
print(" - Feature schema:", schema_path)
print(" - Cleaned snapshot:", snapshot_path if EDA_FLAG else "(disabled)")
print(" - Preprocessed train data:", preprocessed_path)
