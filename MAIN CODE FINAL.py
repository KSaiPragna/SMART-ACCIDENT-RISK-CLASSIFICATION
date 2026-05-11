
# ========================================================================================
         # === SMART ACCIDENT RISK CLASSIFICATION USING TRAFFIC AND SIGNAL DATA​ === #
# ========================================================================================


# =========================================================
# IMPORT LIBRARIES
# =========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import warnings
from pathlib import Path
import math


from sklearn.model_selection import train_test_split
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

warnings.simplefilter(action='ignore', category=FutureWarning)

# =========================================================
# CONFIG
# =========================================================

CSV_PATH = Path(r"C:\Users\MyPc\OneDrive\Desktop\LIVE PROJECT\PROJECT DATASET.csv")
OUTPUT_DIR = Path(r"C:\Users\MyPc\OneDrive\Desktop\LIVE PROJECT\CODES\RTA_OUTPUT_FILES")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # <-- ensure the directory exists

EDA_FLAG = True

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(CSV_PATH)
print("Loaded dataset with shape:", df.shape)
print("\nColumns and Data Types:\n", df.dtypes)


print(df.head())       # first 5 rows
print(df.info())       # column types & non-null counts
print(df.describe())   # summary statistics for numeric columns


# =========================================================
# BASIC CLEANING
# =========================================================
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# =========================================================
# EDA (Display + Save)
# =========================================================
if EDA_FLAG:
    # --- Categorical Features ---
    categorical_cols = df.select_dtypes(include='object').columns
    n_cols = 3
    n_rows = math.ceil(len(categorical_cols)/n_cols)
    plt.figure(figsize=(n_cols*6, n_rows*4))
    for idx, col in enumerate(categorical_cols, 1):
        plt.subplot(n_rows, n_cols, idx)
        sns.countplot(data=df, x=col, hue=col, palette="viridis",
                  order=df[col].value_counts().index, legend=False)
    plt.title(f"{col}", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.suptitle("UNIVARIATE ANALYSIS - CATEGORICAL FEATURES", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # --- Numerical Features ---
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cols = 3
    rows = math.ceil(len(num_cols)/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f"{col}")
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
        plt.suptitle("NUMERICAL FEATURE DISTRIBUTIONS", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

# --- Correlation ---
    plt.figure(figsize=(15, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (Red-Blue Diverging)")
    plt.show()

# --- Pairplot (selected columns) ---
    pair_cols = ['avg_speed_kmph','vehicle_count_per_hr','blackspot_score','accident_occurred']
    if all(x in df.columns for x in pair_cols):
        sns.pairplot(df[pair_cols], hue='accident_occurred')
        plt.show()

# --- Outlier analysis ---
    plt.figure(figsize=(15, 8))
    for i, col in enumerate(num_cols, 1):
        plt.subplot(2, (len(num_cols)+1)//2, i)
        sns.boxplot(x=df[col], color="skyblue")
        plt.title(f"{col}", fontsize=10)
    plt.suptitle("Outlier Analysis (Boxplots of Numerical Features)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# =========================================================
# 4. BUSINESS DECISION MOMENTS (BDMs)
# =========================================================
print("\n" + "="*80)
print("BUSINESS DECISION MOMENTS (BDMs) - ROAD TRAFFIC ACCIDENTS")
print("="*80 + "\n")

# BDM 1: Accident Hotspots
bdm1 = df.groupby("location_id")["accident_occurred"].sum().sort_values(ascending=False).head(10)
print("BDM 1: Top 10 Accident-Prone Locations")
print(bdm1)
print("\nDecision: Install cameras & enforce stricter traffic rules.\n" + "-"*80)

# BDM 2: Peak Hours & Days
bdm2 = df.groupby(["day_of_week","hour_of_day"])["accident_occurred"].sum().unstack()
print("BDM 2: Accidents by Day and Hour (Peak Times)")
print(bdm2)
print("\nDecision: Deploy more traffic police during peak times.\n" + "-"*80)

# BDM 3: Risk Factors
bdm3 = df.groupby(["weather","lighting","road_type"])["accident_occurred"].sum().sort_values(ascending=False).head(10)
print("BDM 3: Top Risk Factors for Accidents")
print(bdm3)
print("\nDecision: Improve lighting, provide warning signs, redesign risky roads/junctions.\n" + "-"*80)

# BDM 4: Severity & Vehicles Involved
bdm4 = df.groupby("severity")["vehicles_involved"].mean().sort_values(ascending=False)
print("BDM 4: Severity vs Average Number of Vehicles Involved")
print(bdm4)
print("\nDecision: Hospitals & insurance companies can prepare resources.\n" + "="*80)




# ================================================================= #

# ----------------------------------------------------------------
#   SQL CONNECTION
# ----------------------------------------------------------------

import pandas as pd
from sqlalchemy import create_engine
import urllib.parse

# Load CSV
csv_path = r"C:\Users\MyPc\OneDrive\Desktop\LIVE PROJECT\PROJECT DATASET.csv"
df = pd.read_csv(csv_path)


from sqlalchemy import create_engine
import urllib.parse

# MySQL credentials
username = 'root'
password = urllib.parse.quote_plus('root@sai')  # encode special characters
host = 'localhost'
port = 3306
database = 'PROJECT'

# Create SQLAlchemy engine
engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')

# Push dataframe to SQL table
df.to_sql('accident_data', con=engine, if_exists='replace', index=False)
print("DataFrame successfully written to MySQL!")

# ================================================================= #


print(f"\n✅ EDA completed and saved to: {OUTPUT_DIR}\n")

# =========================================================
# FEATURES & TARGET
# =========================================================
TARGET = 'accident_occurred'
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================================================
# PREPROCESSING
# =========================================================
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Preprocess train/test
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# =========================================================
# SMOTE
# =========================================================
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_proc, y_train)

print("After SMOTE, class distribution:")
print(pd.Series(y_train_res).value_counts())

# =========================================================
# MODELS
# =========================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}


# =========================================================
# TRAIN & EVALUATE
# =========================================================
results = []

for name, model in models.items():
    print("-"*70)
    print(f"Training {name}")
    
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)[:,1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": roc,
        "Trained_Model": model
    })

# =========================================================
# SAVE METRICS
# =========================================================


metrics_df = pd.DataFrame(results).drop(columns=["Trained_Model"])


# Round values for better readability
metrics_df_rounded = metrics_df.copy()
metrics_df_rounded[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]] = metrics_df_rounded[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]].round(4)

print("\n========== MODEL PERFORMANCE COMPARISON ==========")
print(metrics_df_rounded)


metrics_path = OUTPUT_DIR / "rta_model_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"Saved metrics to: {metrics_path}")


# =========================================================
# MODEL COMPARISON TABLE & PLOTS
# =========================================================


# ---- Save comparison table ----
comparison_path = OUTPUT_DIR / "rta_model_comparison.csv"
metrics_df_rounded.to_csv(comparison_path, index=False)
print(f"✅ Comparison table saved at: {comparison_path}")


# Plot F1-score comparison
sns.set_style("whitegrid")
try:
    plt.figure(figsize=(12,6))
    sns.barplot(data=metrics_df_rounded, x="F1", y="Model", palette="viridis")
    plt.title("Model Comparison (F1 Score)")
    plt.xlabel("F1-Score")
    plt.ylabel("Model")
    plt.tight_layout()
    f1_plot_path = OUTPUT_DIR / "rta_model_comparison_f1.png"
    plt.savefig(f1_plot_path, bbox_inches="tight")  # Save first
    plt.show()  # Display (optional)
    plt.close()  # Close figure

    print(f"✅ Saved F1 score plot: {f1_plot_path}")
except Exception as e:
    print("F1 score plotting failed:", e)
    plt.show()

# Heatmap of all metrics
try:
    plt.figure(figsize=(8,5))
    sns.heatmap(metrics_df_rounded.set_index("Model"), annot=True, fmt=".3f", cmap="RdBu")
    plt.title("Model Performance Heatmap")
    plt.tight_layout()
    heatmap_path = OUTPUT_DIR / "rta_model_comparison_heatmap.png"
    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"✅ Saved heatmap: {heatmap_path}")
except Exception as e:
    print("Heatmap plotting failed:", e)


# =========================================================
# SAVE BEST MODEL
# =========================================================

best_row = metrics_df.loc[metrics_df['F1'].idxmax()]
best_model_name = best_row['Model']
best_model = [r["Trained_Model"] for r in results if r["Model"] == best_model_name][0]

best_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', best_model)
])

with open(OUTPUT_DIR / "rta_best_pipeline.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)

print(f"✅ Best pipeline saved at: {OUTPUT_DIR / 'rta_best_pipeline.pkl'}")

# =========================================================
# SAVE SCHEMA
# =========================================================
schema = {"numeric": numeric_features, "categorical": categorical_features, "target": TARGET}
with open(OUTPUT_DIR / "rta_feature_schema.json", "w") as f:
    json.dump(schema, f, indent=4)
print(f"Saved schema to: {OUTPUT_DIR / 'rta_feature_schema.json'}")