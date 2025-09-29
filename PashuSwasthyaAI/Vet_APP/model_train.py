import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier

# ------------------ 1. Load Dataset ------------------
def load_dataset(file_path, file_type="json"):
    if file_type == "json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data["records"])
    else:
        df = pd.read_csv(file_path)
    
    # Create Disease+Stage column
    df["Disease_Stage"] = df.apply(
        lambda row: f"{row['Disease']}_Critical" if row["Critical"] else f"{row['Disease']}_NonCritical",
        axis=1
    )
    
    return df

# ------------------ 2. Clean cell helper ------------------
def clean_cell(cell):
    if cell is None:
        return []
    if isinstance(cell, (list, np.ndarray)):
        return [str(s).strip() for s in cell if s is not None and str(s).strip() != ""]
    val = str(cell).strip()
    if val.lower() in ["", "nan", "none"]:
        return []
    return [val]

# ------------------ 3. Preprocess Features ------------------
def preprocess_features(df):
    # Symptom columns
    symptom_cols = [col for col in df.columns if "symptom" in col.lower() and "critical" not in col.lower()]
    if not symptom_cols:
        raise ValueError("No normal symptom columns found. Check dataset column names.")

    df["All_Symptoms"] = df[symptom_cols].apply(
        lambda row: sum([clean_cell(cell) for cell in row], []), axis=1
    )

    # Critical symptoms
    critical_cols = [col for col in df.columns if "critical" in col.lower() and "symptom" in col.lower()]
    if critical_cols:
        df["All_Critical_Symptoms"] = df[critical_cols].apply(
            lambda row: sum([clean_cell(cell) for cell in row], []), axis=1
        )
    else:
        df["All_Critical_Symptoms"] = [[] for _ in range(len(df))]

    # Merge all symptoms
    df["Merged_Symptoms"] = df["All_Symptoms"] + df["All_Critical_Symptoms"]

    # MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    X_symptoms = mlb.fit_transform(df["Merged_Symptoms"])
    symptom_features = pd.DataFrame(X_symptoms, columns=mlb.classes_)

    # Categorical features
    categorical_cols = [col for col in ["Animal Type", "Sub-Type", "Age Group"] if col in df.columns]
    if categorical_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat = ohe.fit_transform(df[categorical_cols])
        cat_features = pd.DataFrame(X_cat, columns=ohe.get_feature_names_out(categorical_cols))
    else:
        ohe = None
        cat_features = pd.DataFrame()

    # Numeric features
    num_features = df[[col for col in ["Age", "Duration (days)"] if col in df.columns]].reset_index(drop=True)

    # Critical flag
    critical_feature = df[["Critical"]].astype(int).reset_index(drop=True) if "Critical" in df.columns else pd.DataFrame([0]*len(df), columns=["Critical"])

    # Combine all features
    X = pd.concat([symptom_features, cat_features, num_features, critical_feature], axis=1)

    return X, mlb, ohe

# ------------------ 4. Train LightGBM ------------------
def train_lightgbm(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            class_weight="balanced",
            random_state=42
        )
        model.fit(X_train, y_train)
        models.append(model)

    return models

# ------------------ 5. Evaluate ------------------
def evaluate_models(models, X, y):
    skf = StratifiedKFold(n_splits=len(models), shuffle=True, random_state=42)
    accuracies, precisions, recalls, f1s = [], [], [], []

    for (train_idx, val_idx), model in zip(skf.split(X, y), models):
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        y_pred = model.predict(X_val)

        accuracies.append(accuracy_score(y_val, y_pred))
        precisions.append(precision_score(y_val, y_pred, average="weighted", zero_division=0))
        recalls.append(recall_score(y_val, y_pred, average="weighted", zero_division=0))
        f1s.append(f1_score(y_val, y_pred, average="weighted", zero_division=0))

        print("Classification Report (Fold):")
        print(classification_report(y_val, y_pred, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        print("-" * 50)

    results = {
        "Average Accuracy": np.mean(accuracies),
        "Average Precision": np.mean(precisions),
        "Average Recall": np.mean(recalls),
        "Average F1-Score": np.mean(f1s)
    }
    return results

# ------------------ 6. Main Training + Save ------------------
if __name__ == "__main__":
    df = load_dataset("veterinary_dataset.json")
    X, mlb, ohe = preprocess_features(df)
    y = df["Disease_Stage"]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train
    models = train_lightgbm(X, y_encoded, n_splits=5)
    results = evaluate_models(models, X, y_encoded)
    print("Overall Results:", results)

    # Save everything for inference
    feature_names = X.columns.tolist()
    joblib.dump((models, mlb, ohe, le, feature_names), "trained_model.pkl")
    print("✅ Model + encoders + feature names saved to trained_model.pkl")
