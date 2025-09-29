import numpy as np
import pandas as pd
from rapidfuzz import process  # fast fuzzy matching

# ------------------- Symptom Utilities -------------------
def normalize_symptom(symptom, all_symptoms):
    """Normalize user symptom to closest match from dataset symptoms using fuzzy matching."""
    symptom = symptom.strip()
    match = process.extractOne(symptom, all_symptoms, score_cutoff=70)  # 70% similarity threshold
    if match:
        return match[0]
    else:
        # If completely unknown, just use as-is
        print(f"⚠️ Symptom '{symptom}' not recognized, using as-is.")
        return symptom

def get_disease_symptoms(disease_name, df):
    """Return merged symptoms for a disease."""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    rows = df[df["Disease"] == disease_name]
    if rows.empty:
        return []
    return list(rows["Merged_Symptoms"].iloc[0])

# ------------------- Interactive Prediction -------------------
def interactive_prediction(user_symptoms, animal_type, age, age_group, Sub_Type, duration,
                           models, symptom_encoder, ohe, df, le, feature_names,
                           confidence_threshold=0.65, min_symptoms=6):

    # 1️⃣ Gather all known symptoms from dataset (handles lists)
    all_symptoms = set()
    for col in df.columns:
        if "symptom" in col.lower():
            for cell in df[col].dropna():
                if isinstance(cell, list):
                    all_symptoms.update([str(s).strip().title() for s in cell if s])
                else:
                    all_symptoms.add(str(cell).strip().title())
    all_symptoms = list(all_symptoms)

    # 2️⃣ Normalize user symptoms
    user_symptoms = [normalize_symptom(s, all_symptoms) for s in user_symptoms if s.strip()]
    asked_symptoms = set()

    # 3️⃣ Counter-questioning loop
    while len(user_symptoms) < min_symptoms:
        print(f"⚠️ Only {len(user_symptoms)} symptom(s) provided. Need at least {min_symptoms}.")

        # Encode features
        X_sym = pd.DataFrame(symptom_encoder.transform([user_symptoms]), columns=symptom_encoder.classes_)
        X_cat = pd.DataFrame(ohe.transform([[animal_type, age_group, Sub_Type]]), columns=ohe.get_feature_names_out())
        X_num = pd.DataFrame([[age, duration]], columns=["Age", "Duration (days)"])
        X_input = pd.concat([X_sym, X_cat, X_num], axis=1)
        X_input = X_input.reindex(columns=feature_names, fill_value=0)

        # Ensemble prediction
        preds_proba = np.mean([m.predict_proba(X_input) for m in models], axis=0)
        top_indices = np.argsort(preds_proba[0])[::-1][:3]
        probable_diseases = le.inverse_transform(top_indices)

        # Ask additional symptoms for most probable disease
        asked = False
        for disease in probable_diseases:
            disease_name_only = disease.split("_")[0]  # remove stage
            common_symptoms = get_disease_symptoms(disease_name_only, df)
            for symptom in common_symptoms:
                if symptom not in user_symptoms and symptom not in asked_symptoms:
                    ans = input(f"Does the animal also have '{symptom}'? (yes/no): ").strip().lower()
                    asked_symptoms.add(symptom)
                    if ans == "yes":
                        user_symptoms.append(symptom)
                    asked = True
                    break
            if asked:
                break
        if not asked:
            break

    # 4️⃣ Final prediction
    X_sym = pd.DataFrame(symptom_encoder.transform([user_symptoms]), columns=symptom_encoder.classes_)
    X_cat = pd.DataFrame(ohe.transform([[animal_type, age_group, Sub_Type]]), columns=ohe.get_feature_names_out())
    X_num = pd.DataFrame([[age, duration]], columns=["Age", "Duration (days)"])
    X_input = pd.concat([X_sym, X_cat, X_num], axis=1)
    X_input = X_input.reindex(columns=feature_names, fill_value=0)

    preds_proba = np.mean([m.predict_proba(X_input) for m in models], axis=0)
    top_idx = np.argmax(preds_proba[0])
    confidence = preds_proba[0][top_idx]
    predicted_stage_label = le.inverse_transform([top_idx])[0]

    # Extract disease and stage separately
    if "_" in predicted_stage_label:
        predicted_disease, stage = predicted_stage_label.split("_")
    else:
        predicted_disease = predicted_stage_label
        stage = "Unknown"

    # 5️⃣ Low-confidence extra question
    if confidence < confidence_threshold:
        print("⚠️ Confidence is low. Asking one more symptom...")
        for disease_label in le.classes_:
            disease_name_only = disease_label.split("_")[0]
            common_symptoms = get_disease_symptoms(disease_name_only, df)
            for symptom in common_symptoms:
                if symptom not in user_symptoms and symptom not in asked_symptoms:
                    ans = input(f"Does the animal also have '{symptom}'? (yes/no): ").strip().lower()
                    asked_symptoms.add(symptom)
                    if ans == "yes":
                        user_symptoms.append(symptom)
                        # Recursively re-predict with new symptom
                        return interactive_prediction(user_symptoms, animal_type, age, age_group, Sub_Type, duration,
                                                      models, symptom_encoder, ohe, df, le, feature_names,
                                                      confidence_threshold, min_symptoms)
                    break

    return predicted_disease, stage
