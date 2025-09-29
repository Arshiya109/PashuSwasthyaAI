# ------------------ complete_ai_vet.py ------------------

import numpy as np
import pandas as pd
from rapidfuzz import process
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from model_train import load_dataset, preprocess_features, train_lightgbm, evaluate_models

# ------------------- Generative AI Setup -------------------
# Flan-T5 model for generating disease report
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def generate_report(disease_name, symptoms, lang="en"):
    prompt = f"""
    Disease: {disease_name}
    Symptoms: {', '.join(symptoms)}
    Generate a veterinary report with the following fields in {lang}:
    - Cause of Disease
    - Precautions
    - Care
    - Home Remedies
    - Treatment/Medicine
    - Expected Recovery Time (days)
    Provide concise info suitable for a farmer.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=400)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# ------------------- Translation Pipelines -------------------
translator_hi_en = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
translator_mr_en = pipeline("translation", model="Helsinki-NLP/opus-mt-mr-en")
translator_en_hi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
translator_en_mr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-mr")

def translate_text(text, lang="en"):
    if lang == "hi": return translator_en_hi(text)[0]['translation_text']
    elif lang == "mr": return translator_en_mr(text)[0]['translation_text']
    return text

# ------------------- Symptom Utilities -------------------
def normalize_symptom(symptom, all_symptoms):
    symptom = symptom.strip()
    match = process.extractOne(symptom, all_symptoms, score_cutoff=70)
    if match: return match[0]
    print(f"⚠️ Symptom '{symptom}' not recognized, using as-is.")
    return symptom

def get_disease_symptoms(disease_name, df):
    rows = df[df["Disease"] == disease_name]
    if rows.empty: return []
    return list(rows["Merged_Symptoms"].iloc[0])

# ------------------- Interactive Prediction -------------------
def interactive_prediction(user_symptoms, animal_type, age, age_group, Sub_Type, duration,
                           models, symptom_encoder, ohe, df, le, feature_names,
                           confidence_threshold=0.65, min_symptoms=6, user_lang="en"):

    # Gather all known symptoms
    all_symptoms = set()
    for col in df.columns:
        if "symptom" in col.lower():
            for cell in df[col].dropna():
                if isinstance(cell, list):
                    all_symptoms.update([str(s).strip().title() for s in cell if s])
                else:
                    all_symptoms.add(str(cell).strip().title())
    all_symptoms = list(all_symptoms)

    user_symptoms = [normalize_symptom(s, all_symptoms) for s in user_symptoms if s.strip()]
    asked_symptoms = set()

    while len(user_symptoms) < min_symptoms:
        print(f"⚠️ Only {len(user_symptoms)} symptom(s) provided. Need at least {min_symptoms}.")
        X_sym = pd.DataFrame(symptom_encoder.transform([user_symptoms]), columns=symptom_encoder.classes_)
        X_cat = pd.DataFrame(ohe.transform([[animal_type, age_group, Sub_Type]]), columns=ohe.get_feature_names_out())
        X_num = pd.DataFrame([[age, duration]], columns=["Age", "Duration (days)"])
        X_input = pd.concat([X_sym, X_cat, X_num], axis=1).reindex(columns=feature_names, fill_value=0)

        preds_proba = np.mean([m.predict_proba(X_input) for m in models], axis=0)
        top_indices = np.argsort(preds_proba[0])[::-1][:3]
        probable_diseases = le.inverse_transform(top_indices)

        asked = False
        for disease in probable_diseases:
            disease_name_only = disease.split("_")[0]
            common_symptoms = get_disease_symptoms(disease_name_only, df)
            for symptom in common_symptoms:
                if symptom not in user_symptoms and symptom not in asked_symptoms:
                    question = f"Does the animal also have '{symptom}'? (yes/no): "
                    question_translated = translate_text(question, user_lang)
                    ans = input(question_translated).strip().lower()
                    asked_symptoms.add(symptom)
                    if ans == "yes": user_symptoms.append(symptom)
                    asked = True
                    break
            if asked: break
        if not asked: break

    # Final prediction
    X_sym = pd.DataFrame(symptom_encoder.transform([user_symptoms]), columns=symptom_encoder.classes_)
    X_cat = pd.DataFrame(ohe.transform([[animal_type, age_group, Sub_Type]]), columns=ohe.get_feature_names_out())
    X_num = pd.DataFrame([[age, duration]], columns=["Age", "Duration (days)"])
    X_input = pd.concat([X_sym, X_cat, X_num], axis=1).reindex(columns=feature_names, fill_value=0)

    preds_proba = np.mean([m.predict_proba(X_input) for m in models], axis=0)
    top_idx = np.argmax(preds_proba[0])
    confidence = preds_proba[0][top_idx]
    predicted_stage_label = le.inverse_transform([top_idx])[0]

    if "_" in predicted_stage_label:
        predicted_disease, stage = predicted_stage_label.split("_")
    else:
        predicted_disease = predicted_stage_label
        stage = "Unknown"

    if confidence < confidence_threshold:
        print("⚠️ Confidence is low. Asking one more symptom...")
        for disease_label in le.classes_:
            disease_name_only = disease_label.split("_")[0]
            common_symptoms = get_disease_symptoms(disease_name_only, df)
            for symptom in common_symptoms:
                if symptom not in user_symptoms and symptom not in asked_symptoms:
                    question = f"Does the animal also have '{symptom}'? (yes/no): "
                    question_translated = translate_text(question, user_lang)
                    ans = input(question_translated).strip().lower()
                    asked_symptoms.add(symptom)
                    if ans == "yes":
                        user_symptoms.append(symptom)
                        return interactive_prediction(user_symptoms, animal_type, age, age_group, Sub_Type, duration,
                                                      models, symptom_encoder, ohe, df, le, feature_names,
                                                      confidence_threshold, min_symptoms, user_lang)
                    break

    return predicted_disease, stage, user_symptoms

import json
import re

def parse_report_to_json(report_text, predicted_disease, final_symptoms, critical=False):
    """
    Convert LLM text report into structured JSON.
    Uses simple regex-based extraction to fill fields.
    """
    def extract_field(field_name, text):
        # Try to find the line starting with field_name
        pattern = rf"{field_name}[:\-]\s*(.*)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    report_json = {
        "Animal Type": animal_type,
        "Sub-Type": Sub_Type,
        "Age": age,
        "Age Group": age_group,
        "Symptoms": final_symptoms,
        "Disease": predicted_disease,
        "Critical": critical,
        "Expected_Recovery_Time (days)": extract_field("Expected Recovery Time", report_text) or "",
        "Cause of Disease": extract_field("Cause", report_text) or "",
        "Precautions": extract_field("Precautions", report_text) or "",
        "Care": extract_field("Care", report_text) or "",
        "Home Remedies": extract_field("Home Remedies", report_text) or "",
        "Treatment/Medicine": extract_field("Treatment", report_text) or ""
    }

    return report_json


# ------------------- MAIN -------------------
if __name__ == "__main__":
    df = load_dataset("veterinary_dataset.json", file_type="json")
    le = LabelEncoder()
    df["Disease_Stage_Label"] = le.fit_transform(df["Disease_Stage"])
    X, symptom_encoder, ohe = preprocess_features(df)
    y = df["Disease_Stage_Label"]
    feature_names = X.columns.tolist()
    print("⏳ Training models...")
    models = train_lightgbm(X, y, n_splits=5)
    print("📊 Evaluating models...")
    results = evaluate_models(models, X, y)
    print("Final Evaluation Metrics:", results)

    print("\n--- Animal Details ---")
    animal_type = input("Enter animal type (Cattle/Goat): ").strip()
    age = float(input("Enter age (in years): ").strip())
    age_group = input("Enter age group (Young/Adult): ").strip()
    Sub_Type = input("Enter Sub-Type ([For Cattle: Cow/Buffalo/Bull/Calf/Bullock], [For Goat: Common]): ").strip()
    duration = int(input("Enter duration of illness (in days): ").strip())
    user_symptoms_input = input("Enter initial symptoms (comma separated): ")
    user_symptoms = [s.strip() for s in user_symptoms_input.split(",") if s.strip()]
    user_lang = input("Enter language (en/hi/mr): ").strip().lower() or "en"

    predicted_disease, stage, final_symptoms = interactive_prediction(
        user_symptoms, animal_type, age, age_group, Sub_Type, duration,
        models, symptom_encoder, ohe, df, le, feature_names,
        confidence_threshold=0.65, min_symptoms=6, user_lang=user_lang
    )

    print(f"\n✅ Final Predicted Disease: {predicted_disease}")
    print(f"🚨 Condition Stage: {stage}")
    
    # Generate text report
    report_text = generate_report(predicted_disease, final_symptoms, lang=user_lang)
    print("\n📄 Generated Disease Report (Text):")
    print(report_text)

    # Convert to structured JSON
    report_json = parse_report_to_json(report_text, predicted_disease, final_symptoms, critical=(stage.lower() == "critical"))
    print("\n📄 Structured JSON Report:")
    print(json.dumps(report_json, indent=4))
