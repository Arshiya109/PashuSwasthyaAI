from model_train import load_dataset, preprocess_features, train_lightgbm, evaluate_models
from inter_prediction import interactive_prediction
from sklearn.preprocessing import LabelEncoder
import json

if __name__ == "__main__":
    # 1️. Load dataset
    df = load_dataset("veterinary_dataset.json", file_type="json")

    # 2️. Encode Disease + Stage labels
    le = LabelEncoder()
    df["Disease_Stage_Label"] = le.fit_transform(df["Disease_Stage"])

    # 3️. Preprocess features
    X, symptom_encoder, ohe = preprocess_features(df)
    y = df["Disease_Stage_Label"]

    # 4️. Save feature names for prediction alignment
    feature_names = X.columns.tolist()

    # 5️. Train models
    print("⏳ Training models...")
    models = train_lightgbm(X, y, n_splits=5)

    # 6️. Evaluate models
    print("📊 Evaluating models...")
    results = evaluate_models(models, X, y)
    print("Final Evaluation Metrics:", results)

    # 7️. User input
    print("\n--- Animal Details ---")
    animal_type = input("Enter animal type (Cattle/Goat): ").strip()
    age = float(input("Enter age (in years): ").strip())
    age_group = input("Enter age group (Young/Adult): ").strip()
    Sub_Type = input("Enter Sub-Type ([For Cattle: Cow/Buffalo/Bull/Calf/Bullock], [For Goat: Common]): ").strip()
    duration = int(input("Enter duration of illness (in days): ").strip())
    user_symptoms = [s.strip() for s in input("Enter initial symptoms (comma separated): ").split(",") if s.strip()]

    # 8️. Predict interactively
    predicted_disease, stage = interactive_prediction(
        user_symptoms, animal_type, age, age_group, Sub_Type, duration,
        models, symptom_encoder, ohe, df, le, feature_names,
        confidence_threshold=0.65
    )

    # 9️. Show final prediction
    print("\n✅ Final Predicted Disease:", predicted_disease)
    print("🚨 Condition Stage:", stage)

    # -------- Report Generation ------------ #

    # 10. Fetch disease details directly from dataset
    try:
        disease_data = df[df["Disease"] == predicted_disease].iloc[0][
            [
                "Expected_Recovery_Time (days)",
                "Cause of Disease",
                "Precautions",
                "Care",
                "Home Remedies",
                "Treatment/Medicine"
            ]
        ].to_dict()
    except IndexError:
        disease_data = {
            "Expected_Recovery_Time (days)": "Unknown",
            "Cause of Disease": "Unknown",
            "Precautions": "Unknown",
            "Care": "Unknown",
            "Home Remedies": "Unknown",
            "Treatment/Medicine": "Unknown"
        }

    # 11. Build JSON report
    report = {
        "Disease": predicted_disease,
        "Stage": stage,
        "Expected_Recovery_Time (days)": disease_data["Expected_Recovery_Time (days)"],
        "Cause of Disease": disease_data["Cause of Disease"],
        "Precautions": disease_data["Precautions"],
        "Care": disease_data["Care"],
        "Home Remedies": disease_data["Home Remedies"],
        "Treatment/Medicine": disease_data["Treatment/Medicine"]
    }

    # 12️. Print and Save
    import json
    print("\nGenerated Report:")
    print(json.dumps(report, indent=4, ensure_ascii=False))

    with open("disease_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
