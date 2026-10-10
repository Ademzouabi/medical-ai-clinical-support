"""Load the locked V1 model and return ranked predictions for 26 binary features."""

import joblib
import pandas as pd
from pathlib import Path


# ============================================================
# 1. Load trained model
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
# Keeping this path relative to source makes model loading work outside the project directory.
MODEL_PATH = PROJECT_DIR / "models" / "final_logistic_regression.pkl"

model = joblib.load(MODEL_PATH)


# ============================================================
# 2. Canonical V1 features
# ============================================================

FEATURES = [
    # This ordering is part of the trained-model contract, not merely display order.
    "shortness_of_breath",
    "difficulty_breathing",
    "pain_with_breathing",
    "cough",
    "productive_cough",
    "wheezing",
    "chest_tightness",
    "chest_congestion",
    "hemoptysis",
    "apnea",
    "hoarseness",
    "sore_throat",
    "nasal_congestion",
    "sinus_congestion",
    "coryza",
    "sharp_chest_pain",
    "palpitations",
    "irregular_heartbeat",
    "increased_heart_rate",
    "dizziness",
    "fever",
    "chills",
    "fatigue",
    "malaise",
    "general_weakness",
    "vomiting",
]


# ============================================================
# 3. Prediction function
# ============================================================


def predict_differential(features):
    """
    Predict a ranked differential from a complete
    26-feature binary clinical vector.

    Each feature must be:
        0 = absent
        1 = present
    """

    # Check that all features exist
    # Presence validation prevents pandas from silently constructing an incomplete model row.
    missing_features = [feature for feature in FEATURES if feature not in features]

    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Check values
    # `None`, booleans, strings, and other values must not enter the binary model.
    invalid_features = {
        feature: features[feature]
        for feature in FEATURES
        if features[feature] not in [0, 1]
    }

    if invalid_features:
        raise ValueError(
            f"All features must currently be 0 or 1. Invalid values: {invalid_features}"
        )

    # Create dataframe in EXACT training feature order
    # One row represents one clinical case, with columns aligned to training order.
    X = pd.DataFrame([[features[feature] for feature in FEATURES]], columns=FEATURES)

    # Predict probabilities
    probabilities = model.predict_proba(X)[0]

    classes = model.classes_

    # Sort from highest to lowest probability
    ranked_indices = probabilities.argsort()[::-1]

    # Convert NumPy/scikit-learn values into JSON-friendly Python dictionaries.
    results = []

    for index in ranked_indices:
        results.append(
            {"disease": classes[index], "score": float(probabilities[index])}
        )

    return results


# ============================================================
# 4. Test case
# ============================================================

if __name__ == "__main__":
    # This manual example is separate from the production pipeline and bypasses text matching.
    test_case = {
        "shortness_of_breath": 1,
        "difficulty_breathing": 1,
        "pain_with_breathing": 0,
        "cough": 1,
        "productive_cough": 0,
        "wheezing": 1,
        "chest_tightness": 1,
        "chest_congestion": 0,
        "hemoptysis": 0,
        "apnea": 0,
        "hoarseness": 0,
        "sore_throat": 0,
        "nasal_congestion": 0,
        "sinus_congestion": 0,
        "coryza": 0,
        "sharp_chest_pain": 0,
        "palpitations": 0,
        "irregular_heartbeat": 0,
        "increased_heart_rate": 0,
        "dizziness": 0,
        "fever": 0,
        "chills": 0,
        "fatigue": 0,
        "malaise": 0,
        "general_weakness": 0,
        "vomiting": 0,
    }

    results = predict_differential(test_case)

    print("=" * 60)
    print("RANKED DIFFERENTIAL")
    print("=" * 60)

    for rank, result in enumerate(results[:5], start=1):
        print(f"{rank}. {result['disease']} ({result['score']:.4f})")
