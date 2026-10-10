"""Compare selected tuned classifiers against the V1 validation data."""

import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. Load data
# ============================================================

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
train = pd.read_csv(DATA_DIR / "v1_train.csv")
validation = pd.read_csv(DATA_DIR / "v1_validation.csv")

X_train = train.drop(columns=["disease"])
y_train = train["disease"]

X_val = validation.drop(columns=["disease"])
y_val = validation["disease"]


# ============================================================
# 2. Top-K accuracy function
# ============================================================

def top_k_accuracy(y_true, probabilities, classes, k):
    correct = 0

    for true_label, probs in zip(y_true, probabilities):
        top_k_indices = probs.argsort()[-k:][::-1]
        top_k_labels = classes[top_k_indices]

        if true_label in top_k_labels:
            correct += 1

    return correct / len(y_true)


# ============================================================
# 3. Define tuned models
# ============================================================

logistic = LogisticRegression(
    C=8,
    penalty="l2",
    solver="lbfgs",
    class_weight="balanced",
    max_iter=5000,
    random_state=42
)

random_forest = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight=None,
    random_state=42,
    n_jobs=-1
)


models = {
    "Logistic Regression": logistic,
    "Random Forest": random_forest
}


# ============================================================
# 4. Train and evaluate
# ============================================================

results = {}

for name, model in models.items():

    print("\n" + "=" * 80)
    print(f"TRAINING: {name}")
    print("=" * 80)

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)

    top1 = top_k_accuracy(
        y_val,
        probabilities,
        model.classes_,
        1
    )

    top3 = top_k_accuracy(
        y_val,
        probabilities,
        model.classes_,
        3
    )

    top5 = top_k_accuracy(
        y_val,
        probabilities,
        model.classes_,
        5
    )

    accuracy = accuracy_score(y_val, predictions)

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro"
    )

    results[name] = {
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Top-1": top1,
        "Top-3": top3,
        "Top-5": top5
    }

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    print("\nOverall performance:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")
    print(f"Top-1    : {top1:.4f}")
    print(f"Top-3    : {top3:.4f}")
    print(f"Top-5    : {top5:.4f}")

    # --------------------------------------------------------
    # Per-disease report
    # --------------------------------------------------------

    print("\nPer-disease classification report:")
    print(
        classification_report(
            y_val,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    labels = sorted(y_val.unique())

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    print("\nConfusion matrix:")
    print(cm_df)


# ============================================================
# 5. Final comparison
# ============================================================

comparison = pd.DataFrame(results).T

print("\n\n")
print("=" * 100)
print("FINAL VALIDATION COMPARISON")
print("=" * 100)

print(
    comparison.to_string(
        formatters={
            "Accuracy": "{:.4f}".format,
            "Macro F1": "{:.4f}".format,
            "Top-1": "{:.4f}".format,
            "Top-3": "{:.4f}".format,
            "Top-5": "{:.4f}".format
        }
    )
)


# ============================================================
# 6. Determine which model leads each metric
# ============================================================

print("\n")
print("=" * 100)
print("METRIC LEADERS")
print("=" * 100)

for metric in comparison.columns:

    best_model = comparison[metric].idxmax()
    best_value = comparison.loc[best_model, metric]

    print(
        f"{metric:10s}: "
        f"{best_model} ({best_value:.4f})"
    )
