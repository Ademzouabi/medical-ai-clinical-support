import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. Load datasets
# ============================================================

train = pd.read_csv("v1_train.csv")
validation = pd.read_csv("v1_validation.csv")
test = pd.read_csv("v1_test.csv")


# ============================================================
# 2. Combine TRAIN + VALIDATION
# ============================================================

train_validation = pd.concat(
    [train, validation],
    ignore_index=True
)


X_train = train_validation.drop(columns=["disease"])
y_train = train_validation["disease"]

X_test = test.drop(columns=["disease"])
y_test = test["disease"]


print("=" * 80)
print("FINAL MODEL EVALUATION")
print("=" * 80)

print(f"Training rows: {len(X_train)}")
print(f"Test rows:     {len(X_test)}")
print(f"Features:      {X_train.shape[1]}")
print()


# ============================================================
# 3. Top-K accuracy function
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
# 4. Create FINAL locked Logistic Regression
# ============================================================

model = LogisticRegression(
    C=8,
    class_weight="balanced",
    solver="lbfgs",
    max_iter=5000,
    random_state=42
)


# ============================================================
# 5. Train on TRAIN + VALIDATION
# ============================================================

print("Training final Logistic Regression...")

model.fit(X_train, y_train)

print("Training complete.")
print()


# ============================================================
# 6. Predictions on untouched TEST set
# ============================================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)


# ============================================================
# 7. Overall metrics
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro"
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)

top1 = top_k_accuracy(
    y_test,
    probabilities,
    model.classes_,
    1
)

top3 = top_k_accuracy(
    y_test,
    probabilities,
    model.classes_,
    3
)

top5 = top_k_accuracy(
    y_test,
    probabilities,
    model.classes_,
    5
)


# ============================================================
# 8. Print final results
# ============================================================

print("=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)

print(f"Accuracy : {accuracy:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Top-1    : {top1:.4f}")
print(f"Top-3    : {top3:.4f}")
print(f"Top-5    : {top5:.4f}")


# ============================================================
# 9. Per-disease classification report
# ============================================================

print("\n")
print("=" * 80)
print("PER-DISEASE CLASSIFICATION REPORT")
print("=" * 80)

print(
    classification_report(
        y_test,
        predictions,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 10. Confusion matrix
# ============================================================

labels = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\n")
print("=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(cm_df)


# ============================================================
# 11. Save the final model
# ============================================================

model_filename = "final_logistic_regression.pkl"

joblib.dump(
    model,
    model_filename
)

print("\n")
print("=" * 80)
print("MODEL SAVED")
print("=" * 80)

print(f"Saved as: {model_filename}")


# ============================================================
# 12. Save test predictions for later analysis
# ============================================================

test_results = test.copy()

test_results["predicted_disease"] = predictions

# Store the highest model score as well
test_results["prediction_confidence"] = probabilities.max(axis=1)

test_results.to_csv(
    "final_test_predictions.csv",
    index=False
)

print("Test predictions saved as: final_test_predictions.csv")