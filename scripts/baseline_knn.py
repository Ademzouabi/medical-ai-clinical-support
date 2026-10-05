import pandas as pd  # Import pandas to read CSV files and handle tabular data.
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
# Import the random forest model for multi-class classification.
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)  # Import metrics for evaluation.


# Load data  # Read the train, validation, and test datasets from CSV files.

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
train = pd.read_csv(DATA_DIR / "v1_train.csv")  # Load the training split into a DataFrame.
validation = pd.read_csv(
DATA_DIR / "v1_validation.csv"
)  # Load the validation split into a DataFrame.
test = pd.read_csv(DATA_DIR / "v1_test.csv")  # Load the test split into a DataFrame.

# Separate features and target  # Split each dataset into inputs (X) and output label (y).


X_train = train.drop(
    columns=["disease"]
)  # Keep all symptom columns as the training features.
y_train = train["disease"]  # Keep the disease label as the training target.


X_val = validation.drop(
    columns=["disease"]
)  # Keep all symptom columns as the validation features.
y_val = validation["disease"]  # Keep the disease label as the validation target.

X_test = test.drop(
    columns=["disease"]
)  # Keep all symptom columns as the test features.
y_test = test["disease"]  # Keep the disease label as the test target.

# Create model  # Initialize the logistic regression classifier.

model = KNeighborsClassifier(
    n_neighbors=5,
    metric="hamming"
)


# Train  # Fit the model on the training data.

model.fit(
    X_train, y_train
)  # Train the logistic regression classifier using the training features and labels.


# Validation evaluation  # Measure model performance on the validation set before the final test set.


val_predictions = model.predict(X_val)  # Predict disease labels for the validation set.
print("=== VALIDATION ===")  # Print a heading for the validation section.
print(
    "Accuracy:", round(accuracy_score(y_val, val_predictions), 4)
)  # Print the validation accuracy rounded to 4 decimals.

print(
    "\nClassification report:"
)  # Print a label before the validation classification report.
print(
    classification_report(y_val, val_predictions)
)  # Print precision, recall, F1-score, and support for validation predictions.


# Test evaluation  # Assess model performance on the final unseen test data.

test_predictions = model.predict(X_test)  # Predict disease labels for the test set.

print("\n=== TEST ===")  # Print a heading for the test section.
print(
    "Accuracy:", round(accuracy_score(y_test, test_predictions), 4)
)  # Print the test accuracy rounded to 4 decimals.

print(
    "\nClassification report:"
)  # Print a label before the test classification report.
print(
    classification_report(y_test, test_predictions)
)  # Print precision, recall, F1-score, and support for test predictions.


# Confusion matrix  # Show how often the model confused one disease with another.


print("\n=== CONFUSION MATRIX ===")  # Print a heading for the confusion matrix section.

cm = confusion_matrix(  # Compute the confusion matrix from true labels and predicted labels.
    y_test,  # True disease labels from the test set.
    test_predictions,  # Predicted disease labels from the test set.
    labels=model.classes_,  # Ensure the matrix rows and columns follow the model's class order.
)

cm_df = pd.DataFrame(  # Create a DataFrame version of the confusion matrix for readable output.
    cm,  # The numeric confusion matrix values.
    index=model.classes_,  # Set the row labels to the disease classes.
    columns=model.classes_,  # Set the column labels to the disease classes.
)

print(cm_df)  # Display the confusion matrix in a tabular format.

# --------------------------------------------------
# Top-K evaluation
# --------------------------------------------------

probabilities = model.predict_proba(X_test)

classes = model.classes_

def top_k_accuracy(y_true, probabilities, classes, k):
    correct = 0

    for true_label, probs in zip(y_true, probabilities):
        top_k_indices = probs.argsort()[-k:][::-1]
        top_k_labels = classes[top_k_indices]

        if true_label in top_k_labels:
            correct += 1

    return correct / len(y_true)


print("\n=== TOP-K TEST ACCURACY ===")

for k in [1, 3, 5]:
    score = top_k_accuracy(
        y_test,
        probabilities,
        classes,
        k
    )

    print(f"Top-{k}: {score:.4f}")
