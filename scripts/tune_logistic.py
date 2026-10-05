import pandas as pd  # Import pandas to read CSV data and store evaluation results in a DataFrame.
from pathlib import Path
from sklearn.linear_model import (
    LogisticRegression,
)  # Import the logistic regression model used for tuning.
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)  # Import metrics to compare model quality across parameter settings.


# --------------------------------------------------
# Load data
# --------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
train = pd.read_csv(DATA_DIR / "v1_train.csv")  # Read the training split from the CSV file.
validation = pd.read_csv(
    DATA_DIR / "v1_validation.csv"
)  # Read the validation split from the CSV file.

X_train = train.drop(
    columns=["disease"]
)  # Use all symptom columns as the training input features.
y_train = train["disease"]  # Use the disease label as the training target.

X_val = validation.drop(
    columns=["disease"]
)  # Use all symptom columns as validation features.
y_val = validation["disease"]  # Use the disease label as the validation target.


# --------------------------------------------------
# Hyperparameters to test
# --------------------------------------------------

C_values = [
    0.01,
    0.1,
    0.3,
    1,
    3,
    10,
    30,
    100,
]  # Try a range of regularization strengths to find the best logistic model.

results = []  # Create an empty list to store evaluation results for each parameter setting.


# --------------------------------------------------
# Helper: Top-K accuracy
# --------------------------------------------------


def top_k_accuracy(
    y_true, probabilities, classes, k
):  # Measure whether the true label appears in the top-k predicted diseases.
    correct = 0  # Start a counter for the number of correct top-k predictions.

    for true_label, probs in zip(
        y_true, probabilities
    ):  # Loop through each true label and its predicted probabilities.
        top_k_indices = probs.argsort()[-k:][
            ::-1
        ]  # Select the highest-probability labels for the current row.
        top_k_labels = classes[top_k_indices]  # Map indices back to disease names.

        if (
            true_label in top_k_labels
        ):  # Count this as a correct top-k prediction if the true label is included.
            correct += 1  # Increase the count when the true disease is in the top-k predictions.

    return correct / len(
        y_true
    )  # Return the fraction of rows where the true label is in the top-k list.


# --------------------------------------------------
# Test each C value
# --------------------------------------------------

for C in C_values:  # Iterate over each regularization strength to evaluate the model.
    model = LogisticRegression(  # Create a logistic regression model with the current C value.
        C=C,  # Set the inverse-regularization strength.
        max_iter=2000,  # Allow enough iterations for convergence.
        random_state=42,  # Keep the result reproducible.
    )

    model.fit(X_train, y_train)  # Fit the model to the training data.

    predictions = model.predict(X_val)  # Predict disease labels for the validation set.
    probabilities = model.predict_proba(
        X_val
    )  # Get class probabilities to compute top-k metrics.

    accuracy = accuracy_score(
        y_val, predictions
    )  # Compute standard validation accuracy.

    macro_f1 = f1_score(  # Compute macro F1 across all disease classes.
        y_val, predictions, average="macro"
    )

    top_1 = top_k_accuracy(  # Measure top-1 accuracy using model probabilities.
        y_val, probabilities, model.classes_, 1
    )

    top_3 = top_k_accuracy(  # Measure top-3 accuracy.
        y_val, probabilities, model.classes_, 3
    )

    top_5 = top_k_accuracy(  # Measure top-5 accuracy.
        y_val, probabilities, model.classes_, 5
    )

    results.append(
        {  # Save the metrics for the current C value.
            "C": C,
            "Accuracy": accuracy,
            "Macro_F1": macro_f1,
            "Top_1": top_1,
            "Top_3": top_3,
            "Top_5": top_5,
        }
    )


# --------------------------------------------------
# Display results
# --------------------------------------------------

results_df = pd.DataFrame(results)  # Convert the tuning results list into a DataFrame.

print("\nLogistic Regression tuning results")  # Print a heading for the tuning summary.
print("=" * 80)  # Print a separator line.

print(  # Print the tuning table with formatted metric values.
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Macro_F1": "{:.4f}".format,
            "Top_1": "{:.4f}".format,
            "Top_3": "{:.4f}".format,
            "Top_5": "{:.4f}".format,
        },
    )
)


# --------------------------------------------------
# Best configurations
# --------------------------------------------------

print("\nBest by Top-1 accuracy:")  # Print the best C value based on top-1 accuracy.
print(results_df.loc[results_df["Top_1"].idxmax()])

print("\nBest by Macro F1:")  # Print the best C value based on macro F1.
print(results_df.loc[results_df["Macro_F1"].idxmax()])

print("\nBest by Top-3:")  # Print the best C value based on top-3 accuracy.
print(results_df.loc[results_df["Top_3"].idxmax()])
