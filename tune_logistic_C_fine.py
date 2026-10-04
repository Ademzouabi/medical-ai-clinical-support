import pandas as pd  # Import pandas to load the split data and summarize the tuning results.
from sklearn.linear_model import (
    LogisticRegression,
)  # Import logistic regression for the C-value grid search.
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)  # Import accuracy and macro F1 for comparison.


# Load data  # Read the training and validation sets from their CSV files.
train = pd.read_csv("v1_train.csv")  # Load the training set.
validation = pd.read_csv("v1_validation.csv")  # Load the validation set.

X_train = train.drop(
    columns=["disease"]
)  # Use all symptom columns as training features.
y_train = train["disease"]  # Use disease as the target variable.

X_val = validation.drop(
    columns=["disease"]
)  # Use all symptom columns as validation features.
y_val = validation["disease"]  # Use disease as the validation target.


# Fine-grained C values around the promising region  # Search a tighter range of regularization values around likely good settings.
C_values = [
    1,
    1.5,
    2,
    2.5,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    12,
    15,
    20,
]  # Candidate C values to test.


def top_k_accuracy(
    y_true, probabilities, classes, k
):  # Compute whether the true class is in the top-k predictions.
    correct = 0  # Reset the count for each evaluation.

    for true_label, probs in zip(
        y_true, probabilities
    ):  # Loop through each row's true label and model probabilities.
        top_k_indices = probs.argsort()[-k:][
            ::-1
        ]  # Select the k highest-probability classes.
        top_k_labels = classes[
            top_k_indices
        ]  # Convert the index positions back to class names.

        if (
            true_label in top_k_labels
        ):  # Count a hit if the true label appears in the top-k predictions.
            correct += 1  # Increase the correct count.

    return correct / len(y_true)  # Return the proportion of correct top-k predictions.


results = []  # Store the evaluation metrics for every tested C value.


for C in C_values:  # Evaluate each candidate regularization strength.
    model = LogisticRegression(  # Create a new model for the current regularization value.
        C=C,  # Set the current inverse regularization parameter.
        penalty="l2",  # Use L2 regularization to stabilize the coefficients.
        solver="lbfgs",  # Use the default optimizer for multiclass logistic regression.
        max_iter=5000,  # Allow enough iterations for convergence.
        random_state=42,  # Keep the optimization deterministic.
    )

    model.fit(X_train, y_train)  # Train the model on the training set.

    predictions = model.predict(X_val)  # Predict validation labels.
    probabilities = model.predict_proba(
        X_val
    )  # Get probability scores for top-k evaluation.

    results.append(
        {  # Save metrics for the current model configuration.
            "C": C,
            "Accuracy": accuracy_score(y_val, predictions),
            "Macro_F1": f1_score(y_val, predictions, average="macro"),
            "Top_1": top_k_accuracy(y_val, probabilities, model.classes_, 1),
            "Top_3": top_k_accuracy(y_val, probabilities, model.classes_, 3),
            "Top_5": top_k_accuracy(y_val, probabilities, model.classes_, 5),
        }
    )


results_df = pd.DataFrame(
    results
)  # Convert the stored results into a DataFrame for display.


print(
    "\nFine Logistic Regression C search"
)  # Print the title for this tuning experiment.
print("=" * 90)  # Print a separator line.

print(  # Display all C values and their validation metrics.
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


print("\nBest by Top-1:")  # Print the best result based on top-1 accuracy.
print(results_df.loc[results_df["Top_1"].idxmax()].to_string())

print("\nBest by Macro F1:")  # Print the best result based on macro F1.
print(results_df.loc[results_df["Macro_F1"].idxmax()].to_string())

print("\nBest by Top-3:")  # Print the best result based on top-3 accuracy.
print(results_df.loc[results_df["Top_3"].idxmax()].to_string())
