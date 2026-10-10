"""Compare Logistic Regression class-weight settings on validation data."""

import pandas as pd  # Import pandas to read the split CSV files and store results in a table.
from pathlib import Path
from sklearn.linear_model import (
    LogisticRegression,
)  # Import the logistic regression model for testing different class-weight settings.
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)  # Import performance metrics for the comparison.


# Load data  # Read the training and validation sets from the CSV files.
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
train = pd.read_csv(DATA_DIR / "v1_train.csv")  # Load the training split.
validation = pd.read_csv(DATA_DIR / "v1_validation.csv")  # Load the validation split.

X_train = train.drop(
    columns=["disease"]
)  # Remove the label column to form the feature matrix.
y_train = train["disease"]  # Keep disease as the target variable.

X_val = validation.drop(
    columns=["disease"]
)  # Remove the label column from validation features.
y_val = validation["disease"]  # Keep the true disease labels for validation.


def top_k_accuracy(
    y_true, probabilities, classes, k
):  # Compute whether the true label appears within the top-k predictions.
    correct = 0  # Start with zero correct top-k predictions.

    for true_label, probs in zip(
        y_true, probabilities
    ):  # Loop through each row and its probability scores.
        top_k_indices = probs.argsort()[-k:][
            ::-1
        ]  # Take the highest-probability classes.
        top_k_labels = classes[
            top_k_indices
        ]  # Convert the indices back to disease names.

        if (
            true_label in top_k_labels
        ):  # Count a hit if the true disease is in the top-k set.
            correct += 1  # Increase the total correct count.

    return correct / len(y_true)  # Return the top-k accuracy ratio.


results = []  # Store results from each class-weight setting.


for weight in [None, "balanced"]:  # Test the default behavior versus class balancing.
    model = LogisticRegression(  # Create a logistic regression model with the current class weighting.
        C=8,  # Use the already promising regularization strength.
        penalty="l2",  # Use L2 regularization for stable coefficients.
        solver="lbfgs",  # Use the optimizer for multinomial logistic regression.
        class_weight=weight,  # Optionally reweight classes to deal with imbalance.
        max_iter=5000,  # Allow enough iterations to converge.
        random_state=42,  # Keep the result deterministic.
    )

    model.fit(X_train, y_train)  # Train the model on the training set.

    predictions = model.predict(X_val)  # Predict labels for validation samples.
    probabilities = model.predict_proba(
        X_val
    )  # Generate probability scores for top-k metrics.

    results.append(
        {  # Store the model performance for this class-weight setting.
            "Class_Weight": str(weight),
            "Accuracy": accuracy_score(y_val, predictions),
            "Macro_F1": f1_score(y_val, predictions, average="macro"),
            "Top_1": top_k_accuracy(y_val, probabilities, model.classes_, 1),
            "Top_3": top_k_accuracy(y_val, probabilities, model.classes_, 3),
            "Top_5": top_k_accuracy(y_val, probabilities, model.classes_, 5),
        }
    )


results_df = pd.DataFrame(results)  # Combine the results into a DataFrame for display.

print(
    "\nLogistic Regression class-weight comparison"
)  # Print the title for the comparison.
print("=" * 80)  # Print a separator line.

print(  # Display the comparison table in a readable format.
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
