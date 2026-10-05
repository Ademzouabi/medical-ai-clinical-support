import pandas as pd  # Import pandas to read the CSV files and organize the tuning results.
from pathlib import Path
from sklearn.ensemble import (
    RandomForestClassifier,
)  # Import the random forest model for the depth and leaf-size tuning grid.
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)  # Import metrics used to compare model performance.


# Load data  # Read the training and validation splits from their CSV files.
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
train = pd.read_csv(DATA_DIR / "v1_train.csv")  # Load the training set.
validation = pd.read_csv(DATA_DIR / "v1_validation.csv")  # Load the validation set.

X_train = train.drop(
    columns=["disease"]
)  # Build the feature matrix from the symptom columns.
y_train = train["disease"]  # Use the diagnosis as the training target.

X_val = validation.drop(
    columns=["disease"]
)  # Build feature values for the validation set.
y_val = validation["disease"]  # Use the validation labels as the ground truth.


def top_k_accuracy(
    y_true, probabilities, classes, k
):  # Return the fraction of examples where the true label is in the top-k predictions.
    correct = 0  # Initialize the successful top-k count.

    for true_label, probs in zip(
        y_true, probabilities
    ):  # Iterate over each row and its probability distribution.
        top_k_indices = probs.argsort()[-k:][
            ::-1
        ]  # Select the k highest-probability classes.
        top_k_labels = classes[
            top_k_indices
        ]  # Convert the indices back to disease names.

        if (
            true_label in top_k_labels
        ):  # Count a hit if the true disease appears in the top-k list.
            correct += 1  # Increase the correct count.

    return correct / len(y_true)  # Return the ratio of correct top-k predictions.


configs = []  # Create a list of forest settings to evaluate in the finer grid search.

for max_depth in [
    15,
    18,
    20,
    22,
    25,
    30,
]:  # Try several tree depths to balance complexity and generalization.
    for min_samples_leaf in [
        1,
        2,
        3,
    ]:  # Try a few minimum leaf sizes to reduce overfitting.
        configs.append(
            {  # Save one configuration at a time.
                "n_estimators": 500,
                "max_depth": max_depth,
                "min_samples_split": 2,
                "min_samples_leaf": min_samples_leaf,
                "max_features": "sqrt",
                "class_weight": None,
            }
        )


results = []  # Store all performance values for the tuned forest settings.


for i, config in enumerate(
    configs, start=1
):  # Test each configuration in the grid search.
    print(
        f"Testing configuration {i}/{len(configs)}..."
    )  # Report progress during the search.

    model = (
        RandomForestClassifier(  # Create a new random forest with the current settings.
            random_state=42,  # Fix the random seed for reproducibility.
            n_jobs=-1,  # Use all CPU cores for faster model training.
            **config,  # Apply the current tuning configuration.
        )
    )

    model.fit(X_train, y_train)  # Train the random forest on the training data.

    predictions = model.predict(X_val)  # Predict validation labels.
    probabilities = model.predict_proba(
        X_val
    )  # Compute probability scores for top-k evaluation.

    results.append(
        {  # Save metrics for this configuration.
            "Config": i,
            "max_depth": config["max_depth"],
            "min_samples_leaf": config["min_samples_leaf"],
            "Accuracy": accuracy_score(y_val, predictions),
            "Macro_F1": f1_score(y_val, predictions, average="macro"),
            "Top_1": top_k_accuracy(y_val, probabilities, model.classes_, 1),
            "Top_3": top_k_accuracy(y_val, probabilities, model.classes_, 3),
            "Top_5": top_k_accuracy(y_val, probabilities, model.classes_, 5),
        }
    )


results_df = pd.DataFrame(
    results
)  # Convert the results list into a DataFrame for readable output.

print(
    "\n\nFine Random Forest tuning results"
)  # Print the title for the fine-grained tuning summary.
print("=" * 100)  # Print a separator line.

print(  # Display the complete tuning table.
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


print("\nBest by Top-1:")  # Show the best result based on top-1 accuracy.
print(results_df.loc[results_df["Top_1"].idxmax()].to_string())

print("\nBest by Macro F1:")  # Show the best result based on macro F1.
print(results_df.loc[results_df["Macro_F1"].idxmax()].to_string())

print("\nBest by Top-3:")  # Show the best result based on top-3 accuracy.
print(results_df.loc[results_df["Top_3"].idxmax()].to_string())
