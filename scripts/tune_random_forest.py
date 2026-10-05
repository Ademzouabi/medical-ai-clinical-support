import pandas as pd  # Import pandas to load data and display evaluation tables.
from pathlib import Path
from sklearn.ensemble import (
    RandomForestClassifier,
)  # Import the random forest model for tuning experiments.
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)  # Import metrics used to evaluate each forest configuration.


# --------------------------------------------------
# Load data
# --------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
train = pd.read_csv(DATA_DIR / "v1_train.csv")  # Load the training split.
validation = pd.read_csv(DATA_DIR / "v1_validation.csv")  # Load the validation split.

X_train = train.drop(
    columns=["disease"]
)  # Create the feature matrix from all symptom columns.
y_train = train["disease"]  # Keep the disease label as the target variable.

X_val = validation.drop(columns=["disease"])  # Create validation features.
y_val = validation["disease"]  # Store the validation labels.


# --------------------------------------------------
# Top-K accuracy
# --------------------------------------------------


def top_k_accuracy(
    y_true, probabilities, classes, k
):  # Measure whether the true class appears in the top-k predictions.
    correct = 0  # Initialize counter for correct top-k predictions.

    for true_label, probs in zip(
        y_true, probabilities
    ):  # Loop through each row and its prediction probabilities.
        top_k_indices = probs.argsort()[-k:][
            ::-1
        ]  # Select the k most likely disease classes.
        top_k_labels = classes[top_k_indices]  # Convert indices into class names.

        if (
            true_label in top_k_labels
        ):  # If the true disease is in the top-k set, count it as correct.
            correct += 1  # Increase the success total.

    return correct / len(y_true)  # Return the top-k accuracy proportion.


# --------------------------------------------------
# Random Forest configurations
# --------------------------------------------------

configs = [  # List a small set of forest settings to compare for performance.
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None,
    },
    {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None,
    },
    {
        "n_estimators": 500,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None,
    },
    {
        "n_estimators": 500,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None,
    },
    {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None,
    },
    {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": None,
    },
    {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "log2",
        "class_weight": None,
    },
    {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
]


# --------------------------------------------------
# Train and evaluate configurations
# --------------------------------------------------

results = []  # Store the evaluation results for every forest configuration.


for i, config in enumerate(
    configs, start=1
):  # Check each random forest configuration one by one.
    print(
        f"\nTesting configuration {i}/{len(configs)}..."
    )  # Print progress for the tuning loop.

    model = RandomForestClassifier(  # Create a random forest model with the current settings.
        random_state=42,  # Keep the model deterministic.
        n_jobs=-1,  # Use all available CPU cores to speed up training.
        **config,  # Apply the current configuration dictionary.
    )

    model.fit(X_train, y_train)  # Train the forest on the training data.

    predictions = model.predict(X_val)  # Predict labels for validation rows.
    probabilities = model.predict_proba(
        X_val
    )  # Get class probabilities for top-k checks.

    results.append(
        {  # Save performance statistics for this configuration.
            "Config": i,
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "min_samples_split": config["min_samples_split"],
            "min_samples_leaf": config["min_samples_leaf"],
            "max_features": config["max_features"],
            "class_weight": config["class_weight"],
            "Accuracy": accuracy_score(y_val, predictions),
            "Macro_F1": f1_score(y_val, predictions, average="macro"),
            "Top_1": top_k_accuracy(y_val, probabilities, model.classes_, 1),
            "Top_3": top_k_accuracy(y_val, probabilities, model.classes_, 3),
            "Top_5": top_k_accuracy(y_val, probabilities, model.classes_, 5),
        }
    )


# --------------------------------------------------
# Display results
# --------------------------------------------------

results_df = pd.DataFrame(results)  # Create a DataFrame from the recorded results.

print(
    "\n\nRandom Forest tuning results"
)  # Print a heading for the forest tuning summary.
print("=" * 130)  # Print a wide separator line.

print(  # Display the performance table for all tested configurations.
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

print("\nBest by Top-1:")  # Print the best configuration for top-1 accuracy.
print(results_df.loc[results_df["Top_1"].idxmax()].to_string())

print("\nBest by Macro F1:")  # Print the best configuration for macro F1.
print(results_df.loc[results_df["Macro_F1"].idxmax()].to_string())

print("\nBest by Top-3:")  # Print the best configuration for top-3 accuracy.
print(results_df.loc[results_df["Top_3"].idxmax()].to_string())
