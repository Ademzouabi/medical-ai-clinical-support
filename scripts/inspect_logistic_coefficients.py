import pandas as pd  # Import pandas to read the training CSV and manipulate DataFrames.
from pathlib import Path
from sklearn.linear_model import (
    LogisticRegression,
)  # Import the logistic regression model to inspect learned feature weights.


# --------------------------------------------------
# Load training data only
# --------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
train = pd.read_csv(DATA_DIR / "v1_train.csv")  # Read only the training split from disk.

X_train = train.drop(columns=["disease"])  # Keep all symptom columns as model inputs.
y_train = train["disease"]  # Keep the disease label as the target variable.


# --------------------------------------------------
# Train the same baseline Logistic Regression
# --------------------------------------------------

model = (
    LogisticRegression(  # Create a baseline model similar to the main training setup.
        max_iter=2000,  # Allow enough optimization steps for convergence.
        random_state=42,  # Use a fixed seed to keep the model reproducible.
    )
)

model.fit(X_train, y_train)  # Fit the logistic regression model to the training data.


# --------------------------------------------------
# Extract coefficients
# --------------------------------------------------

coefficients = (
    pd.DataFrame(  # Store the learned coefficients in a table for easier inspection.
        model.coef_,  # The numeric coefficient matrix from the trained model.
        index=model.classes_,  # Label each row by the disease class.
        columns=X_train.columns,  # Label each column by the feature name.
    )
)


# --------------------------------------------------
# Display strongest features for each disease
# --------------------------------------------------

for disease in (
    model.classes_
):  # Loop through each disease class to inspect its most important features.
    disease_coefficients = coefficients.loc[
        disease
    ]  # Get the coefficient vector for the current disease.

    strongest_positive = (  # Select the top 5 features with the strongest positive contribution.
        disease_coefficients.sort_values(ascending=False).head(5)
    )

    strongest_negative = (  # Select the top 5 features with the strongest negative contribution.
        disease_coefficients.sort_values().head(5)
    )

    print("\n" + "=" * 70)  # Print a separator before each disease section.
    print(disease)  # Print the current disease name.
    print("=" * 70)  # Close the section separator.

    print(
        "\nStrongest positive features:"
    )  # Explain that these are features increasing the chance of the disease.
    for (
        feature,
        value,
    ) in (
        strongest_positive.items()
    ):  # Print the feature name and its coefficient value.
        print(
            f"  {feature:30s} {value:+.4f}"
        )  # Show a formatted coefficient with a sign.

    print(
        "\nStrongest negative features:"
    )  # Explain that these are features reducing the chance of the disease.
    for (
        feature,
        value,
    ) in strongest_negative.items():  # Print the most negative coefficients.
        print(
            f"  {feature:30s} {value:+.4f}"
        )  # Show the negative coefficient value in a readable format.
