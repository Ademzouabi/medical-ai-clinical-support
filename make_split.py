import pandas as pd  # Import pandas so we can work with tabular CSV data using DataFrames.
from sklearn.model_selection import (
    GroupShuffleSplit,
)  # Import GroupShuffleSplit to split data while keeping identical symptom vectors in the same group.

INPUT_FILE = "v1_standardized.csv"  # Path to the cleaned standard dataset that will be split into train/validation/test.

TRAIN_FILE = "v1_train.csv"  # Output filename for the training split.
VAL_FILE = "v1_validation.csv"  # Output filename for the validation split.
TEST_FILE = "v1_test.csv"  # Output filename for the test split.


# Load standardized dataset  # Read the cleaned dataset into a DataFrame so we can split it.
df = pd.read_csv(
    INPUT_FILE
)  # Load the CSV file from the current folder into a pandas DataFrame.


# Features = everything except target  # Define the input feature columns as all columns except the disease label.
features = [
    c for c in df.columns if c != "disease"
]  # Create a list of feature column names excluding the target variable.


# create a group identifier from the complete symptom vector  # Build a grouping key from all symptom columns; identical symptom patterns will receive the same group ID.
# Identical symptom vectors receive the same group.  # This ensures that all rows sharing the same symptom feature combination stay together across splits.
groups = df[features].apply(
    lambda row: tuple(row.tolist()),
    axis=1
)
# Convert each row's symptom values to strings and concatenate them into one group label.

# First split: 70% train, 30% temporary  # Use an initial split to reserve 30% of the data for a temporary holdout before creating validation/test sets.

split1 = GroupShuffleSplit(  # Create the first group-aware splitter for the initial data partition.
    n_splits=1,  # Use exactly one split.
    test_size=0.30,  # Reserve 30% of the data for the temporary set.
    random_state=42,  # Fix the random seed so the split is reproducible.
)

train_idx, temp_idx = (
    next(  # Generate the indices for the training set and the temporary holdout set.
        split1.split(
            df, df["disease"], groups
        )  # Split while grouping rows by identical symptom vectors and respecting disease labels.
    )
)

train = df.iloc[train_idx].copy()  # Save the training rows as a new DataFrame copy.
temp = df.iloc[
    temp_idx
].copy()  # Save the temporary holdout rows as a new DataFrame copy.

temp_groups = groups.iloc[
    temp_idx
]  # Keep the group labels for the temporary split so the second split also respects grouping.

# Second split: 15% validation, 15% test  # Split the temporary set into validation and test sets with equal size.

split2 = GroupShuffleSplit(  # Create the second group-aware splitter for the validation/test split.
    n_splits=1,  # Use exactly one split.
    test_size=0.50,  # Assign half of the temporary set to the test set, leaving the other half for validation.
    random_state=42,  # Fix the random seed so the split is reproducible.
)

val_idx, test_idx = (
    next(  # Generate the indices for the validation and test sets from the temporary set.
        split2.split(
            temp, temp["disease"], temp_groups
        )  # Split the temp data while preserving the group-based constraints.
    )
)

validation = temp.iloc[
    val_idx
].copy()  # Save the validation rows as a new DataFrame copy.
test = temp.iloc[test_idx].copy()  # Save the test rows as a new DataFrame copy.

# Save  # Export the split datasets for later model training and evaluation.
train.to_csv(
    TRAIN_FILE, index=False
)  # Write the training DataFrame to its CSV output file.
validation.to_csv(
    VAL_FILE, index=False
)  # Write the validation DataFrame to its CSV output file.
test.to_csv(TEST_FILE, index=False)  # Write the test DataFrame to its CSV output file.

# Verify no symptom-vector leakage  # Check that the same symptom pattern does not appear across train/validation/test sets.


train_groups = set(  # Convert the training feature-summary strings into a set of unique groups.
    train[features]
    .astype(str)
    .agg(
        "|".join, axis=1
    )  # Build one string per row from all symptom columns and collect unique group values.
)

val_groups = set(  # Convert the validation feature-summary strings into a set of unique groups.
    validation[features]
    .astype(str)
    .agg(
        "|".join, axis=1
    )  # Build one string per row from all symptom columns and collect unique group values.
)

test_groups = set(  # Convert the test feature-summary strings into a set of unique groups.
    test[features]
    .astype(str)
    .agg(
        "|".join, axis=1
    )  # Build one string per row from all symptom columns and collect unique group values.
)

print("TRAIN:", len(train))  # Print the number of training examples.
print("VALIDATION:", len(validation))  # Print the number of validation examples.
print("TEST:", len(test))  # Print the number of test examples.

print("\nGroup overlap:")  # Label the next section as group overlap results.
print(
    "Train ∩ Validation:", len(train_groups & val_groups)
)  # Print how many symptom groups overlap between training and validation.
print(
    "Train ∩ Test:", len(train_groups & test_groups)
)  # Print how many symptom groups overlap between training and test.
print(
    "Validation ∩ Test:", len(val_groups & test_groups)
)  # Print how many symptom groups overlap between validation and test.

print("\nFiles saved:")  # Label the next output as the saved file names.
print(TRAIN_FILE)  # Print the path/name of the training file.
print(VAL_FILE)  # Print the path/name of the validation file.
print(TEST_FILE)  # Print the path/name of the test file.
