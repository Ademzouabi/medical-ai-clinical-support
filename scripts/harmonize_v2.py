"""Check the V2 schema and disease scope against the raw dataset."""

import json
from pathlib import Path

import pandas as pd

# Find the project folder by starting from this script's location.
ROOT = Path(__file__).resolve().parents[1]

# These JSON files define the approved V2 features and disease labels.
FEATURE_SCHEMA = ROOT / "config" / "v2_feature_schema.json"
DISEASE_SCOPE = ROOT / "config" / "v2_disease_scope.json"

# The raw source and the future harmonized dataset live in the project.
RAW_DATA = (
    ROOT
    / "data"
    / "raw"
    / "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
    / "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
)
OUTPUT_DATA = ROOT / "data" / "processed" / "v2_standardized.csv"


def main():
    # Read each JSON file and convert its contents into a Python dictionary.
    feature_config = json.loads(FEATURE_SCHEMA.read_text(encoding="utf-8"))
    disease_config = json.loads(DISEASE_SCOPE.read_text(encoding="utf-8"))

    # Confirm the number of features and diseases in the two configs.
    print("Features in the schema:", len(feature_config["features"]))
    print("Diseases in the scope:", len(disease_config["diseases"]))

    # Read only the CSV header here; nrows=0 avoids loading data rows.
    raw_header = pd.read_csv(RAW_DATA, nrows=0).columns.tolist()
    print("Raw CSV columns:", len(raw_header))

    # Find the raw target column name from the disease-scope config.
    target_source = disease_config["raw_target_column"]
    print("Target column exists:", target_source in raw_header)

    # Gather the exact raw source-column names for the 47 approved features.
    required_features = [
        feature["source_column"] for feature in feature_config["features"]
    ]

    # Report any approved feature source that is absent from the raw header.
    missing_features = [name for name in required_features if name not in raw_header]
    print("Required feature sources:", len(required_features))
    print("Missing feature sources:", missing_features)

    # Read the target column so we can count raw examples per disease label.
    raw_diseases = pd.read_csv(
        RAW_DATA,
        usecols=[target_source],
    )[target_source]
    actual_counts = raw_diseases.value_counts()

    # Expected label names and counts come directly from the approved config.
    expected_counts = {
        disease["raw_label"]: disease["expected_count"]
        for disease in disease_config["diseases"]
    }

    # Find any configured class whose raw row count differs from expectation.
    count_mismatches = {
        label: {
            "expected": expected_count,
            "actual": int(actual_counts.get(label, 0)),
        }
        for label, expected_count in expected_counts.items()
        if int(actual_counts.get(label, 0)) != expected_count
    }
    print("Configured disease labels:", len(expected_counts))
    print("Disease count mismatches:", count_mismatches)

    # Load only the target plus approved raw feature columns, not all 378 fields.
    raw_v2 = pd.read_csv(
        RAW_DATA,
        usecols=[target_source, *required_features],
    )

    # Keep rows whose raw disease label belongs to the V2 disease scope.
    selected_rows = raw_v2[raw_v2[target_source].isin(expected_counts)]

    # Check the approved feature values for missing and non-binary values.
    feature_values = selected_rows[required_features]
    missing_value_count = int(feature_values.isna().sum().sum())
    non_binary_value_count = int((~feature_values.isin([0, 1])).sum().sum())

    print("Selected rows:", len(selected_rows))
    print("Missing feature values:", missing_value_count)
    print("Non-binary feature values:", non_binary_value_count)

    # Start the output table with the unchanged raw disease labels.
    v2_data = pd.DataFrame(
        {"disease": selected_rows[target_source].copy()}
    )

    # Keep the configured feature order for the next construction step.
    feature_names = [
        feature["name"]
        for feature in feature_config["features"]
    ]

    # Copy each approved raw feature into its configured V2 feature name.
    for feature in feature_config["features"]:
        v2_data[feature["name"]] = selected_rows[
            feature["source_column"]
        ].copy()
    # Verify that the output table follows the target-plus-feature order in config.
    expected_columns = ["disease"] + feature_names
    column_order_matches = v2_data.columns.tolist() == expected_columns
    print("Column order valid:", column_order_matches)

    # Save only the target and feature columns. index=False avoids an extra CSV field.
    v2_data.to_csv(OUTPUT_DATA, index=False)
    print("Saved V2 dataset to:", OUTPUT_DATA)
# Run the checks only when this script is launched directly.
if __name__ == "__main__":
    main()










