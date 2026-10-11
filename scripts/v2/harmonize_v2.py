"""Harmonize the audited raw dataset into the V2 training table.

This script validates the complete V2 data contract before writing the dataset.
It does not train a model.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

# Find the project folder from this script's location.
ROOT = Path(__file__).resolve().parents[2]

# V2 contract files and dataset paths.
FEATURE_SCHEMA = ROOT / "config" / "v2" / "feature_schema.json"
DISEASE_SCOPE = ROOT / "config" / "v2" / "disease_scope.json"
RAW_DATA = (
    ROOT
    / "data"
    / "raw"
    / "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
    / "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
)
OUTPUT_DATA = ROOT / "data" / "processed" / "v2" / "standardized.csv"
PROVENANCE_DATA = ROOT / "data" / "processed" / "v2" / "provenance.json"


def read_json(path):
    """Read a JSON file and return its Python dictionary."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path):
    """Calculate a file's SHA-256 hash without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stop_if_errors(errors):
    """Stop the pipeline if one or more contract checks failed."""
    if errors:
        details = "\n- ".join(errors)
        raise ValueError(f"V2 validation failed:\n- {details}")


def validate_saved_dataset(frame, expected_columns, expected_counts, expected_rows):
    """Check the dataset read back from disk against the V2 contract."""
    errors = []
    target_column = expected_columns[0]
    feature_columns = expected_columns[1:]
    expected_shape = (expected_rows, len(expected_columns))
    actual_columns = frame.columns.tolist()

    if actual_columns != expected_columns:
        errors.append("saved column names or order differ from the V2 schema")
    if frame.shape != expected_shape:
        errors.append(f"saved dimensions are {frame.shape}; expected {expected_shape}")

    # Only inspect target/features when those columns are present.
    if target_column not in frame.columns:
        errors.append(f"saved target column {target_column!r} is missing")
    else:
        if frame[target_column].isna().any():
            errors.append("saved disease target contains missing values")
        saved_counts = {
            str(label): int(count)
            for label, count in frame[target_column].value_counts().items()
        }
        if saved_counts != expected_counts:
            errors.append("saved disease counts differ from the audited class counts")

    missing_features = [name for name in feature_columns if name not in frame.columns]
    if missing_features:
        errors.append(f"saved feature columns are missing: {missing_features}")
    else:
        feature_values = frame[feature_columns]
        if feature_values.isna().any().any():
            errors.append("saved feature matrix contains missing values")
        elif not feature_values.isin([0, 1]).all().all():
            errors.append("saved feature matrix contains values other than 0 and 1")

    stop_if_errors(errors)
    return saved_counts


def main():
    # Load the ordered feature definitions and exact raw disease-label contract.
    feature_config = read_json(FEATURE_SCHEMA)
    disease_config = read_json(DISEASE_SCOPE)

    features = feature_config["features"]
    diseases = disease_config["diseases"]
    target_column = feature_config["target_column"]
    target_source = disease_config["raw_target_column"]
    source_features = [feature["source_column"] for feature in features]
    feature_names = [feature["name"] for feature in features]
    raw_labels = [disease["raw_label"] for disease in diseases]
    expected_counts = {
        disease["raw_label"]: int(disease["expected_count"]) for disease in diseases
    }
    expected_rows = int(disease_config["expected_rows"])
    expected_columns = [target_column, *feature_names]

    # Check that the two configs agree and contain the required contract sizes.
    errors = []
    if feature_config["target_source_column"] != target_source:
        errors.append("feature schema and disease scope disagree on the raw target")
    if len(features) != 47 or feature_config["feature_count"] != 47:
        errors.append(f"expected 47 features; config contains {len(features)}")
    if len(set(feature_names)) != len(feature_names):
        errors.append("V2 feature names are not unique")
    if len(set(source_features)) != len(source_features):
        errors.append("V2 raw feature source columns are not unique")
    if len(diseases) != 36 or len(set(raw_labels)) != len(raw_labels):
        errors.append("disease scope must contain 36 unique raw labels")
    if sum(expected_counts.values()) != expected_rows:
        errors.append("configured disease counts do not sum to expected_rows")
    if min(expected_counts.values()) < int(disease_config["minimum_class_support"]):
        errors.append("a disease is below the configured minimum class support")

    for feature in features:
        mapping_type = feature["transformation"]
        if mapping_type not in {"identity", "identity_rename"}:
            errors.append(f"{feature['name']}: unsupported mapping {mapping_type!r}")
        if (feature["name"] != feature["source_column"]) != (
            mapping_type == "identity_rename"
        ):
            errors.append(
                f"{feature['name']}: mapping is not a declared one-column identity mapping"
            )
    stop_if_errors(errors)

    if not RAW_DATA.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATA}")

    # Read the header first so missing or duplicate raw columns fail clearly.
    raw_header = pd.read_csv(RAW_DATA, nrows=0).columns.tolist()
    duplicate_headers = [
        name for name, count in Counter(raw_header).items() if count > 1
    ]
    required_raw_columns = [target_source, *source_features]
    missing_columns = [name for name in required_raw_columns if name not in raw_header]
    errors = []
    if duplicate_headers:
        errors.append(f"raw CSV contains duplicate headers: {duplicate_headers}")
    if missing_columns:
        errors.append(f"required raw columns are missing: {missing_columns}")
    stop_if_errors(errors)

    # Load only the raw target and the 47 contracted source columns.
    raw_data = pd.read_csv(
        RAW_DATA,
        usecols=required_raw_columns,
        low_memory=False,
    )

    # Require every exact raw disease label and its audited support count.
    actual_counts = raw_data[target_source].value_counts()
    unresolved_labels = [
        label for label in raw_labels if int(actual_counts.get(label, 0)) == 0
    ]
    count_mismatches = {
        label: {
            "expected": expected_count,
            "actual": int(actual_counts.get(label, 0)),
        }
        for label, expected_count in expected_counts.items()
        if int(actual_counts.get(label, 0)) != expected_count
    }
    errors = []
    if unresolved_labels:
        errors.append(f"raw disease labels do not resolve: {unresolved_labels}")
    if count_mismatches:
        errors.append(f"raw disease counts differ from audit: {count_mismatches}")
    stop_if_errors(errors)

    # Select only the rows for the 36 exact raw labels, preserving source order.
    selected_rows = raw_data[raw_data[target_source].isin(raw_labels)].copy()
    if len(selected_rows) != expected_rows:
        stop_if_errors(
            [f"selected {len(selected_rows)} rows; expected {expected_rows}"]
        )

    # Check values before renaming: no missing values and only binary 0/1 values.
    source_values = selected_rows[source_features]
    missing_value_count = int(source_values.isna().sum().sum())
    non_binary_value_count = int((~source_values.isin([0, 1])).sum().sum())
    errors = []
    if missing_value_count:
        errors.append(f"found {missing_value_count} missing feature values")
    if non_binary_value_count:
        errors.append(f"found {non_binary_value_count} non-binary feature values")
    stop_if_errors(errors)

    # Keep raw disease labels unchanged, then copy each mapped feature in order.
    v2_data = pd.DataFrame({target_column: selected_rows[target_source].copy()})
    for feature in features:
        v2_data[feature["name"]] = selected_rows[feature["source_column"]].copy()

    # Validate the in-memory table before writing anything to disk.
    if v2_data.columns.tolist() != expected_columns:
        stop_if_errors(["model table columns or order differ from the V2 schema"])
    if v2_data.shape != (expected_rows, 48):
        stop_if_errors(
            [f"model table shape is {v2_data.shape}; expected ({expected_rows}, 48)"]
        )
    in_memory_counts = {
        str(label): int(count)
        for label, count in v2_data[target_column].value_counts().items()
    }
    if in_memory_counts != expected_counts:
        stop_if_errors(["in-memory disease counts differ from the audit config"])

    # Write without pandas' index so the CSV contains only the target and features.
    OUTPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    v2_data.to_csv(OUTPUT_DATA, index=False)

    # Reopen the saved CSV and validate the artifact that future code will read.
    saved_data = pd.read_csv(OUTPUT_DATA, low_memory=False)
    saved_counts = validate_saved_dataset(
        saved_data,
        expected_columns,
        expected_counts,
        expected_rows,
    )

    # Record source/config/output hashes and the exact feature/class ordering.
    provenance = {
        "schema_version": "2.0",
        "source_file": str(RAW_DATA.resolve()),
        "source_sha256": sha256_file(RAW_DATA),
        "feature_schema_file": str(FEATURE_SCHEMA.resolve()),
        "feature_schema_sha256": sha256_file(FEATURE_SCHEMA),
        "disease_scope_file": str(DISEASE_SCOPE.resolve()),
        "disease_scope_sha256": sha256_file(DISEASE_SCOPE),
        "output_file": str(OUTPUT_DATA.resolve()),
        "output_sha256": sha256_file(OUTPUT_DATA),
        "raw_target_column": target_source,
        "output_target_column": target_column,
        "target_labels_preserved_exactly": True,
        "feature_order": feature_names,
        "feature_source_order": source_features,
        "disease_order": [disease["raw_label"] for disease in diseases],
        "row_count": len(saved_data),
        "column_count": len(saved_data.columns),
        "class_counts": saved_counts,
        "validation": {
            "36_raw_disease_labels_resolve": True,
            "36_class_counts_match_audit": True,
            "47_raw_feature_columns_exist": True,
            "features_are_binary_and_nonnull": True,
            "no_unexpected_columns": True,
            "feature_and_target_order_matches_schema": True,
            "saved_dimensions_match_audit": True,
            "training_performed": False,
        },
    }
    PROVENANCE_DATA.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    # Report a compact summary only after every check passes.
    print("V2 validation passed.")
    print("Rows and columns:", saved_data.shape)
    print("Feature count:", len(feature_names))
    print("Disease classes:", len(diseases))
    print("Output CSV:", OUTPUT_DATA)
    print("Provenance record:", PROVENANCE_DATA)


if __name__ == "__main__":
    main()
