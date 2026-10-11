"""Create and validate the deterministic, group-isolated V2 data split.

This script prepares data partitions only. It does not train a model or change
the accepted V2 features, labels, or values.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# __file__ is this script's path. Its grandparent is the repository root because
# the script lives in scripts/v2/. Building paths from ROOT means the script can
# be launched from any working directory.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "v2"
FEATURE_SCHEMA = ROOT / "config" / "v2" / "feature_schema.json"
DISEASE_SCOPE = ROOT / "config" / "v2" / "disease_scope.json"
INPUT_FILE = DATA_DIR / "standardized.csv"


# The input remains the complete standardized dataset. These are separate output
# files for the three partitions; the script does not overwrite standardized.csv.
TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"
SPLIT_MANIFEST = DATA_DIR / "split_manifest.csv"
AUDIT_REPORT = DATA_DIR / "split_audit.json"


def sha256_file(path):
    """Return a SHA-256 checksum for identifying the exact input/config files."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stop_if_errors(errors):
    """Stop processing if one or more split integrity checks fail.

    The caller collects problems into a list so one run can report several
    failures together. Before writing, this prevents partition files from being
    created. After writing, it prevents the run from reporting success or writing
    a later audit report when the saved files fail read-back checks.
    """
    if errors:
        details = "\n- ".join(errors)
        raise ValueError(f"V2 split validation failed:\n- {details}")


def main():
    # Read the approved schema and disease list instead of inventing feature or
    # label names inside the split script.
    feature_config = json.loads(FEATURE_SCHEMA.read_text(encoding="utf-8"))
    disease_config = json.loads(DISEASE_SCOPE.read_text(encoding="utf-8"))

    # Keep the exact feature order from the schema. The target is kept separate
    # from the feature vector so it cannot influence group identity.
    features = [feature["name"] for feature in feature_config["features"]]
    target = feature_config["target_column"]
    expected_labels = {disease["raw_label"] for disease in disease_config["diseases"]}

    # The selected grouped candidate was previously inspected: only these two
    # low-diversity diseases are absent from its test partition.
    expected_test_missing = {"carpal tunnel syndrome", "gout"}

    # Load the harmonized V2 table. low_memory=False avoids mixed type guesses
    # while pandas reads the full CSV.
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    # Refuse to split a table with unexpected columns, wrong feature ordering,
    # incorrect dimensions, or labels outside the audited 36-class scope.
    expected_columns = [target, *features]
    if df.columns.tolist() != expected_columns:
        raise ValueError("Input columns or order do not match the V2 feature schema.")
    if len(features) != 47 or len(df) != int(disease_config["expected_rows"]):
        raise ValueError("Input dimensions do not match the accepted V2 data contract.")
    if set(df[target].unique()) != expected_labels:
        raise ValueError("Input disease labels do not match the V2 disease scope.")

    # Give each exact 47-feature vector one integer group ID. The disease label
    # is deliberately excluded: identical inputs stay together even when their
    # rows have conflicting disease labels.
    groups = pd.MultiIndex.from_frame(df[features]).factorize(sort=False)[0]

    # First make five group-isolated outer folds. Each outer fold is used once
    # as a possible test partition. We select fold 4 (Python index 3), the
    # candidate previously reviewed for class coverage and test-vector support.
    # Stratification tries to balance disease labels while groups remain intact.
    outer_splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    outer_splits = list(outer_splitter.split(df[features], df[target], groups))
    train_pool_idx, test_idx = outer_splits[3]

    # Keep the outer training rows together as the pool for the next step.
    # The outer test rows are held aside and are not used to select validation.
    train_pool = df.iloc[train_pool_idx]
    train_pool_groups = groups[train_pool_idx]

    # Split only the training pool into five group-isolated folds. Fold 1
    # (Python index 0) becomes validation; its complement becomes training.
    inner_splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    inner_splits = list(
        inner_splitter.split(
            train_pool[features],
            train_pool[target],
            train_pool_groups,
        )
    )
    train_local_idx, val_local_idx = inner_splits[0]

    # These inner positions refer to rows inside train_pool, so translate them
    # back to positions in the original standardized table.
    train_idx = train_pool_idx[train_local_idx]
    val_idx = train_pool_idx[val_local_idx]

    # Copy the selected original rows. No feature values or target labels are
    # transformed as part of partitioning.
    train = df.iloc[train_idx].copy()
    validation = df.iloc[val_idx].copy()
    test = df.iloc[test_idx].copy()
    partitions = {
        "train": (train_idx, train),
        "validation": (val_idx, validation),
        "test": (test_idx, test),
    }

    # Check completeness and uniqueness of row assignment. These indices refer
    # to the original CSV row positions, so their union must be exactly 0..N-1.
    errors = []
    assigned_indices = [index for indices, _ in partitions.values() for index in indices]
    if len(assigned_indices) != len(df) or len(set(assigned_indices)) != len(df):
        errors.append("rows are missing from or assigned more than once across partitions")
    if set(assigned_indices) != set(range(len(df))):
        errors.append("partition row indices do not cover the complete input dataset")

    # Reassemble all partitions and compare them to the source table. This catches
    # accidental row loss, duplication, feature changes, or label changes.
    reconstructed = pd.concat([part for _, part in partitions.values()]).sort_index()
    try:
        pd.testing.assert_frame_equal(reconstructed, df.sort_index(), check_dtype=True)
    except AssertionError:
        errors.append("partition rows or disease labels differ from the original dataset")

    # Compare the integer vector IDs in each partition. Every intersection must
    # be empty; otherwise an identical symptom vector would leak across splits.
    group_sets = {
        name: set(groups[indices]) for name, (indices, _) in partitions.items()
    }
    overlap_pairs = {
        "train/validation": group_sets["train"] & group_sets["validation"],
        "train/test": group_sets["train"] & group_sets["test"],
        "validation/test": group_sets["validation"] & group_sets["test"],
    }
    for pair_name, overlapping_groups in overlap_pairs.items():
        if overlapping_groups:
            errors.append(
                f"{pair_name} share {len(overlapping_groups)} identical symptom vectors"
            )

    # Check which labels appear in every partition. The chosen split must retain
    # all classes for model fitting and validation, while its test set has 34.
    labels_by_partition = {
        name: set(part[target].unique()) for name, (_, part) in partitions.items()
    }
    for name in ("train", "validation"):
        missing = expected_labels - labels_by_partition[name]
        if missing:
            errors.append(f"{name} is missing disease labels: {sorted(missing)}")

    # Do not force the two low-diversity classes into test by breaking groups.
    # Their absence is explicit and their test metrics will be not estimable.
    actual_test_missing = expected_labels - labels_by_partition["test"]
    if actual_test_missing != expected_test_missing:
        errors.append(
            "test must omit exactly carpal tunnel syndrome and gout for the reviewed candidate; "
            f"actual omissions: {sorted(actual_test_missing)}"
        )

    # Count test support two ways. Rows show how many records are present; distinct
    # vectors show how many independent input patterns support each class. Many
    # repeated rows can still represent very few distinct test vectors.
    test_support = test.assign(_vector_group=groups[test_idx]).groupby(target).agg(
        rows=(target, "size"),
        distinct_vectors=("_vector_group", "nunique"),
    )
    if len(labels_by_partition["test"]) != 34:
        errors.append(
            f"test contains {len(labels_by_partition['test'])} diseases; expected 34"
        )
    if test_support["distinct_vectors"].lt(1).any():
        errors.append("a test-present disease has no distinct test vectors")

    # This is the gate before any output is written. If a check failed, raise an
    # error now and leave the split CSVs untouched.
    stop_if_errors(errors)

    # Build a row-level map back to the original standardized CSV. The source
    # row number is meaningful together with the standardized-file hash below.
    partition_by_row = pd.Series(index=range(len(df)), dtype="object")
    partition_by_row.iloc[train_idx] = "train"
    partition_by_row.iloc[val_idx] = "validation"
    partition_by_row.iloc[test_idx] = "test"
    split_manifest = pd.DataFrame(
        {
            "source_row_index": range(len(df)),
            "symptom_vector_group_id": groups,
            target: df[target].to_numpy(),
            "partition": partition_by_row.to_numpy(),
        }
    )
    if split_manifest["partition"].isna().any():
        raise ValueError("Split manifest does not assign every source row to a partition.")

    # Write files only after all integrity and class-support checks pass. index=False
    # prevents pandas from adding an extra, unintended CSV column.
    train.to_csv(TRAIN_FILE, index=False)
    validation.to_csv(VAL_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)

    # Read the files back from disk. The earlier checks tested the in-memory
    # partitions; these checks verify the actual CSV artifacts future code will use.
    saved_partitions = {
        "train": pd.read_csv(TRAIN_FILE, low_memory=False),
        "validation": pd.read_csv(VAL_FILE, low_memory=False),
        "test": pd.read_csv(TEST_FILE, low_memory=False),
    }
    saved_errors = []

    # Confirm each saved file has the exact model-input columns in schema order,
    # retains the expected row count, and contains the same values as its source
    # in-memory partition. The dataframe comparison includes disease labels.
    for name, saved in saved_partitions.items():
        original_partition = partitions[name][1].reset_index(drop=True)
        if saved.columns.tolist() != expected_columns:
            saved_errors.append(f"saved {name} columns or order differ from the schema")
        missing_columns = [column for column in expected_columns if column not in saved.columns]
        if missing_columns:
            saved_errors.append(f"saved {name} is missing required columns: {missing_columns}")
            continue
        if len(saved) != len(original_partition):
            saved_errors.append(f"saved {name} row count differs from its in-memory partition")
        try:
            pd.testing.assert_frame_equal(
                saved.reset_index(drop=True),
                original_partition,
                check_dtype=True,
            )
        except AssertionError:
            saved_errors.append(
                f"saved {name} rows, feature values, or disease labels differ from memory"
            )

        if saved[features].isna().any().any():
            saved_errors.append(f"saved {name} has missing feature values")
        elif not saved[features].isin([0, 1]).all().all():
            saved_errors.append(f"saved {name} has feature values other than 0 and 1")

    # Missing columns make later group and label checks impossible, so stop with
    # the clear schema error instead of raising a less helpful KeyError.
    stop_if_errors(saved_errors)

    # Rebuild exact feature-vector sets from the saved files and confirm the CSVs
    # still keep every identical vector in only one partition.
    saved_group_sets = {
        name: set(part[features].itertuples(index=False, name=None))
        for name, part in saved_partitions.items()
    }
    saved_overlap_pairs = {
        "train/validation": saved_group_sets["train"] & saved_group_sets["validation"],
        "train/test": saved_group_sets["train"] & saved_group_sets["test"],
        "validation/test": saved_group_sets["validation"] & saved_group_sets["test"],
    }
    for pair_name, overlapping_groups in saved_overlap_pairs.items():
        if overlapping_groups:
            saved_errors.append(
                f"saved {pair_name} share {len(overlapping_groups)} identical symptom vectors"
            )

    # Verify the on-disk label coverage too, rather than relying only on the
    # in-memory class checks above.
    saved_labels = {
        name: set(part[target].unique()) for name, part in saved_partitions.items()
    }
    if saved_labels["train"] != expected_labels:
        saved_errors.append("saved training labels do not contain exactly all 36 diseases")
    if saved_labels["validation"] != expected_labels:
        saved_errors.append("saved validation labels do not contain exactly all 36 diseases")
    if expected_labels - saved_labels["test"] != expected_test_missing:
        saved_errors.append("saved test labels do not omit exactly the two expected diseases")
    if sum(len(part) for part in saved_partitions.values()) != len(df):
        saved_errors.append("saved split row counts do not sum to the standardized dataset")

    # Do not proceed to reporting success unless every saved-file check passes.
    stop_if_errors(saved_errors)

    # Save and read back the row-level manifest. It allows each split row to be
    # traced to its original position, disease label, and exact symptom vector.
    split_manifest.to_csv(SPLIT_MANIFEST, index=False)
    saved_manifest = pd.read_csv(SPLIT_MANIFEST, low_memory=False)
    manifest_errors = []
    if len(saved_manifest) != len(df):
        manifest_errors.append("saved split manifest does not have one row per source row")
    if saved_manifest["source_row_index"].nunique() != len(df):
        manifest_errors.append("saved split manifest has missing or duplicate source row indices")
    if saved_manifest["partition"].isna().any():
        manifest_errors.append("saved split manifest has rows without a partition assignment")
    try:
        pd.testing.assert_frame_equal(
            saved_manifest,
            split_manifest,
            check_dtype=True,
        )
    except AssertionError:
        manifest_errors.append("saved split manifest differs from the in-memory assignments")
    stop_if_errors(manifest_errors)

    # Count conflicting vectors globally and inside each partition. A vector is
    # conflicting when its rows have more than one disease label; all rows remain.
    all_group_labels = pd.DataFrame({"group_id": groups, target: df[target].to_numpy()})
    global_group_label_counts = all_group_labels.groupby("group_id")[target].nunique()
    global_conflicting_ids = set(
        global_group_label_counts[global_group_label_counts > 1].index
    )
    global_conflicting_rows = int(pd.Series(groups).isin(global_conflicting_ids).sum())

    # Build per-partition row/vector/class/conflict summaries for the JSON report.
    partition_report = {}
    for name, (indices, _) in partitions.items():
        saved = saved_partitions[name]
        partition_group_ids = groups[indices]
        label_frame = pd.DataFrame(
            {"group_id": partition_group_ids, target: saved[target].to_numpy()}
        )
        group_label_counts = label_frame.groupby("group_id")[target].nunique()
        conflicting_ids = set(group_label_counts[group_label_counts > 1].index)
        conflicting_rows = int(label_frame["group_id"].isin(conflicting_ids).sum())
        support_frame = label_frame.groupby(target).agg(
            rows=(target, "size"),
            distinct_vectors=("group_id", "nunique"),
        )
        class_support = {
            label: {
                "rows": int(support_frame.loc[label, "rows"])
                if label in support_frame.index
                else 0,
                "distinct_vectors": int(support_frame.loc[label, "distinct_vectors"])
                if label in support_frame.index
                else 0,
            }
            for label in sorted(expected_labels)
        }
        partition_report[name] = {
            "row_count": len(saved),
            "unique_vector_count": len(saved_group_sets[name]),
            "disease_class_count": int(saved[target].nunique()),
            "conflicting_vector_count": len(conflicting_ids),
            "rows_in_conflicting_vectors": conflicting_rows,
            "class_support": class_support,
        }

    # Record enough information to reproduce and independently review this split.
    audit_report = {
        "report_type": "v2_grouped_split_audit",
        "schema_version": feature_config["schema_version"],
        "standardized_file": str(INPUT_FILE.resolve()),
        "standardized_sha256": sha256_file(INPUT_FILE),
        "feature_schema_file": str(FEATURE_SCHEMA.resolve()),
        "feature_schema_sha256": sha256_file(FEATURE_SCHEMA),
        "disease_scope_file": str(DISEASE_SCOPE.resolve()),
        "disease_scope_sha256": sha256_file(DISEASE_SCOPE),
        "split_manifest_file": str(SPLIT_MANIFEST.resolve()),
        "split_method": {
            "name": "StratifiedGroupKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "outer_test_fold": 4,
            "inner_validation_fold": 1,
        },
        "contract": {
            "row_count": len(df),
            "feature_count": len(features),
            "feature_order": features,
            "disease_count": len(expected_labels),
            "disease_order": [disease["raw_label"] for disease in disease_config["diseases"]],
        },
        "checks": {
            "input_columns_match_schema": True,
            "all_rows_assigned_once": True,
            "all_rows_and_labels_preserved": True,
            "identical_vectors_do_not_cross_partitions": True,
            "all_diseases_present_in_train_and_validation": True,
            "test_contains_34_diseases": True,
            "test_missing_labels": sorted(expected_test_missing),
            "saved_partition_csvs_read_back_and_match": True,
            "saved_features_are_binary_and_nonnull": True,
            "saved_vector_groups_do_not_overlap": True,
            "saved_manifest_matches_assignments": True,
            "training_performed": False,
        },
        "global_conflicts": {
            "conflicting_vector_count": len(global_conflicting_ids),
            "rows_in_conflicting_vectors": global_conflicting_rows,
        },
        "partitions": partition_report,
        "group_overlap_counts": {
            name: len(overlapping_groups)
            for name, overlapping_groups in saved_overlap_pairs.items()
        },
    }

    # Write the audit report only after the split CSVs and manifest have passed
    # their read-back checks. The report is machine-generated, not an LLM review.
    AUDIT_REPORT.write_text(
        json.dumps(audit_report, indent=2) + "\n",
        encoding="utf-8",
    )

    # Print a compact audit summary so the split can be reviewed after execution.
    print("V2 grouped split validation passed.")
    for name, (_, part) in partitions.items():
        print(f"{name.title()} rows: {len(part)}")
        print(f"{name.title()} unique vectors: {len(group_sets[name])}")
    print("Group overlap:", {name: len(value) for name, value in overlap_pairs.items()})
    print("Test support (rows, distinct vectors):")
    for label in sorted(expected_labels):
        if label in test_support.index:
            row = test_support.loc[label]
            print(f"  {label}: {int(row['rows'])}, {int(row['distinct_vectors'])}")
        else:
            print(f"  {label}: not represented in test")
    print("Saved split files:")
    print(TRAIN_FILE)
    print(VAL_FILE)
    print(TEST_FILE)
    print("Split manifest:", SPLIT_MANIFEST)
    print("Split audit report:", AUDIT_REPORT)
    print("Saved-file read-back checks passed.")


if __name__ == "__main__":
    main()
